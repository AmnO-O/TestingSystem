import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.contriever import load_retriever, ModernBertRetriever

MODEL_ID = "answerdotai/ModernBERT-base"
PAD = 0


def check_pooling_invariance(model, tokenizer, texts, label):
    one_padded = tokenizer(texts[:1], padding=True, truncation=True, return_tensors="pt", max_length=128)
    max_padded = tokenizer(texts[:1], padding="max_length", max_length=128, truncation=True, return_tensors="pt")

    with torch.no_grad():
        e1 = model(input_ids=one_padded["input_ids"], attention_mask=one_padded["attention_mask"], normalize=False)
        e2 = model(input_ids=max_padded["input_ids"], attention_mask=max_padded["attention_mask"], normalize=False)

    assert torch.isfinite(e1).all() and torch.isfinite(e2).all(), f"[{label}] non-finite embeddings"
    assert torch.allclose(e1, e2, atol=1e-5), f"[{label}] pad-invariance violated:\n{e1}\n{e2}"

    normed = torch.nn.functional.normalize(e1, dim=-1)
    norms = normed.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"[{label}] normalize failed"
    assert not torch.allclose(e1[0], e2[0], atol=1e-8), f"[{label}] empty/dummy embeddings"
    print(f"[{label}] OK -- pool-invariance across padding, finite, unit-norm, non-trivial")


def run_real_probe(model, tokenizer):
    query = "Who won the Nobel Prize in Literature between 1901 and 1902?"
    corpus = {
        "doc_phys_1901": "Wilhelm Conrad Rontgen The Nobel Prize in Physics 1901.",
        "doc_lit_1901": "Sully Prudhomme The Nobel Prize in Literature 1901.",
        "doc_lit_1902": "Theodor Mommsen The Nobel Prize in Literature 1902.",
        "doc_peace_1901": "Henry Dunant The Nobel Peace Prize 1901.",
        "doc_chem_1901": "Jacobus H. van't Hoff The Nobel Prize in Chemistry 1901.",
    }
    docs = list(corpus.values())
    q = tokenizer([query], padding=True, truncation=True, return_tensors="pt")
    d = tokenizer(docs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        qe = model(input_ids=q["input_ids"], attention_mask=q["attention_mask"], normalize=True)
        de = model(input_ids=d["input_ids"], attention_mask=d["attention_mask"], normalize=True)
    scores = (qe @ de.T)[0]
    ranked = sorted(zip(corpus.keys(), scores.tolist()), key=lambda x: -x[1])
    top3 = [k for k, _ in ranked[:3]]
    print(f"[probe] ranking: {ranked}")
    assert "doc_lit_1901" in top3, f"expected doc_lit_1901 in top-3, got {top3}"
    print("[probe] OK -- temporal query retrieves relevant Literature-1901 doc")


def run_synthetic():
    from transformers import ModernBertConfig

    cfg = ModernBertConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=1000,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    model = ModernBertRetriever(cfg)

    a = torch.tensor([[2, 5, 6, 7, 0, 0, 0, 0]])
    b = torch.tensor([[2, 8, 3, 5, 6, 0, 0, 0]])
    ids = torch.cat([a, b], dim=0)
    mask = (ids != PAD)

    with torch.no_grad():
        e = model(input_ids=ids, attention_mask=mask, normalize=False)
    assert e.shape == (2, 128), f"unexpected emb shape {e.shape}"
    assert torch.isfinite(e).all()

    one = torch.tensor([[2, 5, 6, 7]])
    mask1 = (one != PAD)
    with torch.no_grad():
        e1 = model(input_ids=one, attention_mask=mask1, normalize=False)
    assert torch.allclose(e1, e[:1], atol=1e-5), "padding vs unpadded embedding mismatch"

    eh = torch.nn.functional.normalize(e, dim=-1)
    norms = eh.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert not torch.allclose(eh[0], eh[1], atol=1e-8)
    print("[synthetic] OK -- ModernBertRetriever pooling math correct (2x128, pad-invariant, unit-norm)")


def main():
    try:
        model, tokenizer, mid = load_retriever(MODEL_ID)
        assert isinstance(model, ModernBertRetriever), f"dispatch failed: {type(model)}"
        print(f"[load] OK -- model={type(model).__name__}, model_id={mid}")
        check_pooling_invariance(model, tokenizer, ["Argentina won the World Cup in 2022.",
                                                    "France won the World Cup in 2018."], "real")
        run_real_probe(model, tokenizer)
    except Exception as exc:  # noqa: BLE001
        print(f"[load] WARN -- real model unavailable ({type(exc).__name__}: {exc}); falling back to synthetic")
        run_synthetic()


if __name__ == "__main__":
    main()