# Plan: ModernBERT-based TSM (Time-Specifier Model Merging)

Replaces the BERT-based Contriever encoder in the TSM retrieval pipeline with a
ModernBERT-based encoder, built and merged by ourselves. Scope: code changes,
training, merging, retrieval, and evaluation.

Baseline: published `seungyoonee/tsm-contriever` (KnowFM@ACL 2025), trained by
finetuning `facebook/contriever` per time-specifier then merging checkpoints via
parameter averaging.

---

## 1. Goals

- **Primary**: a ModernBERT-based dense retriever for temporal QA, trained per
  time-specifier (`after, before, between_and, from_to, in, in_early, in_late`)
  and merged into one checkpoint.
- **Secondary**: keep the existing codebase functional (backward compatible) so
  the Contriever path still works, and reuse as much infrastructure as possible
  (MoCo/InBatch, BEIR eval, FAISS retrieval, data generation).
- Open decision: starting checkpoint (see §5). We control the whole pipeline, so
  no dependency on released TSM checkpoints.

---

## 2. Environment setup (first milestone)

On this machine (`win32`, Python 3.12.10, no packages installed):

1. Create a venv: `python -m venv .venv`
2. `pip install -r requirements.txt`
   - Heavy: torch 2.9.1 + CUDA wheels (~several GB). If no GPU, install a
     CPU-only torch instead for logic/unit-test phases.
3. Verify: `python -c "from transformers import ModernBertModel; print('ok')"`
4. GPU check: `python -c "import torch; print(torch.cuda.is_available())"` —
   fp16 indexing/training steps require GPU.

**Runtime decision: train/index on Kaggle or Google Colab** (GPU, torch +
transformers pre-installed). Implications:
- Repo (with `contriever/`, `train/`, `dataset/train/TimeQA`) is uploaded or
  git-cloned into the notebook session; install only the missing extras
  (`pip install -r requirements.txt` or subsets).
- Notebook sessions are ephemeral and time-limited → run each of the 7
  specifier trainings as its **own** session/notebook (script entry point
  `finetuning.py`), persisting each best checkpoint immediately to HuggingFace
  Hub (or Drive/`/kaggle/working`) before the session ends.
- Merging + eval are cheap → run in one final session after all specifier
  checkpoints are collected.
- Verify `finetuning.py`/`passage_retrieval.py` don't assume SLURM env
  (`slurm.py` already falls back to single-rank local when `SLURM_JOB_ID` absent).

---

## 3. Architecture changes (code)

All under `contriever/`. Kept alongside existing BERT classes for compat.

### 3.1 New encoder class — `ModernBertRetriever`

File: `contriever/src/contriever.py`

- Add `class ModernBertRetriever(ModernBertModel)` mirroring `Contriever.forward`:
  - `forward(input_ids, attention_mask, position_ids, inputs_embeds,
    output_attentions, output_hidden_states, normalize=False)` — note ModernBERT
    does NOT accept `token_type_ids` / `head_mask` / `encoder_*` kwargs, so the
    override must not forward them.
  - Mean pooling over last hidden state with `masked_fill(~attention_mask)`,
    optional L2 normalize — same pattern as `Contriever` (`contriever.py:45-55`).
    ModernBERT internally unpads/repads and zero-fills pad rows; masking + mean
    over the mask is still correct (pad positions excluded from the sum/divisor),
    but must be verified (§4).
- Set `retriever.config.pooling = "average"` on load (ModernBertModel may not
  have a `pooling` attr; `setattr` guard).

### 3.2 Model-class dispatch (string based)

Dispatch currently keys off the model id substring
(`contriever.py:136-143`; also in `inbatch.py:36-39`, `moco.py:48-51`):

```python
if "xlm" in model_id:          -> XLMRetriever
elif "dpr-question" in id:     -> DPRQuestionEncoderWrapper
elif "dpr-context" in id:      -> DPRContextEncoderWrapper
elif "modernbert" in model_id: -> ModernBertRetriever   # NEW
else:                          -> Contriever
```

- Apply in `load_retriever` (both the local `checkpoint.pth` branch and the HF
  branch) and in `InBatch._load_retriever` / `MoCo._load_retriever`.
- Also handle **local config path**: `load_retriever` builds `cfg` via
  `AutoConfig.from_pretrained(retriever_model_id)`. For merged checkpoints saved
  locally, prefer loading `config.json` from the checkpoint dir so `"modernbert"`
  is present in the id string used for dispatch.

### 3.3 Tokenizer differences

- `ModernBertTokenizer` (BPE, ~50k vocab) already defines bos/eos/pad token ids,
  no `token_type_ids`. No code change needed in `data.py add_bos_eos`
  (`data.py:212-223`) — it keys off `tokenizer.bos_token_id` which exists, and
  the `"bert-"` fallback branch (`inbatch.py:46`, `moco.py:58`) is simply skipped.
- `batch_encode_plus(..., add_special_tokens=True)` works; verify ModernBERT
  accepts `padding=True, truncation=True` with `max_length` (it does).
- Keep `chunk_length`/`passage_maxlength` at defaults initially (256 / 512);
  ModernBERT supports 8k context — revisit as an optional quality bump later.

### 3.4 `tsm_model.py`

No structural change required: it calls `load_retriever` + `model(**inputs,
normalize=True)` (input_ids + attention_mask only) → works with
`ModernBertRetriever`. Verify via §4.

---

## 4. Unit verification (before any training)

Script: `contriever/verify_modernbert.py` (new, temporary):
1. Load `answerai/ModernBERT-base` through `load_retriever` → assert returned
   model is `ModernBertRetriever`.
2. Mean-pooling correctness: batch of 2 short encodings padded to same length;
   assert no NaN, and that a padded row's contribution is excluded (compare
   mean-pool vs. explicit per-row average over nonzero-masked tokens).
3. `normalize=True` returns unit-norm rows (float32 approx).
4. Compare intuition on a tiny retrieval case (query vs. gold/non-gold docs)
   from `dataset/test/timeqa`.

Acceptance: assertions pass; TSM scoring via `TsMModel.compute_score` runs end
to end.

---

## 5. Base-model decision (open, requires user confirmation)

`facebook/contriever` is itself **contrastively pretrained**; raw MLM-only
ModernBERT is not a strong retriever out of the box. Options:

- **(A) Raw start** *(DECIDED)*: finetune directly from
  `answerai/ModernBERT-base` with the existing hard-negative fine-tuning recipe.
  Simplest; may need many negatives / longer training to compensate for no
  contrastive pretraining.
- **(B) Retriever-oriented base** *(recommended, if task allows)*: start from an
  MLM-tuned ModernBERT embedding checkpoint (e.g. GTE-ModernBERT family), which
  is already a strong dense retriever. Caveat: such checkpoints often use
  non-mean pooling (a pooling token / "query is last token" convention) and may
  change the pooling head — requires validating §3.1 mean-pool choice or
  adapting the pooling to the checkpoint's convention.
- **(C) Two-phase**: reuse our own MoCo/InBatch contrastive pretraining
  (`moco.py`, `inbatch.py`) on ModernBERT to reproduce a "contriever-style"
  base, then per-specifier finetune. Most faithful to the original; most work.

**Recommendation**: start with (A) to validate the pipeline cheaply; if quality
lags baseline, move to (B) or (C). **Decision confirmed: (A) — raw
`answerai/ModernBERT-base`.**

---

## 6. Training per specifier

Reuse `contriever/finetuning.py` + `finetuning_data.py` unchanged
(encoder-agnostic; only calls `get_encoder()` then the encoder).

- Data: `dataset/train/TimeQA/{train,dev}_{specifier}.jsonl` (already generated
  by `process_data.py`; regenerate if needed).
- Recipe (mirror `train/train_and_merge.sh`):
  - `--model_path <base per §5>`
  - `--train_data .../train_{spec}.jsonl` / `--eval_data .../dev_{spec}.jsonl`
  - `--negative_ctxs 5 --negative_hard_ratio 1.0`
  - `--total_epochs 5 --eval_freq 50 --log_freq 10`
  - `--output_dir train/model/modernbert_{spec}`
  - `--per_gpu_batch_size 16 --per_gpu_eval_batch_size 64`
- Checkpointing: `finetuning.py` saves best model by dev accuracy/MRR to
  `<out>/checkpoint/best_model` via `utils.save` (checkpoint.pth + HF format) —
  reused unchanged.

**Phased training (DECIDED — small subset first, not all 7 at once):**

- **Phase 1 (subset):** train a small subset of specifiers end-to-end first to
  validate the whole loop (train → merge → eval) cheaply.
  - Recommended subset: `in`, `after`, `before` (3 specifiers — covers a point
    in time and forward/backward single anchors; all present in the TimeQA data).
    Subset is configurable (see `train/train_small.sh` below).
  - Gate: merged subset model evaluated on `nobel_prize` + `timeqa` (§9); if the
    pipeline behaves and quality is reasonable → Phase 2.
- **Phase 2 (full):** remaining specifiers (`between_and`, `from_to`,
  `in_early`, `in_late`) added, all 7 merged into the final TSM checkpoint.

New script `train/train_small.sh`: same content as `train_and_merge.sh` but the
`SPECIFIERS` list is the Phase-1 subset (edit the file to pick the subset). The
full run remains `train_and_merge.sh`.

Notes:
- `utils.save` stores `opt` including `retriever_model_id` — ensure it contains
  `"modernbert"` for correct re-dispatch upstream.
- SAM optimizer is already wired (`optim == "sam"/"asam"` in `finetuning.py`),
  optional.

---

## 7. Merging (by ourselves)

`train/merge_models.py` currently delegates to `LM_Cocktail.mix_models`
(`model_type='encoder'`). Two paths:

### 7.1 Custom merger (primary, preferred)
Write our own `train/merge_models.py` implementing **parameter averaging** of the
7 best-model state dicts — **DECIDED: plain equal-weight averaging is the
default mode**:

```
    merged = 1/N * sum(state_dict_i)   # arg-namespace identical across checks
```

- Deterministic, no LM_Cocktail dependency or ModernBERT-compat question.
- Equal weights by default (mirrors `weights = [1/N]*N`); accept `--weights`
  option for future task-vector / weighted variants.
- Also expose `--mode task_vector` (start from base model, add averaged deltas)
  as a fallback.
- Save merged encoder via `utils.save`-compatible layout: `config.json`,
  `tokenizer` files, `pytorch_model.bin`, plus `checkpoint.pth` with
  `opt.retriever_model_id` containing `"modernbert"`.

### 7.2 LM_Cocktail (optional validation)
Keep `mix_models` path behind `--tool lm_cocktail` to cross-check against the
custom merger. Open risk: LM_Cocktail 0.0.4 may assume BERT-style
`AutoModelForMaskedLM`/pooler config; verify it loads a ModernBERT checkpoint; if
not, drop this path.

### Acceptance
Merged checkpoint loads via `load_retriever` and produces stable embeddings
(no NaN; unit-norm rows correlate with per-specifier members on a few probes).

---

## 8. Retrieval / indexing

Unchanged scripts, they call `load_retriever`:
- `generate_passage_embeddings.py` — corpus embedding, sharded, fp16.
  - ModernBERT fp16 is fine; verify no unstable `_compute_loss` floats (fp16
    underflow over long sequences is possible → keep fp32 option `--no_fp16`
    if needed).
- `passage_retrieval.py` — FAISS `IndexFlatIP` (or `IndexPQ`) search → top-k ctxs
  + `hasanswer`.
- Optional: raise `question_maxlength` / `passage_maxlength` (512) using
  ModernBERT's 8k context for longer temporal passages; measure recall impact.

---

## 9. Evaluation

BEIR setup already present:
- Query/corpus/qrels: `dataset/test/nobel_prize`, `dataset/test/timeqa`.
- `test/test_beir_tsm-contriever.sh` → adapt to our merged checkpoint
  (`eval_beir.py --model_name_or_path <merged>`).
- Metrics: nDCG@10, Recall@5/10/20/100, MRR; plus R@k from
  `evaluate_retrieved_passages.py`.

Comparison table target:
| system | timeqa R@10 | nobel_prize R@10 | nDCG@10 |
|---|---|---|---|
| facebook/contriever (base) | ... | ... | ... |
| seungyoonee/tsm-contriever (baseline) | ... | ... | ... |
| Our ModernBERT-TSM | ... | ... | ... |

Decision gate: if ModernBERT-TSM < baseline on both datasets, revisit §5 base
choice before shipping.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Raw MLM ModernBERT is a weak retriever start | Contrastive pretraining phase (C) or retriever-oriented base (B) in §5 |
| Mean pooling vs. ModernBERT repadding surprise | §4 unit test; fall back to CLS/First-token pooling if mean shows dilution |
| `load_retriever`/dispatch breaks for saved ModernBERT checkpoints | Ensure `retriever_model_id` contains `"modernbert"`; local `config.json` path in §3.2 |
| LM_Cocktail incompatibility | Primary custom merger (§7.1); LM_Cocktail optional only |
| fp16 underflow / NaN on ModernBERT | `--no_fp16` fallback; add fp16 sanity check in §4 |
| Heavy install / no GPU on this machine | CPU torch for unit tests; training/indexing steps assume GPU box |

---

## 11. Milestones & task breakdown

1. **M0 Setup** — venv, install deps, verify imports (§2).
2. **M1 Code** — `ModernBertRetriever` + dispatch in `contriever.py`,
   `inbatch.py`, `moco.py` (§3); verify script (§4) passing.
3. **M2 Dry-run** — 1 specifier, 1–2 epoch on sample data; validate
   finetuning loop + best-model saving.
4. **M3 Phase 1 (subset)** — train small subset (`in`, `after`, `before`) via
   `train/train_small.sh`; merge; evaluate. **Gate → Phase 2 if quality ok.**
5. **M3b Phase 2 (full)** — train remaining 4 specifiers; final 7-specifier
   merge + evaluation.
6. **M4 Merge** — custom merger; load + embed sanity checks (§7).
7. **M5 Retrieval & eval** — corpus embed, index, BEIR eval on
   nobel_prize/timeqa; comparison table (§9).
8. **M6 Doc/cleanup** — remove temp verify script or keep as test under
   `contriever/`; update README with ModernBERT variant; record final numbers.

## 12. Decisions (recorded)

1. **Base checkpoint: (A) `answerai/ModernBERT-base`** — direct finetune from
   raw MLM model; revisit (B) retriever-oriented base or (C) our own contrastive
   pretraining only if quality lags baseline at M5.
2. **Merge mode: plain equal-weight parameter averaging**; task-vector kept as a
   `--mode` fallback.
3. **Backward compat: yes** — `facebook/contriever` path + released checkpoints
   stay working; ModernBERT code is additive.
4. **Runtime: Kaggle / Google Colab** for training + indexing (see §2); local
   machine used for M0–M1 code + unit verification.