# Walkthrough — ModernBERT-TSM migration, file-by-file

Living document. Each entry = what was changed and why.

---

## 1. `plan.md` (new)

**Why**: full build/migration plan for replacing the Contriever encoder with a
ModernBERT encoder (base `answerai/ModernBERT-base`, plain parameter-averaging
merge, phased training — small specifier subset first, runtime on Kaggle/Colab).
Recorded decisions:
- base = raw `answerai/ModernBERT-base` (retriever-oriented/contrastive bases
  kept as fallbacks if Phase-1 quality lags);
- merge = equal-weight parameter averaging (custom merger, no LM_Cocktail dep);
- training = Phase 1 subset (`in`, `after`, `before`) → gate → Phase 2 (all 7);
- environment = code/unit tests locally (CPU), train/index on Kaggle/Colab.

---

## 2. `.gitignore` (new)

**Why**: exclude the local `.venv/` (pip environment for unit tests) from the
repo.

---

## 3. `contriever/src/contriever.py`

### 3.1 Import `ModernBertModel`
`Contriever` subclasses `BertModel`; ModernBERT is not a `BertModel`, so the
import line now also pulls `ModernBertModel` from `transformers`.

### 3.2 New class `ModernBertRetriever(ModernBertModel)`
**Why**: the encoder wrapper for the ModernBERT path. Mirrors `Contriever.forward`:
- mean pooling over `last_hidden_state` with `masked_fill(~attention_mask)` so
  pad positions are excluded from the sum/divisor;
- optional L2 normalize;
- pooling mode stored on `config.pooling` (`average` | `cls`).
ModernBERT's `forward` does **not** accept `token_type_ids` / `head_mask` /
`encoder_hidden_states`, so the override only forwards supported kwargs
(`input_ids`, `attention_mask`, `position_ids`, `inputs_embeds`,
`output_attentions`, `output_hidden_states`).

### 3.3 Dispatch in `load_retriever` (both branches)
**Why**: `load_retriever` picks the encoder class by substrings in the model id.
Added `elif "modernbert" in retriever_model_id.lower():` (case-insensitive, since
the hub id is `answerai/ModernBERT-base`) in both the local `checkpoint.pth`
branch and the straight HuggingFace branch, so ModernBERT checkpoints build a
`ModernBertRetriever` instead of falling through to the BERT-backed `Contriever`.

### 3.4 Fixed `XLMRetriever` (restored its `def forward(`)
**Why**: an earlier edit accidentally deleted the `forward` definition line.
Re-added it so the class body is intact.

---

## 4. `contriever/src/inbatch.py`

**Why**: `InBatch._load_retriever` has the same string-based dispatch. Added the
`modernbert` case so in-batch contrastive models can build a
`ModernBertRetriever`.

---

## 5. `contriever/src/moco.py`

**Why**: `MoCo._load_retriever` has the same dispatch; added the `modernbert`
case so the MoCo momentum-encoder training path supports ModernBERT.

---

## 6. `contriever/verify_modernbert.py` (new)

**Why**: M1 unit verification. Runs:
- real-model path: `load_retriever("answerai/ModernBERT-base")` → asserts the
  returned model is a `ModernBertRetriever` (dispatch), embeds a padded batch
  (finite, pad-invariance: same text with different padding lengths → same
  embedding), checks normalized embeddings are unit-norm, and probes a temporal
  query against synthetic Nobel docs (relevant doc in top-3).
- synthetic fallback (offline): a small random `ModernBertRetriever`
  (hidden_size=128, 2 layers) exercising the same pooling math
  (shape, pad-invariance, unit-norm) without downloading the 574 MB checkpoint.

---

## 7. `train/merge_models.py` (planned, not yet edited)

Will be rewritten to a self-contained parameter-averaging merger (default
`--mode avg`, uniform weights). LM_Cocktail stays behind `--tool lm_cocktail`
as a cross-check.

## 8. `train/train_small.sh` (planned, not yet created)

Phase-1 driver: finetune the subset (`in`, `after`, `before`) per the
`finetuning.py` recipe, then merge via §7.

## Verification results (M1)

`python contriever/verify_modernbert.py` (local, offline, CPU):
- **Load (real model):** skipped on this machine — HF hub unreachable
  (`401`/offline). Will run on Kaggle/Colab.
- **Synthetic path: PASS** — a tiny random `ModernBertRetriever` (hidden 128,
  2 layers) produces correct shapes, is **pad-invariant** (longer padding does
  not change the embedding), finite, and `normalize=True` yields unit-norm rows.
  This validates the pooling math and the class wiring end to end.
- Fix applied during verification: synthetic `ModernBertConfig` needed explicit
  `pad_token_id=0` (ModernBERT defaults 50283, out of range for a tiny vocab).

## 9. `train/merge_models.py` (rewritten)

**Why**: replace the LM_Cocktail-only merger with our own parameter-averaging
merger (decision: plain equal-weight average). Custom path:
- loads each best-model `checkpoint.pth` state dict (CPU),
- `--mode avg`: `merged = Σ w_i * sd_i` (default uniform),
- `--mode task_vector`: `merged = base + Σ w_i * (sd_i − base)` (needs `--base_model`),
- saves HF-format `pytorch_model.bin` + `config.json` + tokenizer + a
  `checkpoint.pth` (with `opt.retriever_model_id` containing `"modernbert"`) so
  `load_retriever` can dispatch to `ModernBertRetriever`.
- `--tool lm_cocktail` keeps the old LM_Cocktail path for cross-checking.
- Tested end-to-end offline on synthetic tiny ModernBERT checkpoints.

## 10. `KaggleModernBert.ipynb` (new)

**Why**: a single, self-contained Kaggle notebook so the whole Phase-1 run is
"upload and Run All" — no shell babysitting. Cells (12):
0-1 md intro/config; 3 clone repo; 4 upgrade transformers + patch
`utils.save` (guard `tokenizer.save_vocabulary`, which may be missing in newer
transformers); 5 regenerate git-ignored `train_/dev_*.jsonl` via
`process_data.py` (skips if present); 8 train the `in/after/before` subset with
resumable per-specifier checkpoints (`RESUME` skips existing `best_model`);
9 parameter-average merge (`train/merge_models.py --mode avg`);
10 BEIR eval `timeqa` + `nobel_prize`; 11 copy artifacts to `/kaggle/output`.
Config dict at the top (epochs, batch, specifiers, flags). Smoke-test hint:
`TOTAL_EPOCHS=1, SPECIFIERS=["in"]`. Requires selector **Internet ON** + **GPU**.

<!-- Append new entries at the end as work progresses. -->