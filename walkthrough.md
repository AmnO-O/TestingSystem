# Walkthrough — ModernBERT-TSM migration, file-by-file

Living document. Each entry = what was changed and why.

---

## 1. `plan.md` (new)

**Why**: full build/migration plan for replacing the Contriever encoder with a
ModernBERT encoder (base `answerdotai/ModernBERT-base`, plain parameter-averaging
merge, phased training — small specifier subset first, runtime on Kaggle/Colab).
Recorded decisions:
- base = raw `answerdotai/ModernBERT-base` (retriever-oriented/contrastive bases
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
the hub id is `answerdotai/ModernBERT-base`) in both the local `checkpoint.pth`
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
- real-model path: `load_retriever("answerdotai/ModernBERT-base")` → asserts the
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

## 11. Per-epoch logging in `contriever/finetuning.py`

**Why**: the stock loop only logged every `--log_freq` steps and dev-evaluated
every `--eval_freq` steps, so nothing fired reliably at epoch boundaries. Now,
when training by epochs (`opt.total_epochs > 0` — our recipe), at the end of
every epoch `finetuning()`:
- averages that epoch's rollups into a `[epoch N] train loss: .. | train
  accuracy: ..` line (also pushed to TB as `epoch/train/...`),
- runs the dev `evaluate()` and prints `eval acc / eval mrr`,
- feeds that dev acc into the `best_model` selection (so best-model tracking is
  per-epoch rather than only every `eval_freq` steps).

No new CLI flags; the Kaggle notebook's "Train per specifier" md notes this.

## 12. BEIR dependency fix (Kaggle failure)

**Symptom**: first smoke run died in 20s with
`ModuleNotFoundError: No module named 'beir'` — `contriever/finetuning.py` imports
`src/beir_utils.py`, which imports the `beir` package unconditionally; the
notebook never installed it.

**Root cause (worse than just a missing package)**: root `requirements.txt`
pinned `beir==2.2.0`, but `src/beir_utils.py` only works with the **beir 1.x**
layout (`beir.util`, `beir.datasets.data_loader`, `beir.retrieval.search.dense`,
`beir.reranking`). PyPI `beir` 1.0.0 was verified to ship all those modules and
its distribution carries **no `requires_dist`** (so `pip` won't downgrade
`transformers`/`torch`). `beir==2.2.0` is an incompatible rewrite.

**Fix**: root `requirements.txt` re-pinned to `beir==1.0.0`; notebook install
cell now runs
`pip install -q -U transformers tokenizers safetensors beir==1.0.0 faiss-cpu`
(`faiss-cpu` for the dense-search eval path, preinstalled on Kaggle anyway).

**Second failure (numpy 2)**: with `beir` installed, the next crash was
`beir/retrieval/evaluation.py` → `.search.lexical.BM25Search` →
`elasticsearch` package → `np.float_` removed in NumPy 2 (Kaggle ships numpy
2.x + a numpy-2-incompatible `elasticsearch` client). The dense retrieval
pipeline never uses BM25/ES, so `src/beir_utils.py` now **stubs the
`elasticsearch` module in `sys.modules` before any `beir` import** (verified
locally under numpy 2.5: the ES import error disappears; the stub is never
instantiated). `pytrec_eval` and `faiss` are preinstalled on Kaggle (beir
1.0.0 ships no `requires_dist` metadata, so pip does not pull them); notebook
install cell also adds `pytrec-eval` defensively.

**Third failure (stub not package-like)**: with the plain-module stub, the very
same `beir.retrieval.search.lexical.elastic_search` hit
`ModuleNotFoundError: No module named 'elasticsearch.helpers'; 'elasticsearch'
is not a package`. It imports BOTH `from elasticsearch import Elasticsearch`
AND `from elasticsearch.helpers import streaming_bulk`, so the stub must look
like a package: `_fake_es.__path__ = []` plus a registered
`elasticsearch.helpers` submodule exposing `streaming_bulk`. Verified locally
(numpy 2.5): the lexical chain now imports cleanly; the stub is only ever
pulled for BM25/ES, which the dense pipeline never invokes.

**Fourth failure (datasets + find_spec)**: next, `beir_utils.py` imported
`beir.reranking.models.CrossEncoder` eagerly → `sentence_transformers` →
`datasets`, and `datasets/search.py` calls `importlib.util.find_spec("elasticsearch")`
which raises `ValueError: elasticsearch.__spec__ is None` for our hand-built
stub module. Two fixes in `src/beir_utils.py`:
1. **Deleted the unused `beir.reranking` imports** (`CrossEncoder`, `Rerank`
   were imported but never referenced) — removes the whole
   `sentence_transformers`→`datasets` chain from the training import path.
2. Give the stub a real `__spec__` (`importlib.util.spec_from_loader`) on both
   `elasticsearch` and `elasticsearch.helpers`, so `find_spec` probes return a
   spec instead of raising.

Verified locally (mirroring beir_utils import order — `torch` first, then
stub, then beir): `find_spec` OK, lexical/dense/evaluation/data_loader chains
all import. `beir.util` needs `requests`/`tqdm` (beir 1.0.0 ships no
`requires_dist`) — both already preinstalled on Kaggle.

**Fifth failure (wrong base-model id)**: with imports fixed, training finally
ran but died before the first step: `401 Unauthorized` /
`RepositoryNotFoundError` downloading `answerai/ModernBERT-base`. The official
Hub repo id is **`answerdotai/ModernBERT-base`** (owner `answerdotai`).
Renamed in all six places: notebook CONFIG, `README.md`, `plan.md`,
`walkthrough.md`, `train/train_small.sh`, `contriever/verify_modernbert.py`.
The retriever dispatch (`"modernbert" in model_id.lower()`) is unaffected.

<!-- Append new entries at the end as work progresses. -->