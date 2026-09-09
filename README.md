# Temporal Information Retrieval via Time-Specifier Model Merging (TSM)

This repository contains the code and resources for the paper ["Temporal Information Retrieval via Time-Specifier Model Merging"](https://arxiv.org/abs/2507.06782) (KnowFM@ACL 2025).

## Overview

The proposed TSM (Time-Specifier Model Merging) approach enhances temporal information retrieval by training separate retrieval models for different time specifiers (e.g., after, between, in early). These specialized models are then merged into a single, robust model using parameter averaging. This allows the final model to effectively handle various temporal granularities without the computational cost of maintaining multiple models at inference time.

## Model

The fine-tuned models used in the paper is available on HuggingFace.

### Contriever-based
- [seungyoonee/tsm-contriever](https://huggingface.co/seungyoonee/tsm-contriever)

### DPR-based
- [seungyoonee/tsm-dpr-question](https://huggingface.co/seungyoonee/tsm-dpr-question)
- [seungyoonee/tsm-dpr-context](https://huggingface.co/seungyoonee/tsm-dpr-context)

## Installation

### Requirements
- Python 3.11.14 or higher

1. Clone this repository:
   ```bash
   git clone https://github.com/seungyoonee/TSM.git
   cd TSM
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quickstart

To run the model with an example query and document, you can use the provided `tsm.py` script:

```bash
python tsm.py
```

This script demonstrates how to load the model and compute relevance scores for a list of documents given a query.

### Usage in Python

You can also use the `TSMModel` class in your own scripts:

```python
from tsm_model import TSMModel

# Initialize model
tsm = TSMModel("seungyoonee/tsm-contriever")

# Compute score
score = tsm.compute_score("Query text", "Document text")
print(score)
```

## ModernBERT variant (this fork)

This fork extends the contriever-based pipeline with a ModernBERT encoder
(`contriever/src/contriever.py::ModernBertRetriever`). It keeps the original
Contriever path fully working (dispatch is by model id substring).

- **Verify** (local, offline-safe): `python contriever/verify_modernbert.py`
- **Train Phase 1 subset** (`in`, `after`, `before`): `bash train/train_small.sh`
  (base model: `answerdotai/ModernBERT-base`; see `plan.md` for the phased plan)
- **Merge** the per-specifier best checkpoints: `python train/merge_models.py
  --model_paths <dir1> <dir2> ... --output_dir <merged> --mode avg`
- **Evaluate** (BEIR): `python contriever/eval_beir.py --model_name_or_path
  <merged> --beir_dir dataset/test --dataset timeqa`

Note: the per-specifier `train_*.jsonl` / `dev_*.jsonl` files are git-ignored
(regenerable). Re-create them on a fresh machine with:

```bash
cd dataset/train/TimeQA && python process_data.py   # reads ./original/*
```

## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@inproceedings{han-etal-2025-temporal,
    title = "Temporal Information Retrieval via Time-Specifier Model Merging",
    author = "Han, SeungYoon  and
      Hwang, Taeho  and
      Cho, Sukmin  and
      Jeong, Soyeong  and
      Song, Hoyun  and
      Lee, Huije  and
      Park, Jong C.",
    booktitle = "Proceedings of the 3rd Workshop on Towards Knowledgeable Foundation Models (KnowFM)",
    month = aug,
    year = "2025",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.knowllm-1.1/",
}
```
