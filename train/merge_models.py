import os
import copy
import json
import shutil
import argparse
import logging
import types
from collections import OrderedDict

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checkpoint(path):
    checkpoint_path = os.path.join(path, "checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"No checkpoint.pth in {path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt["model"], ckpt.get("opt")


def average_state_dicts(model_paths, weights=None):
    if not model_paths:
        raise ValueError("No model paths provided")

    n = len(model_paths)
    if weights is None:
        weights = [1.0 / n] * n
    assert len(weights) == n, "weights must match number of model paths"

    acc = None
    for path, w in zip(model_paths, weights):
        sd, _ = load_checkpoint(path)
        if acc is None:
            acc = OrderedDict((k, torch.zeros_like(v)) for k, v in sd.items())
        for k, v in sd.items():
            acc[k].add_(v, alpha=w)

    return acc


def task_vector_state_dicts(model_paths, base_model, weights=None):
    base_sd, _ = load_checkpoint(base_model)
    n = len(model_paths)
    if weights is None:
        weights = [1.0 / n] * n
    assert len(weights) == n

    merged_sd = OrderedDict((k, torch.zeros_like(v)) for k, v in base_sd.items())
    for path, w in zip(model_paths, weights):
        sd, _ = load_checkpoint(path)
        for k in merged_sd:
            merged_sd[k].add_(sd[k] - base_sd[k], alpha=w)
    for k in merged_sd:
        merged_sd[k].add_(base_sd[k])
    return merged_sd


def merge_models(model_paths, output_dir, weights=None, mode="avg", base_model=None):
    logger.info(f"Merging {len(model_paths)} models ({mode}) -> {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if mode == "avg":
        merged_sd = average_state_dicts(model_paths, weights)
    elif mode == "task_vector":
        if base_model is None:
            raise ValueError("--base_model is required for task_vector mode")
        merged_sd = task_vector_state_dicts(model_paths, base_model, weights)
    else:
        raise ValueError(f"Unknown merge mode: {mode}")

    ref_path = model_paths[0]
    retriever_model_id = None
    _, ref_opt = load_checkpoint(ref_path)
    if ref_opt is not None and hasattr(ref_opt, "retriever_model_id"):
        retriever_model_id = ref_opt.retriever_model_id

    torch.save(merged_sd, os.path.join(output_dir, "pytorch_model.bin"))

    config_src = os.path.join(ref_path, "config.json")
    if os.path.exists(config_src):
        shutil.copy(config_src, os.path.join(output_dir, "config.json"))

    for name in ("tokenizer_config.json", "special_tokens_map.json", "tokenizer.json",
                 "vocab.txt", "merges.txt", "tokenizer.model"):
        src = os.path.join(ref_path, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, name))

    opt = types.SimpleNamespace(retriever_model_id=retriever_model_id)
    checkpoint = {
        "step": 0,
        "model": merged_sd,
        "optimizer": None,
        "scheduler": None,
        "opt": opt,
    }
    torch.save(checkpoint, os.path.join(output_dir, "checkpoint.pth"))

    logger.info(f"Merged checkpoint saved to {output_dir} "
                f"(retriever_model_id={retriever_model_id})")


def merge_with_lm_cocktail(model_paths, output_dir, weights=None):
    from LM_Cocktail import mix_models

    os.makedirs(output_dir, exist_ok=True)
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)

    mix_models(
        model_names_or_paths=model_paths,
        model_type="encoder",
        weights=weights,
        output_path=output_dir,
    )
    logger.info("LM_Cocktail merge complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs="+", required=True, help="List of model directories to merge")
    parser.add_argument("--output_dir", required=True, help="Output directory for merged model")
    parser.add_argument("--mode", choices=["avg", "task_vector"], default="avg",
                        help="avg: weighted parameter average; task_vector: averaged deltas added to base")
    parser.add_argument("--base_model", default=None, help="Base model dir (required for task_vector mode)")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Optional per-model weights (default: uniform)")
    parser.add_argument("--tool", choices=["custom", "lm_cocktail"], default="custom",
                        help="Merge implementation to use")
    args = parser.parse_args()

    if args.tool == "lm_cocktail":
        merge_with_lm_cocktail(args.model_paths, args.output_dir, args.weights)
    else:
        merge_models(args.model_paths, args.output_dir, args.weights, args.mode, args.base_model)