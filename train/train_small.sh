#!/bin/bash
set -e

DATA_PATH="../dataset/train/TimeQA"
SPECIFIERS="in after before"
TRAINED_MODEL_PATH="./model"
MERGED_MODEL_PATH="./model/merged_model_small"

MODEL_PATHS=""

for specifier in $SPECIFIERS; do
    echo "Training for specifier: $specifier"
    OUTPUT_DIR="$TRAINED_MODEL_PATH/modernbert_$specifier"

    python ../contriever/finetuning.py \
        --model_path answerdotai/ModernBERT-base \
        --train_data $DATA_PATH/train_$specifier.jsonl \
        --eval_data $DATA_PATH/dev_$specifier.jsonl \
        --negative_ctxs 5 \
        --negative_hard_ratio 1.0 \
        --total_epochs 5 \
        --eval_freq 50 \
        --log_freq 10 \
        --output_dir $OUTPUT_DIR \
        --per_gpu_batch_size 16 \
        --per_gpu_eval_batch_size 64

    MODEL_PATHS="$MODEL_PATHS $OUTPUT_DIR/checkpoint/best_model"
done

echo "Training complete. Merging models..."
echo "Model paths: $MODEL_PATHS"

python ./merge_models.py \
    --model_paths $MODEL_PATHS \
    --output_dir $MERGED_MODEL_PATH

echo "Merging complete. Merged model saved to $MERGED_MODEL_PATH"