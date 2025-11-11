#!/bin/bash

# EECS 595 HW3: Tiny GPT Training Script
# This script trains a very small GPT model for local testing
# Designed to run quickly on CPU for student verification

echo "Starting tiny GPT training for local testing..."
echo "=============================================="

# Training hyperparameters for tiny model (CPU-friendly)
python pretrain_gpt.py \
    --batch_size 4 \
    --learning_rate 1e-3 \
    --max_epochs 1 \
    --emb_dim 32 \
    --n_layers 2 \
    --n_heads 4 \
    --context_length 64 \
    --drop_rate 0.0 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 1 \
    --warmup_steps 10 \
    --max_docs 100 \
    --stride 64 \
    --save_every_n_steps 50 \
    --eval_every_n_steps 25 \
    --device cpu \
    --mixed_precision false \
    --compile_model false \
    --data_dir ./data/fineweb-edu-sample-1M.jsonl.gz \
    --output_dir ./models/tiny/ \
    --wandb_run_name "gpt-tiny-test-$(date +%Y%m%d-%H%M%S)"

echo "Tiny training completed!"
echo "This should run in a few minutes on CPU."
echo "Check ./models/tiny/ for the saved model."
