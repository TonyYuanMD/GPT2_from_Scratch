#!/bin/bash

# Script to run SFT training on a tiny dataset for local verification.
# This configuration is designed to run quickly on a CPU for a few steps.

echo "Starting SFT training on tiny dataset for local verification..."

python sft_gpt.py \
    --train_data_path "data/smol-smoltalk-train.jsonl.gz" \
    --val_data_path "data/smol-smoltalk-dev.jsonl.gz" \
    --model_path "models/tiny/gpt.10M.1-epoch.model.pth" \
    --max_docs 100 \
    --context_length 64 \
    --emb_dim 32 \
    --n_heads 4 \
    --n_layers 2 \
    --drop_rate 0.0 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_epochs 1 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 10 \
    --output_dir "models/tiny-sft/" \
    --save_every 50 \
    --eval_every 25 \
    --wandb_project "gpt-sft-tiny-local" \
    --device "cpu" \
    --num_workers 0 \
    --seed 42

echo "SFT training on tiny dataset finished."
echo "This should run in a few minutes on CPU."
echo "Check models/tiny-sft/ for the saved model."
