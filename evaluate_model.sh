#!/bin/bash

# Script to evaluate SFT-trained GPT model on test questions
# This script runs the evaluation with default parameters

echo "🚀 Starting GPT Model Evaluation..."

python score_gpt.py \
    --model_path "models/sft-models/sft-gpt-7000-step.pth" \
    --questions_file "test_questions.jsonl" \
    --output_file "evaluation_results_$(date +'%Y%m%d_%H%M%S').csv" \
    --vocab_size 50262 \
    --context_length 1024 \
    --emb_dim 512 \
    --n_heads 8 \
    --n_layers 12 \
    --drop_rate 0.1 \
    --max_tokens 200 \
    --temperature 0.3 \
    --device "auto"

echo "✅ Evaluation completed!"
