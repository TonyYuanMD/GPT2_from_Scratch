#!/bin/bash

#SBATCH --job-name=sft
#SBATCH --account=ling702w25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# The application(s) to execute along with its input arguments and options:

module load cuda

source ~/.bashrc
conda activate CSE595

# EECS 595 HW3: Full GPT Training Script
# This script trains the GPT model with production-ready hyperparameters
# Designed for use on Great Lakes cluster with GPU resources

echo "Starting SFT training on full dataset..."
echo "=================================="

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="eecs595-gpt-sft"
export DATA_PATH="Data/"
export MODEL_PATH="/scratch/eecs595f25_class_root/eecs595f25_class/yhongda/models/pretrained-models"
export TOKENIZERS_PARALLELISM=false

# Use these hyperparameters for your full SFT training
python 'sft_gpt(1).py' \
    --train_data_path $DATA_PATH/sft_data_packed.arrow \
    --val_data_path $DATA_PATH/smol-smoltalk-dev.jsonl.gz \
    --model_path $MODEL_PATH/gpt-pretraining-20251117-233818/model_step_3000.pth \
    --context_length 1024 \
    --emb_dim 512 \
    --n_heads 8 \
    --n_layers 12 \
    --drop_rate 0.1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --max_epochs 3 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 100 \
    --output_dir "/scratch/eecs595f25_class_root/eecs595f25_class/yhongda/models/sft-models/" \
    --save_every 1000 \
    --eval_every 500 \
    --wandb_project "gpt-sft-full" \
    --device "cuda" \
    --num_workers 4 \
    --seed 42 \
    --train_data_format arrow \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "gpt-sft-$(date +%Y%m%d-%H%M%S)" \
    --val_data_format jsonl

echo "SFT training on full dataset finished."
