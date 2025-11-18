#!/bin/bash

#SBATCH --job-name=pretrain
#SBATCH --account=ling702w25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=2
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# The application(s) to execute along with its input arguments and options:

module load cuda
# module load pytorch

source ~/.bashrc
conda activate CSE595

# EECS 595 HW3: Full GPT Training Script
# This script trains the GPT model with production-ready hyperparameters
# Designed for use on Great Lakes cluster with GPU resources

echo "Starting full GPT training..."
echo "=================================="

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="eecs595-gpt-pretraining"
export DATA_PATH="Data/"
export OUTPUT_DIR="models/pretrained-models/"
export TOKENIZERS_PARALLELISM=false

# Use these hyperparameters for your full pretraining

# Training hyperparameters for full model
python pretrain_gpt.py \
    --batch_size 16 \
    --learning_rate 6e-4 \
    --max_epochs 2 \
    --emb_dim 512 \
    --n_layers 12 \
    --n_heads 8 \
    --context_length 1024 \
    --save_every 1000 \
    --eval_every 500 \
    --device cuda \
    --data_path $DATA_PATH/fineweb-edu-sample-1B-hf/ \
    --data_format arrow \
    --output_dir $OUTPUT_DIR \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "gpt-pretraining-$(date +%Y%m%d-%H%M%S)" \
    --eval_data_path $DATA_PATH/fineweb-edu-eval-3M.jsonl.gz \
    --eval_data_format jsonl

echo "Training completed!"
echo "Check the output directory for saved models and logs."
