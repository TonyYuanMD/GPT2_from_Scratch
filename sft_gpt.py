"""
EECS 595 HW3: SFT Training Script

This script contains the complete training loop for supervised fine-tuning (SFT) of GPT models.
Students need to implement the core components in sft.py before running this script.

Usage:
    python sft_gpt.py

The script will:
1. Load a pre-trained GPT model
2. Load SFT conversation data
3. Fine-tune the model with masked loss computation
4. Save checkpoints and log to wandb

TODO: Students need to implement the following components in sft.py:
- SFTDataset: Load and format conversational data with proper token masking
- SFTDatasetFast: Fast tokenization version for better performance
- Data collators: Handle batching for SFT training
- Generation functions: Conversational text generation
- Utility functions: Model loading and validation
"""

import os
import json
import math
import numpy as np
import random
import logging
import argparse
from typing import Optional, Callable, List, Tuple, Dict, Any, Iterable
from copy import deepcopy
import gzip

# PyTorch imports
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn import RMSNorm
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Transformers and tokenization
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, default_data_collator

# Data handling
from datasets import load_from_disk
import orjson
import datetime

# Progress tracking
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import wandb

# Import our implementations
import gpt
import sft
from gpt import setup_tokenizer as gpt_setup_tokenizer

# Set CuPy/CUDA to allow TF32 computations
# This can provide a speedup on compatible GPUs (RTX 4000 series, etc.)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GPT model with SFT')

    # Data arguments
    parser.add_argument('--train_data_path', type=str,
                       default='./Data/sft_data_packed.arrow/',
                       help='Path to training data')
    parser.add_argument('--train_data_format', type=str, choices=['jsonl', 'arrow'], default='jsonl',
                       help='Format of training data: jsonl (for .jsonl/.gz files) or arrow (for arrow datasets)')
    parser.add_argument('--val_data_format', type=str, choices=['jsonl', 'arrow'], default='jsonl',
                       help='Format of validation data: jsonl (for .jsonl/.gz files) or arrow (for arrow datasets)')
    parser.add_argument('--model_path', type=str,
                       default='./models/sfted-models/',
                       help='Path to pre-trained model')

    # Validation arguments
    parser.add_argument('--val_data_path', type=str,
                       default='./Data/smol-smoltalk-dev.jsonl.gz',
                       help='Path to validation data')
    parser.add_argument('--eval_max_docs', type=int, default=None,
                       help='Maximum number of documents to load for validation (only for raw text)')
    parser.add_argument('--eval_max_docs_step', type=int, default=None,
                       help='Maximum number of validation documents to use during step evaluation (None = use all)')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                       help='Validation batch size')

    # Model arguments
    parser.add_argument('--context_length', type=int, default=1024,
                       help='Context length')
    parser.add_argument('--emb_dim', type=int, default=512,
                       help='Embedding dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--drop_rate', type=float, default=0.1,
                       help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=3,
                       help='Maximum number of epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Warmup steps')

    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Evaluation arguments
    parser.add_argument('--eval_every', type=int, default=500,
                       help='Evaluate every N steps')


    # Logging and saving
    parser.add_argument('--output_dir', type=str,
                       default='./models/sft-models/',
                       help='Output directory for models')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='Save model every N steps')
    parser.add_argument('--wandb_project', type=str, default='gpt-sft',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str,
    # Provide a unique, dynamic default value
    default=f"sft-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    help='Wandb run name')

    # Data arguments
    parser.add_argument('--max_train_docs', type=int, default=None,
                       help='Maximum number of documents to load (for testing)')

    return parser.parse_args()


def setup_device(device_arg):
    """Set up the device for training."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = device_arg

    print(f"Using device: {device}")
    return device


def setup_tokenizer():
    """Set up tokenizer with special tokens for SFT."""
    # Call the setup_tokenizer function from gpt.py
    tokenizer = gpt_setup_tokenizer()

    # Calculate actual vocabulary size
    special_tokens = ["<|user|>", "<|assistant|>", "<|end|>", "<|system|>", "<|pad|>"]
    max_token_id = max(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    actual_vocab_size = max_token_id + 1

    print(f"✅ Tokenizer initialized with {actual_vocab_size} tokens")
    return tokenizer, actual_vocab_size


def create_model_config(args, vocab_size):
    """Create model configuration."""
    config = {
        "vocab_size": vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": False
    }
    return config


def load_model(model_path, config):
    """Load pre-trained model."""
    print(f"Loading pre-trained model from {model_path}...")
    state_dict = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k

        if new_k.startswith("_orig_mod."):
            new_k = new_k[len("_orig_mod."):]

        if new_k.startswith("_wrapped_module."):
            new_k = new_k[len("_wrapped_module."):]

        if new_k.startswith("_compile_cache."):
          
            parts = new_k.split(".")
            if len(parts) > 2:
                new_k = ".".join(parts[2:])

        cleaned[new_k] = v

    state_dict = cleaned
    # Create model with correct configuration
    model = gpt.GPTModel(config)

    # Check vocabulary size compatibility
    original_vocab_size = state_dict['embedding.token_embeddings.weight'].shape[0]
    new_vocab_size = config['vocab_size']

    if original_vocab_size != new_vocab_size:
        print(f"❌ ERROR: Vocabulary size mismatch!")
        print(f"   Model vocab size: {original_vocab_size}")
        print(f"   Expected vocab size: {new_vocab_size}")
        raise ValueError("Vocabulary size mismatch - use the corrected model")
    else:
        print(f"✅ Vocabulary sizes match: {original_vocab_size}")

    # Load the state dict
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Model loaded successfully!")

    return model


def create_dataloaders(args, tokenizer):
    """Create training and validation dataloaders."""
    print("Creating dataloaders...")

    ###########################################################################
    #                            TODO 4.1: YOUR CODE HERE                     #
    #                                                                         #
    # Implement SFT DataLoader creation:                                      #
    #                                                                         #
    # 1. Create training DataLoader using sft.create_sft_dataloader():        #
    #    - Use args.train_data_path for data file                             #
    #    - Pass tokenizer, batch_size, context_length                         #
    #    - Set shuffle=True, drop_last=True for training                      #
    #    - Use args.num_workers and args.use_packed                           #
    # 2. Create validation DataLoader using sft.create_sft_dataloader():      #
    #    - Use args.val_data_path for data file                               #
    #    - Pass tokenizer, batch_size, context_length                         #
    #    - Set shuffle=False, drop_last=False for validation                  #
    #    - Use args.num_workers and args.use_packed                           #
    # 3. Print success messages with batch counts                             #
    # 4. Return both dataloaders                                              #
    #                                                                         #
    # This function sets up the data pipeline for SFT training.               #
    ###########################################################################

    train_loader = sft.create_sft_dataloader(data_file=args.train_data_path,
                                             tokenizer=tokenizer,
                                             batch_size=args.batch_size,
                                             max_length=args.context_length,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=args.num_workers,
                                             use_packed=(args.train_data_format == 'arrow')
                                             )

    val_loader = sft.create_sft_dataloader(data_file=args.val_data_path,
                                           tokenizer=tokenizer,
                                           batch_size=args.batch_size,
                                           max_length=args.context_length,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=args.num_workers,
                                           use_packed=(args.val_data_format == 'arrow')
                                           )
    
    print(f"✅ Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    return train_loader, val_loader


def get_amp_dtype(device):
    '''Get the appropriate AMP dtype for mixed precision training on the device.'''

    # Check for CUDA (GPU) support
    if device.startswith('cuda'):
        # bfloat16 is preferred for stability on modern GPUs (e.g., A100, H100)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
    # Check for Apple Silicon (MPS) support
    elif device == 'mps':
        # MPS typically supports float16 (half precision)
        amp_dtype = torch.float16
        
    else:
        # Default to standard float32 on CPU, as low precision is not efficient/supported
        amp_dtype = torch.float32 
        
    return amp_dtype

def save_model(model, optimizer, scheduler, args, global_step, epoch, is_epoch_end=False):
    """
    Saves the complete training checkpoint to enable resumption.
    """
    
    # Define filename based on the step or epoch end flag
    if is_epoch_end:
        filename = f"checkpoint-epoch-{epoch}.pth"
    else:
        filename = f"checkpoint-step-{global_step}.pth"

    # Define the save path
    save_dir = os.path.join(args.output_dir, args.wandb_project, args.wandb_run_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    print(f"Saving checkpoint to {save_path}...")
    
    # 1. Assemble the state dictionary
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'global_step': global_step,
        'epoch': epoch,
        # Save any other essential metadata
    }

    # 2. Save the checkpoint file
    torch.save(checkpoint, save_path)
    print(f"✅ Checkpoint saved at step {global_step}")

def train_model(model, train_loader, val_loader, args, device):
    """Train the model with SFT."""
    print("Starting SFT training...")

    # Move model to device
    model = model.to(device)

    # Loss function - automatically ignores -100 labels
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * args.max_epochs // args.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5
    )

    # Training tracking
    train_losses = []
    val_losses = []
    step = 0
    opt_step = 0
    global_step = 0  # Track total steps across all epochs

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config=vars(args)
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    model.train()
    try:
        # Note: This attempt to create a CUDA scaler will fail if CUDA is not available
        scaler = GradScaler(device="cuda" if device.startswith("cuda") else "cpu") 
    except TypeError:
        scaler = GradScaler()
    
    running_loss = 0.0

    last_save_step = -1
    use_amp = (device.startswith("cuda") or device == "mps")
    amp_dtype = get_amp_dtype(device)

    for epoch in trange(args.max_epochs, desc="Epoch"):
        epoch_losses = []

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):

            ###########################################################################
            #                            TODO 4.2: YOUR CODE HERE                     #
            #                                                                         #
            # Implement SFT forward pass:                                             #
            #                                                                         #
            # 1. Handle different batch formats (dict vs tuple)                      #
            # 2. Move input_ids and labels to the correct device                      #
            # 3. Forward pass with mixed precision:                                   #
            #    - Use torch.amp.autocast() for mixed precision                       #
            #    - Call model(input_ids) to get logits                               #
            #    - Shift logits and labels for next-token prediction                  #
            #    - Compute loss using CrossEntropyLoss with ignore_index=-100         #
            #    - Scale loss by gradient accumulation steps                          #
            #                                                                         #
            # Key insight: SFT uses masked loss where only assistant tokens           #
            # contribute to the loss (labels != -100).                               #
            #                                                                         #
            # Example:                                                                #
            # input_ids: [<|user|>, "Hello", <|end|>, <|assistant|>, "Hi", <|end|>]  #
            # labels:    [-100,     -100,    -100,    50257,        50258, 50259]   #
            # Only tokens 50257, 50258, 50259 contribute to loss!                    #
            ###########################################################################

            # code goes here
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # 2. Forward Pass and Loss Calculation (TODO 4.2)
            with autocast(device_type=device, enabled=use_amp, dtype=amp_dtype):
                
                # 3. Model call and Reshaping
                logits = model(input_ids)
                B, T, V = logits.shape
                
                # Loss calculation: CrossEntropyLoss expects (N, V) and (N)
                logits_flat = logits.view(B * T, V)
                labels_flat = labels.view(B * T)
                
                # Compute loss and scale by accumulation factor
                loss = loss_fn(logits_flat, labels_flat) / args.gradient_accumulation_steps


            ###########################################################################
            #                            TODO 4.3: YOUR CODE HERE                     #
            #                                                                         #
            # Implement SFT backward pass and optimization:                           #
            #                                                                         #
            # 1. Compute gradients with loss.backward()                               #
            # 2. Update weights only every gradient_accumulation_steps:               #
            #    - Apply gradient clipping with torch.nn.utils.clip_grad_norm_()      #
            #    - Call optimizer.step() to update parameters                         #
            #    - Call scheduler.step() to update learning rate                      #
            #    - Call optimizer.zero_grad() to clear gradients                      #
            #    - Increment opt_step counter                                         #
            # 3. Track loss for logging                                               #
            # 4. Evaluate the model on the validation set every eval_every steps      #
            # 5. Save the model every save_every steps                                #
            # 6. Evaluate the model on the validation set at the end of each epoch    #
            # 7. Save the model at the end of each epoch                              #
            #                                                                         #
            #                                                                         #
            # Example with gradient_accumulation_steps=4:                             #
            # - Steps 1-3: Only compute gradients (no optimizer.step())               #
            # - Step 4: Apply gradients and update parameters                         #
            ###########################################################################

            ###########################################################################
            # (TODO 4.2: Forward Pass & Loss is assumed to be defined above this block) #
            ###########################################################################
            
            # --- Backward Pass ---
            # Use appropriate backward pass based on device/AMP
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 1. Track loss for logging (unscaled by accum factor)
            running_loss += loss.item() * args.gradient_accumulation_steps

            ###########################################################################
            #                            TODO 4.3: Optimization Block                 #
            ###########################################################################

            # 2. Conditional Optimization Step (Macro-Batch Update)
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                
                # a. Gradient Clipping and Unscaling
                if use_amp:
                    scaler.unscale_(optimizer) 
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer) # Scaler updates parameters
                    scaler.update()        # Updates scaling factor
                else:
                    # CPU/Standard Path: Clip and Step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # b. Clear Gradients and Update LR
                scheduler.step()
                optimizer.zero_grad(set_to_none=True) # Use set_to_none=True for efficiency
                
                # c. Track Steps
                opt_step += 1
                global_step += 1
                
                # 3. Periodic Logging
                if opt_step % args.eval_every == 0:
                    
                    # Log Training Loss
                    wandb.log({
                        "train/loss": running_loss / args.eval_every,
                        "train/lr": optimizer.param_groups[0]['lr'],
                    }, step=global_step)
                    
                    # Reset loss accumulator
                    running_loss = 0.0 
                    
                    # 4. Step Evaluation
                    if val_loader is not None:
                        val_loss = sft.evaluate_validation_loss(
                            model, val_loader, loss_fn, device
                        )
                        wandb.log({"val/loss": val_loss}, step=global_step)

                # 5. Checkpointing
                if opt_step % args.save_every == 0:
                    # Assume save_model utility is available
                    save_model(model, optimizer, scheduler, args, global_step, epoch)

        # code goes here
        if val_loader is not None:
            full_val_loss = sft.evaluate_validation_loss(model, val_loader, loss_fn, device)
            wandb.log({"epoch/val_loss": full_val_loss, "epoch/global_step": global_step})
        save_model(model, optimizer, scheduler, args, global_step, epoch, is_epoch_end=True)
    # code goes here

    wandb.finish()
    print("✅ SFT training complete!")


def main():
    """Main training function."""
    args = parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set up device
    device = setup_device(args.device)

    # Set up tokenizer
    tokenizer, vocab_size = setup_tokenizer()

    # Create model configuration
    config = create_model_config(args, vocab_size)

    ###########################################################################
    #                            TODO 4.4: YOUR CODE HERE                     #
    #                                                                         #
    # Implement model loading and setup:                                      #
    #                                                                         #
    # 1. Load pre-trained GPT model from checkpoint                           #
    # 2. Move model to the correct device (CPU/GPU)                           #
    # 3. Optionally compile model for better performance                      #
    # 4. Verify model is ready for SFT training                               #
    #                                                                         #
    # This ensures your pre-trained model is properly loaded!                 #
    ###########################################################################

    model = load_model(args.model_path, config)
    model.to(device)

    # 4. Optionally compile model for performance
    if hasattr(torch, 'compile'):
        print("Compiling model for performance...")
        try:
            # Note: We assign the compiled function back to the model variable
            model = torch.compile(model, mode="default")
            print("✅ Model compiled successfully!")
        except Exception as e:
            # We skip compilation if the environment (like Windows/CPU) doesn't support it
            print(f"Warning: Model compilation failed, using eager mode. Error: {e}")

    train_loader, val_loader = create_dataloaders(args, tokenizer)

    # 5. Start Training
    # train_model is called with the setup model and device
    train_model(model, train_loader=train_loader, val_loader=val_loader, args=args, device=device)


if __name__ == "__main__":
    main()
