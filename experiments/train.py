"""
Example command:
uv run experiments/train.py \
--train_filename ../a1-data/ts-train-encoded-tiktoken.npy \
--val_filename ../a1-data/ts-valid-encoded-tiktoken.npy \
--vocab_size 10000 \
--num_heads 16 \
--d_model 512 \
--d_ff 1344 \
--context_length 256 \
--num_layers 4 \
--rope_theta 10000 \
--batch_size 64 \
--save_every_n_iterations 100 \
--total_tokens_processed 40_000_000 \
--resume \
--load_checkpoint_path ../a1-checkpoints/lr_max_1e-2_checkpoint_iter18999.pt \
--save_checkpoint_path ../a1-checkpoints/checkpoint.pt \
--wandb_run_name xxx \
--wandb_proj_name yyy
"""

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse
import numpy as np
import random
import math
import torch
import wandb
from src.transformer import Transformer
from src.adamw import AdamW
from src.cross_entropy import cross_entropy
from src.data_loading import get_batch
from src.gradient_clipping import clip_gradient
from src.cosine_lr import cosine_lr_schedule_with_warmup
from src.checkpointing import save_checkpoint, load_checkpoint
from src.logger import logging

# Seed for deterministic training
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def train(
    vocab_size,
    num_heads,
    d_model,
    d_ff,
    rope_theta,
    context_length,
    num_layers,
    betas,
    eps,
    weight_decay,
    lr_max,
    lr_min,
    T_w,
    T_c,
    max_l2_norm,
    batch_size,
    num_iterations,
    train_filename,
    val_filename,
    save_checkpoint_path: str,
    resume,
    load_checkpoint_path: str,
    save_every_n_iterations,
    device,
    wandb_run_name,
    wandb_proj_name,
):
    run = wandb.init(
        project=wandb_proj_name,
        name=wandb_run_name,
        config={
            "vocab_size": vocab_size,
            "num_heads": num_heads,
            "d_model": d_model,
            "d_ff": d_ff,
            "rope_theta": rope_theta,
            "context_length": context_length,
            "num_layers": num_layers,
            "lr_max": lr_max,
            "lr_min": lr_min,
            "T_w": T_w,
            "T_c": T_c,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "max_l2_norm": max_l2_norm,
            "betas": betas,
            "eps": eps,
        },
    )

    # .to(device) moves all parameters and buffers to the device. It does so
    # by recursively traversing all parameters and buffers and call .to(device).
    # However, tensors created outside of nn.Parameter() or register_buffer()
    # won't be moved automatically.
    # The best practice is to use .register_buffer() for fixed tensors like
    # RoPE or positional encoding, s.t. they will be moved.
    model = Transformer(
        vocab_size,
        num_heads,
        d_model,
        d_ff,
        rope_theta,
        context_length,
        num_layers,
    )

    # Initialize opt with lr_max
    opt = AdamW(model.parameters(), lr_max, betas, eps, weight_decay)

    # Load from checkpoint if needed
    start_iter = 0
    if resume and os.path.exists(load_checkpoint_path):
        checkpoint_iter = load_checkpoint(load_checkpoint_path, model, opt)
        logging.info(f"Loaded from checkpoint at iteration {checkpoint_iter}")

        start_iter = checkpoint_iter + 1

    # LambdaLR expects the function to return a ratio of the initial lr
    # Return multiplier w.r.t lr_max
    lr_lambda = lambda step: (
        cosine_lr_schedule_with_warmup(step, lr_max, lr_min, T_w, T_c) / lr_max
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda, last_epoch=start_iter - 1
    )

    # Compile model for speedup
    if device == "mps":
        model.compile(backend="aot_eager")
    else:
        torch.set_float32_matmul_precision("high")
        model.compile()

    # Move model and optimizer to device
    model = model.to(device)
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # Sanity check
    for param in model.parameters():
        assert param.device.type == device

    # Load as uint16 - the encoded data uses this dtype
    train_dataset = np.memmap(train_filename, mode="r", dtype=np.uint16)
    val_dataset = np.memmap(val_filename, mode="r", dtype=np.uint16)

    model.train()

    for t in range(start_iter, num_iterations):

        logging.info(f"--- Starting iteration {t} ---")

        inputs, targets = get_batch(train_dataset, batch_size, context_length, device)
        opt.zero_grad()

        outputs = model(inputs)
        train_loss = cross_entropy(outputs, targets)
        train_loss.backward()

        clip_gradient(model.parameters(), max_l2_norm)
        opt.step()
        scheduler.step()

        train_perplexity = torch.exp(train_loss)

        logging.info(f"Iteration {t} finished")
        logging.info(f"Train Loss: {train_loss.item():.4f}")
        logging.info(f"Train Perplexity: {train_perplexity.item():.4f}")
        logging.info(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        wandb_data = {
            "iteration": t,
            "train/loss": train_loss.item(),
            "train/perplexity": train_perplexity.item(),
            "lr": scheduler.get_last_lr()[0],
        }

        # Validation loss
        if (t + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = get_batch(
                    val_dataset, batch_size, context_length, device
                )
                val_outputs = model(val_inputs)
                val_loss = cross_entropy(val_outputs, val_targets)
                val_perplexity = torch.exp(val_loss)

            # Revert to train mode
            model.train()

            wandb_data["val/loss"] = val_loss.item()
            wandb_data["val/perplexity"] = val_perplexity.item()

            logging.info(f"Val Loss: {val_loss.item():.4f}")
            logging.info(f"Val Perplexity: {val_perplexity.item():.4f}")

        wandb.log(wandb_data)

        if (t + 1) % save_every_n_iterations == 0 and save_checkpoint_path:
            checkpoint_file = save_checkpoint_path.replace(".pt", f"_iter{t}.pt")
            save_checkpoint(model, opt, t, checkpoint_file)
            logging.info(f"Checkpoint saved to {checkpoint_file}")

        logging.info("================================")

    run.finish()


def main():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--train_filename", type=str, required=True)
    parser.add_argument("--val_filename", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)

    # Model params
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Training hyperparams
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--total_tokens_processed", type=int, required=True)
    parser.add_argument("--max_l2_norm", type=float, default=1.0)

    # Optimizer
    # For AdamW, some rough rules for lr_max:
    # Small transformers (≤100M params): 3e-4 to 5e-4
    # Medium (∼300M): 1e-4 to 3e-4
    # Large (∼1B+): 2e-5 to 6e-5
    parser.add_argument("--lr_max", type=float, default=3e-4)
    # For lr_min: Usually 1e-5 or 0.1 * lr_max
    parser.add_argument("--lr_min", type=float, default=1e-5)
    # T_w usually 3%-6% of total steps.
    parser.add_argument("--T_w", type=int)
    # T_c usually is the same as total steps
    parser.add_argument("--T_c", type=int)
    # Betas common default: (0.9, 0.95)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    # Eps common default: 1e-8
    parser.add_argument("--eps", type=float, default=1e-8)
    # Weight decay common default: 0.01
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Checkpointing
    parser.add_argument("--save_checkpoint_path", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_checkpoint_path", type=str)
    parser.add_argument("--save_every_n_iterations", type=int, default=1)

    # Wandb naming
    parser.add_argument("--wandb_proj_name", type=str, required=True)
    parser.add_argument("--wandb_run_name", type=str, required=True)

    # Device
    default_device = "cpu"
    if torch.cuda.is_available():
        logging.info("cuda available")
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        logging.info("mps available")
        default_device = "mps"

    logging.info(f"Default device: {default_device}")
    parser.add_argument("--device", type=str, default=default_device)

    args = parser.parse_args()

    if bool(args.resume) != bool(args.load_checkpoint_path):
        raise ValueError(
            "--resume and --load_checkpoint_path must both be passed in or neither!"
        )

    num_iterations = math.ceil(
        int(args.total_tokens_processed)
        / (int(args.context_length) * int(args.batch_size))
    )
    logging.info(
        f"Total tokens processec: {args.total_tokens_processed:,}. Number of iterations to run: {num_iterations:,}"
    )

    if args.T_w:
        T_w = args.T_w
    else:
        T_w = int(0.05 * num_iterations)
        logging.info(f"Deriving T_w from num_iterations: {T_w}")

    if args.T_c:
        T_c = args.T_c
    else:
        T_c = num_iterations
        logging.info(f"Deriving T_c from num_iterations: {T_c}")

    train(
        vocab_size=args.vocab_size,
        num_heads=args.num_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        context_length=args.context_length,
        num_layers=args.num_layers,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
        T_w=T_w,
        T_c=T_c,
        max_l2_norm=args.max_l2_norm,
        batch_size=args.batch_size,
        num_iterations=num_iterations,
        train_filename=args.train_filename,
        val_filename=args.val_filename,
        save_checkpoint_path=args.save_checkpoint_path,
        resume=args.resume,
        load_checkpoint_path=args.load_checkpoint_path,
        save_every_n_iterations=args.save_every_n_iterations,
        device=args.device,
        wandb_proj_name=args.wandb_proj_name,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
