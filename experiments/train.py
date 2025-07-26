import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse
import numpy as np
import random
import torch
from src.transformer import Transformer
from src.adamw import AdamW
from src.cross_entropy import cross_entropy
from src.data_loading import get_batch
from src.gradient_clipping import clip_gradient
from src.cosine_lr import cosine_lr_schedule_with_warmup
from src.checkpointing import save_checkpoint, load_checkpoint

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
    checkpoint_path,
    resume,
    save_every_n_iterations,
    device,
):
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
    # LambdaLR expects the function to return a ratio of the initial lr
    # Return multiplier w.r.t lr_max
    lr_lambda = lambda step: (
        cosine_lr_schedule_with_warmup(step, lr_max, lr_min, T_w, T_c) / lr_max
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Load from checkpoint if needed
    start_iter = 0
    if resume and os.path.exists(checkpoint_path):
        checkpoint_iter = load_checkpoint(checkpoint_path, model, opt)
        print(f"Loaded from checkpoint at iteration {checkpoint_iter}")

        start_iter = checkpoint_iter + 1

    # Move model and optimizer to device
    model = model.to(device)
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # Sanity check
    for param in model.parameters():
        assert param.device == device

    # TODO: verify dtype is right
    train_dataset = np.memmap(train_filename, dtype=np.uint16, mode="r")
    val_dataset = np.memmap(val_filename, dtype=np.uint16, mode="r")

    model.train()

    for t in range(start_iter, num_iterations):

        print(f"--- Starting iteration {t} ---")

        inputs, targets = get_batch(train_dataset, batch_size, context_length, device)
        opt.zero_grad()

        outputs = model(inputs)
        train_loss = cross_entropy(outputs, targets)
        train_loss.backward()

        clip_gradient(model.parameters(), max_l2_norm)
        opt.step()
        scheduler.step()

        train_perplexity = torch.exp(train_loss)

        print(f"Iteration {t} finished")
        print(f"Train Loss: {train_loss.item():.4f}")
        print(f"Train Perplexity: {train_perplexity.item():.4f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Validation loss, after warmup
        if t >= T_w:
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

            print(f"Val Loss: {val_loss.item():.4f}")
            print(f"Val Perplexity: {val_perplexity.item():.4f}")

        print("================================")

        if (t + 1) % save_every_n_iterations == 0 and checkpoint_path:
            checkpoint_file = checkpoint_path.replace(".pt", f"_iter{t}.pt")
            save_checkpoint(model, opt, t, checkpoint_file)
            print(f"Checkpoint saved to {checkpoint_file}")

        # TODO: wandb logging


def main():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--training_filename", type=str, required=True)
    parser.add_argument("--val_filename", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)

    # Model params
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Training hyperparams
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--max_l2_norm", type=float, default=1.0)

    # Optimizer
    parser.add_argument("--lr_max", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--T_w", type=int, default=1000)
    parser.add_argument("--T_c", type=int, default=10000)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Checkpointing
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every_n_iterations", type=int, default=1)

    # Device
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

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
        T_w=args.T_w,
        T_c=args.T_c,
        max_l2_norm=args.max_l2_norm,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        train_filename=args.train_filename,
        val_filename=args.val_filename,
        checkpoint_path=args.checkpoint_path,
        resume=args.resume,
        save_every_n_iterations=args.save_every_n_iterations,
        device=args.device,
    )


if __name__ == "__main__":
    main()
