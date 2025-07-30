#!/bin/bash

# Define parameter ranges
temperatures=(0.5 0.7 1.0 2.0 5.0 10.0)
top_ps=(0.7 0.8 0.9 0.95 1.0)

# Common arguments
BPE=../a1-log/ts-train-bpe.pkl
CHECKPOINT=../a1-checkpoints/generate_text_checkpoint_iter17999.pt
PROMPT="Once upon a time"
MAX_TOKENS=64

# Sweep
for temp in "${temperatures[@]}"; do
  for top_p in "${top_ps[@]}"; do
    echo
    echo "=============================="
    echo "temperature=$temp, top-p=$top_p"
    echo "------------------------------"

    uv run experiments/generate_text.py \
      --bpe "$BPE" \
      --model-checkpoint "$CHECKPOINT" \
      --max-generated-tokens "$MAX_TOKENS" \
      --prompt "$PROMPT" \
      --temperature "$temp" \
      --top-p-sampling-threshold "$top_p"
  done
done
