#!/bin/bash

# Nanocode Speedrun: Full pipeline from scratch to coding agent
# Designed to run on 8xH100 node in ~4 hours
#
# Usage:
#   bash runs/speedrun.sh
#   WANDB_RUN=nanocode screen -L -Logfile speedrun.log -S speedrun bash runs/speedrun.sh

set -e  # exit on error

export OMP_NUM_THREADS=1
export NANOCODE_BASE_DIR="$HOME/.cache/nanocode"
mkdir -p $NANOCODE_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup (optional)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

echo "============================================"
echo "  Nanocode Speedrun"
echo "  Base dir: $NANOCODE_BASE_DIR"
echo "  wandb run: $WANDB_RUN"
echo "============================================"

# -----------------------------------------------------------------------------
# Step 1: Download code data
echo ""
echo ">>> Step 1: Downloading code data..."
echo ""

# Download a small initial set for tokenizer training
python -m nanocode.dataset -l python,javascript,typescript,java,c,go,rust -n 2

# Kick off larger download in background for pretraining
python -m nanocode.dataset -l python,javascript,typescript,java,c,go,rust -n 20 &
DATASET_DOWNLOAD_PID=$!

# -----------------------------------------------------------------------------
# Step 2: Train tokenizer
echo ""
echo ">>> Step 2: Training tokenizer..."
echo ""

python -m scripts.tok_train --max_chars=2000000000 --vocab_size=32768

# Evaluate tokenizer compression
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Step 3: Pretrain base model with FIM
echo ""
echo ">>> Step 3: Pretraining base model..."
echo ""

# Wait for data download
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Pretrain d12 model (~124M params) with 50% FIM rate
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --fim_rate=0.5 \
    --fim_spm_rate=0.5 \
    --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Step 4: Evaluate base model
echo ""
echo ">>> Step 4: Evaluating base model..."
echo ""

# Perplexity + FIM samples
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# HumanEval + MBPP (expect near-zero for base model)
torchrun --standalone --nproc_per_node=8 -m scripts.code_eval -- \
    -i base -a all --num-samples=1

# -----------------------------------------------------------------------------
# Step 5: SFT on code instructions
echo ""
echo ">>> Step 5: Running code SFT..."
echo ""

torchrun --standalone --nproc_per_node=8 -m scripts.code_sft -- --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Step 6: Evaluate SFT model
echo ""
echo ">>> Step 6: Evaluating SFT model..."
echo ""

# HumanEval + MBPP with multiple samples for pass@k
torchrun --standalone --nproc_per_node=8 -m scripts.code_eval -- \
    -i sft -a all --num-samples=10 --temperature=0.8

# -----------------------------------------------------------------------------
# Step 7: Agent demo
echo ""
echo ">>> Step 7: Agent demo..."
echo ""

# Quick demo: ask the agent to write and run a simple program
python -m scripts.code_cli -p "Write a Python function that checks if a number is prime, save it to prime.py, then test it with a few numbers."

echo ""
echo "============================================"
echo "  Nanocode Speedrun Complete!"
echo "============================================"
