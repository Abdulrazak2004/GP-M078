#!/bin/bash
# ===========================================================================
# One-command vast.ai server setup for Casing RUL experiments.
#
# Usage:
#   bash setup_vastai.sh
#
# Prerequisites:
#   - vast.ai instance with 4x RTX 4090 GPUs
#   - NVIDIA drivers + CUDA installed (standard vast.ai PyTorch template)
#   - Git available
# ===========================================================================

set -euo pipefail

echo "============================================================"
echo "  Casing RUL Pipeline — vast.ai Setup"
echo "============================================================"

# 1. Clone repo (if not already present)
REPO_DIR="/workspace/GP"
if [ -d "$REPO_DIR" ]; then
    echo "[1/6] Repo already exists at $REPO_DIR, pulling latest..."
    cd "$REPO_DIR"
    git pull || echo "  (git pull failed — continuing with existing code)"
else
    echo "[1/6] Cloning repository..."
    cd /workspace
    git clone https://github.com/$(git config user.name 2>/dev/null || echo "user")/GP.git "$REPO_DIR" \
        || { echo "ERROR: git clone failed. Clone manually and re-run."; exit 1; }
    cd "$REPO_DIR"
fi

# 2. Install dependencies
echo "[2/6] Installing Python dependencies..."
pip install --quiet --upgrade pip 2>/dev/null || true  # may fail on Debian-managed pip
pip install --quiet -r requirements.txt

# 3. Regenerate dataset (~10-15 min for 500 wells)
echo "[3/6] Generating synthetic dataset (500 wells, 30-year horizon)..."
DATA_DIR="$REPO_DIR/data"
mkdir -p "$DATA_DIR"

# Always regenerate to match current config (well count may have changed)
if [ -f "$DATA_DIR/synthetic_corrosion_dataset.csv" ]; then
    echo "  Removing old dataset..."
    rm "$DATA_DIR/synthetic_corrosion_dataset.csv"
fi
python -m data_generation.generate_dataset
echo "  Dataset generated."

# Verify dataset
if [ ! -f "$DATA_DIR/synthetic_corrosion_dataset.csv" ]; then
    echo "ERROR: Dataset not found at $DATA_DIR/synthetic_corrosion_dataset.csv"
    exit 1
fi
FILE_SIZE=$(du -h "$DATA_DIR/synthetic_corrosion_dataset.csv" | cut -f1)
echo "  Dataset size: $FILE_SIZE"

# 4. Verify GPUs
echo "[4/6] Verifying GPUs..."
python -c "
import torch
n = torch.cuda.device_count()
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU count: {n}')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'    GPU {i}: {name} ({mem:.1f} GB)')
if n < 1:
    print('  WARNING: No GPUs detected!')
"

# 5. Create output directories
echo "[5/6] Creating output directories..."
mkdir -p outputs/logs

# 6. Launch all experiments
echo "[6/6] Launching parallel experiments..."
echo ""
python run_all.py

echo ""
echo "============================================================"
echo "  All experiments complete!"
echo "  Results: $REPO_DIR/outputs/"
echo "============================================================"
