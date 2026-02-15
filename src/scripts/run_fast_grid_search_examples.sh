#!/bin/bash
# Example script to run fast grid search with early stopping

# Navigate to scripts directory
cd /workspace/tesi-laurea/src/scripts

# EXAMPLE 1: Quick test (minimal combinations to test the script)
echo "=== QUICK TEST ==="
python grid_search_fast.py \
    --quick_test \
    --checkpoint_epoch 15 \
    --min_threshold 80.0 \
    --epochs 20

# EXAMPLE 2: Standard fast search (recommended)
# Tests 5 lr × 6 width × 6 weight_decay = 180 combinations
# With early stopping at epoch 20, skips poor configs early
# Estimated time: ~8-10 hours (vs ~90 hours without early stopping)
echo "=== STANDARD FAST SEARCH ==="
python grid_search_fast.py \
    --lr 0.01 0.025 0.05 0.075 0.1 \
    --width_mult 0.3 0.35 0.4 0.42 0.45 0.5 \
    --weight_decay 0.0001 0.0002 0.0003 0.0005 0.0007 0.001 \
    --checkpoint_epoch 20 \
    --min_threshold 85.0 \
    --epochs 25

# EXAMPLE 3: Very extensive search with lower threshold
# Tests even more combinations with relaxed early stopping
# Use this if you want to explore wider parameter space
echo "=== EXTENSIVE SEARCH ==="
python grid_search_fast.py \
    --lr 0.005 0.01 0.015 0.025 0.035 0.05 0.075 0.1 \
    --width_mult 0.25 0.3 0.35 0.4 0.42 0.45 0.5 0.55 \
    --weight_decay 0.00005 0.0001 0.0002 0.0003 0.0005 0.0007 0.001 0.0015 \
    --checkpoint_epoch 18 \
    --min_threshold 83.0 \
    --epochs 22

# EXAMPLE 4: Fine-tuning around best config from previous search
# Narrow search around promising region
echo "=== FINE-TUNING SEARCH ==="
python grid_search_fast.py \
    --lr 0.04 0.045 0.05 0.055 0.06 \
    --width_mult 0.45 0.475 0.5 0.525 0.55 \
    --weight_decay 0.0004 0.0005 0.0006 0.0007 \
    --checkpoint_epoch 20 \
    --min_threshold 87.0 \
    --epochs 30

# EXAMPLE 5: Adding lr_scale and momentum to search space
# Test different gradient scaling strategies
echo "=== TESTING LR_SCALE AND MOMENTUM ==="
python grid_search_fast.py \
    --lr 0.025 0.05 \
    --width_mult 0.42 0.5 \
    --weight_decay 0.0003 0.0005 \
    --lr_scale 1.2 1.4 1.54 1.7 \
    --momentum 0.85 0.9 0.95 \
    --checkpoint_epoch 20 \
    --min_threshold 85.0 \
    --epochs 25
