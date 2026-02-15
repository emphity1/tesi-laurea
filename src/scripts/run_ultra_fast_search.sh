#!/bin/bash

# Ultra-Fast Grid Search - Example Run
# Tests many hyperparameters in 2-3 hours using two-phase strategy

echo "========================================================================"
echo "ULTRA-FAST GRID SEARCH - Example Configuration"
echo "========================================================================"
echo ""
echo "This will test approximately 125 configurations (5x5x5) in ~2-3 hours:"
echo "  - Phase 1: Screen all 125 configs with 25 epochs (~2 hours)"
echo "  - Phase 2: Refine top-5 configs with 50 epochs (~40 min)"
echo ""
echo "Search Space:"
echo "  - Learning rates: [0.01, 0.025, 0.05, 0.075, 0.1]"
echo "  - Width multipliers: [0.3, 0.35, 0.4, 0.45, 0.5]"
echo "  - Weight decay: [1e-4, 2e-4, 3e-4, 5e-4, 1e-3]"
echo ""
echo "Strategy:"
echo "  - Checkpoints at epochs [15, 20, 25] for trend analysis"
echo "  - Early stop at epoch 15 if val_acc < 78%"
echo "  - Smart scoring: 70% current + 30% predicted final accuracy"
echo "========================================================================"
echo ""

cd /workspace/tesi-laurea/src/scripts

python grid_search_ultra_fast.py \
  --lr 0.005 0.01 0.025 0.05 0.075 0.1 \
  --width_mult 0.3 0.35 0.4 0.45 0.5 0.55 0.6 \
  --weight_decay 0.0001 0.0002 0.0003 0.0005 0.001 \
  --screening_epochs 25 \
  --checkpoint_epochs 15 20 25 \
  --min_acc_epoch_15 80.0 \
  --top_n 5 \
  --refinement_epochs 50 \
  --batch_size 128 \
  --num_workers 16

echo ""
echo "========================================================================"
echo "Grid search complete! Check the results in:"
echo "  /workspace/tesi-laurea/reports/grid_search_ultra_fast/"
echo "========================================================================"
