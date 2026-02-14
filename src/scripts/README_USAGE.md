# Training Scripts - Usage Guide

## Overview

This directory contains modular, production-ready scripts for training MobileNetECA on CIFAR-10.

## Scripts

### 1. `model.py`
Model architecture definition. Can be imported in other scripts or run standalone for testing.

```bash
# Test model instantiation
python model.py
```

### 2. `train.py`
Single training run with specified hyperparameters.

**Usage:**
```bash
# Basic training with default parameters
python train.py

# Custom configuration
python train.py \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.025 \
    --width_mult 0.42 \
    --weight_decay 3e-4 \
    --save_dir ../../models \
    --log_dir ../../reports
```

**Features:**
- Compact progress display (updates every 50 batches)
- Saves best model based on validation accuracy
- Exports training metrics to JSON
- Proper data augmentation (training only)

### 3. `grid_search.py`
Systematic hyperparameter optimization via grid search.

**Usage:**
```bash
# Full grid search (warning: computationally expensive)
python grid_search.py \
    --lr 0.01 0.025 0.05 0.1 \
    --width_mult 0.35 0.42 0.5 0.75 \
    --batch_size 128 256 \
    --weight_decay 1e-4 3e-4 5e-4 \
    --epochs 50 \
    --output_dir ../../reports/grid_search

# Quick test mode (1 combination only)
python grid_search.py --quick_test

# Targeted search (fewer combinations)
python grid_search.py \
    --lr 0.025 0.05 \
    --width_mult 0.42 0.5 \
    --epochs 50
```

**Output:**
- `grid_search_results.json`: All run results
- `best_config.json`: Optimal configuration for final training
- Individual run directories with models and metrics

## Progress Display Examples

### Training (compact format)
```
Epoch 25/200 [ 80.0%] Loss: 0.4523 | Acc:  84.23%
  Val:  82.15% ↑ | Loss: 0.5234 | LR: 0.024500 | Time: 45.2s
```

### Grid Search
```
======================================================================
RUN 12/54
======================================================================
Configuration:
  lr: 0.025
  width_mult: 0.42
  batch_size: 128
  weight_decay: 0.0003

[Training output...]

🏆 NEW BEST: 93.45% validation accuracy!

Progress: 12/54 runs complete
```

## Expected Output Structure

```
tesi-laurea/
├── models/
│   ├── best_model.pt         # Best checkpoint (highest val acc)
│   └── final_model.pt        # Final epoch checkpoint
│
├── reports/
│   ├── training_metrics.json # Full training history
│   └── grid_search/
│       └── search_TIMESTAMP/
│           ├── grid_search_results.json
│           ├── best_config.json
│           └── run_XXX_*/
│               ├── models/
│               └── logs/
```

## Tips

1. **Start with quick test:** `python grid_search.py --quick_test` to verify setup
2. **Monitor GPU:** Grid search is GPU-intensive; monitor with `nvidia-smi`
3. **Reduce epochs for search:** Use `--epochs 50` for grid search, save full 200 epochs for final training
4. **Use best config:** After grid search, train final model with `best_config.json` values

## Thesis Integration

For thesis experiments:

1. Run grid search to find optimal hyperparameters
2. Train final model with best configuration (200 epochs)
3. Use `training_metrics.json` for plotting accuracy/loss curves
4. Compare with baseline models using same protocol
