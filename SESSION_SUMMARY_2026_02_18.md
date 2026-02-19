# Session Summary — 2026-02-18

## Context
**Thesis**: Bachelor's thesis on efficient image classification using MobileNetECA-Rep on CIFAR-10.
**Author**: Dmytro Kozak, Roma Tre University.
**Repo**: https://github.com/emphity1/tesi-laurea.git (branch: main, commit: `4d7efda`)

---

## What Was Done

### 1. Critical Code Fix: Data Leakage
**Problem**: All 4 original training scripts used the test set (10k images) for model selection (early stopping), causing data leakage.
**Solution**: Created `src/train/shared_config.py` with a proper **45k train / 5k val / 10k test** split. The validation set is used for model selection; the test set is evaluated only once at the end.

### 2. Ablation Study — Clean Re-Implementation
Created 4 new standardized training scripts in `src/train/`:

| Script | Model | What it adds |
|--------|-------|-------------|
| `train_A_baseline.py` | MobileNetV2-Micro | Pure baseline (no ECA, no Rep) |
| `train_B_eca.py` | MobileNetECA | Adds ECA attention blocks |
| `train_C_eca_rep.py` | MobileNetECA-Rep | Adds structural reparameterization |
| `train_D_eca_rep_advaug.py` | MobileNetECA-Rep-AdvAug | Adds AutoAugment + RandomErasing |

**All 4 share identical settings via `shared_config.py`**:
- Seed: 42
- Normalization: mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)
- Optimizer: SGD, LR=0.05, momentum=0.9, weight_decay=5e-4
- Scheduler: CosineAnnealingLR, 200 epochs
- Batch size: 128
- No `lr_scale` trick (was inconsistent before)
- MIN_CHANNELS = 12

### 3. Training Results (new, clean)

| Model | Test Acc | Val Acc | Params (deploy) |
|-------|----------|---------|-----------------|
| A Baseline | 91.65% | 92.66% | 77,934 |
| B +ECA | 92.17% | 92.58% | 78,014 |
| C +Rep | 92.43% | 93.84% | 76,602 |
| D +AdvAug | **93.23%** | **93.80%** | **76,602** |

> **Note**: Previous accuracy was 93.50% (with data leakage). New clean result is 93.23%.

### 4. Generated All Thesis Figures
Script: `src/train/generate_plots.py` — produces 9 PNG files:
- `accuracy_comparison.png` — val acc curves for all 4 models
- `loss_comparison.png` — train loss curves for all 4
- `v3_train_vs_val_acc.png` — train vs val acc for model D
- `v3_train_loss.png` — train loss for model D
- `v3_lr_schedule.png` — cosine annealing LR curve
- `confusion_matrix.png` — 10×10 confusion matrix for model D
- `roc_curve_zoomed.png` — per-class ROC curves (zoomed)
- `top_errors.png` — high-confidence misclassifications
- `efficiency_params.png` — accuracy vs params (log scale) with SOTA

Script: `src/train/generate_occlusion.py` — produces:
- `occlusion_sensitivity.png` — 3-row visualization (original, heatmap, overlay)

All figures copied to `Tesi_Dmytro_Kozak/figure/`.

### 5. LaTeX Thesis Updates

#### Files modified with new numbers (93.50% → 93.23%):
- `main.tex` — abstract
- `capitolo1.tex` — contributions section
- `capitolo5.tex` — validation protocol, expanded ECA section
- `capitolo6.tex` — ablation table, training dynamics text, grid search note
- `conclusioni.tex` — results summary (two places)
- `reports/tables/sota_comparison.tex` — SOTA table entry
- `appendice.tex` — per-class precision/recall/F1 updated

#### Structural improvements:
- **`capitolo5.tex`**: Replaced data leakage disclaimer with proper validation protocol description. Added "Why ECA and not SE?" subsection with concrete parameter comparison table. Added PyTorch code snippet for ECA block implementation.
- **`capitolo6.tex`**: Ablation table now has both Test Acc and Val Acc columns. Added footnote about deploy vs training params. Fixed LaTeX formatting (`*word*` → `\textit{word}`, triple quotes → `\textit{}`). All figures use `[H]` float specifier for exact placement.
- **`bibliografia.bib`**: Removed fake "ThriftyNet" citation (unverifiable arXiv ID).
- **`main.tex`**: Removed `\nocite{*}`.

### 6. Saved Artifacts for Future Analysis
Each model run saves:
```
results_X_*/
├── history.json          # per-epoch: train_loss, train_acc, val_acc, lr, time
├── test_predictions.npz  # targets, predictions, probabilities (10k × 10)
├── best_model.pth        # best checkpoint (by val acc)
├── best_model_deploy.pth # (C, D only) fused-kernel model
└── training.log          # full training log
```

---

## File Map (key files)

```
/workspace/tesi-laurea/
├── Tesi_Dmytro_Kozak/          # LaTeX thesis (compile with: latexmk -pdf main.tex)
│   ├── main.tex                # Entry point
│   ├── capitolo1-6.tex         # Chapters
│   ├── conclusioni.tex
│   ├── appendice.tex
│   ├── bibliografia.bib
│   ├── figure/                 # All PNG/JPG figures
│   └── reports/tables/         # SOTA comparison table
├── src/
│   ├── train/                  # NEW clean training code
│   │   ├── shared_config.py    # Shared config + train_and_evaluate()
│   │   ├── train_A_baseline.py
│   │   ├── train_B_eca.py
│   │   ├── train_C_eca_rep.py
│   │   ├── train_D_eca_rep_advaug.py
│   │   ├── generate_plots.py   # Generates 9 thesis figures
│   │   ├── generate_occlusion.py
│   │   ├── results_A_baseline/ # Training outputs
│   │   ├── results_B_eca/
│   │   ├── results_C_eca_rep/
│   │   └── results_D_eca_rep_advaug/
│   ├── legacy/                 # OLD scripts (kept for reference, NOT used)
│   └── scripts/                # Grid search scripts (separate from ablation)
└── reports/                    # Old grid search results
```

---

## What Was NOT Done / Still TODO

1. **heatmap_lr_wd.png**: Old grid search heatmap kept as-is (same hyperparameters, pattern unchanged with 45k vs 50k). Could be redone for full rigor but not strictly necessary.
2. **LaTeX compilation**: Cannot compile locally (no LaTeX installed in container). User compiles on their machine.
3. **Occlusion sensitivity**: Script created and user ran it manually, but the exact images shown depend on the model checkpoint loaded.
4. **FLOPs recalculation**: The 10.73M FLOPs figure in the SOTA table was kept from the old model. Should be verified with `thop` on the new deploy model if needed.

---

## Key Design Decisions

- **Accuracy dropped from 93.50% to 93.23%**: Expected consequence of fixing data leakage (5k fewer training samples + no test-set peeking). This is the *correct* number to report.
- **ECA adds only ~80 params total** across the entire model (vs SE which would add 21k+ per block). This is now explicitly shown in a table in the thesis.
- **RepConv adds ~8k training params** but they disappear at deploy time (kernel fusion). Deploy params: 76,602.
- **All old training scripts preserved** in `src/legacy/` for reference but should not be used.
