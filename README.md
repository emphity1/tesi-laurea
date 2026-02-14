# Efficient Deep Learning Architectures for CIFAR-10 Classification

Bachelor Thesis Project - Implementation and Analysis of MobileNetECA with Attention Mechanisms

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Objectives](#research-objectives)
3. [Methodology](#methodology)
4. [Experimental Design](#experimental-design)
5. [Baseline Comparisons](#baseline-comparisons)
6. [Directory Structure](#directory-structure)
7. [Requirements](#requirements)
8. [Usage](#usage)
9. [Results and Metrics](#results-and-metrics)
10. [Thesis Structure](#thesis-structure)
11. [References](#references)

---

## Project Overview

This thesis investigates the effectiveness of lightweight convolutional neural networks with attention mechanisms for image classification on the CIFAR-10 dataset. The primary focus is on **MobileNetECA**, an efficient architecture combining:

- **Inverted Residual Blocks**: Compact feature representation with expansion in intermediate layers
- **ECA Modules** (Efficient Channel Attention): Lightweight attention mechanism for channel recalibration
- **Depthwise Separable Convolutions**: Reduced computational complexity with minimal accuracy loss

### Research Context

Modern deep learning applications require models that balance accuracy with computational efficiency, particularly for deployment on edge devices with limited resources. This work explores the trade-off between model size, computational cost (FLOPs), and classification accuracy on CIFAR-10, demonstrating that attention mechanisms can significantly improve performance with minimal overhead.

---

## Research Objectives

### Primary Objectives

1. **Design and implement** an efficient neural architecture (MobileNetECA) optimized for CIFAR-10 classification
2. **Conduct systematic hyperparameter tuning** using grid search to find optimal configuration
3. **Compare performance** against state-of-the-art baseline models with similar computational budgets
4. **Analyze the contribution** of ECA blocks to classification accuracy
5. **Demonstrate efficiency** in terms of parameters, FLOPs, and inference time

### Key Research Questions

- Can attention mechanisms improve lightweight networks without significant computational overhead?
- How does MobileNetECA compare to MobileNetV2, ShuffleNetV2, and ResNet baselines?
- What is the optimal configuration of width multiplier, learning rate, and regularization?
- Which CIFAR-10 classes are most challenging, and how does the model handle inter-class confusion?

---

## Methodology

### Architecture: MobileNetECA

The proposed architecture builds upon MobileNetV2 with the following enhancements:

#### Core Components

**1. Inverted Residual Block**
- Structure: narrow → wide → narrow (e.g., 24 → 144 → 24 channels)
- Expansion factor: typically 6x
- Activation: GELU (Gaussian Error Linear Unit) for smoother gradients
- Residual connection when input/output dimensions match

**2. ECA Module (Efficient Channel Attention)**
- Adaptive kernel size based on channel dimensionality: k = |log₂(C)/γ + b/γ|
- 1D convolution for cross-channel interaction
- Sigmoid activation for channel-wise weighting
- Computational overhead: ~100 parameters per block

**3. Depthwise Separable Convolutions**
- Depthwise: spatial filtering per channel
- Pointwise: 1x1 convolutions for channel mixing
- FLOPs reduction: ~8-9x compared to standard convolutions

### Dataset: CIFAR-10

- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Images**: 60,000 total (50,000 train, 10,000 test)
- **Resolution**: 32x32 RGB
- **Distribution**: Perfectly balanced (6,000 images per class)

### Data Augmentation Strategy

**Training Set Only**:
- Random crop with padding (pad=4)
- Random horizontal flip (p=0.5)
- Normalization (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

**Test/Validation Set**:
- Center crop (if needed)
- Normalization only (same statistics as training)

Critical: Data augmentation is applied exclusively to training data to prevent data leakage and ensure fair evaluation.

---

## Experimental Design

### Hyperparameter Optimization

**Grid Search Parameters**:

| Parameter | Search Space | Optimal Value |
|-----------|--------------|---------------|
| Learning Rate | [0.01, 0.025, 0.05, 0.1] | To be determined |
| Width Multiplier | [0.35, 0.42, 0.5, 0.75] | To be determined |
| Batch Size | [64, 128, 256] | To be determined |
| Weight Decay | [1e-4, 5e-4, 1e-3] | To be determined |
| Optimizer | SGD with momentum (0.9) | Fixed |
| LR Scheduler | Cosine annealing | Fixed |
| Epochs | 200 | Fixed |

**Validation Protocol**:
- K-fold cross-validation on training set for hyperparameter selection
- Final evaluation on held-out test set
- Seeds fixed for reproducibility

### Training Configuration

```python
# Optimizer
optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

# Learning Rate Schedule
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# Loss Function
criterion = CrossEntropyLoss()

# Training Duration
epochs = 200
```

### Evaluation Metrics

**Primary Metrics**:
- Top-1 Accuracy
- Top-5 Accuracy (if applicable)
- Cross-Entropy Loss

**Efficiency Metrics**:
- Number of Parameters (millions)
- FLOPs / MACs (multiply-accumulate operations)
- Inference Time (ms per image)
- Memory Footprint (MB)

**Analysis Metrics**:
- Per-class Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curves (one-vs-all for each class)
- AUC (Area Under Curve)

---

## Baseline Comparisons

### Selected Baseline Models

The following state-of-the-art architectures serve as comparison benchmarks:

#### 1. ResNet Family

| Model | Parameters | FLOPs | Expected Accuracy |
|-------|-----------|-------|-------------------|
| ResNet-20 | 0.27M | 40.8M | ~92.6% |
| ResNet-32 | 0.47M | 69.1M | ~93.5% |
| ResNet-56 | 0.86M | 125.8M | ~94.4% |

**Rationale**: Standard baseline for deep learning on CIFAR-10; demonstrates pure depth scaling.

#### 2. MobileNetV2

| Model | Parameters | FLOPs | Expected Accuracy |
|-------|-----------|-------|-------------------|
| MobileNetV2 x0.5 | 0.70M | 28.0M | ~92.9% |
| MobileNetV2 x1.0 | 2.24M | 88.0M | ~93.8% |

**Rationale**: Direct architectural predecessor; enables ablation study of ECA module contribution.

#### 3. ShuffleNetV2

| Model | Parameters | FLOPs | Expected Accuracy |
|-------|-----------|-------|-------------------|
| ShuffleNetV2 x0.5 | 0.35M | 10.9M | ~90.1% |
| ShuffleNetV2 x1.0 | 1.26M | 45.0M | ~93.0% |

**Rationale**: Competitor in efficient architecture space; different design philosophy (channel shuffle).

#### 4. EfficientNet-B0

| Model | Parameters | FLOPs | Expected Accuracy |
|-------|-----------|-------|-------------------|
| EfficientNet-B0 | 5.3M | 390M | ~91.7% (from scratch) |

**Rationale**: State-of-the-art compound scaling; heavier baseline for comparison.

### Comparison Protocol

**Fair Comparison Requirements**:
- Same dataset split (50k train, 10k test)
- Same data augmentation strategy
- Hyperparameter values from published literature
- Same evaluation metrics
- No test-time augmentation (unless explicitly stated)
- No pre-training on external datasets

**Validation**:
- If grid search finds similar hyperparameters to published values, it validates the optimization approach
- Results should be within ±2% of reported accuracy for fair implementation

---

## Directory Structure

```
tesi-laurea/
├── README.md                           # This file
├── NOTE_PROFESSORE.md                  # Advisor meeting notes and guidelines
├── requirements.txt                    # Python dependencies
├── profile.txt                         # Profiling results
│
├── data/                               # Dataset storage
│   ├── cifar-10-batches-py/           # CIFAR-10 dataset files
│   └── cifar-10-python.tar.gz         # Original archive
│
├── src/                                # Source code
│   ├── scripts/                       # Main training and evaluation scripts
│   │   ├── train.py                   # Training pipeline
│   │   ├── evaluate.py                # Model evaluation
│   │   ├── grid_search.py             # Hyperparameter search
│   │   └── visualize.py               # Generate plots and figures
│   ├── experiments/                   # Experimental code (NAS, ablations)
│   ├── legacy/                        # Previous iterations and archived code
│   └── sn31-v2/                       # External repositories or shared code
│
├── models/                             # Saved model checkpoints
│   ├── resnet_light_check1.pt
│   ├── resnet_se_check2.pt
│   └── [model_name]_best.pt
│
├── reports/                            # Training logs and reports
│   ├── training_report_LAST_USED.txt
│   ├── resnet_light_report.txt
│   └── grid_search_results.json
│
├── grafici/                            # Generated plots and visualizations
│   ├── LAST_USED/                     # Most recent experiment plots
│   ├── accuracy_curves.png
│   ├── loss_curves.png
│   ├── confusion_matrix.png
│   └── roc_curves.png
│
└── docs/                               # Documentation and thesis materials
    ├── README_STRUTTURA.md            # Structure documentation
    ├── checklist_tesi.md              # Thesis writing checklist
    ├── template_capitoli/             # Chapter templates
    └── bibliografia/                  # Bibliography and references
```

---

## Requirements

### Software Dependencies

```
Python >= 3.8
PyTorch >= 1.12.0
torchvision >= 0.13.0
numpy >= 1.20.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
scikit-learn >= 0.24.0
tensorboard >= 2.8.0
tqdm >= 4.60.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Hardware Requirements

**Minimum**:
- GPU: NVIDIA GPU with 4GB VRAM (e.g., GTX 1650)
- RAM: 8GB
- Storage: 2GB

**Recommended**:
- GPU: NVIDIA GPU with 8GB VRAM (e.g., RTX 3060)
- RAM: 16GB
- Storage: 10GB

---

## Usage

### 1. Data Preparation

```bash
# CIFAR-10 will be automatically downloaded if not present
python src/scripts/train.py --download
```

### 2. Grid Search for Hyperparameters

```bash
python src/scripts/grid_search.py \
    --model mobilenet_eca \
    --lr 0.01 0.025 0.05 0.1 \
    --width_mult 0.35 0.42 0.5 0.75 \
    --batch_size 128 256 \
    --weight_decay 1e-4 5e-4 1e-3 \
    --epochs 200 \
    --output_dir reports/grid_search/
```

### 3. Train Model with Optimal Hyperparameters

```bash
python src/scripts/train.py \
    --model mobilenet_eca \
    --lr 0.025 \
    --width_mult 0.42 \
    --batch_size 128 \
    --weight_decay 5e-4 \
    --epochs 200 \
    --save_dir models/ \
    --log_dir reports/
```

### 4. Evaluate Model

```bash
python src/scripts/evaluate.py \
    --model_path models/mobilenet_eca_best.pt \
    --batch_size 256 \
    --output_dir results/
```

### 5. Generate Visualizations

```bash
python src/scripts/visualize.py \
    --log_file reports/training_report_LAST_USED.txt \
    --model_path models/mobilenet_eca_best.pt \
    --output_dir grafici/
```

This will generate:
- Training and validation accuracy curves
- Training and validation loss curves
- Confusion matrix (10x10 for CIFAR-10 classes)
- ROC curves (one-vs-all for each class)
- Per-class accuracy bar charts

---

## Results and Metrics

### Expected Deliverables

**Quantitative Results**:
- Final test accuracy with confidence intervals
- Training convergence plots (accuracy and loss)
- Computational efficiency comparison table
- Per-class performance analysis

**Qualitative Analysis**:
- Confusion matrix interpretation
- Error analysis on misclassified samples
- Feature map visualizations (if applicable)
- Attention weight distributions

**Comparison Table Example**:

| Model | Params (M) | FLOPs (M) | Accuracy (%) | Inference Time (ms) |
|-------|-----------|-----------|--------------|---------------------|
| MobileNetECA (ours) | TBD | TBD | TBD | TBD |
| MobileNetV2 x0.5 | 0.70 | 28.0 | 92.88 | TBD |
| ResNet-20 | 0.27 | 40.8 | 92.60 | TBD |
| ShuffleNetV2 x1.0 | 1.26 | 45.0 | 92.98 | TBD |

---

## Thesis Structure

### Proposed Chapter Outline

**Chapter 1: Introduction** (5-6 pages)
- 1.1 Motivation and Context
  - Deep learning in computer vision
  - Need for efficient architectures
  - Edge computing challenges
- 1.2 Research Objectives
  - Primary and secondary objectives
  - Research questions
- 1.3 Contributions
  - Novel aspects of this work
  - Improvements over baselines
- 1.4 Thesis Organization

**Chapter 2: Background and Related Work** (8-10 pages)
- 2.1 CIFAR-10 Dataset
  - Dataset characteristics
  - Challenges and benchmarks
  - Historical context
- 2.2 Evolution of CNN Architectures
  - ImageNet Challenge era (AlexNet, VGG, ResNet, Inception)
  - Transition to efficiency (MobileNet, ShuffleNet, EfficientNet)
  - Current state-of-the-art
- 2.3 Attention Mechanisms in CNNs
  - Squeeze-and-Excitation Networks (SE-Net)
  - Efficient Channel Attention (ECA-Net)
  - Comparison of attention strategies
- 2.4 Efficient Neural Architectures
  - Depthwise separable convolutions
  - Inverted residual blocks
  - Width and depth scaling

**Chapter 3: Methodology** (10-12 pages)
- 3.1 MobileNetECA Architecture
  - Overall design philosophy
  - Layer-by-layer specification
  - Architectural diagrams
- 3.2 Key Components
  - Inverted Residual Blocks (detailed explanation)
  - ECA Module (mathematical formulation and implementation)
  - Activation functions (GELU vs ReLU)
- 3.3 Design Choices and Justification
  - ECA vs SE: why ECA is more efficient
  - Width multiplier selection
  - Kernel size adaptation
- 3.4 Experimental Setup
  - Dataset preparation and augmentation
  - Training protocol
  - Hyperparameter search strategy (grid search)
  - Hardware and software environment

**Chapter 4: Experimental Results** (8-10 pages)
- 4.1 Grid Search Results
  - Hyperparameter sensitivity analysis
  - Optimal configuration identification
  - Validation curves
- 4.2 Performance Metrics
  - Test accuracy and loss
  - Training convergence analysis
  - Computational efficiency (FLOPs, parameters, inference time)
- 4.3 Baseline Comparison
  - vs MobileNetV2: effect of ECA module
  - vs ShuffleNetV2: architectural philosophy comparison
  - vs ResNet-20/32: efficiency vs depth
  - vs EfficientNet-B0: lightweight vs compound scaling
- 4.4 Ablation Studies
  - Impact of ECA module (with/without comparison)
  - Effect of width multiplier
  - Contribution of GELU activation
- 4.5 Detailed Analysis
  - Confusion matrix and error patterns
  - ROC curves and AUC scores
  - Per-class performance breakdown
  - Challenging class pairs (e.g., cat vs dog, truck vs automobile)

**Chapter 5: Conclusions and Future Work** (3-4 pages)
- 5.1 Summary of Findings
  - Main results recap
  - Achievement of objectives
- 5.2 Limitations
  - Dataset scope (only CIFAR-10)
  - Computational constraints
  - Potential improvements
- 5.3 Future Research Directions
  - Scaling to higher resolutions (CIFAR-100, ImageNet)
  - Deployment on edge devices
  - Neural Architecture Search integration
  - Adversarial robustness

**Additional Elements**:
- Abstract (Italian and English)
- Table of Contents
- List of Figures
- List of Tables
- Bibliography (IEEE or ACM format)
- Appendices (if needed: code snippets, additional results)

---

## Key Visualizations Required

### Mandatory Figures

1. **Training Curves**
   - Dual-axis plot: training and validation accuracy over epochs
   - Dual-axis plot: training and validation loss over epochs
   - Demonstrates convergence and absence of overfitting

2. **Confusion Matrix**
   - 10x10 heatmap for CIFAR-10 classes
   - Normalized by row (true class)
   - Identifies frequent misclassifications

3. **ROC Curves**
   - One-vs-all curves for each of the 10 classes
   - AUC (Area Under Curve) values labeled
   - Demonstrates discriminative power per class

4. **Efficiency Comparison**
   - Scatter plot: Accuracy vs FLOPs
   - Scatter plot: Accuracy vs Parameters
   - Positions of baseline models and MobileNetECA

5. **Architecture Diagram**
   - Block diagram of MobileNetECA
   - Detailed illustration of Inverted Residual + ECA block
   - Data flow visualization

6. **Ablation Study Results**
   - Bar chart comparing variants (with/without ECA, different width multipliers)
   - Table of quantitative results

### Optional But Recommended

- Per-class accuracy bar chart
- Precision-Recall curves
- Learning rate schedule visualization
- Feature map visualizations from intermediate layers
- Activation distribution histograms

---

## Grid Search Validation Strategy

### Purpose

The grid search serves two critical purposes:

1. **Optimization**: Find the best hyperparameter configuration for MobileNetECA on CIFAR-10
2. **Validation**: Verify that the search process converges to values similar to those reported in literature for comparable architectures

### Validation Criterion

After completing grid search for MobileNetECA, train baseline models (ResNet-20, MobileNetV2, etc.) using hyperparameters reported in their respective papers.

**Expected Outcome**:
- If our grid search finds `lr ≈ 0.025`, `wd ≈ 5e-4`, and literature reports similar values, this validates the search procedure
- Baseline models with literature hyperparameters should achieve ±2% of reported accuracy
- This demonstrates **fair comparison** and **reproducibility**

### Reporting

In the thesis, document:
- Grid search space explored
- Best configuration found
- Comparison with literature values for similar models
- Justification for any significant deviations

---

## References and Bibliography

### Essential Papers

**Architectures**:
- He et al., "Deep Residual Learning for Image Recognition" (ResNet)
- Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
- Ma et al., "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
- Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

**Attention Mechanisms**:
- Hu et al., "Squeeze-and-Excitation Networks"
- Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"

**Dataset and Benchmarks**:
- Krizhevsky & Hinton, "Learning Multiple Layers of Features from Tiny Images" (CIFAR-10)

**Training Techniques**:
- Zhang et al., "mixup: Beyond Empirical Risk Minimization"
- DeVries & Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout"

See `docs/bibliografia/` for complete BibTeX entries.

---

## Notes and Important Guidelines

### From Advisor Meetings

Critical points emphasized by the thesis advisor (see [NOTE_PROFESSORE.md](NOTE_PROFESSORE.md)):

1. **Data Augmentation**: Apply ONLY to training set, never to test/validation
2. **Grid Search**: Implement personally, do not rely solely on pre-existing tools
3. **Fair Comparison**: Use hyperparameters from published papers for baselines
4. **Visualizations**: All figures (confusion matrix, ROC curves, training plots) are mandatory
5. **Reproducibility**: Fix random seeds, document all configuration details
6. **Writing Style**: Professional tone, avoid first person, use passive voice
7. **Timeline**: Prioritize Chapters 3-4 (Methodology and Results) first

### Common Pitfalls to Avoid

- Applying data augmentation to test set (causes inflated accuracy)
- Unfair comparison with different training protocols
- Missing citations for baseline model implementations
- Inconsistent terminology throughout the thesis
- Figures without proper captions or references in text
- Overfitting to validation set during grid search

---

## Contact and Support

**Student**: [Your Name]
**Advisor**: [Advisor Name]
**University**: [University Name]
**Department**: [Department Name]
**Academic Year**: [Year]

---

## License

This code is provided for academic purposes as part of a bachelor thesis. If you use or build upon this work, please provide appropriate citation.

---

## Acknowledgments

Special thanks to:
- Thesis advisor for guidance and feedback
- Authors of baseline architectures for publishing reproducible implementations
- PyTorch and torchvision teams for excellent frameworks
- CIFAR-10 dataset creators for providing a valuable benchmark

---

**Document Status**: Living document, updated throughout thesis development
**Last Updated**: [Current Date]
**Version**: 1.0
