
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import numpy as np
import os
import re

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

OUTPUT_DIR = "/workspace/tesi-laurea/docs/scrittura-tesi/tesi/immagini"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. LOAD GRID SEARCH DATA
print("Loading Grid Search Results...")
try:
    with open("/workspace/tesi-laurea/reports/grid_search/search_20260213_193330/grid_search_results.json", "r") as f:
        grid_data = json.load(f)
    
    df_grid = pd.DataFrame([run['config'] | {'val_acc': run['best_val_acc']} for run in grid_data['runs']])
    
    # Heatmap 1: LR vs Width (Fixed WD = 0.0005 - best)
    best_wd = 0.0005
    df_hm1 = df_grid[df_grid['weight_decay'] == best_wd].pivot(index='lr', columns='width_mult', values='val_acc')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_hm1, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)
    plt.title(f"Validation Accuracy: LR vs Width (Weight Decay={best_wd})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_lr_width.png"), dpi=300)
    plt.close()

    # Heatmap 2: LR vs Weight Decay (Fixed Width = 0.5 - best)
    best_width = 0.5
    df_hm2 = df_grid[df_grid['width_mult'] == best_width].pivot(index='lr', columns='weight_decay', values='val_acc')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_hm2, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)
    plt.title(f"Validation Accuracy: LR vs Weight Decay (Width={best_width})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_lr_wd.png"), dpi=300)
    plt.close()
    
    # Heatmap 3: Width vs Weight Decay (Fixed LR = 0.05 - best)
    best_lr = 0.05
    df_hm3 = df_grid[df_grid['lr'] == best_lr].pivot(index='width_mult', columns='weight_decay', values='val_acc')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_hm3, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)
    plt.title(f"Validation Accuracy: Width vs Weight Decay (LR={best_lr})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_width_wd.png"), dpi=300)
    plt.close()
    print("Grid Search Heatmaps generated.")

except Exception as e:
    print(f"Error generating Grid Search Heatmaps: {e}")


# 2. LOAD TRAINING HISTORY
print("Loading Training History...")
try:
    history_path = "/workspace/tesi-laurea/reports/final_run_reparam/training_history.json"
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        
        epochs = history['epochs']
        train_acc = history['train_acc']
        val_acc = history['val_acc']
        loss = history['loss']
        lr = history['lr']
        
        # Accuracy Plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_acc, label='Training Accuracy', linewidth=2)
        plt.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2)
        plt.title('Training vs Validation Accuracy (200 Epochs)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve_200.png"), dpi=300)
        plt.close()
        
        # Loss Plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss, label='Training Loss', color='red', linewidth=2)
        plt.title('Training Loss Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve_200.png"), dpi=300)
        plt.close()
        
        # LR Schedule Plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, lr, label='Learning Rate', color='purple', linewidth=2)
        plt.title('Cosine Annealing Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "lr_schedule_200.png"), dpi=300)
        plt.close()
        print("Training Curves generated.")
    else:
        print("Training history file not found.")

except Exception as e:
    print(f"Error generating Training Curves: {e}")


# 3. PER-CLASS ACCURACY BAR CHART
print("Generating Per-Class Accuracy Chart...")
try:
    report_path = "/workspace/tesi-laurea/docs/scrittura-tesi/tesi/immagini/classification_report.txt"
    classes = []
    f1_scores = []
    
    with open(report_path, "r") as f:
        lines = f.readlines()
        # Skip header lines (0 and 1)
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                classes.append(parts[0])
                f1_scores.append(float(parts[3])) # Using F1-Score as a proxy for per-class performance summary
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, f1_scores, color=sns.color_palette("viridis", len(classes)))
    plt.ylim(0.8, 1.0) # Start from 0.8 to highlight differences
    plt.title('F1-Score per Class')
    plt.xlabel('Class')
    plt.ylabel('F1-Score')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "per_class_f1_bars.png"), dpi=300)
    plt.close()
    print("Per-Class Bar Chart generated.")

except Exception as e:
    print(f"Error generating Per-Class Chart: {e}")

print("All missing charts generated successfully.")
