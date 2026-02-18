
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Path setting
script_dir = os.path.dirname(os.path.abspath(__file__))
# Correct path to V3 JSON (which is in legacy folder)
json_path = os.path.join(script_dir, "../legacy/mobilnet_eca_rep_advaug/v3.json")
save_dir = os.path.join(script_dir, "../legacy/mobilnet_eca_rep_advaug/v3_results")
os.makedirs(save_dir, exist_ok=True)

# Load data
with open(json_path, 'r') as f:
    data = json.load(f)

epochs = data["epoch"]
train_loss = data["train_loss"]
train_acc = data["train_acc"]
val_acc = data["val_acc"]
lr = data["lr"]

# Improved Style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif' # Matches Thesis LaTeX font

# 1. Train vs Val Accuracy (Classic thesis plot)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, label='Training Accuracy', color='#1f77b4', linewidth=2)
plt.plot(epochs, val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
plt.title(f'Training Dynamics (V3 Clean) - Best Val Acc: {max(val_acc):.2f}%')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "v3_train_vs_val_acc.png"))
plt.close()

# 2. Training Loss (Smoothed)
def smooth(scalars, weight=0.6):  # Weight between 0 and 1
    last = scalars[0] 
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss (Raw)', color='blue', alpha=0.3)
plt.plot(epochs, smooth(train_loss), label='Training Loss (Smoothed)', color='red', linewidth=2.5)
plt.title('Training Loss Progression')
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "v3_train_loss.png"))
plt.close()

# 3. Learning Rate Schedule
plt.figure(figsize=(10, 6))
plt.plot(epochs, lr, label='Learning Rate', color='#2ca02c', linewidth=2)
plt.title('Cosine Annealing Schedule')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "v3_lr_schedule.png"))
plt.close()

print(f"Grafici generati in: {save_dir}")
