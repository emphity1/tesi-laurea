import json
import matplotlib.pyplot as plt
import os
import numpy as np

# Check if file exists to avoid errors
def check_file(path):
    if not os.path.exists(path):
        print(f"File not found: {path} - Skipping...")
        return False
    return True

files = {
    "Baseline (MobileNetV2 0.5x)": "/workspace/tesi-laurea/src/legacy/standard/MobileNetPure_50_history.json",
    "+ ECA Attention": "/workspace/tesi-laurea/src/legacy/mobilnet_eca/MobileNetECA_200_history.json",
    "+ Reparameterization": "/workspace/tesi-laurea/src/legacy/mobilnet_eca_rep/MobileNetECA_Rep_200_history.json",
    "+ Advanced Augmentation (Final)": "/workspace/tesi-laurea/reports/adv_aug_test/MobileNetECA_Rep_200_history.json"
}

output_dir = "/workspace/tesi-laurea/reports/figures"
os.makedirs(output_dir, exist_ok=True)

# Load data
histories = {}
for label, path in files.items():
    if check_file(path):
        with open(path, 'r') as f:
            histories[label] = json.load(f)

# Plot Validation Accuracy
plt.figure(figsize=(10, 6))
for label, history in histories.items():
    # Fix potential key mismatch
    epochs = history.get('epochs', list(range(1, len(history['val_acc']) + 1)))
    val_acc = history['val_acc']
    plt.plot(epochs, val_acc, label=f"{label} (Best: {max(val_acc):.2f}%)")

plt.title("Confronto Accuratezza Validazione tra Configurazioni")
plt.xlabel("Epoche")
plt.ylabel("Accuratezza (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
plt.close()
print(f"Saved accuracy_comparison.png to {output_dir}")

# Plot Validation Loss
plt.figure(figsize=(10, 6))
for label, history in histories.items():
    # Fix potential key mismatch
    epochs = history.get('epochs', list(range(1, len(history.get('loss', [])) + 1)))
    # Some older history files might not have loss recorded or different keys
    if 'loss' in history:
        loss = history['loss']
        # Smooth loss for better visualization
        loss_smooth = np.convolve(loss, np.ones(5)/5, mode='valid')
        plt.plot(epochs[:len(loss_smooth)], loss_smooth, label=label)

plt.title("Confronto Loss di Training (Smoothed)")
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, "loss_comparison.png"))
plt.close()
print(f"Saved loss_comparison.png to {output_dir}")

# Efficiency Comparison (Pareto Frontier)
# Data from "Modelli CIFAR-10_ Accuratezza, FLOPs, Parametri.txt" and our results
models_data = [
    {"name": "ResNet-20", "acc": 92.60, "params": 0.27, "flops": 40.81, "color": "gray", "marker": "o"},
    {"name": "ResNet-32", "acc": 93.53, "params": 0.47, "flops": 69.12, "color": "gray", "marker": "o"},
    {"name": "ResNet-44", "acc": 94.01, "params": 0.66, "flops": 97.44, "color": "gray", "marker": "o"},
    {"name": "ResNet-56", "acc": 94.37, "params": 0.86, "flops": 125.75, "color": "gray", "marker": "o"},
    {"name": "VGG-16 (bn)", "acc": 94.16, "params": 15.25, "flops": 313.73, "color": "purple", "marker": "s"},
    {"name": "MobileNetV2 x0.5", "acc": 92.88, "params": 0.70, "flops": 27.97, "color": "blue", "marker": "^"},
    {"name": "MobileNetV2 x1.0", "acc": 93.79, "params": 2.24, "flops": 87.98, "color": "blue", "marker": "^"},
    {"name": "ShuffleNetV2 x0.5", "acc": 90.13, "params": 0.35, "flops": 10.90, "color": "green", "marker": "D"},
    {"name": "ShuffleNetV2 x1.0", "acc": 92.98, "params": 1.26, "flops": 45.00, "color": "green", "marker": "D"},
    {"name": "RepVGG-A0", "acc": 94.39, "params": 7.84, "flops": 489.08, "color": "orange", "marker": "v"},
    # Our Model
    {"name": "Ours (MobileNetECA-Rep)", "acc": 93.49, "params": 0.0766, "flops": 10.73, "color": "red", "marker": "*", "size": 200}
]

# Plot Accuracy vs Parameters
plt.figure(figsize=(12, 8))
for m in models_data:
    size = m.get("size", 100)
    plt.scatter(m["params"], m["acc"], c=m["color"], marker=m["marker"], s=size, label=m["name"] if "ResNet" not in m["name"] and "Shuffle" not in m["name"] else "") # Simple legend logic
    # Annotate significant points
    if m["name"] in ["Ours (MobileNetECA-Rep)", "ResNet-20", "MobileNetV2 x0.5", "ShuffleNetV2 x0.5"]:
        plt.annotate(m["name"], (m["params"], m["acc"]), xytext=(5, 5), textcoords='offset points', fontsize=9)

# Add Legend manually to avoid duplicates
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='*', color='w', label='Ours', markerfacecolor='red', markersize=15),
    Line2D([0], [0], marker='o', color='w', label='ResNet Family', markerfacecolor='gray', markersize=10),
    Line2D([0], [0], marker='^', color='w', label='MobileNetV2', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='D', color='w', label='ShuffleNetV2', markerfacecolor='green', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='VGG', markerfacecolor='purple', markersize=10),
]
plt.legend(handles=legend_elements, loc='lower right')

plt.xscale('log')
plt.title("Efficienza Parametrica: Accuratezza vs Numero di Parametri (Log Scale)")
plt.xlabel("Parametri (Milioni) - Scala Logaritmica")
plt.ylabel("Accuratezza (%)")
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.savefig(os.path.join(output_dir, "efficiency_params.png"))
plt.close()
print(f"Saved efficiency_params.png to {output_dir}")

# Plot Accuracy vs FLOPs
plt.figure(figsize=(12, 8))
for m in models_data:
    size = m.get("size", 100)
    plt.scatter(m["flops"], m["acc"], c=m["color"], marker=m["marker"], s=size)
    if m["name"] in ["Ours (MobileNetECA-Rep)", "ResNet-20", "MobileNetV2 x0.5", "ShuffleNetV2 x0.5"]:
        plt.annotate(m["name"], (m["flops"], m["acc"]), xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.legend(handles=legend_elements, loc='lower right')
plt.xscale('log')
plt.title("Efficienza Computazionale: Accuratezza vs FLOPs (Log Scale)")
plt.xlabel("FLOPs (Milioni) - Scala Logaritmica")
plt.ylabel("Accuratezza (%)")
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.savefig(os.path.join(output_dir, "efficiency_flops.png"))
plt.close()
print(f"Saved efficiency_flops.png to {output_dir}")
