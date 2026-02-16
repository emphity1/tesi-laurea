
import matplotlib.pyplot as plt
import json
import numpy as np

# Load real data
history_path = '/workspace/tesi-laurea/reports/adv_aug_test/MobileNetECA_Rep_200_history.json'
with open(history_path, 'r') as f:
    history = json.load(f)

epochs = np.arange(1, 201)
train_acc = history['train_acc']
val_acc = history['val_acc']
train_loss = history['loss']

# Plot 1: Train vs Val Accuracy (Real Data)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, label='Training Accuracy', color='#1f77b4', linewidth=2, alpha=0.8)
plt.plot(epochs, val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
plt.title('Training vs Validation Accuracy (MobileNetECA-Rep + AdvAug)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('/workspace/tesi-laurea/Tesi_Dmytro_Kozak/figure/train_vs_val_acc_real.png', dpi=300, bbox_inches='tight')
print("Generated figure/train_vs_val_acc_real.png")

# Plot 2: Training Loss (Real Data)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss', color='#d62728', linewidth=2)
plt.title('Training Loss Progression')
plt.xlabel('Epochs')
plt.ylabel('Loss (CrossEntropy)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('/workspace/tesi-laurea/Tesi_Dmytro_Kozak/figure/train_loss_real.png', dpi=300, bbox_inches='tight')
print("Generated figure/train_loss_real.png")
