
import re
import matplotlib.pyplot as plt

log_path = '/workspace/tesi-laurea/src/legacy/mobilnet_eca_rep_advaug/eca_rep_advaug.log'
with open(log_path, 'r') as f:
    text = f.read()

# Pattern: Epoca 001/200 - Loss: 1.8531 | Train: 30.39% | Val: 48.34%
matches = re.findall(r"Epoca (\d+)/200 - Loss: ([\d.]+) \| Train: ([\d.]+)% \| Val: ([\d.]+)%", text)

if not matches:
    print("No data found in log!")
    exit(1)

epochs = [int(m[0]) for m in matches]
train_loss = [float(m[1]) for m in matches]
train_acc = [float(m[2]) for m in matches]
val_acc = [float(m[3]) for m in matches]

# Plot 1: Accuracy (Real)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, label='Training Accuracy', color='#1f77b4', linewidth=2, alpha=0.7)
plt.plot(epochs, val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
plt.title('Dinamiche di Accuratezza (Dati Reali dal Log)')
plt.xlabel('Epoca')
plt.ylabel('Accuratezza (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('/workspace/tesi-laurea/Tesi_Dmytro_Kozak/figure/train_vs_val_acc.png', dpi=300, bbox_inches='tight')

# Plot 2: Training Loss (Real) - The log only has Train Loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss', color='#d62728', linewidth=2)
plt.title('Dinamiche di Loss di Addestramento (Dati Reali dal Log)')
plt.xlabel('Epoca')
plt.ylabel('Loss (CrossEntropy)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('/workspace/tesi-laurea/Tesi_Dmytro_Kozak/figure/train_loss.png', dpi=300, bbox_inches='tight')

print(f"Generated plots for {len(epochs)} epochs.")
