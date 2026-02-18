"""
Genera tutti i grafici per la tesi triennale.
Legge i risultati dai 4 modelli A/B/C/D e produce le figure.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
import os

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = {
    'A': ('results_A_baseline', 'MobileNet Baseline'),
    'B': ('results_B_eca', 'MobileNetECA'),
    'C': ('results_C_eca_rep', 'MobileNetECA-Rep'),
    'D': ('results_D_eca_rep_advaug', 'MobileNetECA-Rep-AdvAug'),
}
CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
COLORS = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c', 'D': '#d62728'}

OUTPUT_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Carica dati ---
histories = {}
predictions = {}

for key, (folder, label) in RESULTS.items():
    path = os.path.join(BASE_DIR, folder)
    with open(os.path.join(path, 'history.json')) as f:
        histories[key] = json.load(f)
    predictions[key] = np.load(os.path.join(path, 'test_predictions.npz'))

print("Dati caricati. Genero grafici...\n")

# ============================================================
# 1. ACCURACY COMPARISON (val_acc per tutti e 4)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for key, (_, label) in RESULTS.items():
    h = histories[key]
    ax.plot(h['epoch'], h['val_acc'], label=f"{label} ({h['test_acc_final']:.2f}%)", color=COLORS[key], linewidth=1.5)
ax.set_xlabel('Epoca', fontsize=12)
ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax.set_title('Confronto Accuratezza di Validazione', fontsize=14)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 200)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'accuracy_comparison.png'), dpi=200)
plt.close()
print("✓ accuracy_comparison.png")

# ============================================================
# 2. LOSS COMPARISON (train_loss per tutti e 4)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for key, (_, label) in RESULTS.items():
    h = histories[key]
    # Smoothing con media mobile
    losses = np.array(h['train_loss'])
    window = 5
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    ax.plot(range(window, len(losses)+1), smoothed, label=label, color=COLORS[key], linewidth=1.5)
ax.set_xlabel('Epoca', fontsize=12)
ax.set_ylabel('Training Loss (Smoothed)', fontsize=12)
ax.set_title('Confronto Loss di Training', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'loss_comparison.png'), dpi=200)
plt.close()
print("✓ loss_comparison.png")

# ============================================================
# 3. TRAIN vs VAL ACC per modello finale (D)
# ============================================================
h = histories['D']
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(h['epoch'], h['train_acc'], label='Training Accuracy', color='#1f77b4', linewidth=1.5)
ax.plot(h['epoch'], h['val_acc'], label='Validation Accuracy', color='#ff7f0e', linewidth=1.5)
ax.set_xlabel('Epoca', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Training vs Validation Accuracy (MobileNetECA-Rep-AdvAug)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 200)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'v3_train_vs_val_acc.png'), dpi=200)
plt.close()
print("✓ v3_train_vs_val_acc.png")

# ============================================================
# 4. TRAIN LOSS per modello finale (D)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(h['epoch'], h['train_loss'], color='#d62728', linewidth=1.5)
ax.set_xlabel('Epoca', fontsize=12)
ax.set_ylabel('Training Loss', fontsize=12)
ax.set_title('Progressione Loss di Training (MobileNetECA-Rep-AdvAug)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 200)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'v3_train_loss.png'), dpi=200)
plt.close()
print("✓ v3_train_loss.png")

# ============================================================
# 5. LR SCHEDULE
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(h['epoch'], h['lr'], color='#9467bd', linewidth=2)
ax.set_xlabel('Epoca', fontsize=12)
ax.set_ylabel('Learning Rate', fontsize=12)
ax.set_title('Cosine Annealing Learning Rate Schedule', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 200)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'v3_lr_schedule.png'), dpi=200)
plt.close()
print("✓ v3_lr_schedule.png")

# ============================================================
# 6. CONFUSION MATRIX (modello D)
# ============================================================
pred_D = predictions['D']
cm = confusion_matrix(pred_D['targets'], pred_D['predictions'])

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(ax=ax, cmap='Blues', values_format='d', xticks_rotation=45)
ax.set_title('Matrice di Confusione - MobileNetECA-Rep-AdvAug', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=200)
plt.close()
print("✓ confusion_matrix.png")

# ============================================================
# 7. ROC CURVES (modello D, zoom)
# ============================================================
targets_bin = label_binarize(pred_D['targets'], classes=range(10))
probs = pred_D['probabilities']

fig, ax = plt.subplots(figsize=(10, 8))
for i, cls_name in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(targets_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=1.5, label=f'{cls_name} (AUC={roc_auc:.4f})')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Curve ROC per classe (Zoom)', fontsize=14)
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim(-0.01, 0.15)
ax.set_ylim(0.80, 1.01)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'roc_curve_zoomed.png'), dpi=200)
plt.close()
print("✓ roc_curve_zoomed.png")

# ============================================================
# 8. TOP ERRORS (alta confidenza, modello D)
# ============================================================
import torchvision

# Carica immagini test originali (senza normalizzazione per visualizzazione)
testset_raw = torchvision.datasets.CIFAR10(
    root=os.path.join(os.path.dirname(BASE_DIR), 'data'),
    train=False, download=True,
    transform=torchvision.transforms.ToTensor()
)

targets_D = pred_D['targets']
preds_D = pred_D['predictions']
probs_D = pred_D['probabilities']

# Trova errori con alta confidenza
wrong_mask = targets_D != preds_D
wrong_indices = np.where(wrong_mask)[0]
wrong_confidences = probs_D[wrong_indices].max(axis=1)
top_wrong_order = np.argsort(-wrong_confidences)[:10]  # top 10 errori
top_wrong_indices = wrong_indices[top_wrong_order]

fig, axes = plt.subplots(2, 5, figsize=(15, 7))
for idx, ax in zip(top_wrong_indices, axes.flat):
    img, _ = testset_raw[idx]
    img = img.permute(1, 2, 0).numpy()
    ax.imshow(img)
    true_cls = CLASSES[targets_D[idx]]
    pred_cls = CLASSES[preds_D[idx]]
    conf = probs_D[idx].max() * 100
    ax.set_title(f'True: {true_cls}\nPred: {pred_cls} ({conf:.1f}%)', fontsize=9)
    ax.axis('off')
fig.suptitle('Errori ad Alta Confidenza (>90%)', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'top_errors.png'), dpi=200, bbox_inches='tight')
plt.close()
print("✓ top_errors.png")

# ============================================================
# 9. EFFICIENCY FRONTIER (Accuracy vs Params)
# ============================================================
# Nostri modelli
our_models = {
    'A Baseline': (histories['A']['total_params'], histories['A']['test_acc_final']),
    'B ECA': (histories['B']['total_params'], histories['B']['test_acc_final']),
    'C ECA-Rep': (histories['C'].get('deploy_params', histories['C']['total_params']), histories['C']['test_acc_final']),
    'D Final': (histories['D'].get('deploy_params', histories['D']['total_params']), histories['D']['test_acc_final']),
}

# Modelli SOTA di riferimento (dalla tabella tesi)
sota = {
    'ResNet-20': (270_000, 92.60),
    'ResNet-32': (470_000, 93.53),
    'ResNet-44': (660_000, 94.01),
    'ResNet-56': (860_000, 94.37),
    'VGG-16 (bn)': (15_250_000, 94.16),
    'MobileNetV2 (0.5x)': (700_000, 92.88),
    'MobileNetV2 (1.0x)': (2_240_000, 93.79),
    'ShuffleNetV2 (0.5x)': (350_000, 90.13),
    'RepVGG-A0': (7_840_000, 94.39),
}

fig, ax = plt.subplots(figsize=(12, 7))

# SOTA scatter
for name, (params, acc) in sota.items():
    ax.scatter(params, acc, s=80, alpha=0.7, zorder=3, color='gray')
    ax.annotate(name, (params, acc), textcoords="offset points", xytext=(5, 5), fontsize=8, color='gray')

# Nostri modelli
for name, (params, acc) in our_models.items():
    ax.scatter(params, acc, s=150, zorder=5, color='red', marker='*', edgecolors='darkred', linewidths=0.5)
    ax.annotate(name, (params, acc), textcoords="offset points", xytext=(5, -12), fontsize=9, fontweight='bold', color='red')

ax.set_xscale('log')
ax.set_xlabel('Numero di Parametri (scala log)', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Efficiency Frontier: Accuracy vs Parametri', fontsize=14)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'efficiency_params.png'), dpi=200)
plt.close()
print("✓ efficiency_params.png")

# ============================================================
# RIEPILOGO TESTUALE PER LA TESI
# ============================================================
print("\n" + "=" * 60)
print("RIEPILOGO ABLATION STUDY (per aggiornare Tabella 6.1)")
print("=" * 60)
print(f"{'Config':<12} {'Nome':<30} {'Val Acc':<12} {'Test Acc':<12} {'Params':<12}")
print("-" * 78)
for key, (_, label) in RESULTS.items():
    h = histories[key]
    deploy_p = h.get('deploy_params', h['total_params'])
    print(f"{key:<12} {label:<30} {h['best_val_acc']:.2f}%      {h['test_acc_final']:.2f}%      {deploy_p:,}")

print(f"\nTutti i grafici salvati in: {OUTPUT_DIR}")
print("Copia i file in Tesi_Dmytro_Kozak/figure/ per la tesi.")
