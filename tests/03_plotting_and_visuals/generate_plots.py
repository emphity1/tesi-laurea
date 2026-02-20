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
    'E': ('results_F_kd_ema', 'MobileNetECA-Rep-AdvAug (KD+EMA)'),
}
CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
COLORS = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c', 'D': '#d62728', 'E': '#9467bd'}

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
# 1. ACCURACY COMPARISON (val_acc per tutti e 5)
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
# 2. LOSS COMPARISON (train_loss per tutti e 5)
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
# 3. TRAIN vs VAL ACC per modello finale (E)
# ============================================================
h = histories['E']
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(h['epoch'], h['train_acc'], label='Training Accuracy', color='#1f77b4', linewidth=1.5)
ax.plot(h['epoch'], h['val_acc'], label='Validation Accuracy', color='#ff7f0e', linewidth=1.5)
ax.set_xlabel('Epoca', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Training vs Validation Accuracy (MobileNetECA-Rep-AdvAug (KD+EMA))', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 200)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'v3_train_vs_val_acc.png'), dpi=200)
plt.close()
print("✓ v3_train_vs_val_acc.png")

# ============================================================
# 4. TRAIN LOSS per modello finale (E)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(h['epoch'], h['train_loss'], color='#d62728', linewidth=1.5)
ax.set_xlabel('Epoca', fontsize=12)
ax.set_ylabel('Training Loss', fontsize=12)
ax.set_title('Progressione Loss di Training (MobileNetECA-Rep-AdvAug (KD+EMA))', fontsize=14)
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
# 6. CONFUSION MATRIX (modello E)
# ============================================================
pred_E = predictions['E']
cm = confusion_matrix(pred_E['targets'], pred_E['predictions'])

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(ax=ax, cmap='Blues', values_format='d', xticks_rotation=45)
ax.set_title('Matrice di Confusione - MobileNetECA-Rep-AdvAug (KD+EMA)', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=200)
plt.close()
print("✓ confusion_matrix.png")

# ============================================================
# 7. ROC CURVES (modello E, zoom)
# ============================================================
targets_bin = label_binarize(pred_E['targets'], classes=range(10))
probs = pred_E['probabilities']

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

targets_E = pred_E['targets']
preds_E = pred_E['predictions']
probs_E = pred_E['probabilities']

# Trova errori con alta confidenza
wrong_mask = targets_E != preds_E
wrong_indices = np.where(wrong_mask)[0]
wrong_confidences = probs_E[wrong_indices].max(axis=1)
top_wrong_order = np.argsort(-wrong_confidences)[:10]  # top 10 errori
top_wrong_indices = wrong_indices[top_wrong_order]

fig, axes = plt.subplots(2, 5, figsize=(15, 7))
for idx, ax in zip(top_wrong_indices, axes.flat):
    img, _ = testset_raw[idx]
    img = img.permute(1, 2, 0).numpy()
    ax.imshow(img)
    true_cls = CLASSES[targets_E[idx]]
    pred_cls = CLASSES[preds_E[idx]]
    conf = probs_E[idx].max() * 100
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
    'A Baseline': (77934, histories['A']['test_acc_final']),
    'B ECA': (78014, histories['B']['test_acc_final']),
    'C ECA-Rep': (62194, histories['C']['test_acc_final']),
    'D AdvAug': (62194, histories['D']['test_acc_final']),
    'E Final': (62194, 93.76),
}

# Modelli SOTA di riferimento (dalla tabella tesi - valori misurati)
sota = {
    'ResNet-20': (270_000, 92.60),
    'ResNet-32': (470_000, 93.53),
    'ResNet-44': (660_000, 94.01),
    'ResNet-56': (860_000, 94.37),
    'DenseNet-BC (k=12)': (770_000, 95.49),
    'MobileNetV2 (0.5x)': (700_000, 92.06),
    'MobileNetV2 (1.0x)': (2_240_000, 94.11),
    'MobileNetV3-Small': (2_540_000, 92.97),
    'ShuffleNetV2 (0.5x)': (350_000, 90.45),
    'GhostNet (ResNet-56)': (430_000, 93.38),
    'RepVGG-A0': (7_040_000, 94.19),
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
# GRAFICO 10: Accuracy vs FLOPs
# ============================================================

# FLOPs per i nostri modelli (misurati)
our_flops = {
    'A Baseline': (9_362_640, histories['A']['test_acc_final']),
    'B ECA': (9_362_640, histories['B']['test_acc_final']),
    'C ECA-Rep': (9_362_640, histories['C']['test_acc_final']),
    'D AdvAug': (9_362_640, histories['D']['test_acc_final']),
    'E Final': (9_362_640, 93.76),
}

# FLOPs SOTA (dalla tabella tesi - valori misurati su CIFAR-10)
sota_flops = {
    'ResNet-20': (40_810_000, 92.60),
    'ResNet-32': (69_120_000, 93.53),
    'ResNet-44': (97_440_000, 94.01),
    'ResNet-56': (125_750_000, 94.37),
    'MobileNetV2 (0.5x)': (25_330_000, 92.06),
    'MobileNetV2 (1.0x)': (87_980_000, 94.11),
    'MobileNetV3-Small': (60_000_000, 92.97),
    'ShuffleNetV2 (0.5x)': (10_900_000, 90.45),
    'GhostNet (ResNet-56)': (63_000_000, 93.38),
    'RepVGG-A0': (489_080_000, 94.19),
}

fig, ax = plt.subplots(figsize=(12, 7))

# SOTA scatter
for name, (flops, acc) in sota_flops.items():
    ax.scatter(flops, acc, s=80, alpha=0.7, zorder=3, color='gray')
    ax.annotate(name, (flops, acc), textcoords="offset points", xytext=(5, 5), fontsize=8, color='gray')

# Nostri modelli (solo E Final per chiarezza)
d_flops, d_acc = our_flops['E Final']
ax.scatter(d_flops, d_acc, s=200, zorder=5, color='red', marker='*', edgecolors='darkred', linewidths=0.5)
ax.annotate('MobileNetECA-Rep-AdvAug\n(Ours Final)', (d_flops, d_acc),
            textcoords="offset points", xytext=(8, -15), fontsize=10, fontweight='bold', color='red')

# Evidenzia parità FLOPs con ShuffleNetV2
shuffle_flops, shuffle_acc = sota_flops['ShuffleNetV2 (0.5x)']
ax.annotate('',
            xy=(d_flops, d_acc), xytext=(shuffle_flops, shuffle_acc),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5, ls='--'))
mid_flops = (d_flops + shuffle_flops) / 2
mid_acc = (d_acc + shuffle_acc) / 2
ax.annotate(f'+{d_acc - shuffle_acc:.1f}%\nstessi FLOPs',
            (mid_flops, mid_acc), textcoords="offset points", xytext=(30, 0),
            fontsize=9, color='blue', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='blue', alpha=0.8))

ax.set_xscale('log')
ax.set_xlabel('FLOPs (scala log)', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Efficiency Frontier: Accuracy vs FLOPs', fontsize=14)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'efficiency_flops.png'), dpi=200)
plt.close()
print("✓ efficiency_flops.png")

# ============================================================
# GRAFICO 11: Bar chart confronto accuratezza
# ============================================================

# Tutti i modelli per il bar chart (ordinati per accuratezza)
all_models = {
    'ShuffleNetV2\n(0.5x)': (90.45, 0.35, '#9e9e9e'),
    'MobileNetV2\n(0.5x)': (92.06, 0.70, '#9e9e9e'),
    'ResNet-20': (92.60, 0.27, '#9e9e9e'),
    'MobileNetV3\nSmall': (92.97, 2.54, '#9e9e9e'),
    'Ours\n(A Baseline)': (histories['A']['test_acc_final'], 0.078, '#ff9999'),
    'Ours\n(B ECA)': (histories['B']['test_acc_final'], 0.078, '#ff7777'),
    'Ours\n(C ECA-Rep)': (histories['C']['test_acc_final'], 0.062, '#ff5555'),
    'Ours\n(D AdvAug)': (histories['D']['test_acc_final'], 0.062, '#dd3333'),
    'Ours\n(E Final KD)': (93.76, 0.062, '#cc0000'), # KD + EMA Result
    'GhostNet': (93.38, 0.43, '#9e9e9e'),
    'ResNet-32': (93.53, 0.47, '#9e9e9e'),
    'ResNet-44': (94.01, 0.66, '#9e9e9e'),
    'MobileNetV2\n(1.0x)': (94.11, 2.24, '#9e9e9e'),
    'RepVGG-A0': (94.19, 7.04, '#9e9e9e'),
    'ResNet-56': (94.37, 0.86, '#9e9e9e'),
    'DenseNet-BC\n(k=12)': (95.49, 0.77, '#9e9e9e'),
}

# Ordina per accuratezza
sorted_models = sorted(all_models.items(), key=lambda x: x[1][0])

names = [n for n, _ in sorted_models]
accs = [v[0] for _, v in sorted_models]
params_m = [v[1] for _, v in sorted_models]
colors = [v[2] for _, v in sorted_models]

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.barh(names, accs, color=colors, edgecolor='black', linewidth=0.5, height=0.7)

# Annotazione parametri su ogni barra
for bar, acc, pm in zip(bars, accs, params_m):
    ax.text(bar.get_width() - 0.3, bar.get_y() + bar.get_height()/2,
            f'{acc:.2f}%  ({pm:.2f}M)',
            va='center', ha='right', fontsize=8, fontweight='bold',
            color='white' if acc > 91 else 'black')

ax.set_xlabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Confronto Accuratezza CIFAR-10: Modello Proposto vs SOTA', fontsize=14)
ax.set_xlim(89, 96)
ax.grid(True, axis='x', alpha=0.3)
ax.axvline(x=93.76, color='red', linestyle='--', alpha=0.5, label='Ours (Final KD)')
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'accuracy_bar_comparison.png'), dpi=200)
plt.close()
print("✓ accuracy_bar_comparison.png")

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
