"""
Script per generare grafici comparativi tra MobileNetECA e altri modelli.
Include i risultati REALI di LAST_USED.py e confronti con letteratura.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Setup
sns.set_style("whitegrid")
OUTPUT_DIR = Path("/workspace/tesi-laurea/grafici")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Dati REALI
MOBILENET_ECA = {
    'name': 'MobileNetECA\n(Nostro)',
    'params': 0.054,  # 54k
    'macs': 9.4,      # 9.4M
    'accuracy': 89.02,
    'color': '#E63946',
    'marker': '*',
    'size': 500
}

# Altri modelli da letteratura
OTHER_MODELS = [
    {'name': 'ResNet-20', 'params': 0.27, 'macs': 41, 'accuracy': 91.2, 'color': '#457B9D', 'marker': '^', 'size': 200},
    {'name': 'MobileNetV2-0.5x', 'params': 0.70, 'macs': 97, 'accuracy': 90.1, 'color': '#2A9D8F', 'marker': 'o', 'size': 200},
    {'name': 'ShuffleNetV2-0.5x', 'params': 0.35, 'macs': 40, 'accuracy': 87.5, 'color': '#E76F51', 'marker': 's', 'size': 200},
    {'name': 'ResNet-32', 'params': 0.46, 'macs': 69, 'accuracy': 92.5, 'color': '#F4A261', 'marker': 'D', 'size': 200},
]

# Dati training MobileNetECA
TRAINING_DATA = {
    'epochs': [1, 2, 3, 4, 5, 10, 17, 25, 37, 40, 45, 50],
    'train_acc': [37.86, 58.19, 67.48, 72.61, 75.67, 82.47, 85.83, 88.58, 92.29, 92.85, 93.83, 94.18],
    'val_acc': [47.96, 61.87, 68.46, 72.97, 76.11, 80.19, 84.89, 86.02, 88.22, 88.52, 88.73, 89.02]
}


def plot_efficiency_comparison():
    """Grafico 1: Params vs Accuracy - mostra efficienza parametrica"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot altri modelli
    for model in OTHER_MODELS:
        ax.scatter(model['params'], model['accuracy'],
                  c=model['color'], marker=model['marker'], s=model['size'],
                  alpha=0.7, edgecolors='black', linewidths=2,
                  label=model['name'], zorder=2)
    
    # Plot MobileNetECA (evidenziato)
    ax.scatter(MOBILENET_ECA['params'], MOBILENET_ECA['accuracy'],
              c=MOBILENET_ECA['color'], marker=MOBILENET_ECA['marker'],
              s=MOBILENET_ECA['size'], alpha=0.9, edgecolors='black',
              linewidths=2.5, label=MOBILENET_ECA['name'], zorder=3)
    
    # Annotazione MobileNetECA
    ax.annotate(f'{MOBILENET_ECA["accuracy"]:.2f}%\n{int(MOBILENET_ECA["params"]*1000)}k params',
                xy=(MOBILENET_ECA['params'], MOBILENET_ECA['accuracy']),
                xytext=(0.15, 86), fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # Zona efficiente
    ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.4, linewidth=2, label='Budget 250k params')
    ax.fill_between([0, 0.25], 85, 95, alpha=0.15, color='green', label='Zona Efficiente')
    
    ax.set_xlabel('Parametri (Milioni)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Efficienza Parametrica su CIFAR-10\nMobileNetECA vs Modelli SOTA',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 0.8])
    ax.set_ylim([86, 93])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'efficiency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Salvato: {OUTPUT_DIR / 'efficiency_comparison.png'}")
    plt.close()


def plot_params_macs_accuracy():
    """Grafico 2: Confronto triplo (Params, MACs, Accuracy)"""
    models = [MOBILENET_ECA] + OTHER_MODELS
    names = [m['name'] for m in models]
    params = [m['params'] for m in models]
    macs = [m['macs'] / 100 for m in models]  # Scala per visualizzazione
    accuracy = [m['accuracy'] / 10 for m in models]  # Scala per visualizzazione
    
    x = np.arange(len(names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars1 = ax.bar(x - width, params, width, label='Parametri (M)',
                   color='#457B9D', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x, macs, width, label='MACs (×100M)',
                   color='#E76F51', edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x + width, accuracy, width, label='Accuracy (×10%)',
                   color='#2A9D8F', edgecolor='black', linewidth=1.2)
    
    # Evidenzia MobileNetECA
    bars1[0].set_facecolor('#FFD700')
    bars1[0].set_edgecolor('red')
    bars1[0].set_linewidth(2.5)
    
    ax.set_xlabel('Modello', fontsize=14, fontweight='bold')
    ax.set_ylabel('Valore', fontsize=14, fontweight='bold')
    ax.set_title('Confronto Multi-dimensionale: Parametri, MACs e Accuratezza',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Valori sopra le barre
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'multi_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Salvato: {OUTPUT_DIR / 'multi_comparison.png'}")
    plt.close()


def plot_efficiency_score():
    """Grafico 3: Efficiency Score (Accuracy per parametro)"""
    models = [MOBILENET_ECA] + OTHER_MODELS
    names = [m['name'] for m in models]
    
    # Calcola efficiency: accuracy / (params × macs)
    efficiency = []
    for m in models:
        score = m['accuracy'] / (m['params'] * m['macs'])
        efficiency.append(score)
    
    # Normalizza per visualizzazione
    max_eff = max(efficiency)
    efficiency_norm = [e / max_eff * 100 for e in efficiency]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = [m['color'] for m in models]
    bars = ax.barh(names, efficiency_norm, color=colors,
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Evidenzia MobileNetECA
    bars[0].set_facecolor('#FFD700')
    bars[0].set_edgecolor('red')
    bars[0].set_linewidth(2.5)
    bars[0].set_alpha(1.0)
    
    ax.set_xlabel('Efficiency Score (normalizzato, max=100)', fontsize=13, fontweight='bold')
    ax.set_title('Efficiency Score: Accuracy / (Parametri × MACs)\\nMaggiore = Più Efficiente',
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Valori
    for i, (bar, score) in enumerate(zip(bars, efficiency_norm)):
        ax.text(score + 2, bar.get_y() + bar.get_height()/2,
               f'{score:.1f}', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'efficiency_score.png', dpi=300, bbox_inches='tight')
    print(f"✅ Salvato: {OUTPUT_DIR / 'efficiency_score.png'}")
    plt.close()


def plot_training_curve():
    """Grafico 4: Training curve di MobileNetECA"""
    # Interpola per tutte le 50 epoche
    epochs_full = list(range(1, 51))
    train_acc_full = np.interp(epochs_full, TRAINING_DATA['epochs'], TRAINING_DATA['train_acc'])
    val_acc_full = np.interp(epochs_full, TRAINING_DATA['epochs'], TRAINING_DATA['val_acc'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Linee interpolate
    ax.plot(epochs_full, train_acc_full, linewidth=2.5, color='#2E86AB',
           alpha=0.7, label='Training Accuracy')
    ax.plot(epochs_full, val_acc_full, linewidth=2.5, color='#A23B72',
           alpha=0.7, label='Validation Accuracy')
    
    # Punti noti
    ax.scatter(TRAINING_DATA['epochs'], TRAINING_DATA['train_acc'],
              color='#2E86AB', s=80, zorder=5, edgecolors='black', linewidths=1)
    ax.scatter(TRAINING_DATA['epochs'], TRAINING_DATA['val_acc'],
              color='#A23B72', s=80, zorder=5, edgecolors='black', linewidths=1)
    
    # Annotazione finale
    ax.annotate(f'Finale: {TRAINING_DATA["val_acc"][-1]:.2f}%',
                xy=(50, TRAINING_DATA['val_acc'][-1]),
                xytext=(40, 80), fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    ax.set_xlabel('Epoca', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuratezza (%)', fontsize=13, fontweight='bold')
    ax.set_title('Curva di Training - MobileNetECA su CIFAR-10',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([35, 100])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curve.png', dpi=300, bbox_inches='tight')
    print(f"✅ Salvato: {OUTPUT_DIR / 'training_curve.png'}")
    plt.close()


def plot_architecture_comparison():
    """Grafico 5: Confronto architetture (bubble chart)"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = [MOBILENET_ECA] + OTHER_MODELS
    
    for model in models:
        # Size della bolla proporzionale a MACs
        bubble_size = model['macs'] * 10
        alpha = 0.9 if model == MOBILENET_ECA else 0.6
        edge_width = 2.5 if model == MOBILENET_ECA else 1.5
        
        ax.scatter(model['params'], model['accuracy'], s=bubble_size,
                  c=model['color'], marker=model['marker'], alpha=alpha,
                  edgecolors='black', linewidths=edge_width,
                  label=model['name'], zorder=3 if model == MOBILENET_ECA else 2)
    
    ax.set_xlabel('Parametri (M)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Bubble Chart: Params vs Accuracy (bubble size = MACs)',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Zona Pareto-optimal
    ax.axvline(x=0.25, color='green', linestyle=':', alpha=0.5, linewidth=2)
    ax.axhline(y=88, color='green', linestyle=':', alpha=0.5, linewidth=2)
    ax.fill_between([0, 0.25], 88, 95, alpha=0.1, color='green')
    ax.text(0.05, 91, 'Zona\nPareto-Optimal', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'architecture_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Salvato: {OUTPUT_DIR / 'architecture_comparison.png'}")
    plt.close()


if __name__ == "__main__":
    print("🎨 Generazione grafici comparativi...\n")
    
    plot_efficiency_comparison()
    plot_params_macs_accuracy()
    plot_efficiency_score()
    plot_training_curve()
    plot_architecture_comparison()
    
    print(f"\n✅ Tutti i grafici salvati in: {OUTPUT_DIR}")
    print("\n📊 Grafici generati:")
    print("  1. efficiency_comparison.png - Params vs Accuracy")
    print("  2. multi_comparison.png - Confronto triplo (Params, MACs, Acc)")
    print("  3. efficiency_score.png - Score di efficienza")
    print("  4. training_curve.png - Curva training MobileNetECA")
    print("  5. architecture_comparison.png - Bubble chart comparativo")
    
    print("\n🎯 MobileNetECA: 54k parametri, 89.02% accuracy - IL PIÙ EFFICIENTE!")
