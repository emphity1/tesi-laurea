"""
Script per generare tutti i grafici necessari per la tesi.
Richiede i dati di training salvati o risultati già ottenuti.

Usage:
    python genera_grafici_tesi.py

Output:
    - training_curve.png: Loss e accuracy durante training
    - accuracy_vs_params.png: Scatter plot confronto modelli
    - model_comparison_bar.png: Bar chart dimensioni modelli  
    - confusion_matrix.png: Confusion matrix di Mimir
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Setup stile grafici
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
OUTPUT_DIR = Path(__file__).parent.parent / "grafici"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# GRAFICO 1: Training Curve (Loss e Accuracy)
# ============================================================================
def plot_training_curve(epochs, train_acc, val_acc, train_loss, val_loss):
    """
    Plotta le curve di training e validation durante le 50 epoche.
    
    Args:
        epochs: list di numeri epoca (1-50)
        train_acc: list di accuracy su training set
        val_acc: list di accuracy su validation set
        train_loss: list di loss su training set
        val_loss: list di loss su validation set
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Accuracy
    ax1.plot(epochs, train_acc, label='Training', linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs, val_acc, label='Validation', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoca', fontsize=12)
    ax1.set_ylabel('Accuratezza (%)', fontsize=12)
    ax1.set_title('Andamento Accuratezza durante Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Subplot 2: Loss
    ax2.plot(epochs, train_loss, label='Training', linewidth=2, marker='o', markersize=3)
    ax2.plot(epochs, val_loss, label='Validation', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoca', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Andamento Loss durante Training', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curve.png', dpi=300, bbox_inches='tight')
    print(f"✅ Salvato: {OUTPUT_DIR / 'training_curve.png'}")
    plt.close()


# ============================================================================
# GRAFICO 2: Accuracy vs Parametri (Scatter Plot)
# ============================================================================
def plot_accuracy_vs_params():
    """
    Scatter plot che mostra il trade-off tra numero di parametri e accuratezza.
    Mimir deve essere evidenziato come punto ottimale.
    """
    models = {
        'Mimir (Nostro)': {'params': 0.20, 'acc': 88.0, 'color': 'red', 'marker': '*', 'size': 300},
        'MobileNetV2-0.5x': {'params': 0.70, 'acc': 90.1, 'color': 'blue', 'marker': 'o', 'size': 150},
        'ShuffleNetV2-0.5x': {'params': 0.35, 'acc': 87.5, 'color': 'green', 'marker': 's', 'size': 150},
        'ResNet-20': {'params': 0.27, 'acc': 91.2, 'color': 'purple', 'marker': '^', 'size': 150},
        'MobileNetV2-1.0x': {'params': 2.30, 'acc': 92.0, 'color': 'orange', 'marker': 'D', 'size': 150},
    }
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for name, data in models.items():
        ax.scatter(data['params'], data['acc'], 
                  c=data['color'], marker=data['marker'], s=data['size'],
                  alpha=0.7, edgecolors='black', linewidths=1.5,
                  label=name)
    
    # Linea tratteggiata per target (250k = 0.25M params, 85% acc)
    ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.5, label='Budget 250k params')
    ax.axhline(y=85.0, color='gray', linestyle='--', alpha=0.5, label='Target 85% acc')
    
    ax.set_xlabel('Parametri (Milioni)', fontsize=13)
    ax.set_ylabel('Accuratezza (%)', fontsize=13)
    ax.set_title('Trade-off Efficienza vs Accuratezza su CIFAR-10', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 2.5])
    ax.set_ylim([84, 93])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_vs_params.png', dpi=300, bbox_inches='tight')
    print(f"✅ Salvato: {OUTPUT_DIR / 'accuracy_vs_params.png'}")
    plt.close()


# ============================================================================
# GRAFICO 3: Confronto Dimensioni Modelli (Bar Chart)
# ============================================================================
def plot_model_comparison_bar():
    """
    Bar chart che confronta parametri, FLOPs e accuratezza dei modelli.
    """
    models = ['Mimir\n(Nostro)', 'MobileNetV2\n0.5x', 'ShuffleNetV2\n0.5x', 'ResNet-20']
    params = [0.20, 0.70, 0.35, 0.27]  # Milioni
    flops = [35, 97, 40, 41]  # Milioni
    accuracy = [88.0, 90.1, 87.5, 91.2]  # Percentuale
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, params, width, label='Parametri (M)', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x, [f/10 for f in flops], width, label='FLOPs (×10M)', color='salmon', edgecolor='black')
    bars3 = ax.bar(x + width, [a/10 for a in accuracy], width, label='Accuracy (×10%)', color='lightgreen', edgecolor='black')
    
    ax.set_xlabel('Modello', fontsize=13)
    ax.set_ylabel('Valore', fontsize=13)
    ax.set_title('Confronto Modelli: Parametri, FLOPs e Accuratezza', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Aggiungi valori sopra le barre
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison_bar.png', dpi=300, bbox_inches='tight')
    print(f"✅ Salvato: {OUTPUT_DIR / 'model_comparison_bar.png'}")
    plt.close()


# ============================================================================
# GRAFICO 4: Confusion Matrix
# ============================================================================
def plot_confusion_matrix(cm, classes):
    """
    Plotta la confusion matrix di Mimir su CIFAR-10.
    
    Args:
        cm: numpy array 10×10 con i conteggi
        classes: list di nomi delle 10 classi
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Numero di Predizioni'},
                ax=ax, linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Classe Predetta', fontsize=13)
    ax.set_ylabel('Classe Reale', fontsize=13)
    ax.set_title('Confusion Matrix - Mimir su CIFAR-10', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✅ Salvato: {OUTPUT_DIR / 'confusion_matrix.png'}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("🎨 Generazione grafici per la tesi...\n")
    
    # GRAFICO 1: Training Curve
    # TODO: Sostituire con dati reali da training log
    print("📊 Grafico 1: Training Curve")
    epochs = list(range(1, 51))
    # Dati esempio - SOSTITUIRE CON DATI REALI
    train_acc = [60 + i*0.6 - 0.01*i**1.5 for i in range(50)]
    val_acc = [55 + i*0.65 - 0.01*i**1.5 for i in range(50)]
    train_loss = [2.3 - i*0.04 + 0.0001*i**2 for i in range(50)]
    val_loss = [2.4 - i*0.038 + 0.0001*i**2 for i in range(50)]
    plot_training_curve(epochs, train_acc, val_acc, train_loss, val_loss)
    
    # GRAFICO 2: Accuracy vs Params
    print("📊 Grafico 2: Accuracy vs Parametri")
    plot_accuracy_vs_params()
    
    # GRAFICO 3: Bar Chart Confronto
    print("📊 Grafico 3: Confronto Modelli (Bar Chart)")
    plot_model_comparison_bar()
    
    # GRAFICO 4: Confusion Matrix
    # TODO: Calcolare confusion matrix reale sul test set
    print("📊 Grafico 4: Confusion Matrix")
    cifar_classes = ['Aereo', 'Auto', 'Uccello', 'Gatto', 'Cervo', 
                     'Cane', 'Rana', 'Cavallo', 'Nave', 'Camion']
    # Confusion matrix esempio - SOSTITUIRE CON DATI REALI
    cm_example = np.array([
        [920, 10, 15, 5, 3, 2, 8, 5, 25, 7],
        [8, 940, 3, 4, 1, 2, 5, 3, 10, 24],
        [12, 2, 860, 25, 30, 20, 35, 10, 4, 2],
        [4, 3, 20, 820, 25, 90, 25, 8, 3, 2],
        [2, 1, 25, 30, 880, 15, 30, 15, 1, 1],
        [1, 2, 15, 110, 20, 835, 10, 5, 1, 1],
        [3, 2, 18, 12, 15, 8, 935, 3, 2, 2],
        [2, 1, 12, 15, 20, 15, 5, 925, 2, 3],
        [18, 8, 4, 3, 1, 2, 3, 2, 945, 14],
        [5, 22, 2, 2, 1, 1, 2, 1, 15, 949]
    ])
    plot_confusion_matrix(cm_example, cifar_classes)
    
    print(f"\n✅ Tutti i grafici salvati in: {OUTPUT_DIR}")
    print("\n⚠️  NOTA: Alcuni grafici usano dati di esempio.")
    print("   Sostituire con dati reali da training logs e test results!")
