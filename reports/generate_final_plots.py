import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Configurazione Stile
plt.style.use('ggplot')
OUTPUT_DIR = '/workspace/tesi-laurea/reports/final_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dati Ablation Study Hardcoded (dai risultati consolidati)
ablation_data = {
    'Baseline (Vanilla)': 89.92,
    'Golden (Pure)': 90.87,
    'MobileNetECA (Std)': 92.12,
    'MobileNetECA (Rep)': 92.47,
    'Advanced (Aug)': 93.50
}

# 1. Ablation Study Bar Chart
def plot_ablation_bars():
    labels = list(ablation_data.keys())
    values = list(ablation_data.values())
    colors = ['#bdc3c7', '#95a5a6', '#3498db', '#2980b9', '#e74c3c'] 
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors, alpha=0.9, width=0.6)
    
    # Linea Baseline
    plt.axhline(y=90.87, color='gray', linestyle='--', alpha=0.5, label='Reference (Golden)')
    
    # Annotazioni
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.ylim(88, 94.5)
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Ablation Study: Progressive Improvements')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_study_bars.png'), dpi=300)
    plt.close()
    print("Generato: ablation_study_bars.png")

# 2. Comparative Learning Curves
# Carichiamo i JSON reali (se disponibili, altrimenti mockiamo per demo)
# Path definitivi
paths = {
    'Standard': '/workspace/tesi-laurea/reports/eca_vs_eca_parametrized/MobileNetECA_200_history.json',
    'Reparam': '/workspace/tesi-laurea/reports/eca_vs_eca_parametrized/MobileNetECA_Rep_200_history.json',
    'Advanced': '/workspace/tesi-laurea/reports/adv_aug_test/MobileNetECA_Rep_200_history.json'
}

def plot_learning_curves():
    plt.figure(figsize=(12, 6))
    
    for label, path in paths.items():
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    history = json.load(f)
                    acc = history.get('val_acc', [])
                    # Smoothing
                    if len(acc) > 200: acc = acc[:200]
                    epochs = range(1, len(acc) + 1)
                    plt.plot(epochs, acc, label=f'{label} (Max: {max(acc):.2f}%)', linewidth=2)
            except Exception as e:
                print(f"Errore lettura {label}: {e}")
        else:
            print(f"File mancante: {path}")

    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Training Dynamics: Standard vs Reparam vs Advanced')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparative_learning_curves.png'), dpi=300)
    plt.close()
    print("Generato: comparative_learning_curves.png")

if __name__ == "__main__":
    plot_ablation_bars()
    plot_learning_curves()
