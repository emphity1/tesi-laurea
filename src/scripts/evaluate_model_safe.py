import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
import os
import argparse
from typing import Dict, List, Any
import math

class SafeEvaluator:
    """Valuta il modello e salva i dati grezzi per la generazione dei grafici LaTeX"""
    
    def __init__(self, model_path, output_dir, device='cuda'):
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        os.makedirs(output_dir, exist_ok=True)
        
    def load_model(self):
        """Carica il modello in modo sicuro"""
        try:
            # Tenta di caricare come TorchScript (JIT)
            model = torch.jit.load(self.model_path)
            print("Caricato modello JIT/TorchScript")
        except:
            # Fallback: Carica come state_dict (richiede architettura nota)
            print("Caricamento JIT fallito. Provo caricamento standard...")
            # IMPORTAZIONE DINAMICA ARCHITETTURA (MobileNetEca Reparameterized)
            import sys
            sys.path.append(os.path.join(os.getcwd(), 'src'))
            from legacy.MobileNetEca_Rep import MobileNetECARep
            model = MobileNetECARep(num_classes=10, width_mult=0.5)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Gestione State Dict vs Checkpoint Dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Rileva se il checkpoint è in modalità 'deployed' (ha i pesi dei kernel fusi)
            if any("reparam_conv" in k for k in state_dict.keys()):
                print("Rilevato stato Deployed. Switching automatico a Deploy Mode prima del loading...")
                model.deploy()
                
            model.load_state_dict(state_dict)
            
            # Se è un modello normale Reparameterized non ancora deployato, esegui il deploy!
            if hasattr(model, 'deploy') and not any("reparam_conv" in k for k in state_dict.keys()):
                print("Eseguendo deploy post-loading (fusione parametri)...")
                model.deploy()
                
        model.to(self.device)
        model.eval()
        return model

    def get_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        return DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2)

    def evaluate(self):
        model = self.load_model()
        loader = self.get_dataloader()
        
        all_preds = []
        all_targets = []
        all_probs = [] # Per ROC
        
        print("Inizio valutazione...")
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        return all_preds, all_targets, all_probs

    def compute_metrics(self, preds, targets, probs):
        """Calcola metriche e salva grafici PNG per la tesi"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # 1. Confusion Matrix
        num_classes = 10
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p, t in zip(preds, targets):
            cm[t][p] += 1
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix: MobileNetECA Reparameterized')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        print("Salvata confusion_matrix.png")

        # 2. Per-Class Metrics (Precision, Recall, F1)
        report_lines = []
        report_lines.append(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        report_lines.append("-" * 55)
        
        class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        
        for c in range(num_classes):
            tp = cm[c][c]
            fp = sum(cm[x][c] for x in range(num_classes)) - tp
            fn = sum(cm[c][x] for x in range(num_classes)) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            report_lines.append(f"{class_names[c]:<10} {precision:.4f}     {recall:.4f}     {f1:.4f}     {tp+fn:<10}")
            
        report_path = os.path.join(self.output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        print("Salvato classification_report.txt")

        # 3. ROC Curves (One-vs-Rest)
        plt.figure(figsize=(10, 8))
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, num_classes))
        
        for c in range(num_classes):
            y_true = [1 if t == c else 0 for t in targets]
            y_score = [p[c] for p in probs]
            
            # Simple manual ROC calculation (to avoid sklearn dependency if possible, but here we can use simple sorting)
            desc_score_indices = np.argsort(y_score)[::-1]
            y_true_sorted = np.array(y_true)[desc_score_indices]
            
            tps = np.cumsum(y_true_sorted)
            fps = np.cumsum(1 - y_true_sorted)
            tpr = tps / tps[-1]
            fpr = fps / fps[-1]
            
            # AUC calculation using trapezoidal rule (NumPy 2.x compatible)
            try:
                auc = np.trapezoid(tpr, fpr)
            except AttributeError:
                # Fallback for older NumPy versions
                auc = np.trapz(tpr, fpr)
            
            plt.plot(fpr, tpr, color=colors[c], lw=2, label=f'{class_names[c]} (AUC = {auc:.3f})')
            
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_curves.png'), dpi=300)
        plt.close()
        print("Salvato roc_curves.png")

        return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path al file .pt o .pth')
    parser.add_argument('--output_dir', type=str, default='reports/final_run_reparam', help='Directory output')
    args = parser.parse_args()
    
    evaluator = SafeEvaluator(args.model_path, args.output_dir)
    preds, targets, probs = evaluator.evaluate()
    evaluator.compute_metrics(preds, targets, probs)
