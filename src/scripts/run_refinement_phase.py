import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from datetime import datetime
import argparse

# Import our modular training components
from train import train_epoch, validate, get_cifar10_loaders, save_checkpoint, save_metrics
from model import MobileNetECA

def run_refinement(configs, output_dir, epochs=50):
    """
    Run full training for selected configurations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Pre-load data loaders (shared)
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=128, num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dataset loaded. Training on {device}.\n")
    
    results = []
    
    print(f"{'='*80}")
    print(f"STARTING REFINEMENT PHASE - {len(configs)} CONFIGURATIONS")
    print(f"Epochs: {epochs} (No early stopping)")
    print(f"{'='*80}\n")
    
    for idx, config in enumerate(configs, 1):
        print(f"--- Config {idx}/{len(configs)} ---")
        print(f"LR: {config['lr']}, Width: {config['width_mult']}, WD: {config['weight_decay']}")
        
        # Setup directories
        run_name = f"refine_{idx:02d}_lr{config['lr']}_w{config['width_mult']}_wd{config['weight_decay']}"
        save_dir = os.path.join(output_dir, run_name, 'models')
        log_dir = os.path.join(output_dir, run_name, 'logs')
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Model
        model = MobileNetECA(
            num_classes=10,
            width_mult=config['width_mult'],
            lr_scale=1.54
        ).to(device)
        
        # Optimizer
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=0.9,
            weight_decay=config['weight_decay']
        )
        
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_acc = 0.0
        best_epoch = 0
        metrics = {'epochs': [], 'config': config}
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            
            # Train & Validate
            train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
            val_acc, val_loss = validate(model, test_loader, criterion, device)
            
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            ep_time = time.time() - ep_start
            
            # Track best
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                save_checkpoint(model, optimizer, epoch, train_acc, best_acc, os.path.join(save_dir, 'best_model.pth'))
            
            # Log
            metrics['epochs'].append({
                'epoch': epoch,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'lr': current_lr,
                'time': ep_time
            })
            
            indicator = "🏆" if val_acc == best_acc else " "
            print(f"  Ep {epoch:2d}: Train {train_acc:5.2f}% | Val {val_acc:5.2f}% {indicator} | Best: {best_acc:.2f}%")
            
        total_time = (time.time() - start_time) / 60
        print(f"\n✓ Completed in {total_time:.1f} min. Best Acc: {best_acc:.2f}% @ Ep {best_epoch}\n")
        
        # Save final metrics
        save_metrics(metrics, os.path.join(log_dir, 'final_metrics.json'))
        
        results.append({
            'rank': idx,
            'config': config,
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'total_time_min': total_time
        })
        
    # Save overall summary
    summary_path = os.path.join(output_dir, 'refinement_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Print final leaderboard
    print(f"{'='*80}")
    print(f"REFINEMENT RESULTS (Sorted by Accuracy)")
    print(f"{'='*80}")
    results.sort(key=lambda x: x['best_acc'], reverse=True)
    
    for i, res in enumerate(results, 1):
        cfg = res['config']
        print(f"{i}. {res['best_acc']:.2f}% | lr={cfg['lr']}, w={cfg['width_mult']}, wd={cfg['weight_decay']} | (Ep {res['best_epoch']})")
        
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    # TOP 5 CONFIGs from partial analysis
    top_5_configs = [
        {'lr': 0.025, 'width_mult': 0.4, 'weight_decay': 0.001},     # Run 85 (Rank 1 score)
        {'lr': 0.025, 'width_mult': 0.45, 'weight_decay': 0.0005},   # Run 89 
        {'lr': 0.025, 'width_mult': 0.35, 'weight_decay': 0.001},    # Run 80
        {'lr': 0.025, 'width_mult': 0.45, 'weight_decay': 0.001},    # Run 90 (Rank 1 acc)
        {'lr': 0.025, 'width_mult': 0.5, 'weight_decay': 0.0002},    # Run 92 (Your favorite)
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/workspace/tesi-laurea/reports/refinement_phase/run_{timestamp}"
    
    run_refinement(top_5_configs, output_dir, epochs=50)
