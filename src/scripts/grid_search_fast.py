"""
Fast Grid Search with Early Stopping for MobileNetECA
Optimized for testing many hyperparameter combinations quickly

Strategy:
- Train for reduced epochs (e.g., 20-25)
- Apply early stopping based on accuracy threshold at checkpoint epoch
- Discard configurations that don't reach minimum threshold
- Test significantly more combinations in less time
"""

import torch
import itertools
import json
import os
from datetime import datetime
import argparse

from train import train_epoch, validate, get_cifar10_loaders, save_checkpoint, save_metrics
from model import MobileNetECA, count_parameters, format_number

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time


def fast_train_model(config, checkpoint_epoch=20, min_threshold=85.0):
    """
    Fast training with early stopping based on threshold
    
    Args:
        config: Training configuration dict
        checkpoint_epoch: Epoch at which to check threshold (default: 20)
        min_threshold: Minimum validation accuracy required at checkpoint (default: 85%)
    
    Returns:
        metrics dict if successful, None if stopped early
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Data loaders (reuse existing ones if passed)
    if 'train_loader' not in config or 'test_loader' not in config:
        train_loader, test_loader = get_cifar10_loaders(
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 2)
        )
    else:
        train_loader = config['train_loader']
        test_loader = config['test_loader']
    
    # Model
    model = MobileNetECA(
        num_classes=10,
        width_mult=config['width_mult'],
        lr_scale=config.get('lr_scale', 1.54)
    ).to(device)
    
    # Optimizer & Scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.CrossEntropyLoss()
    
    # Training metrics
    best_val_acc = 0.0
    metrics = {
        'config': {k: v for k, v in config.items() if k not in ['train_loader', 'test_loader']},
        'epochs': [],
        'best_val_acc': 0.0,
        'best_epoch': 0,
        'early_stopped': False,
        'stopped_at_epoch': None
    }
    
    start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        train_acc, train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config['epochs']
        )
        
        # Validate
        val_acc, val_loss = validate(model, test_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - epoch_start
        
        # Save epoch metrics
        epoch_metrics = {
            'epoch': epoch,
            'train_acc': round(train_acc, 2),
            'train_loss': round(train_loss, 4),
            'val_acc': round(val_acc, 2),
            'val_loss': round(val_loss, 4),
            'lr': current_lr,
            'time': round(epoch_time, 1)
        }
        metrics['epochs'].append(epoch_metrics)
        
        # Update best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            metrics['best_val_acc'] = round(best_val_acc, 2)
            metrics['best_epoch'] = epoch
        
        # Print compact progress
        val_indicator = "↑" if val_acc > best_val_acc else " "
        print(f"  Ep {epoch:2d}: Train {train_acc:5.2f}% | Val {val_acc:5.2f}% {val_indicator} | Time {epoch_time:.1f}s", end='')
        
        # Early stopping check at checkpoint epoch
        if epoch == checkpoint_epoch:
            if val_acc < min_threshold:
                print(f" ❌ STOPPED (Val {val_acc:.2f}% < threshold {min_threshold}%)")
                metrics['early_stopped'] = True
                metrics['stopped_at_epoch'] = epoch
                metrics['total_time_minutes'] = round((time.time() - start_time) / 60, 2)
                return None  # Signal to skip this configuration
            else:
                print(f" ✓ Passed threshold ({val_acc:.2f}% >= {min_threshold}%)")
        else:
            print()
    
    total_time = time.time() - start_time
    metrics['total_time_minutes'] = round(total_time / 60, 2)
    
    # Save final metrics
    metrics_path = os.path.join(config['log_dir'], 'training_metrics.json')
    save_metrics(metrics, metrics_path)
    
    return metrics


def grid_search_fast(search_space, base_config, output_dir, checkpoint_epoch=20, min_threshold=85.0):
    """
    Perform fast grid search with early stopping
    
    Args:
        search_space: Dict mapping parameter names to lists of values
        base_config: Base configuration dict
        output_dir: Directory to save results
        checkpoint_epoch: Epoch to evaluate threshold (default: 20)
        min_threshold: Minimum accuracy threshold at checkpoint (default: 85%)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all combinations
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    combinations = list(itertools.product(*param_values))
    
    total_runs = len(combinations)
    
    print(f"\n{'='*70}")
    print(f"FAST GRID SEARCH FOR MOBILENETECA")
    print(f"{'='*70}")
    print(f"Total combinations: {total_runs}")
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"Minimum threshold: {min_threshold}%")
    print(f"\nSearch space:")
    for param, values in search_space.items():
        print(f"  {param}: {values}")
    print(f"{'='*70}\n")
    
    # Pre-load data loaders once (reuse for all runs)
    print("Loading CIFAR-10 dataset (shared across all runs)...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=base_config.get('batch_size', 128),
        num_workers=base_config.get('num_workers', 2)
    )
    print("Dataset loaded.\n")
    
    # Results storage
    results = {
        'search_space': search_space,
        'base_config': base_config,
        'checkpoint_epoch': checkpoint_epoch,
        'min_threshold': min_threshold,
        'runs': [],
        'successful_runs': [],
        'stopped_runs': [],
        'best_config': None,
        'best_val_acc': 0.0
    }
    
    # Run grid search
    successful_count = 0
    stopped_count = 0
    
    for run_idx, combination in enumerate(combinations, 1):
        print(f"\n{'='*70}")
        print(f"RUN {run_idx}/{total_runs}")
        print(f"{'='*70}")
        
        # Create config for this run
        config = base_config.copy()
        for param_name, param_value in zip(param_names, combination):
            config[param_name] = param_value
        
        print(f"Configuration:")
        for param_name in param_names:
            print(f"  {param_name}: {config[param_name]}")
        print()
        
        # Add shared data loaders
        config['train_loader'] = train_loader
        config['test_loader'] = test_loader
        
        # Create run-specific directories
        run_name = f"run_{run_idx:03d}_" + "_".join([f"{pn}_{pv}".replace(".", "p") for pn, pv in zip(param_names, combination)])
        
        config['save_dir'] = os.path.join(output_dir, run_name, 'models')
        config['log_dir'] = os.path.join(output_dir, run_name, 'logs')
        
        try:
            # Train with early stopping
            metrics = fast_train_model(config, checkpoint_epoch, min_threshold)
            
            if metrics is None:
                # Early stopped
                stopped_count += 1
                run_result = {
                    'run_id': run_idx,
                    'config': {pn: pv for pn, pv in zip(param_names, combination)},
                    'status': 'stopped_early',
                    'stopped_at_epoch': checkpoint_epoch
                }
                results['stopped_runs'].append(run_result)
                
            else:
                # Completed successfully
                successful_count += 1
                run_result = {
                    'run_id': run_idx,
                    'config': {pn: pv for pn, pv in zip(param_names, combination)},
                    'status': 'completed',
                    'best_val_acc': metrics['best_val_acc'],
                    'best_epoch': metrics['best_epoch'],
                    'final_train_acc': metrics['epochs'][-1]['train_acc'],
                    'final_val_acc': metrics['epochs'][-1]['val_acc'],
                    'total_time_minutes': metrics['total_time_minutes']
                }
                results['successful_runs'].append(run_result)
                
                # Update best config
                if metrics['best_val_acc'] > results['best_val_acc']:
                    results['best_val_acc'] = metrics['best_val_acc']
                    results['best_config'] = run_result['config'].copy()
                    results['best_run_id'] = run_idx
                    
                    print(f"\n🏆 NEW BEST: {metrics['best_val_acc']:.2f}% validation accuracy!")
            
            results['runs'].append(run_result)
            
        except Exception as e:
            print(f"\n❌ Run {run_idx} failed with error: {str(e)}")
            results['runs'].append({
                'run_id': run_idx,
                'config': {pn: pv for pn, pv in zip(param_names, combination)},
                'status': 'error',
                'error': str(e)
            })
        
        # Save intermediate results
        results_path = os.path.join(output_dir, 'grid_search_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nProgress: {run_idx}/{total_runs} | ✓ {successful_count} | ❌ {stopped_count}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"FAST GRID SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {successful_count} ({successful_count/total_runs*100:.1f}%)")
    print(f"Early stopped: {stopped_count} ({stopped_count/total_runs*100:.1f}%)")
    print(f"Failed: {total_runs - successful_count - stopped_count}")
    
    if results['best_config']:
        print(f"\nBest configuration (Run {results.get('best_run_id', 'N/A')}):")
        for param, value in results['best_config'].items():
            print(f"  {param}: {value}")
        print(f"\nBest validation accuracy: {results['best_val_acc']:.2f}%")
        
        # Save best config separately
        best_config_full = base_config.copy()
        best_config_full.update(results['best_config'])
        # Remove data loaders from config before saving
        best_config_full.pop('train_loader', None)
        best_config_full.pop('test_loader', None)
        
        best_config_path = os.path.join(output_dir, 'best_config.json')
        with open(best_config_path, 'w') as f:
            json.dump(best_config_full, f, indent=2)
        
        print(f"\nBest configuration saved to: {best_config_path}")
    else:
        print("\n  No successful runs")
    
    # Save final results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Full results saved to: {results_path}")
    print(f"{'='*70}\n")
    
    # Create summary file
    summary_path = os.path.join(output_dir, 'SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FAST GRID SEARCH SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Checkpoint epoch: {checkpoint_epoch}\n")
        f.write(f"Minimum threshold: {min_threshold}%\n")
        f.write(f"Total combinations: {total_runs}\n")
        f.write(f"Successful: {successful_count} ({successful_count/total_runs*100:.1f}%)\n")
        f.write(f"Early stopped: {stopped_count} ({stopped_count/total_runs*100:.1f}%)\n\n")
        
        f.write("="*70 + "\n")
        f.write("TOP 10 CONFIGURATIONS\n")
        f.write("="*70 + "\n\n")
        
        # Sort successful runs by accuracy
        sorted_runs = sorted(results['successful_runs'], 
                           key=lambda x: x['best_val_acc'], 
                           reverse=True)
        
        for rank, run in enumerate(sorted_runs[:10], 1):
            f.write(f"RANK {rank} - Run {run['run_id']}\n")
            f.write(f"  Validation Accuracy: {run['best_val_acc']}%\n")
            for param, value in run['config'].items():
                f.write(f"  {param}={value}\n")
            f.write(f"  Best epoch: {run['best_epoch']}\n")
            f.write("\n")
    
    print(f"Summary saved to: {summary_path}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Fast grid search with early stopping')
    
    # Search space - EXPANDED options
    parser.add_argument('--lr', nargs='+', type=float,
                       default=[0.01, 0.025, 0.05, 0.075, 0.1],
                       help='Learning rates to test')
    parser.add_argument('--width_mult', nargs='+', type=float,
                       default=[0.3, 0.35, 0.4, 0.42, 0.45, 0.5],
                       help='Width multipliers to test')
    parser.add_argument('--weight_decay', nargs='+', type=float,
                       default=[1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3],
                       help='Weight decay values to test')
    parser.add_argument('--lr_scale', nargs='+', type=float,
                       default=[1.54],
                       help='LR scale values to test (default: fixed at 1.54)')
    parser.add_argument('--momentum', nargs='+', type=float,
                       default=[0.9],
                       help='Momentum values to test (default: fixed at 0.9)')
    
    # Early stopping parameters
    parser.add_argument('--checkpoint_epoch', type=int, default=20,
                       help='Epoch to check minimum threshold (default: 20)')
    parser.add_argument('--min_threshold', type=float, default=85.0,
                       help='Minimum validation accuracy at checkpoint (default: 85%)')
    
    # Fixed parameters
    parser.add_argument('--epochs', type=int, default=25,
                       help='Maximum epochs per run (default: 25)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (fixed at 128)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Data loading workers')
    
    # Output
    parser.add_argument('--output_dir', type=str,
                       default='../../reports/grid_search_fast',
                       help='Directory to save results')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test with minimal combinations')
    
    args = parser.parse_args()
    
    # Define search space
    if args.quick_test:
        print("\n⚡ QUICK TEST MODE: Minimal search space")
        search_space = {
            'lr': [args.lr[0], args.lr[-1]],  # First and last
            'width_mult': [args.width_mult[0], args.width_mult[-1]],
            'weight_decay': [args.weight_decay[0], args.weight_decay[-1]],
        }
    else:
        search_space = {
            'lr': args.lr,
            'width_mult': args.width_mult,
            'weight_decay': args.weight_decay,
        }
        
        # Add lr_scale and momentum if varying
        if len(args.lr_scale) > 1:
            search_space['lr_scale'] = args.lr_scale
        if len(args.momentum) > 1:
            search_space['momentum'] = args.momentum
    
    # Base configuration
    base_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'lr_scale': args.lr_scale[0] if len(args.lr_scale) == 1 else None,
        'momentum': args.momentum[0] if len(args.momentum) == 1 else None,
        'save_dir': '',
        'log_dir': '',
    }
    
    # Remove None values
    base_config = {k: v for k, v in base_config.items() if v is not None}
    
    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"search_{timestamp}")
    
    # Run fast grid search
    results = grid_search_fast(
        search_space,
        base_config,
        output_dir,
        checkpoint_epoch=args.checkpoint_epoch,
        min_threshold=args.min_threshold
    )
    
    return results


if __name__ == '__main__':
    main()
