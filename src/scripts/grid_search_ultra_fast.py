"""
Ultra-Fast Grid Search with Two-Phase Strategy
Designed to test many hyperparameters in 2-3 hours

Strategy:
Phase 1 - SCREENING (1.5-2h): Test many configs with 25 epochs
  - Multi-checkpoint evaluation (epochs 15, 20, 25)
  - Smart scoring: accuracy + improvement trend
  - Early stopping for clearly bad configs
  
Phase 2 - REFINEMENT (0.5-1h): Train top-5 configs for 50 epochs
  - Full training to get final performance
  - Find the truly best configuration
"""

import torch
import itertools
import json
import os
from datetime import datetime
import argparse
import numpy as np

from train import train_epoch, validate, get_cifar10_loaders, save_checkpoint, save_metrics
from model import MobileNetECA, count_parameters, format_number

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time


def calculate_score(metrics_history, checkpoint_epochs=[8, 12, 15]):
    """
    Calculate a score that predicts final performance
    
    Args:
        metrics_history: List of epoch metrics
        checkpoint_epochs: Epochs to evaluate
    
    Returns:
        score: Predicted final performance score
        details: Dict with scoring breakdown
    """
    
    # Get validation accuracies at checkpoints
    val_accs = []
    for epoch in checkpoint_epochs:
        if epoch <= len(metrics_history):
            val_accs.append(metrics_history[epoch - 1]['val_acc'])
    
    if len(val_accs) == 0:
        return 0.0, {}
    
    # Latest validation accuracy (most important)
    latest_val_acc = val_accs[-1]
    
    # Calculate improvement trend
    if len(val_accs) >= 2:
        # Average improvement per epoch between checkpoints
        improvements = []
        for i in range(1, len(val_accs)):
            epochs_diff = checkpoint_epochs[i] - checkpoint_epochs[i-1]
            acc_diff = val_accs[i] - val_accs[i-1]
            improvements.append(acc_diff / epochs_diff)
        
        avg_improvement_rate = np.mean(improvements)
        
        # Predict improvement over next 35 epochs (to reach 50 total)
        predicted_final = latest_val_acc + (avg_improvement_rate * 35)
    else:
        avg_improvement_rate = 0.0
        predicted_final = latest_val_acc
    
    # Combined score: 70% current accuracy + 30% predicted final
    score = 0.7 * latest_val_acc + 0.3 * predicted_final
    
    details = {
        'latest_val_acc': round(latest_val_acc, 2),
        'improvement_rate': round(avg_improvement_rate, 3),
        'predicted_final': round(predicted_final, 2),
        'score': round(score, 2)
    }
    
    return score, details


def ultra_fast_train(config, screening_epochs=25, checkpoint_epochs=[15, 20, 25], min_acc_epoch_15=78.0):
    """
    Ultra-fast training for screening phase
    
    Args:
        config: Training configuration
        screening_epochs: Number of epochs for screening (default: 25)
        checkpoint_epochs: Epochs to evaluate (default: [15, 20, 25])
        min_acc_epoch_15: Minimum accuracy at epoch 15 to continue (default: 78%)
    
    Returns:
        metrics dict with score, or None if stopped early
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Data loaders
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
    
    scheduler = CosineAnnealingLR(optimizer, T_max=screening_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training metrics
    best_val_acc = 0.0
    metrics = {
        'config': {k: v for k, v in config.items() if k not in ['train_loader', 'test_loader']},
        'epochs': [],
        'best_val_acc': 0.0,
        'best_epoch': 0,
        'early_stopped': False,
        'stopped_at_epoch': None,
        'score': 0.0,
        'score_details': {}
    }
    
    start_time = time.time()
    
    for epoch in range(1, screening_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_acc, train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, screening_epochs
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
        print(f"  Ep {epoch:2d}: Train {train_acc:5.2f}% | Val {val_acc:5.2f}% | Time {epoch_time:.1f}s", end='')
        
        # Early stopping at epoch 15 for clearly bad configs
        if epoch == 15 and val_acc < min_acc_epoch_15:
            print(f" ❌ STOPPED (Val {val_acc:.2f}% < {min_acc_epoch_15}%)")
            metrics['early_stopped'] = True
            metrics['stopped_at_epoch'] = epoch
            metrics['total_time_minutes'] = round((time.time() - start_time) / 60, 2)
            return None
        
        # Calculate and display score at checkpoints
        if epoch in checkpoint_epochs:
            score, score_details = calculate_score(metrics['epochs'], checkpoint_epochs)
            metrics['score'] = score
            metrics['score_details'] = score_details
            print(f" | Score: {score:.1f} (pred: {score_details['predicted_final']:.1f}%)")
        else:
            print()
    
    # Final score calculation
    score, score_details = calculate_score(metrics['epochs'], checkpoint_epochs)
    metrics['score'] = score
    metrics['score_details'] = score_details
    
    total_time = time.time() - start_time
    metrics['total_time_minutes'] = round(total_time / 60, 2)
    
    # Save metrics
    metrics_path = os.path.join(config['log_dir'], 'screening_metrics.json')
    save_metrics(metrics, metrics_path)
    
    return metrics


def full_train(config, epochs=50):
    """
    Full training for refinement phase
    
    Args:
        config: Training configuration
        epochs: Number of epochs (default: 50)
    
    Returns:
        metrics dict
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Data loaders
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
    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training metrics
    best_val_acc = 0.0
    metrics = {
        'config': {k: v for k, v in config.items() if k not in ['train_loader', 'test_loader']},
        'epochs': [],
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"FULL TRAINING - {epochs} epochs")
    print(f"{'='*70}")
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_acc, train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs
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
            
            # Save best model
            checkpoint_path = os.path.join(config['save_dir'], 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, best_val_acc, checkpoint_path)
        
        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            val_indicator = "🏆" if val_acc == best_val_acc else " "
            print(f"  Ep {epoch:2d}: Train {train_acc:5.2f}% | Val {val_acc:5.2f}% {val_indicator} | Best {best_val_acc:.2f}% @ Ep{metrics['best_epoch']}")
    
    total_time = time.time() - start_time
    metrics['total_time_minutes'] = round(total_time / 60, 2)
    
    # Save metrics
    metrics_path = os.path.join(config['log_dir'], 'final_metrics.json')
    save_metrics(metrics, metrics_path)
    
    print(f"\n✓ Training complete! Best: {best_val_acc:.2f}% at epoch {metrics['best_epoch']}")
    print(f"  Time: {metrics['total_time_minutes']:.1f} minutes\n")
    
    return metrics


def two_phase_grid_search(search_space, base_config, output_dir, 
                          screening_epochs=25, 
                          checkpoint_epochs=[15, 20, 25],
                          min_acc_epoch_15=78.0,
                          top_n_refinement=5,
                          refinement_epochs=50):
    """
    Two-phase grid search: fast screening + refinement of top configs
    
    Args:
        search_space: Dict mapping parameter names to lists of values
        base_config: Base configuration dict
        output_dir: Directory to save results
        screening_epochs: Epochs for phase 1 (default: 25)
        checkpoint_epochs: Epochs to evaluate (default: [15, 20, 25])
        min_acc_epoch_15: Minimum accuracy at epoch 15 (default: 78%)
        top_n_refinement: Number of top configs to refine (default: 5)
        refinement_epochs: Epochs for phase 2 (default: 50)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all combinations
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    combinations = list(itertools.product(*param_values))
    
    total_runs = len(combinations)
    
    print(f"\n{'='*70}")
    print(f"ULTRA-FAST TWO-PHASE GRID SEARCH")
    print(f"{'='*70}")
    print(f"Total combinations: {total_runs}")
    print(f"\nPHASE 1 - SCREENING:")
    print(f"  Epochs per run: {screening_epochs}")
    print(f"  Checkpoints: {checkpoint_epochs}")
    print(f"  Early stop threshold: {min_acc_epoch_15}% @ epoch 15")
    print(f"\nPHASE 2 - REFINEMENT:")
    print(f"  Top configs to refine: {top_n_refinement}")
    print(f"  Epochs for refinement: {refinement_epochs}")
    print(f"\nSearch space:")
    for param, values in search_space.items():
        print(f"  {param}: {values}")
    print(f"{'='*70}\n")
    
    # Pre-load data loaders
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
        'screening_epochs': screening_epochs,
        'checkpoint_epochs': checkpoint_epochs,
        'min_acc_epoch_15': min_acc_epoch_15,
        'phase1_runs': [],
        'phase2_runs': [],
        'successful_screening': [],
        'stopped_screening': [],
        'best_config': None,
        'best_score': 0.0,
        'best_final_acc': 0.0
    }
    
    # =====================================================================
    # PHASE 1: SCREENING
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 1: SCREENING ({total_runs} configurations)")
    print(f"{'='*70}\n")
    
    phase1_start = time.time()
    successful_count = 0
    stopped_count = 0
    
    for run_idx, combination in enumerate(combinations, 1):
        print(f"\n--- Screening Run {run_idx}/{total_runs} ---")
        
        # Create config
        config = base_config.copy()
        for param_name, param_value in zip(param_names, combination):
            config[param_name] = param_value
        
        print(f"Config: " + ", ".join([f"{pn}={pv}" for pn, pv in zip(param_names, combination)]))
        
        # Add shared data loaders
        config['train_loader'] = train_loader
        config['test_loader'] = test_loader
        
        # Create run-specific directories
        run_name = f"screen_{run_idx:03d}_" + "_".join([f"{pn}_{str(pv).replace('.', 'p')}" for pn, pv in zip(param_names, combination)])
        
        config['save_dir'] = os.path.join(output_dir, 'phase1_screening', run_name, 'models')
        config['log_dir'] = os.path.join(output_dir, 'phase1_screening', run_name, 'logs')
        
        try:
            # Ultra-fast training
            metrics = ultra_fast_train(config, screening_epochs, checkpoint_epochs, min_acc_epoch_15)
            
            if metrics is None:
                # Early stopped
                stopped_count += 1
                run_result = {
                    'run_id': run_idx,
                    'config': {pn: pv for pn, pv in zip(param_names, combination)},
                    'status': 'stopped_early'
                }
                results['stopped_screening'].append(run_result)
                
            else:
                # Completed screening
                successful_count += 1
                run_result = {
                    'run_id': run_idx,
                    'config': {pn: pv for pn, pv in zip(param_names, combination)},
                    'status': 'completed',
                    'score': metrics['score'],
                    'score_details': metrics['score_details'],
                    'final_val_acc': metrics['epochs'][-1]['val_acc'],
                    'time_minutes': metrics['total_time_minutes']
                }
                results['successful_screening'].append(run_result)
                
                # Track best score
                if metrics['score'] > results['best_score']:
                    results['best_score'] = metrics['score']
                    print(f"  🌟 NEW BEST SCORE: {metrics['score']:.2f}")
            
            results['phase1_runs'].append(run_result)
            
        except Exception as e:
            print(f"❌ Run {run_idx} failed: {str(e)}")
            results['phase1_runs'].append({
                'run_id': run_idx,
                'config': {pn: pv for pn, pv in zip(param_names, combination)},
                'status': 'error',
                'error': str(e)
            })
    
    phase1_time = (time.time() - phase1_start) / 60
    
    print(f"\n{'='*70}")
    print(f"PHASE 1 COMPLETE ({phase1_time:.1f} minutes)")
    print(f"{'='*70}")
    print(f"Successful: {successful_count}/{total_runs}")
    print(f"Early stopped: {stopped_count}/{total_runs}")
    
    # =====================================================================
    # PHASE 2: REFINEMENT
    # =====================================================================
    if len(results['successful_screening']) > 0:
        # Sort by score and select top N
        sorted_configs = sorted(results['successful_screening'], 
                               key=lambda x: x['score'], 
                               reverse=True)
        
        top_configs = sorted_configs[:min(top_n_refinement, len(sorted_configs))]
        
        print(f"\n{'='*70}")
        print(f"PHASE 2: REFINEMENT (Top {len(top_configs)} configurations)")
        print(f"{'='*70}\n")
        
        print("Selected configurations for refinement:")
        for rank, cfg in enumerate(top_configs, 1):
            print(f"  {rank}. Score {cfg['score']:.2f} - Run {cfg['run_id']} - {cfg['config']}")
        print()
        
        phase2_start = time.time()
        
        for rank, screen_result in enumerate(top_configs, 1):
            print(f"\n{'='*70}")
            print(f"REFINEMENT {rank}/{len(top_configs)} - Original Run {screen_result['run_id']}")
            print(f"Screening Score: {screen_result['score']:.2f}")
            print(f"{'='*70}")
            
            # Create config
            config = base_config.copy()
            config.update(screen_result['config'])
            config['train_loader'] = train_loader
            config['test_loader'] = test_loader
            
            # Create refinement directories
            run_name = f"refine_{rank:02d}_run{screen_result['run_id']:03d}"
            config['save_dir'] = os.path.join(output_dir, 'phase2_refinement', run_name, 'models')
            config['log_dir'] = os.path.join(output_dir, 'phase2_refinement', run_name, 'logs')
            
            try:
                # Full training
                metrics = full_train(config, refinement_epochs)
                
                refine_result = {
                    'rank': rank,
                    'original_run_id': screen_result['run_id'],
                    'config': screen_result['config'],
                    'screening_score': screen_result['score'],
                    'final_val_acc': metrics['best_val_acc'],
                    'final_epoch': metrics['best_epoch'],
                    'time_minutes': metrics['total_time_minutes']
                }
                results['phase2_runs'].append(refine_result)
                
                # Update best
                if metrics['best_val_acc'] > results['best_final_acc']:
                    results['best_final_acc'] = metrics['best_val_acc']
                    results['best_config'] = screen_result['config'].copy()
                    results['best_run_id'] = screen_result['run_id']
                    print(f"\n🏆 NEW BEST FINAL ACCURACY: {metrics['best_val_acc']:.2f}%!\n")
                
            except Exception as e:
                print(f"❌ Refinement {rank} failed: {str(e)}")
                results['phase2_runs'].append({
                    'rank': rank,
                    'original_run_id': screen_result['run_id'],
                    'config': screen_result['config'],
                    'status': 'error',
                    'error': str(e)
                })
        
        phase2_time = (time.time() - phase2_start) / 60
    else:
        phase2_time = 0
        print("\n⚠️  No successful screening runs - skipping Phase 2")
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    total_time = phase1_time + phase2_time
    
    print(f"\n{'='*70}")
    print(f"TWO-PHASE GRID SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.1f} minutes ({total_time/60:.2f} hours)")
    print(f"  Phase 1 (Screening): {phase1_time:.1f} min")
    print(f"  Phase 2 (Refinement): {phase2_time:.1f} min")
    print(f"\nPhase 1 Results:")
    print(f"  Tested: {total_runs} configurations")
    print(f"  Successful: {successful_count}")
    print(f"  Early stopped: {stopped_count}")
    print(f"\nPhase 2 Results:")
    print(f"  Refined: {len(results['phase2_runs'])} configurations")
    
    if results['best_config']:
        print(f"\n{'='*70}")
        print(f"🏆 BEST CONFIGURATION (Run {results.get('best_run_id', 'N/A')})")
        print(f"{'='*70}")
        for param, value in results['best_config'].items():
            print(f"  {param}: {value}")
        print(f"\nFinal Validation Accuracy: {results['best_final_acc']:.2f}%")
        
        # Save best config
        best_config_full = base_config.copy()
        best_config_full.update(results['best_config'])
        best_config_full.pop('train_loader', None)
        best_config_full.pop('test_loader', None)
        
        best_config_path = os.path.join(output_dir, 'best_config.json')
        with open(best_config_path, 'w') as f:
            json.dump(best_config_full, f, indent=2)
        
        print(f"\nBest config saved to: {best_config_path}")
    
    # Save full results
    results_path = os.path.join(output_dir, 'two_phase_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Full results saved to: {results_path}")
    
    # Create summary file
    summary_path = os.path.join(output_dir, 'SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TWO-PHASE GRID SEARCH SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total time: {total_time:.1f} minutes ({total_time/60:.2f} hours)\n")
        f.write(f"Phase 1: {phase1_time:.1f} min | Phase 2: {phase2_time:.1f} min\n\n")
        f.write(f"Configurations tested: {total_runs}\n")
        f.write(f"Successful screening: {successful_count}\n")
        f.write(f"Early stopped: {stopped_count}\n")
        f.write(f"Refined: {len(results['phase2_runs'])}\n\n")
        
        f.write("="*70 + "\n")
        f.write("TOP REFINED CONFIGURATIONS\n")
        f.write("="*70 + "\n\n")
        
        # Sort refined runs by final accuracy
        sorted_refined = sorted(results['phase2_runs'], 
                               key=lambda x: x.get('final_val_acc', 0), 
                               reverse=True)
        
        for rank, run in enumerate(sorted_refined, 1):
            f.write(f"RANK {rank} - Original Run {run['original_run_id']}\n")
            f.write(f"  Final Validation Accuracy: {run.get('final_val_acc', 'N/A')}%\n")
            f.write(f"  Screening Score: {run['screening_score']:.2f}\n")
            for param, value in run['config'].items():
                f.write(f"  {param} = {value}\n")
            f.write(f"  Best epoch: {run.get('final_epoch', 'N/A')}\n")
            f.write("\n")
    
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Ultra-fast two-phase grid search')
    
    # Search space - Can test MANY more combinations now!
    parser.add_argument('--lr', nargs='+', type=float,
                       default=[0.01, 0.025, 0.05, 0.075, 0.1],
                       help='Learning rates to test')
    parser.add_argument('--width_mult', nargs='+', type=float,
                       default=[0.3, 0.35, 0.4, 0.45, 0.5],
                       help='Width multipliers to test')
    parser.add_argument('--weight_decay', nargs='+', type=float,
                       default=[1e-4, 2e-4, 3e-4, 5e-4, 1e-3],
                       help='Weight decay values to test')
    parser.add_argument('--lr_scale', nargs='+', type=float,
                       default=[1.54],
                       help='LR scale values')
    parser.add_argument('--momentum', nargs='+', type=float,
                       default=[0.9],
                       help='Momentum values')
    
    # Phase 1 parameters
    parser.add_argument('--screening_epochs', type=int, default=25,
                       help='Epochs for screening phase (default: 25)')
    parser.add_argument('--checkpoint_epochs', nargs='+', type=int,
                       default=[15, 20, 25],
                       help='Checkpoint epochs for scoring (default: [15, 20, 25])')
    parser.add_argument('--min_acc_epoch_15', type=float, default=78.0,
                       help='Minimum accuracy at epoch 15 to continue (default: 78%)')
    
    # Phase 2 parameters
    parser.add_argument('--top_n', type=int, default=5,
                       help='Number of top configs to refine (default: 5)')
    parser.add_argument('--refinement_epochs', type=int, default=50,
                       help='Epochs for refinement phase (default: 50)')
    
    # Fixed parameters
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Data loading workers')
    
    # Output
    parser.add_argument('--output_dir', type=str,
                       default='../../reports/grid_search_ultra_fast',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Define search space
    search_space = {
        'lr': args.lr,
        'width_mult': args.width_mult,
        'weight_decay': args.weight_decay,
    }
    
    # Add optional parameters if varying
    if len(args.lr_scale) > 1:
        search_space['lr_scale'] = args.lr_scale
    if len(args.momentum) > 1:
        search_space['momentum'] = args.momentum
    
    # Base configuration
    base_config = {
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
    
    # Run two-phase grid search
    results = two_phase_grid_search(
        search_space,
        base_config,
        output_dir,
        screening_epochs=args.screening_epochs,
        checkpoint_epochs=args.checkpoint_epochs,
        min_acc_epoch_15=args.min_acc_epoch_15,
        top_n_refinement=args.top_n,
        refinement_epochs=args.refinement_epochs
    )
    
    return results


if __name__ == '__main__':
    main()
