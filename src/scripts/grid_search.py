"""
Grid Search for MobileNetECA Hyperparameter Optimization
Systematic search over learning rate, width multiplier, batch size, and weight decay
"""

import torch
import itertools
import json
import os
from datetime import datetime
import argparse

from train import train_model


def grid_search(search_space, base_config, output_dir):
    """
    Perform grid search over hyperparameter space
    
    Args:
        search_space: Dict mapping parameter names to lists of values
        base_config: Base configuration dict
        output_dir: Directory to save results
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all parameter combinations
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    combinations = list(itertools.product(*param_values))
    
    total_runs = len(combinations)
    
    print(f"\n{'='*70}")
    print(f"GRID SEARCH FOR MOBILENETECA")
    print(f"{'='*70}")
    print(f"Total combinations to test: {total_runs}")
    print(f"\nSearch space:")
    for param, values in search_space.items():
        print(f"  {param}: {values}")
    print(f"{'='*70}\n")
    
    # Results storage
    results = {
        'search_space': search_space,
        'base_config': base_config,
        'runs': [],
        'best_config': None,
        'best_val_acc': 0.0
    }
    
    # Run grid search
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
        
        # Create run-specific directories
        run_name = f"run_{run_idx:03d}_" + "_".join([f"{pn}_{pv}" for pn, pv in zip(param_names, combination)])
        run_name = run_name.replace(".", "p")  # Replace dots for valid filenames
        
        config['save_dir'] = os.path.join(output_dir, run_name, 'models')
        config['log_dir'] = os.path.join(output_dir, run_name, 'logs')
        
        try:
            # Train model with this configuration
            metrics = train_model(config)
            
            # Store results
            run_result = {
                'run_id': run_idx,
                'config': {pn: pv for pn, pv in zip(param_names, combination)},
                'best_val_acc': metrics['best_val_acc'],
                'best_epoch': metrics['best_epoch'],
                'final_train_acc': metrics['epochs'][-1]['train_acc'],
                'final_val_acc': metrics['epochs'][-1]['val_acc'],
                'total_time_minutes': metrics['total_time_minutes']
            }
            
            results['runs'].append(run_result)
            
            # Update best config if improved
            if metrics['best_val_acc'] > results['best_val_acc']:
                results['best_val_acc'] = metrics['best_val_acc']
                results['best_config'] = run_result['config'].copy()
                results['best_run_id'] = run_idx
                
                print(f"\n🏆 NEW BEST: {metrics['best_val_acc']:.2f}% validation accuracy!")
            
        except Exception as e:
            print(f"\n❌ Run {run_idx} failed with error: {str(e)}")
            results['runs'].append({
                'run_id': run_idx,
                'config': {pn: pv for pn, pv in zip(param_names, combination)},
                'error': str(e)
            })
        
        # Save intermediate results
        results_path = os.path.join(output_dir, 'grid_search_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nProgress: {run_idx}/{total_runs} runs complete")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {len([r for r in results['runs'] if 'error' not in r])}")
    print(f"Failed runs: {len([r for r in results['runs'] if 'error' in r])}")
    print(f"\nBest configuration (Run {results.get('best_run_id', 'N/A')}):")
    if results['best_config']:
        for param, value in results['best_config'].items():
            print(f"  {param}: {value}")
        print(f"\nBest validation accuracy: {results['best_val_acc']:.2f}%")
    else:
        print("  No successful runs")
    
    # Save final results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save best config separately for easy use
    if results['best_config']:
        best_config_full = base_config.copy()
        best_config_full.update(results['best_config'])
        
        best_config_path = os.path.join(output_dir, 'best_config.json')
        with open(best_config_path, 'w') as f:
            json.dump(best_config_full, f, indent=2)
        
        print(f"\nBest configuration saved to: {best_config_path}")
    
    print(f"Full results saved to: {results_path}")
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Grid search for MobileNetECA hyperparameters')
    
    # Search space
    parser.add_argument('--lr', nargs='+', type=float, 
                       default=[0.01, 0.025, 0.05],
                       help='Learning rates to test')
    parser.add_argument('--width_mult', nargs='+', type=float,
                       default=[0.35, 0.42, 0.5],
                       help='Width multipliers to test')
    parser.add_argument('--batch_size', nargs='+', type=int,
                       default=[128],
                       help='Batch sizes to test (default: fixed at 128)')
    parser.add_argument('--weight_decay', nargs='+', type=float,
                       default=[1e-4, 3e-4, 5e-4],
                       help='Weight decay values to test')
    
    # Fixed parameters (not searched)
    parser.add_argument('--epochs', type=int, default=50,
                       help='Epochs per run (default: 50 for faster search)')
    parser.add_argument('--lr_scale', type=float, default=1.54,
                       help='LR scale (fixed)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (fixed)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Data loading workers')
    
    # Output
    parser.add_argument('--output_dir', type=str, 
                       default='../../reports/grid_search',
                       help='Directory to save results')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test with minimal combinations')
    
    args = parser.parse_args()
    
    # Define search space
    if args.quick_test:
        print("\n⚡ QUICK TEST MODE: Using minimal search space")
        search_space = {
            'lr': [args.lr[0]] if len(args.lr) > 0 else [0.025],
            'width_mult': [args.width_mult[0]] if len(args.width_mult) > 0 else [0.42],
            'batch_size': [args.batch_size[0]] if len(args.batch_size) > 0 else [128],
            'weight_decay': [args.weight_decay[0]] if len(args.weight_decay) > 0 else [3e-4],
        }
    else:
        search_space = {
            'lr': args.lr,
            'width_mult': args.width_mult,
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
        }
    
    # Base configuration (parameters not being searched)
    base_config = {
        'epochs': args.epochs,
        'lr_scale': args.lr_scale,
        'momentum': args.momentum,
        'num_workers': args.num_workers,
        'save_dir': '',  # Will be set per run
        'log_dir': '',   # Will be set per run
    }
    
    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"search_{timestamp}")
    
    # Run grid search
    results = grid_search(search_space, base_config, output_dir)
    
    return results


if __name__ == '__main__':
    main()
