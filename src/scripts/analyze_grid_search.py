"""
Analyze and Visualize Grid Search Results
Creates plots and tables summarizing hyperparameter search
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
import os


def load_results(results_path):
    """Load grid search results from JSON"""
    with open(results_path, 'r') as f:
        return json.load(f)


def create_summary_table(results):
    """Create pandas DataFrame with all runs"""
    runs = results['runs']
    
    data = []
    for run in runs:
        if 'error' not in run:
            row = {
                'run_id': run['run_id'],
                **run['config'],
                'best_val_acc': run['best_val_acc'],
                'best_epoch': run['best_epoch'],
                'final_train_acc': run['final_train_acc'],
                'final_val_acc': run['final_val_acc'],
                'time_minutes': run['total_time_minutes']
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    return df.sort_values('best_val_acc', ascending=False)


def plot_hyperparameter_effects(df, output_dir):
    """Create plots showing effect of each hyperparameter"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    hyperparams = ['lr', 'width_mult', 'weight_decay']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, param in enumerate(hyperparams):
        ax = axes[idx]
        
        # Group by parameter and compute mean/std
        grouped = df.groupby(param)['best_val_acc'].agg(['mean', 'std', 'max'])
        
        # Plot
        x = grouped.index
        y_mean = grouped['mean']
        y_std = grouped['std']
        y_max = grouped['max']
        
        ax.plot(x, y_mean, 'o-', linewidth=2, markersize=8, label='Mean', color='steelblue')
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, color='steelblue')
        ax.plot(x, y_max, 's--', linewidth=1.5, markersize=6, label='Max', color='coral')
        
        ax.set_xlabel(param.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Effect of {param.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for weight_decay (scientific notation)
        if param == 'weight_decay':
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hyperparameter_effects.png'), bbox_inches='tight')
    print(f"✅ Saved: hyperparameter_effects.png")
    plt.close()


def plot_heatmaps(df, output_dir):
    """Create heatmaps for pairwise hyperparameter interactions"""
    
    # LR vs Width Multiplier
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap 1: LR vs Width
    pivot1 = df.pivot_table(values='best_val_acc', index='lr', columns='width_mult', aggfunc='mean')
    sns.heatmap(pivot1, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0], 
                cbar_kws={'label': 'Validation Accuracy (%)'}, vmin=85, vmax=92)
    axes[0].set_title('Learning Rate vs Width Multiplier', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Width Multiplier', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
    
    # Heatmap 2: LR vs Weight Decay
    pivot2 = df.pivot_table(values='best_val_acc', index='lr', columns='weight_decay', aggfunc='mean')
    sns.heatmap(pivot2, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1],
                cbar_kws={'label': 'Validation Accuracy (%)'}, vmin=85, vmax=92)
    axes[1].set_title('Learning Rate vs Weight Decay', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Weight Decay', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0e}'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hyperparameter_heatmaps.png'), bbox_inches='tight')
    print(f"✅ Saved: hyperparameter_heatmaps.png")
    plt.close()


def plot_top_configs(df, output_dir, top_n=10):
    """Bar plot of top N configurations"""
    
    top_df = df.head(top_n).copy()
    top_df['config_name'] = top_df.apply(
        lambda row: f"Run {row['run_id']}\nlr={row['lr']}, w={row['width_mult']}", axis=1
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['gold' if i == 0 else 'silver' if i == 1 else 'peru' if i == 2 else 'steelblue' 
              for i in range(len(top_df))]
    
    bars = ax.bar(range(len(top_df)), top_df['best_val_acc'], color=colors, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Best Validation Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Configurations from Grid Search', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(top_df)))
    ax.set_xticklabels(top_df['config_name'], rotation=45, ha='right', fontsize=9)
    ax.set_ylim([top_df['best_val_acc'].min() - 1, top_df['best_val_acc'].max() + 0.5])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_df['best_val_acc'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_configurations.png'), bbox_inches='tight')
    print(f"✅ Saved: top_configurations.png")
    plt.close()


def generate_report(results, df, output_dir):
    """Generate text report with key findings"""
    
    report = []
    report.append("=" * 70)
    report.append("GRID SEARCH ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Best configuration
    best_config = results['best_config']
    report.append("BEST CONFIGURATION:")
    for param, value in best_config.items():
        report.append(f"  {param}: {value}")
    report.append(f"\nBest Validation Accuracy: {results['best_val_acc']:.2f}%")
    report.append(f"Best Run ID: {results['best_run_id']}")
    report.append("")
    
    # Top 5 configurations
    report.append("TOP 5 CONFIGURATIONS:")
    report.append("-" * 70)
    for idx, row in df.head(5).iterrows():
        report.append(f"\nRank {row.name + 1} (Run {row['run_id']}):")
        report.append(f"  Validation Accuracy: {row['best_val_acc']:.2f}%")
        report.append(f"  lr={row['lr']}, width_mult={row['width_mult']}, weight_decay={row['weight_decay']}")
        report.append(f"  Converged at epoch {row['best_epoch']}/{int(df['best_epoch'].max())}")
    report.append("")
    
    # Hyperparameter insights
    report.append("=" * 70)
    report.append("HYPERPARAMETER INSIGHTS:")
    report.append("-" * 70)
    
    for param in ['lr', 'width_mult', 'weight_decay']:
        grouped = df.groupby(param)['best_val_acc'].agg(['mean', 'std', 'max', 'min'])
        report.append(f"\n{param.upper().replace('_', ' ')}:")
        for val in grouped.index:
            stats = grouped.loc[val]
            report.append(f"  {val}: mean={stats['mean']:.2f}%, max={stats['max']:.2f}%, std={stats['std']:.2f}%")
    
    report.append("")
    report.append("=" * 70)
    
    # Save report
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✅ Saved: analysis_report.txt")
    
    # Print to console
    print("\n" + "\n".join(report))


def main():
    parser = argparse.ArgumentParser(description='Analyze grid search results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to grid_search_results.json')
    parser.add_argument('--output_dir', type=str, default='./analysis',
                       help='Directory to save plots and reports')
    
    args = parser.parse_args()
    
    # Load results
    print(f"\n📊 Loading results from: {args.results}")
    results = load_results(args.results)
    
    # Create summary DataFrame
    df = create_summary_table(results)
    print(f"✅ Loaded {len(df)} successful runs")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    print(f"\n📈 Generating visualizations...")
    plot_hyperparameter_effects(df, args.output_dir)
    plot_heatmaps(df, args.output_dir)
    plot_top_configs(df, args.output_dir, top_n=10)
    
    # Save summary table
    csv_path = os.path.join(args.output_dir, 'all_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved: all_results.csv")
    
    # Generate text report
    print(f"\n📝 Generating analysis report...")
    generate_report(results, df, args.output_dir)
    
    print(f"\n✅ Analysis complete! All outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
