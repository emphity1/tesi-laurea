import os
import json
import glob
import argparse
import numpy as np
import sys
from datetime import datetime

def find_latest_search_dir(base_dir):
    search_dirs = glob.glob(os.path.join(base_dir, "search_*"))
    if not search_dirs:
        return None
    return max(search_dirs, key=os.path.getmtime)

def load_screening_results(search_dir):
    screening_dir = os.path.join(search_dir, "phase1_screening")
    if not os.path.exists(screening_dir):
        print(f"Directory not found: {screening_dir}")
        return []
    
    results = []
    
    # Find all screening_metrics.json files
    metric_files = glob.glob(os.path.join(screening_dir, "screen_*", "logs", "screening_metrics.json"))
    
    print(f"Found {len(metric_files)} completed runs in {screening_dir}")
    
    for file_path in metric_files:
        try:
            with open(file_path, 'r') as f:
                metrics = json.load(f)
            
            # Extract run specific info from path
            # .../screen_073_lr_.../logs/...
            run_dir = os.path.dirname(os.path.dirname(file_path))
            run_name = os.path.basename(run_dir)
            
            # Parse run ID from name (screen_073_...)
            try:
                run_id = int(run_name.split('_')[1])
            except:
                run_id = 0
            
            # Check if it was early stopped or completed
            status = 'stopped_early' if metrics.get('early_stopped', False) else 'completed'
            score = metrics.get('score', 0.0)
            
            # If no score (old format or early stop), try to calc or use val_acc
            if score == 0.0 and metrics['epochs']:
                score = metrics['epochs'][-1]['val_acc']
            
            results.append({
                'run_id': run_id,
                'run_name': run_name,
                'config': metrics.get('config', {}),
                'score': score,
                'best_val_acc': metrics.get('best_val_acc', 0.0),
                'status': status,
                'path': run_dir
            })
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze partial grid search results')
    parser.add_argument('--search_dir', type=str, help='Specific search directory (optional)')
    parser.add_argument('--reports_dir', type=str, default='/workspace/tesi-laurea/reports/grid_search_ultra_fast', 
                        help='Base reports directory')
    parser.add_argument('--top_n', type=int, default=10, help='Show top N configurations')
    
    args = parser.parse_args()
    
    # Determine search directory
    if args.search_dir:
        search_dir = args.search_dir
    else:
        search_dir = find_latest_search_dir(args.reports_dir)
        
    if not search_dir or not os.path.exists(search_dir):
        print(f"Error: Could not find search directory in {args.reports_dir}")
        return
        
    print(f"Analyzing results in: {search_dir}")
    
    results = load_screening_results(search_dir)
    
    if not results:
        print("No results found.")
        return
        
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n" + "="*80)
    print(f"TOP {args.top_n} CONFIGURATIONS (out of {len(results)} analyzed)")
    print("="*80)
    print(f"{'Rank':<5} {'Run ID':<8} {'Score':<8} {'Best Acc':<10} {'Status':<15} {'Config Summary'}")
    print("-" * 80)
    
    for i, res in enumerate(results[:args.top_n], 1):
        cfg = res['config']
        # Format config string
        cfg_str = f"lr={cfg.get('lr')}, w={cfg.get('width_mult')}, wd={cfg.get('weight_decay')}"
        
        print(f"{i:<5} {res['run_id']:<8} {res['score']:<8.2f} {res['best_val_acc']:<10.2f} {res['status']:<15} {cfg_str}")
        
    print("-" * 80)
    
    # Save partial summary to JSON
    output_file = os.path.join(search_dir, "partial_analysis_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full analysis to: {output_file}")
    
    # Suggest next steps
    print("\nTo continue with Phase 2 (Refinement) for the top 5 configs, you can use these parameters:")
    print("Use manual training or modify the grid search script to run only specific configs.")
    
    top_5 = results[:5]
    print("\nTop 5 Configs details:")
    for res in top_5:
        print(f"Run {res['run_id']}: {json.dumps(res['config'])}")

if __name__ == "__main__":
    main()
