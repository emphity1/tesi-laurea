import torch
import torch.nn as nn
import os
import glob
from pathlib import Path

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def inspect_model(model_path):
    print(f"\n{'='*50}")
    print(f"Inspecting: {os.path.basename(model_path)}")
    print(f"{'='*50}")
    
    try:
        # Load map_location=cpu to avoid CUDA errors on Mac
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check what kind of object we loaded
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
            print("Type: Full nn.Module")
        elif isinstance(checkpoint, dict):
            print("Type: State Dict / Checkpoint Dict")
            print(f"Keys found: {list(checkpoint.keys())}")
            if 'model_state_dict' in checkpoint:
                print("Found 'model_state_dict', likely a training checkpoint.")
                # We can't easily instantiate the model info without the class definition
                # but we can count params in the dict
                params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
                print(f"Estimated Parameters (from state dict): {params:,}")
            elif 'state_dict' in checkpoint:
                params = sum(p.numel() for p in checkpoint['state_dict'].values())
                print(f"Estimated Parameters (from state dict): {params:,}")
            
            # If it's just weights, return
            return
        else:
            print(f"Type: {type(checkpoint)}")
            return

        # If it is a full model
        total_params = count_parameters(model)
        print(f"Total Parameters: {total_params:,}")
        
        # Try to guess architecture style
        modules = [m for m in model.modules()]
        conv_layers = len([m for m in modules if isinstance(m, nn.Conv2d)])
        linear_layers = len([m for m in modules if isinstance(m, nn.Linear)])
        print(f"Convolutional Layers: {conv_layers}")
        print(f"Linear Layers: {linear_layers}")
        
    except Exception as e:
        print(f"Error loading {model_path}: {e}")

def main():
    # Adjust path relative to this script or absolute
    base_dir = Path(__file__).parent.parent.parent # Go up to Tesi root
    models_dir = base_dir / "models"
    
    pt_files = list(models_dir.glob("*.pt"))
    
    if not pt_files:
        print(f"No .pt files found in {models_dir}")
        return
        
    print(f"Found {len(pt_files)} models in {models_dir}...")
    
    for pt_file in pt_files:
        inspect_model(str(pt_file))

if __name__ == "__main__":
    main()
