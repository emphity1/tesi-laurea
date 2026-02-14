"""
Calculate Parameters and FLOPs for MobileNetECA configurations
"""

import torch
import torch.onnx
import re
import os
from model import MobileNetECA, count_parameters, format_number

def calculate_flops_onnx(model, input_size=(1, 3, 32, 32)):
    """Calculate FLOPs using ONNX profiling"""
    try:
        import onnx_tool
    except ImportError:
        print("⚠️  onnx_tool not installed. Run: pip install onnx-tool")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_size).to(device)
    
    # Export to ONNX
    onnx_path = "tmp_model.onnx"
    profile_path = "tmp_profile.txt"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None
    )
    
    # Profile with onnx_tool
    onnx_tool.model_profile(onnx_path, save_profile=profile_path)
    
    # Parse profile
    with open(profile_path, 'r') as f:
        profile = f.read()
    
    # Extract total MACs
    match = re.search(r'Total\s+_\s+([\d,]+)\s+100%', profile)
    
    if match:
        total_macs = int(match.group(1).replace(',', ''))
    else:
        total_macs = None
    
    # Cleanup
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    if os.path.exists(profile_path):
        os.remove(profile_path)
    
    return total_macs


def analyze_configuration(width_mult, lr_scale=1.54):
    """Analyze a specific configuration"""
    
    print(f"\n{'='*60}")
    print(f"MobileNetECA Analysis: width_mult={width_mult}")
    print(f"{'='*60}")
    
    # Create model
    model = MobileNetECA(num_classes=10, width_mult=width_mult, lr_scale=lr_scale)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {format_number(total_params)} ({total_params:,})")
    print(f"  Trainable parameters: {format_number(trainable_params)} ({trainable_params:,})")
    
    # Calculate FLOPs
    print(f"\n⚙️  Calculating FLOPs...")
    macs = calculate_flops_onnx(model)
    
    if macs:
        # FLOPs ≈ 2 × MACs for most operations
        flops = macs * 2
        print(f"  MACs (Multiply-Accumulate): {format_number(macs)} ({macs:,})")
        print(f"  FLOPs (approx): {format_number(flops)} ({flops:,})")
    else:
        print(f"  ⚠️  Could not calculate FLOPs")
    
    # Efficiency metrics
    if macs:
        params_per_mac = total_params / macs
        print(f"\n✨ Efficiency Metrics:")
        print(f"  Parameters per MAC: {params_per_mac:.4f}")
        print(f"  Model compactness: {'Very efficient' if params_per_mac < 0.01 else 'Moderate'}")
    
    return {
        'width_mult': width_mult,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'macs': macs,
        'flops': flops if macs else None
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze MobileNetECA configurations')
    parser.add_argument('--width_mult', nargs='+', type=float,
                       default=[0.35, 0.42, 0.5],
                       help='Width multipliers to analyze')
    
    args = parser.parse_args()
    
    results = []
    for width in args.width_mult:
        result = analyze_configuration(width)
        results.append(result)
        print(f"{'='*60}\n")
    
    # Summary table
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON TABLE")
        print(f"{'='*60}")
        print(f"\n{'Width':<10} {'Params':<15} {'MACs':<15} {'FLOPs':<15}")
        print(f"{'-'*60}")
        
        for r in results:
            width_str = f"{r['width_mult']}"
            params_str = format_number(r['total_params'])
            macs_str = format_number(r['macs']) if r['macs'] else 'N/A'
            flops_str = format_number(r['flops']) if r['flops'] else 'N/A'
            
            print(f"{width_str:<10} {params_str:<15} {macs_str:<15} {flops_str:<15}")
        
        print(f"{'='*60}\n")
