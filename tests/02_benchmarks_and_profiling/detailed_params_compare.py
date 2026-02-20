import torch
from train_A_baseline import MobileNetBaseline as MobileNetBaseline
from train_C_eca_rep import MobileNetECARep

def compare_models():
    # Instantiate models
    model_a = MobileNetBaseline(num_classes=10)
    model_c = MobileNetECARep(num_classes=10)
    model_c.deploy()
    
    params_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)
    params_c = sum(p.numel() for p in model_c.parameters() if p.requires_grad)
    
    print(f"Total Params A: {params_a:,}")
    print(f"Total Params C: {params_c:,}")
    print(f"Total Difference: {params_c - params_a:,}\n")

    def analyze_layers(model):
        layer_params = {}
        for name, module in model.named_modules():
            # count if it doesn't have child modules
            if not list(module.children()):
                layer_params[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return layer_params
    
    layers_a = analyze_layers(model_a)
    layers_c = analyze_layers(model_c)

    # Let's aggregate by prefix to see where the drop happens
    prefixes = ["features.0", "features.1", "features.2", "features.3", "features.4", "features.5", "features.6", "features.7", "features.8", "features.9", "features.10", "features.11", "features.12", "features.13", "classifier"]
    
    print(f"{'Section':<15} | {'A Baseline':<12} | {'C ECA-Rep':<12} | {'Diff'}")
    print("-" * 55)
    for p in prefixes:
        a_sum = sum(v for k, v in layers_a.items() if k.startswith(p))
        c_sum = sum(v for k, v in layers_c.items() if k.startswith(p))
        print(f"{p:<15} | {a_sum:<12,} | {c_sum:<12,} | {c_sum - a_sum:,}")

if __name__ == "__main__":
    compare_models()
