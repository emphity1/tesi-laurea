
import torch
import torch.nn as nn
import time
import numpy as np
import sys
import os

# Aggiungi il path per trovare il modulo
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from src.legacy.mobilnet_eca_rep_advaug.MobileNetEca_Rep_AdvAug import MobileNetECARep
except ImportError as e:
    print(f"Error importing model: {e}")
    sys.exit(1)

def measure_latency():
    # Setup del modello (configurazione finale - Reparameterized)
    model = MobileNetECARep(num_classes=10, width_mult=0.5)
    
    # IMPORTANTE: Switch to deploy mode per fondere i branch e misurare la velocità reale
    model.deploy()
    model.eval()
    model.eval()
    
    # Input dummy
    x = torch.randn(1, 3, 32, 32)
    
    # Warmup
    print("Warming up...")
    for _ in range(50):
        _ = model(x)
        
    print("Measuring latency...")
    latencies = []
    with torch.no_grad():
        for _ in range(1000):
            start = time.time()
            _ = model(x)
            end = time.time()
            latencies.append((end - start) * 1000) # ms
            
    avg_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    fps = 1000 / avg_lat
    
    print(f"Average Latency (CPU): {avg_lat:.4f} ms")
    print(f"Std Dev: {std_lat:.4f} ms")
    print(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    measure_latency()
