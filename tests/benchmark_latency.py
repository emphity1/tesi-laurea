"""
Latency Benchmark — MobileNetECA-Rep (Deploy Mode)
Measures inference latency on CPU with proper warmup exclusion.
Outputs: median, mean, p95, p99, and full distribution as JSON.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'train'))

import json
import time
import numpy as np
import torch

# Import model architecture
from train_D_eca_rep_advaug import MobileNetECARep

# --- Config ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'src', 'train', 'results_D_eca_rep_advaug', 'best_model_deploy.pth')
NUM_ITERATIONS = 1000
WARMUP_ITERATIONS = 50
INPUT_SHAPE = (1, 3, 32, 32)  # Batch=1, CIFAR-10 resolution
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 60)
    print("LATENCY BENCHMARK — MobileNetECA-Rep (Deploy)")
    print("=" * 60)

    # Load deploy model
    device = torch.device('cpu')
    model = MobileNetECARep()
    model.deploy()  # Fuse RepConv kernels

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters (deploy)")
    print(f"Device: CPU")
    print(f"Input shape: {INPUT_SHAPE}")
    print(f"Iterations: {NUM_ITERATIONS} (warmup: {WARMUP_ITERATIONS})")
    print()

    # Create random input
    dummy_input = torch.randn(*INPUT_SHAPE)

    # Benchmark
    latencies_ms = []
    with torch.no_grad():
        for i in range(NUM_ITERATIONS):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

    all_latencies = np.array(latencies_ms)
    warmup_latencies = all_latencies[:WARMUP_ITERATIONS]
    stable_latencies = all_latencies[WARMUP_ITERATIONS:]

    # Compute statistics (excluding warmup)
    results = {
        "config": {
            "model": "MobileNetECA-Rep (deploy)",
            "parameters": total_params,
            "input_shape": list(INPUT_SHAPE),
            "device": "CPU",
            "num_iterations": NUM_ITERATIONS,
            "warmup_iterations": WARMUP_ITERATIONS,
        },
        "all_iterations": {
            "mean_ms": float(np.mean(all_latencies)),
            "std_ms": float(np.std(all_latencies)),
            "note": "Includes warmup — NOT representative"
        },
        "stable_iterations": {
            "mean_ms": float(np.mean(stable_latencies)),
            "std_ms": float(np.std(stable_latencies)),
            "median_ms": float(np.median(stable_latencies)),
            "p5_ms": float(np.percentile(stable_latencies, 5)),
            "p25_ms": float(np.percentile(stable_latencies, 25)),
            "p75_ms": float(np.percentile(stable_latencies, 75)),
            "p95_ms": float(np.percentile(stable_latencies, 95)),
            "p99_ms": float(np.percentile(stable_latencies, 99)),
            "min_ms": float(np.min(stable_latencies)),
            "max_ms": float(np.max(stable_latencies)),
            "throughput_fps": float(1000.0 / np.mean(stable_latencies)),
        },
        "warmup_stats": {
            "mean_ms": float(np.mean(warmup_latencies)),
            "max_ms": float(np.max(warmup_latencies)),
            "note": "First 50 iterations — high due to JIT/cache warmup"
        },
        "raw_latencies_ms": latencies_ms,
    }

    # Print summary
    s = results["stable_iterations"]
    print("=== Results (excluding warmup) ===")
    print(f"  Median (p50):  {s['median_ms']:.2f} ms")
    print(f"  Mean:          {s['mean_ms']:.2f} ms  (σ = {s['std_ms']:.2f} ms)")
    print(f"  p5:            {s['p5_ms']:.2f} ms")
    print(f"  p95:           {s['p95_ms']:.2f} ms")
    print(f"  p99:           {s['p99_ms']:.2f} ms")
    print(f"  Min:           {s['min_ms']:.2f} ms")
    print(f"  Max:           {s['max_ms']:.2f} ms")
    print(f"  Throughput:    {s['throughput_fps']:.1f} FPS")
    print()
    w = results["warmup_stats"]
    print(f"  Warmup mean:   {w['mean_ms']:.2f} ms")
    print(f"  Warmup max:    {w['max_ms']:.2f} ms")
    a = results["all_iterations"]
    print(f"  All-iter mean: {a['mean_ms']:.2f} ms  (σ = {a['std_ms']:.2f} ms) ← inflated by warmup")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "latency_benchmark.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
