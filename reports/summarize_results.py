import json
import os
import glob

report_dir = "/workspace/tesi-laurea/reports/test_all"
json_files = glob.glob(os.path.join(report_dir, "*.json"))

results = []
for f in json_files:
    with open(f, 'r') as file:
        data = json.load(file)
        results.append({
            'name': data['name'],
            'eca': data['config']['use_eca'],
            'gelu': data['config']['use_gelu'],
            'params': data['config']['params'],
            'acc': data['results']['best_acc'],
            'time': data['results']['total_time']
        })

# Sort by accuracy
results.sort(key=lambda x: x['acc'], reverse=True)

print(f"{'Name':<25} | {'ECA':<6} | {'GELU':<6} | {'Params':<10} | {'Acc (%)':<10} | {'Time (s)':<10}")
print("-" * 80)
for r in results:
    print(f"{r['name']:<25} | {str(r['eca']):<6} | {str(r['gelu']):<6} | {r['params']:<10} | {r['acc']:<10.2f} | {r['time']:<10.1f}")
