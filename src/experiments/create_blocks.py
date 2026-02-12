import itertools
import random
import yaml
import os
import json

# Definisci i parametri
repetitions_list = [1, 2, 3]
channel_list = [18, 20, 24, 28, 32, 36, 40, 44]
expand_ratio_list = [2, 4, 8, 10]
use_eca = True
num_blocks = 4

# Directory e file di output
output_dir = '/workspace/Dima'
output_file = os.path.join(output_dir, 'block_settings.yaml')
os.makedirs(output_dir, exist_ok=True)

# File per memorizzare le combinazioni generate
generated_block_settings_file = os.path.join(output_dir, 'generated_block_settings.json')

# Classe personalizzata per rappresentare le liste in stile flow (inline) in YAML
class FlowList(list):
    pass

def flow_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(FlowList, flow_list_representer)

# Carica le combinazioni già generate, se esistono
if os.path.exists(generated_block_settings_file):
    with open(generated_block_settings_file, 'r') as f:
        try:
            data = json.load(f)
            generated_block_settings_list = data.get('block_settings_list', [])
            last_idx = data.get('last_idx', 0)
        except json.JSONDecodeError:
            # Se il file è vuoto o malformato, inizializza le variabili
            generated_block_settings_list = []
            last_idx = 0
else:
    generated_block_settings_list = []
    last_idx = 0

# Funzione per generare sequenze crescenti con ripetizioni massime
def generate_increasing_sequences_with_max_repeats(param_list, num_blocks, max_repeats):
    sequences = []
    for seq in itertools.combinations_with_replacement(param_list, num_blocks):
        if all(seq[i] <= seq[i+1] for i in range(len(seq)-1)):
            counts = {}
            for val in seq:
                counts[val] = counts.get(val, 0) + 1
            if all(count <= max_repeats for count in counts.values()):
                sequences.append(list(seq))
    return sequences

# Funzione per generare sequenze crescenti senza ripetizioni per i canali
def generate_strictly_increasing_sequences(param_list, num_blocks):
    sequences = []
    for seq in itertools.combinations(param_list, num_blocks):
        sequences.append(list(seq))
    return sequences

# Genera le sequenze
repetitions_sequences = generate_increasing_sequences_with_max_repeats(repetitions_list, num_blocks, max_repeats=2)
expand_ratio_sequences = generate_increasing_sequences_with_max_repeats(expand_ratio_list, num_blocks, max_repeats=num_blocks)  # No limit on expand_ratio repeats
channel_sequences = generate_strictly_increasing_sequences(channel_list, num_blocks)

# Genera combinazioni di sequenze
combinations = list(itertools.product(repetitions_sequences, expand_ratio_sequences, channel_sequences))
random.shuffle(combinations)

block_settings_dict = {}
max_new_block_settings = 150
new_block_settings_count = 0

for combo in combinations:
    if new_block_settings_count >= max_new_block_settings:
        break
    rep_seq, exp_seq, ch_seq = combo
    block_settings = []
    for i in range(num_blocks):
        block = FlowList([
            exp_seq[i],     # expand_ratio
            ch_seq[i],      # channels
            rep_seq[i],     # repetitions
            1 if i == 0 else 2,  # stride
            use_eca
        ])
        block_settings.append(block)
    # Verifica che le repetitions non si ripetano più di 2 volte
    counts = {}
    for block in block_settings:
        rep = block[2]
        counts[rep] = counts.get(rep, 0) + 1
    if all(count <= 2 for count in counts.values()):
        # Crea una rappresentazione unica delle block_settings
        block_settings_str = str(block_settings)
        if block_settings_str not in generated_block_settings_list:
            last_idx += 1
            block_settings_key = f'block_settings_{last_idx}'
            block_settings_dict[block_settings_key] = block_settings
            generated_block_settings_list.append(block_settings_str)
            new_block_settings_count += 1

# Salva le combinazioni generate aggiornate
with open(generated_block_settings_file, 'w') as f:
    data = {
        'block_settings_list': generated_block_settings_list,
        'last_idx': last_idx
    }
    json.dump(data, f)

# Scrivi le combinazioni nel file YAML
with open(output_file, 'w') as yaml_file:
    yaml.dump(block_settings_dict, yaml_file, default_flow_style=False)

print(f"Generation completed. The new block_settings have been saved in {output_file}")
