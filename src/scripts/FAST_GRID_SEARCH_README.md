# Fast Grid Search con Early Stopping

## 🚀 Panoramica

Questo script implementa una grid search **ottimizzata per velocità** che permette di testare molte più combinazioni di iperparametri in meno tempo, usando una strategia di **early stopping basata su threshold**.

## 💡 Come Funziona

### Strategia Tradizionale vs Fast
- **Grid Search Tradizionale**: Allena ogni configurazione per 50 epoche complete
  - Esempio: 27 config × 50 epoche = ~3 ore
  
- **Fast Grid Search**: Allena per 20-25 epoche, ma scarta configurazioni deboli già all'epoca 20
  - Esempio: 180 config × ~15 epoche medie = ~8-10 ore
  - **180 combinazioni vs 27 = 6.7× più configurazioni testate!**

### Meccanismo Early Stopping

1. **Checkpoint Epoch** (default: 20)
   - A questa epoca, il modello viene valutato contro una soglia minima

2. **Minimum Threshold** (default: 85%)
   - Se la validation accuracy all'epoca checkpoint è < threshold → STOP
   - Altrimenti → continua fino a `epochs` massime

3. **Risparmio Tempo**
   - Config scarse vengono scartate dopo ~20 epoche invece di 50
   - Config promettenti completano il training (25 epoche)
   - In media: ~40-50% risparmio tempo per configurazione

## 📊 Vantaggi

✅ **Più combinazioni**: Testa 6-7× più configurazioni nello stesso tempo  
✅ **Risultati migliori**: Maggiore probabilità di trovare configurazione ottimale  
✅ **Efficiente**: Scarta rapidamente configurazioni poco promettenti  
✅ **Flessibile**: Puoi regolare checkpoint_epoch e threshold basandoti sui dati  

## 🔧 Parametri

### Spazio di Ricerca (Espandibile)
```bash
--lr              # Learning rate (default: [0.01, 0.025, 0.05, 0.075, 0.1])
--width_mult      # Width multiplier (default: [0.3, 0.35, 0.4, 0.42, 0.45, 0.5])
--weight_decay    # Weight decay (default: [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3])
--lr_scale        # LR scale (default: [1.54] - fisso)
--momentum        # Momentum (default: [0.9] - fisso)
```

**Combinazioni default**: 5 × 6 × 6 = **180 configurazioni**

### Early Stopping
```bash
--checkpoint_epoch  # Epoca di verifica threshold (default: 20)
--min_threshold     # Accuratezza minima richiesta (default: 85.0%)
--epochs           # Epoche massime se supera threshold (default: 25)
```

### Altri Parametri
```bash
--batch_size      # Batch size (fisso a 128)
--num_workers     # Data loading workers (default: 2)
--output_dir      # Directory output (default: ../../reports/grid_search_fast)
--quick_test      # Modalità test veloce (poche combinazioni)
```

## 📝 Esempi di Utilizzo

### Esempio 1: Quick Test (per testare lo script)
```bash
cd /workspace/tesi-laurea/src/scripts

python grid_search_fast.py \
    --quick_test \
    --checkpoint_epoch 15 \
    --min_threshold 80.0 \
    --epochs 20
```
- Testa solo 2×2×2 = 8 combinazioni
- Tempo: ~30-40 minuti
- Utile per verificare che tutto funzioni

### Esempio 2: Standard Fast Search (RACCOMANDATO)
```bash
python grid_search_fast.py \
    --lr 0.01 0.025 0.05 0.075 0.1 \
    --width_mult 0.3 0.35 0.4 0.42 0.45 0.5 \
    --weight_decay 0.0001 0.0002 0.0003 0.0005 0.0007 0.001 \
    --checkpoint_epoch 20 \
    --min_threshold 85.0 \
    --epochs 25
```
- Testa 5×6×6 = **180 combinazioni**
- Tempo stimato: **8-10 ore** (vs ~90 ore senza early stopping)
- Early stop: ~50-60% delle config fermate all'epoca 20
- Config promettenti: completano 25 epoche

### Esempio 3: Search Estensiva
```bash
python grid_search_fast.py \
    --lr 0.005 0.01 0.015 0.025 0.035 0.05 0.075 0.1 \
    --width_mult 0.25 0.3 0.35 0.4 0.42 0.45 0.5 0.55 \
    --weight_decay 0.00005 0.0001 0.0002 0.0003 0.0005 0.0007 0.001 0.0015 \
    --checkpoint_epoch 18 \
    --min_threshold 83.0 \
    --epochs 22
```
- Testa 8×8×8 = **512 combinazioni**
- Threshold più bassa (83%) per esplorare più opzioni
- Tempo: ~20-25 ore
- Massima esplorazione spazio iperparametri

### Esempio 4: Fine-Tuning
```bash
# Dopo aver trovato regione promettente (es. lr~0.05, width~0.5)
python grid_search_fast.py \
    --lr 0.04 0.045 0.05 0.055 0.06 \
    --width_mult 0.45 0.475 0.5 0.525 0.55 \
    --weight_decay 0.0004 0.0005 0.0006 0.0007 \
    --checkpoint_epoch 20 \
    --min_threshold 87.0 \
    --epochs 30
```
- Ricerca fine-grained attorno a zona ottimale
- Threshold alta (87%) per tenere solo le migliori
- 5×5×4 = 100 combinazioni

### Esempio 5: Variare lr_scale e momentum
```bash
python grid_search_fast.py \
    --lr 0.025 0.05 \
    --width_mult 0.42 0.5 \
    --weight_decay 0.0003 0.0005 \
    --lr_scale 1.2 1.4 1.54 1.7 \
    --momentum 0.85 0.9 0.95 \
    --checkpoint_epoch 20 \
    --min_threshold 85.0 \
    --epochs 25
```
- Testa anche lr_scale e momentum
- 2×2×2×4×3 = 96 combinazioni
- Utile per ablation study su gradient scaling

## 📈 Come Scegliere i Parametri

### Checkpoint Epoch
- **Troppo presto (es. 10)**: Rischi di scartare config che migliorano dopo
- **Troppo tardi (es. 35)**: Non risparmi molto tempo
- **Raccomandato: 18-22 epoche**
  - A quest'epoca, le curve di training sono già stabilizzate
  - Puoi predire l'andamento finale

### Minimum Threshold
Analizza la grid search precedente per calibrare:

```python
# Dalla tua grid search a 50 epoche:
# - Run peggiore: 85.21% finale
# - Run migliore: 91.47% finale
#
# Accuratezza media all'epoca 20: ~87-88%
# Se imposto threshold=85%, scarto le config che all'epoca 20 sono sotto 85%
# Queste difficilmente supereranno 90% all'epoca 50
```

**Suggerimenti per threshold**:
- **Conservativo (82-83%)**: Tiene quasi tutto, poco risparmio
- **Moderato (85-86%)**: Scarta ~40-50% config, buon compromesso
- **Aggressivo (87-88%)**: Scarta ~60-70% config, rischi di perdere qualcosa

## 📁 Output

La grid search genera:

```
reports/grid_search_fast/search_YYYYMMDD_HHMMSS/
├── grid_search_results.json      # Risultati completi JSON
├── best_config.json               # Migliore configurazione trovata
├── SUMMARY.txt                    # Riassunto human-readable
└── run_XXX_lr_XXX_width_XXX_.../  # Directory per ogni run
    ├── models/
    │   └── (nessun model salvato per config early-stopped)
    └── logs/
        └── training_metrics.json
```

### Struttura `grid_search_results.json`
```json
{
  "search_space": {...},
  "checkpoint_epoch": 20,
  "min_threshold": 85.0,
  "runs": [...],                  // Tutti i run
  "successful_runs": [...],       // Solo run completati
  "stopped_runs": [...],          // Run fermati early
  "best_config": {...},           // Migliore configurazione
  "best_val_acc": 91.5
}
```

## 🎯 Strategia Raccomandata

### Passo 1: Esplorazione Ampia
```bash
# ~180 config, threshold moderata
python grid_search_fast.py \
    --lr 0.01 0.025 0.05 0.075 0.1 \
    --width_mult 0.3 0.35 0.4 0.42 0.45 0.5 \
    --weight_decay 0.0001 0.0002 0.0003 0.0005 0.0007 0.001 \
    --checkpoint_epoch 20 \
    --min_threshold 85.0 \
    --epochs 25
```

### Passo 2: Analizza Top-10
Guarda `SUMMARY.txt` e identifica regioni promettenti:
- lr ~ ?
- width_mult ~ ?
- weight_decay ~ ?

### Passo 3: Fine-Tuning
```bash
# Ricerca più fine attorno ai valori migliori
python grid_search_fast.py \
    --lr 0.04 0.045 0.05 0.055 0.06 \
    --width_mult 0.45 0.475 0.5 0.525 0.55 \
    --weight_decay 0.00045 0.0005 0.00055 0.0006 \
    --checkpoint_epoch 20 \
    --min_threshold 87.0 \
    --epochs 30
```

### Passo 4: Training Finale
```bash
# Prendi la BEST config e allena per 200 epoche
python train.py \
    --lr 0.055 \
    --width_mult 0.525 \
    --weight_decay 0.0005 \
    --epochs 200
```

## ⚡ Ottimizzazioni Implementate

1. **Shared Data Loaders**: Dataset caricato una sola volta, condiviso tra tutti i run
2. **Early Stopping**: Config deboli fermate presto
3. **Incremental Saving**: Risultati salvati dopo ogni run (non perdi nulla se si interrompe)
4. **Compact Logging**: Output pulito e leggibile

## 🔍 Debugging

Se un run fallisce, controlla:
```bash
# Nel file grid_search_results.json
{
  "runs": [
    {
      "run_id": 42,
      "status": "error",
      "error": "CUDA out of memory"  // <-- Messaggio di errore
    }
  ]
}
```

## 📊 Analisi Risultati

### Python Script per Visualizzare Top Configs
```python
import json

with open('reports/grid_search_fast/search_XXXXX/grid_search_results.json') as f:
    results = json.load(f)

# Top 10 configurazioni
top_10 = sorted(results['successful_runs'], 
                key=lambda x: x['best_val_acc'], 
                reverse=True)[:10]

for i, run in enumerate(top_10, 1):
    print(f"Rank {i}: {run['best_val_acc']}% - {run['config']}")

# Statistiche early stopping
total = len(results['runs'])
stopped = len(results['stopped_runs'])
print(f"\nEarly stopped: {stopped}/{total} ({stopped/total*100:.1f}%)")
print(f"Time saved: ~{stopped * 5 / 60:.1f} hours")  # Assume 5min saved per stopped config
```

## 🚨 Note Importanti

1. **Batch size fisso a 128**: Non modificabile in questa grid search (già ottimale)

2. **GPU Memory**: Se hai OOM errors, riduci batch_size a 64 nel codice base

3. **Checkpoint consigliato**: 
   - Per epochs=25 → checkpoint_epoch=20
   - Per epochs=30 → checkpoint_epoch=22-25
   - Per epochs=20 → checkpoint_epoch=15-18

4. **Threshold calibration**: 
   - Prima grid search: usa 85% (moderato)
   - Se troppi run stoppati: abbassa a 82-83%
   - Se troppo pochi stoppati: alza a 87-88%

## 🎓 Per la Tesi

Nella tesi, documenta:
1. **Motivazione**: Perché hai usato early stopping (efficienza)
2. **Criterio**: Come hai scelto checkpoint_epoch e threshold
3. **Risultati**: Quante config stoppate vs completate
4. **Validazione**: Confronta top config fast vs top config grid normale (dovrebbero essere simili!)

Esempio frase tesi:
> "Per esplorare uno spazio di iperparametri più ampio in modo efficiente, abbiamo implementato una strategia di early stopping basata su threshold. Le configurazioni che non raggiungevano l'85% di validation accuracy all'epoca 20 venivano scartate, permettendo di testare 180 combinazioni in ~10 ore invece delle ~90 ore necessarie con grid search completa."
