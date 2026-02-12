# Training MobileNetECA su CIFAR-10

## Avvio Rapido

```bash
cd /workspace/tesi-laurea
python src/legacy/LAST_USED.py
```

## Requisiti

```bash
pip install -r requirements.txt
```

## Configurazione

Parametri modificabili all'inizio del file (righe 28-35):

- `epoche`: Numero di epoche (default: 50)
- `dimensione_batch`: Batch size (default: 128)
- `tasso_iniziale`: Learning rate iniziale (default: 0.025)
- `fattore_larghezza`: Width multiplier (default: 0.42)
- `lr_scale`: Scaling dei gradienti (default: 1.54)

## Output

- **Modello salvato**: `/workspace/tesi-laurea/models/modello_mimir1.pt`
- **Report**: `/workspace/tesi-laurea/reports/training_report.txt`
- **Dataset**: Scaricato automaticamente in `./data/`

## Risultati

Modello: **54k parametri, 9.4M MACs**  
Performance: **~89% validation accuracy** su CIFAR-10

## Architettura

- **MobileNetECA**: MobileNetV2 + ECA attention
- **GELU activation**: Migliori prestazioni vs ReLU in modelli compatti
- **Inverted Residual blocks**: Efficienza parametrica
- **Cosine Annealing LR**: Scheduler per ottimizzazione
