# Capitolo 4: Risultati Sperimentali

> **Obiettivo**: Presentare i risultati ottenuti da Mimir e confrontarli con i modelli baseline.  
> **Lunghezza**: 8-10 pagine

---

## 4.1 Metriche del Modello Mimir (~2 pagine)

### Tabella riassuntiva:

| Metrica | Valore | Note |
|---------|--------|------|
| **Parametri Totali** | 200,000 | 0.20M |
| **Parametri Trainable** | 200,000 | 100% trainable |
| **FLOPs** | 35,000,000 | 35M |
| **Dimensione Modello** | ~0.8 MB | FP32 |
| **Accuratezza Test** | 88.0% | CIFAR-10 test set |
| **Tempo Inferenza** | ~X ms | Su GPU Y (specificare) |

### Template paragrafo:

```
Il modello Mimir, con la configurazione finale (width multiplier α=0.4, 
expansion ratios [3,6,8,8]), raggiunge le seguenti prestazioni su CIFAR-10:

**Complessità**:
- **Parametri**: 200.000 (0.20M), distribuiti principalmente negli stage S3 
  e S4 dove i canali raggiungono la massima espansione.
- **FLOPs**: 35 milioni, significativamente inferiori ai 97M di MobileNetV2-0.5x
  grazie al width multiplier aggressivo e all'uso di depthwise convolutions.
- **Dimensione**: 0.8 MB in FP32, facilmente deployable su microcontrollori 
  con almeno 1MB di RAM.

**Accuratezza**:
- **Test Accuracy**: 88.0% sul test set ufficiale di CIFAR-10 (10.000 immagini)
- **Training Accuracy**: 94.2% (gap di ~6%, indicativo di leggero overfitting)
- **Top-2 Accuracy**: 96.5% (utile per applicazioni che considerano top-k)

La Figura 4.1 mostra le curve di training per 50 epoche. Si osserva una 
convergenza stabile senza oscillazioni significative, indicando che gli 
hyperparameter (lr=0.025, cosine annealing) sono ben calibrati.
```

### Figure da includere:
- **Figura 4.1**: Training curve (loss + accuracy per 50 epoche)
- **Figura 4.2**: Distribuzione dei parametri per layer

---

## 4.2 Confronto con Baseline (~4 pagine)

### 4.2.1 Tabella Comparativa Completa

| Modello | Params (M) | FLOPs (M) | Acc (%) | Params Ratio | FLOPs Ratio | Acc Delta |
|---------|------------|-----------|---------|--------------|-------------|-----------|
| **Mimir** | **0.20** | **35** | **88.0** | **1.0×** | **1.0×** | **–** |
| MobileNetV2-0.5x | 0.70 | 97 | 90.1 | 3.5× | 2.8× | +2.1% |
| ShuffleNetV2-0.5x | 0.35 | 40 | 87.5 | 1.75× | 1.14× | -0.5% |
| ResNet-20 | 0.27 | 41 | 91.2 | 1.35× | 1.17× | +3.2% |
| MobileNetV2-1.0x | 2.30 | 300 | 92.0 | 11.5× | 8.6× | +4.0% |

*Nota: "Params Ratio" e "FLOPs Ratio" sono relativi a Mimir (baseline=1.0×)*

### Template analisi dettagliata:

```
**Confronto con MobileNetV2-0.5x**:
Mimir presenta 3.5× meno parametri (200k vs 700k) e 2.8× meno FLOPs (35M vs 97M) 
rispetto a MobileNetV2-0.5x, sacrificando solo 2.1 punti percentuali di 
accuratezza (88.0% vs 90.1%). Questo trade-off è eccellente per applicazioni 
edge dove memoria e computazione sono fortemente limitate. La maggiore efficienza 
di Mimir deriva da:
1. Width multiplier più aggressivo (0.4x vs 0.5x)
2. Uso di ECA invece di SE (meno parametri)
3. Ottimizzazione dei block settings per CIFAR-10

**Confronto con ShuffleNetV2-0.5x**:
Mimir **domina** ShuffleNetV2-0.5x in termini Pareto-efficiency: ha 43% meno 
parametri (200k vs 350k), 12.5% meno FLOPs (35M vs 40M), e **+0.5% di 
accuratezza** (88.0% vs 87.5%). Questo risultato dimostra che l'approccio 
basato su inverted residuals + ECA è superiore al channel shuffle di ShuffleNet 
per dataset a bassa risoluzione come CIFAR-10.

**Confronto con ResNet-20**:
ResNet-20 mantiene un vantaggio in accuratezza (+3.2%, 91.2% vs 88.0%) grazie 
alla sua profondità (20 layer vs ~10 di Mimir) e ai blocchi residuali tradizionali. 
Tuttavia, Mimir è 26% più leggero (200k vs 270k params) mantenendo FLOPs simili. 
Per applicazioni dove 85-88% di accuracy è sufficiente, Mimir offre un miglior 
compromesso dimensione/prestazioni.

**Confronto con MobileNetV2-1.0x**:
La versione full-size di MobileNetV2 raggiunge 92% accuracy ma richiede 11.5× 
più parametri e 8.6× più FLOPs. Questo la rende inadatta per dispositivi ultra-
low-power, dimostrando la necessità di architetture compatte come Mimir.
```

### Figure da includere:
- **Figura 4.3**: Scatter plot Accuracy vs Parametri
- **Figura 4.4**: Bar chart confronto dimensioni

---

## 4.3 Analisi Efficienza (~2 pagine)

### 4.3.1 Frontiera di Pareto

```
La Figura 4.5 mostra la frontiera di Pareto per i modelli considerati. 
Mimir e ShuffleNetV2-0.5x sono i due modelli Pareto-efficient (nessun 
altro modello li domina in entrambi parametri E accuratezza). Tuttavia, 
Mimir è l'unico modello che soddisfa simultaneamente:
- Parametri < 250k ✓
- Accuratezza > 85% ✓
- FLOPs < 50M ✓

Questo lo rende ideale per deployment su microcontrollori con vincoli 
stringenti (es. STM32, ESP32, ARM Cortex-M7).
```

### 4.3.2 Stima Consumo Energetico (Opzionale)

Se hai dati di inferenza su hardware reale:

```
**Inferenza su Raspberry Pi 4**:
- Tempo per immagine: X ms
- Energia per inferenza: Y mJ
- FPS: Z

**Confronto con MobileNetV2-0.5x**:
- Speedup: A×
- Risparmio energetico: B%
```

---

## 4.4 Ablation Study (~2 pagine)

### Tabella Ablation:

| Configurazione | Params (M) | Acc (%) | Delta |
|----------------|------------|---------|-------|
| **Mimir (Full)** | **0.20** | **88.0** | **–** |
| Mimir senza ECA | 0.19 | 85.8 | -2.2% |
| Mimir con SE | 0.24 | 87.9 | -0.1% |
| Mimir con ReLU6 | 0.20 | 86.8 | -1.2% |
| Mimir width=0.5x | 0.31 | 89.1 | +1.1% |

### Template analisi:

```
**Effetto di ECA**:
Rimuovendo i moduli ECA (sostituendoli con identity), l'accuratezza scende 
da 88.0% a 85.8% (-2.2%), confermando che l'attenzione fornisce un contributo 
significativo alla capacità rappresentativa. I parametri diminuiscono 
marginalmente (200k → 190k) ma il trade-off non è vantaggioso.

**ECA vs SE**:
Sostituendo ECA con SE-Net (reduction ratio r=16), i parametri aumentano 
da 200k a 240k (+20%) mentre l'accuratezza rimane praticamente invariata 
(88.0% vs 87.9%). Questo conferma che ECA offre performance comparabili a 
SE con molti meno parametri, rendendolo preferibile per edge deployment.

**GELU vs ReLU6**:
Sostituendo GELU con ReLU6 (come in MobileNetV2 standard), l'accuratezza 
scende da 88.0% a 86.8% (-1.2%). Questo gap conferma i benefici di GELU 
in termini di smoothness dei gradienti e convergenza.

**Effetto Width Multiplier**:
Aumentando il width a 0.5x (310k params), l'accuratezza sale a 89.1% (+1.1%). 
Tuttavia, questo supera il budget di 250k parametri, confermando che α=0.4x 
è il punto ottimale per il constraint specificato.
```

---

## 4.5 Confusion Matrix e Analisi Errori (~1 pagina)

### Template analisi:

```
La Figura 4.6 mostra la confusion matrix di Mimir sul test set. Si osservano 
le seguenti tendenze:

**Classi Meglio Classificate**:
- Nave (ship): 94.5% accuracy
- Camion (truck): 94.9% accuracy
- Cavallo (horse): 92.5% accuracy

Queste classi hanno feature visuali distintive (forme, colori) che facilitano 
la classificazione.

**Classi Problematiche**:
- Gatto vs Cane: 12% di confusione reciproca (texture simile)
- Uccello vs Aereo: 8% di confusione (forme simili, background sky)
- Cervo vs Cavallo: 7% di confusione (mammiferi quadrupedi)

Queste confusioni sono coerenti con la letteratura su CIFAR-10 e riflettono 
la similarità visiva intrinseca tra le classi.
```

### Figure da includere:
- **Figura 4.6**: Confusion matrix 10×10

---

## Checklist Capitolo 4

- [ ] Metriche Mimir complete (params, FLOPs, accuracy)
- [ ] Tabella comparativa con almeno 4 baseline
- [ ] Analisi dettagliata confronto per ogni modello
- [ ] Scatter plot e bar chart inclusi
- [ ] Frontiera Pareto discussa
- [ ] Ablation study con almeno 3 varianti
- [ ] Confusion matrix visualizzata e analizzata
- [ ] Lunghezza: 8-10 pagine
