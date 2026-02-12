# Note Discussione Tesi - MobileNetECA per CIFAR-10

## Punti Chiave dalla Discussione con il Professore

---

## 1. Metodologia Sperimentale

### Hyperparameter Tuning
- **Approccio**: Grid Search sistematica per trovare configurazione ottimale
- **Parametri da testare**: learning rate, width multiplier, batch size, weight decay
- **Obiettivo**: Dimostrare che i valori scelti (es. `width_mult=0.42`, `lr=0.025`) sono ottimali tramite ricerca empirica

### Data Augmentation
- ⚠️ **Importante**: Applicare augmentation **SOLO su training set**, NON su test/validation
- Tecniche utilizzate: RandomCrop, RandomHorizontalFlip, Normalizzazione
- Motivazione: Evitare data leakage e garantire valutazione corretta

---

## 2. Visualizzazioni e Grafici Richiesti

### Grafici Obbligatori
1. **Accuracy e Loss**
   - Training curve (entrambe train e validation)
   - Loss curve (entrambe train e validation)
   - Mostrare convergenza del modello

2. **Confusion Matrix**
   - Matrice 10×10 per classi CIFAR-10
   - Identificare quali classi vengono confuse più frequentemente
   - Esempio: gatto↔cane, auto↔camion

3. **Curva ROC** (Receiver Operating Characteristic)
   - Per ogni classe (one-vs-all)
   - Calcolare AUC (Area Under Curve)
   - Dimostrare capacità discriminativa del modello

4. **Altri grafici utili per "riempire"**
   - Precision-Recall curves
   - Per-class accuracy comparison
   - Feature map visualizations
   - Qualsiasi visualizzazione che supporti l'analisi

---

## 3. Struttura della Tesi

### Introduzione e Background

#### Storia dell'Image Classification
- **ImageNet Challenge (2012-2017)**
  - AlexNet (2012): Breakthrough delle CNN
  - VGG (2014): Profondità e regolarità
  - ResNet (2015): Skip connections e reti molto profonde
  - Inception/GoogLeNet: Multi-scale features

#### Evoluzione verso l'Efficienza
- MobileNet (2017): Depthwise separable convolutions
- ShuffleNet (2018): Channel shuffle operations
- EfficientNet (2019): Compound scaling
- **Stato dell'Arte attuale**: Modelli compatti per edge deployment

#### Posizionamento del Lavoro
- Confrontare MobileNetECA con SOTA recenti
- Dimostrare competitività su CIFAR-10
- Evidenziare trade-off efficiency/accuracy

---

### Architettura Proposta

#### ECA Block (Efficient Channel Attention)
**Cos'è:**
- Meccanismo di attention leggero per ricalibrazione dei canali
- Utilizza convoluzione 1D con kernel size adattivo
- Overhead computazionale minimale (~100 parametri per blocco)

**Come migliora la classificazione:**
- Permette al modello di focalizzarsi su feature importanti
- Riduce rumore nei canali meno informativi
- +1-2% accuracy con costo quasi zero

#### Inverted Residual Block
**Cos'è:**
- Struttura: stretto → largo → stretto (es. 24→144→24)
- Opposto al bottleneck ResNet (largo → stretto → largo)

**Perché funziona meglio:**
- Mantiene rappresentazioni compatte negli endpoint
- Espansione solo nella fase intermedia (dove serve)
- Più efficiente per modelli con pochi parametri

---

## 4. Validazione Sperimentale

### Grid Search Personale
- **Implementazione propria** della ricerca hyperparameter
- Testare diverse combinazioni sistematicamente
- Documentare processo e risultati

### Confronto con Letteratura
**Obiettivo:**
- Prendere reti SOTA con hyperparametri da articoli
- Confrontare con i parametri trovati tramite grid search personale
- I risultati devono essere comparabili (±2% accuracy)
- **Validazione**: Se grid search trova parametri simili a quelli pubblicati, conferma validità dell'approccio

**Modelli di riferimento:**
- ResNet-20/32 (baseline classica)
- MobileNetV2 (architettura simile)
- ShuffleNetV2 (competitor diretto)
- EfficientNet-B0 (scaled baseline)

---

## 5. Checklist Implementazione

### Da Fare
- [ ] Implementare Grid Search per hyperparameter tuning
- [ ] Generare tutte le curve (training/validation accuracy e loss)
- [ ] Calcolare e visualizzare Confusion Matrix
- [ ] Implementare e plottare Curve ROC per tutte le classi
- [ ] Replicare esperimenti con hyperparametri da letteratura
- [ ] Scrivere sezione introduttiva (storia, evoluzione, SOTA)
- [ ] Documentare architettura (ECA, Inverted Residual)
- [ ] Analisi comparativa dettagliata

### Best Practices
- ✅ Data augmentation **solo** su training
- ✅ Validazione su test set pulito
- ✅ Grid search sistematica e documentata
- ✅ Confronto fair con baseline
- ✅ Grafici chiari e informativi

---

## 6. Obiettivi della Tesi

**Tesi Principale:**
Dimostrare che un'architettura efficiente (MobileNetECA) con attention mechanisms moderni può raggiungere performance competitive con modelli più grandi, utilizzando una frazione dei parametri.

**Contributi:**
1. Implementazione e ottimizzazione di MobileNetECA per CIFAR-10
2. Grid search sistematica per trovare configurazione ottimale
3. Analisi comparativa con modelli SOTA
4. Validazione empirica dell'efficacia di ECA block
5. Dimostrazione del trade-off efficiency/accuracy

**Impatto:**
- Deploy su dispositivi edge con risorse limitate
- Riduzione costi computazionali
- Mantenimento alta accuracy

---

## Note Finali

Il focus deve essere su:
- **Riproducibilità**: Grid search ben documentata
- **Confronto fair**: Stessi dataset, metriche, procedure
- **Analisi approfondita**: Non solo numeri, ma spiegazione del perché
- **Visualizzazioni**: Grafici chiari che supportano le conclusioni
