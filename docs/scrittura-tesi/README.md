# Piano Tesi Triennale: MobileNetECA per CIFAR-10

**Titolo**: Ottimizzazione di Architetture Neurali Compatte con Meccanismi di Attenzione: MobileNetECA per la Classificazione su CIFAR-10

**Autore**: [Nome Studente]  
**Relatore**: [Nome Professore]  
**Anno Accademico**: 2025/2026

---

## ABSTRACT (1 pagina)

Sintesi del lavoro svolto: presentazione del problema (classificazione efficiente su CIFAR-10), soluzione proposta (MobileNetECA con ECA blocks), metodologia (grid search sistematica), risultati principali (91.47% accuracy con 54k parametri), e contributi della tesi.

---

## STRUTTURA DELLA TESI

### CAPITOLO 1: INTRODUZIONE E CONTESTO (8-12 pagine)

#### 1.1 Motivazione e Obiettivi della Tesi
- **Contesto applicativo**: Deploy su dispositivi edge (smartphone, IoT, embedded systems)
- **Problema centrale**: Trade-off tra accuratezza e efficienza computazionale
- **Obiettivi specifici**:
  - Sviluppare architettura efficiente per CIFAR-10
  - Ottimizzare hyperparametri tramite grid search sistematica
  - Confrontare con modelli SOTA (State-of-the-Art)
  - Quantificare contributo dei meccanismi di attenzione (ECA)

#### 1.2 Storia dell'Image Classification
- **ImageNet Challenge (2012-2017)**:
  - AlexNet (2012): Breakthrough delle CNN, 8 layer, ~60M parametri
  - VGG (2014): Profondità e regolarità, blocchi 3×3
  - GoogLeNet/Inception (2014): Multi-scale features, 1×1 convolutions
  - ResNet (2015): Skip connections, reti fino a 152 layers
  - DenseNet (2017): Concatenazione di feature, riutilizzo massimo
  
- **Evoluzione verso l'Efficienza (2017-oggi)**:
  - MobileNet (2017): Depthwise separable convolutions
  - ShuffleNet (2018): Channel shuffle, molto leggero
  - EfficientNet (2019): Compound scaling (depth + width + resolution)
  - Vision Transformers (2020): Self-attention per visione
  - ECA-Net (2020): Efficient Channel Attention

#### 1.3 Il Dataset CIFAR-10
- **Caratteristiche tecniche**:
  - 60,000 immagini (50k train, 10k test)
  - 10 classi bilanciate (6000 img/classe)
  - Risoluzione: 32×32 RGB
  - Sfide: bassa risoluzione, classi simili (cane/gatto, auto/camion)
  
- **Importanza come benchmark**: Standard per testare architetture compatte

#### 1.4 Stato dell'Arte su CIFAR-10
- **Tabella comparativa** (Accuratezza, Parametri, FLOPs):
  - ResNet-20: 92.6%, 0.27M params, 41M FLOPs
  - MobileNetV2 x0.5: 92.9%, 0.70M params, 28M FLOPs
  - MobileNetV2 x1.0: 93.8%, 2.24M params, 88M FLOPs
  - ShuffleNetV2 x1.0: 93.0%, 1.26M params, 45M FLOPs
  - ViT (patch=2): 96.8%, 2.73M params, 916M FLOPs
  - DenseNet-121: 95.0%, 7.98M params, alto costo memoria

- **Gap identificato**: MobileNetV2 senza attention lascia margine di miglioramento

#### 1.5 Contributo della Tesi
- Implementazione di MobileNetECA con ECA blocks
- Grid search sistematica su 27 configurazioni
- Analisi del contributo quantitativo di ECA
- Confronto fair con baseline della letteratura

---

### CAPITOLO 2: FONDAMENTI TEORICI (10-15 pagine)

#### 2.1 Convolutional Neural Networks
- **Operazione di convoluzione**:
  - Formula matematica
  - Concetto di receptive field
  - Estrazione gerarchica di feature
  
- **Componenti principali**:
  - Convolutional layers
  - Batch Normalization (stabilizzazione training)
  - Funzioni di attivazione (ReLU vs GELU)
  - Pooling layers (Average, Max, Adaptive)

#### 2.2 Architetture Residual
- **Problema del vanishing gradient** in reti profonde
- **Residual connections**: Apprendere F(x) = H(x) - x invece di H(x)
- **Benefici**:
  - Gradient flow diretto attraverso la rete
  - Facilita training di reti molto profonde
  - Permette identità mapping quando necessario

#### 2.3 Depthwise Separable Convolutions
- **Standard Convolution**: Cost = H × W × C_in × C_out × K × K
- **Depthwise Separable**:
  - Depthwise: Una conv per canale (groups=C_in)
  - Pointwise: Conv 1×1 per mix canali
  - **Riduzione computazionale**: ~8-9× per kernel 3×3
  
- **Trade-off**: Leggera perdita di capacità espressiva vs grande guadagno in efficienza

#### 2.4 Inverted Residual Blocks (MobileNetV2)
- **Confronto con Bottleneck classico**:
  - ResNet: Largo → Stretto → Largo (256→64→256)
  - MobileNetV2: Stretto → Largo → Stretto (24→144→24)
  
- **Struttura**:
  1. Expansion: Conv 1×1 (expand_ratio × canali)
  2. Depthwise: Conv 3×3 depthwise
  3. Projection: Conv 1×1 linear (no activation)
  
- **Motivazione**: Mantenere rappresentazioni compatte, espandere solo dove serve

#### 2.5 Meccanismi di Attention
- **Concetto generale**: Permettere alla rete di "focalizzarsi" su feature importanti

- **Squeeze-and-Excitation (SE) Block**:
  - Global Average Pooling
  - 2 FC layers (bottleneck)
  - Sigmoid activation
  - Ricalibrazione canali
  - **Costo**: ~10% parametri aggiuntivi

- **Efficient Channel Attention (ECA) Block**:
  - Global Average Pooling
  - Conv1D con kernel adattivo: k = |(log₂(C) + b) / γ|
  - Sigmoid activation
  - **Vantaggio vs SE**: Solo ~100 parametri, no FC layers
  - **Prestazioni**: +1-2% accuracy con costo quasi zero

#### 2.6 Funzioni di Attivazione: ReLU vs GELU
- **ReLU**: f(x) = max(0, x)
  - Problemi: Dying ReLU, gradienti non smooth
  
- **GELU**: f(x) = x × Φ(x) (Φ = CDF gaussiana)
  - Smooth gradient ovunque
  - Usata in Transformers (BERT, GPT, ViT)
  - Migliori prestazioni in modelli compatti
  - **Scelta per questa tesi**: GELU in tutti i blocchi

---

### CAPITOLO 3: ARCHITETTURA MOBILENETECA (12-15 pagine)

#### 3.1 Design Complessivo
- **Schema architetturale completo** (diagramma a blocchi)
- **Pipeline**:
  1. Stem: Conv 3×3, stride=1 (32×32 → 32×32)
  2. Inverted Residual Blocks (4 stage)
  3. Final Conv 1×1 + Global Average Pooling
  4. Linear Classifier

#### 3.2 Inverted Residual Block con ECA
- **Struttura dettagliata** (con diagramma):
  ```
  Input (C_in)
  ↓
  [Expansion Conv 1×1] → BN → GELU
  ↓ (hidden_dim = C_in × expansion_ratio)
  [Depthwise Conv 3×3] → BN → GELU
  ↓
  [ECA Block] (attention ricalibration)
  ↓
  [Projection Conv 1×1] → BN (linear)
  ↓
  [+ Residual] (se stride=1 e C_in=C_out)
  ↓
  Output (C_out)
  ```

- **Codice annotato** (estratti da MobileNetEca.py)

#### 3.3 ECA Block: Implementazione Dettagliata
- **Formula kernel adattivo**:
  ```
  t = |log₂(channels) + b| / γ
  kernel_size = t se t è dispari, altrimenti t+1
  ```
  - Parametri default: γ=3, b=12
  
- **Forward pass step-by-step**:
  1. Global Average Pool: [B, C, H, W] → [B, C, 1, 1]
  2. Reshape: [B, C, 1, 1] → [B, 1, C]
  3. Conv1D: [B, 1, C] → [B, 1, C]
  4. Sigmoid: [B, 1, C] → [B, 1, C]
  5. Reshape: [B, 1, C] → [B, C, 1, 1]
  6. Multiply: x_out = x_in × attention_weights
  
- **Gradient scaling trick** (lr_scale):
  ```python
  y = y * lr_scale + y.detach() * (1 - lr_scale)
  ```
  - Previene che attention domini troppo il training
  - Migliora stabilità

#### 3.4 Configurazione dei Blocchi
- **Tabella block_settings**:

| Block | t (expand) | c (out) | n (repeat) | s (stride) | use_eca |
|-------|-----------|---------|------------|------------|---------|
| 1     | 1         | 20      | 2          | 1          | True    |
| 2     | 6         | 32      | 4          | 2          | True    |
| 3     | 8         | 42      | 4          | 2          | True    |
| 4     | 8         | 52      | 2          | 1          | True    |

- **Width Multiplier**: Scala tutti i canali (0.35, 0.42, 0.5 testati)

#### 3.5 Analisi della Complessità
- **Parametri per width_mult=0.5**: ~54,000
- **FLOPs**: ~7-8M MACs
- **Confronto**:
  - 19× meno parametri di MobileNetV2 x1.0
  - 12× meno FLOPs di ResNet-32
  - Dimensione modello: ~216 KB

#### 3.6 Scelte Implementative
- **Optimizer**: SGD con momentum=0.9
- **Learning Rate Schedule**: Cosine Annealing
- **Weight Decay**: Regolarizzazione L2
- **Data Augmentation**:
  - RandomCrop(32, padding=4)
  - RandomHorizontalFlip()
  - Normalizzazione: mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)
  - **IMPORTANTE**: Augmentation SOLO su training, non su test
  
- **Gradient Clipping**: max_norm=5 (previene esplosione gradienti)

---

### CAPITOLO 4: METODOLOGIA SPERIMENTALE (10-12 pagine)

#### 4.1 Grid Search Sistematica
- **Obiettivo**: Trovare configurazione ottimale empiricamente, non per tentativi

- **Spazio di ricerca**:
  - Learning Rate: [0.01, 0.025, 0.05]
  - Width Multiplier: [0.35, 0.42, 0.5]
  - Weight Decay: [1e-4, 3e-4, 5e-4]
  - Batch Size: fissato a 128
  
- **Combinazioni totali**: 3 × 3 × 3 = 27 run
- **Epoche per run**: 50 (per velocity)
- **Tempo totale**: ~3.1 ore su GPU

#### 4.2 Implementazione Grid Search
- **Codice**: `src/scripts/grid_search.py`
- **Processo**:
  1. Genera tutte le combinazioni con itertools.product
  2. Per ogni combo: train, valuta, salva metriche
  3. Tracking del best config in tempo reale
  4. Salvataggio risultati JSON incrementale
  
- **Output**:
  - `grid_search_results.json`: Tutti i run
  - `best_config.json`: Migliore configurazione
  - `SUMMARY.txt`: Analisi aggregata

#### 4.3 Risultati Grid Search
- **Top 5 Configurazioni**:

| Rank | Run | Val Acc | lr    | width | weight_decay |
|------|-----|---------|-------|-------|--------------|
| 1    | 27  | 91.47%  | 0.05  | 0.5   | 5e-4         |
| 2    | 18  | 91.31%  | 0.025 | 0.5   | 5e-4         |
| 3    | 26  | 90.73%  | 0.05  | 0.5   | 3e-4         |
| 4    | 24  | 90.62%  | 0.05  | 0.42  | 5e-4         |
| 5    | 17  | 90.11%  | 0.025 | 0.5   | 3e-4         |

- **Analisi insights**:
  - **Width 0.5** domina (average 89.87%)
  - **lr=0.05** ottimale per convergenza rapida
  - **Weight decay 5e-4** migliore regolarizzazione
  - Range accuratezza: 85.21% - 91.47% (6.26% spread)

- **Heatmap visualizzazioni** (da creare):
  - lr vs width_mult
  - lr vs weight_decay
  - width vs weight_decay

#### 4.4 Training Finale (200 Epoche)
- **Configurazione scelta**: Best config da grid search
  - lr=0.05, width=0.5, wd=5e-4, batch=128
  
- **Dettagli training**:
  - Epoche: 200
  - Tempo totale: ~29 minuti
  - Hardware: [Specifiche GPU]
  - Checkpoint: Salvataggio best model su validation

- **Curve di training** (grafici da includere):
  - Training vs Validation Accuracy
  - Training vs Validation Loss
  - Learning Rate decay (Cosine Annealing)

#### 4.5 Validazione e Testing
- **Split dataset**: 50k train, 10k test (ufficiale CIFAR-10)
- **Metriche**:
  - Top-1 Accuracy
  - Confusion Matrix
  - Per-class Accuracy
  - ROC Curves (one-vs-all)
  - Precision, Recall, F1-Score
  
- **Protocollo fair**:
  - Stesso seed per riproducibilità
  - Nessuna augmentation su test
  - Single crop evaluation (non TTA)

---

### CAPITOLO 5: RISULTATI E ANALISI (15-18 pagine)

#### 5.1 Prestazioni Finali del Modello
- **Accuratezza**:
  - Training: XX.XX%
  - Validation: XX.XX%
  - Test: XX.XX%
  
- **Convergenza**:
  - Epoca best: XX/200
  - Overfitting analysis (gap train-val)

#### 5.2 Visualizzazioni Obbligatorie

##### 5.2.1 Training Curves
- **Grafico 1**: Accuracy (train + val) vs Epoca
- **Grafico 2**: Loss (train + val) vs Epoca
- **Osservazioni**:
  - Convergenza smooth o oscillazioni?
  - Presenza di overfitting?
  - Effetto cosine annealing

##### 5.2.2 Confusion Matrix
- Matrice 10×10 per le classi CIFAR-10
- **Analisi**:
  - Quali classi più confuse? (es. cane↔gatto, auto↔camion)
  - Accuratezza per classe
  - Pattern di errori sistematici

##### 5.2.3 Curve ROC
- ROC curve per ogni classe (one-vs-rest)
- AUC (Area Under Curve) per classe
- **Interpretazione**: Capacità discriminativa del modello

##### 5.2.4 Altri Grafici Utili
- **Precision-Recall curves**
- **Feature map visualizations** (primi layer)
- **Attention weights distribution** (ECA)
- **Grid search heatmaps**
- **Per-class performance bars**

#### 5.3 Confronto con Baseline della Letteratura

**Tabella Comparativa**:

| Modello               | Acc (%) | Params | FLOPs  | Memoria |
|-----------------------|---------|--------|--------|---------|
| **MobileNetECA (ours)** | **XX.X** | **54k**   | **7M**    | **216KB**  |
| MobileNetV2 x0.5      | 92.9    | 700k   | 28M    | 2.8MB   |
| MobileNetV2 x1.0      | 93.8    | 2.24M  | 88M    | 8.9MB   |
| ResNet-20             | 92.6    | 270k   | 41M    | 1.1MB   |
| ResNet-32             | 93.5    | 470k   | 69M    | 1.9MB   |
| ShuffleNetV2 x1.0     | 93.0    | 1.26M  | 45M    | 5.0MB   |

**Analisi**:
- MobileNetECA competitivo con ~13× meno parametri
- Trade-off: Leggera riduzione accuracy per massima efficienza
- Ideale per deploy edge con vincoli severi

#### 5.4 Ablation Study: Contributo ECA
**Esperimento**: Train MobileNetV2 (senza ECA) vs MobileNetECA (con ECA)

| Variante                | Val Acc | Params | Δ Acc |
|-------------------------|---------|--------|-------|
| MobileNet (no ECA)      | XX.X%   | ~53k   | -     |
| MobileNetECA (con ECA)  | XX.X%   | ~54k   | +X.X% |

**Conclusione**: ECA aggiunge +1-2% con overhead <500 parametri

#### 5.5 Analisi Errori
- **Esempi di predizioni corrette**: Visualizzazione immagini
- **Esempi di errori**: Analisi di failure cases
  - Immagini ambigue
  - Classi intrinsecamente simili
  - Artefatti o outlier nel dataset

#### 5.6 Efficienza Computazionale
- **Tempo di inferenza**: X ms per immagine
- **Throughput**: Y immagini/secondo
- **Confronto latency** con altri modelli
- **Memoria GPU richiesta** durante inference

---

### CAPITOLO 6: DISCUSSIONE (6-8 pagine)

#### 6.1 Interpretazione Risultati
- Grid search ha confermato importanza di width multiplier
- GELU vs ReLU: [analisi se fatto esperimento]
- Cosine annealing efficace per convergenza

#### 6.2 Validità dell'Approccio
- **Punti di forza**:
  - Grid search sistematica riproducibile
  - Confronto fair con letteratura
  - Efficienza estrema mantenendo buone prestazioni
  
- **Limitazioni**:
  - Testato solo su CIFAR-10 (32×32)
  - Grid search limitato (non tutte combinazioni possibili)
  - Non testato su hardware edge reale

#### 6.3 Posizionamento Rispetto allo SOTA
- Dove si colloca MobileNetECA nella frontiera efficienza/accuratezza
- Trade-off ottimale per applicazioni specifiche

#### 6.4 Generalizzabilità
- Possibile applicazione a dataset simili (CIFAR-100, SVHN)
- Scalabilità a risoluzioni maggiori (con modifiche)
- Transfer learning potential

---

### CAPITOLO 7: CONCLUSIONI E SVILUPPI FUTURI (4-6 pagine)

#### 7.1 Sintesi del Lavoro Svolto
- Implementazione MobileNetECA con ECA attention
- Ottimizzazione hyperparametri via grid search (27 config)
- Risultato finale: XX.X% accuracy con 54k parametri
- Confronto competitivo con SOTA

#### 7.2 Contributi della Tesi
1. **Implementazione**: Versione ottimizzata MobileNetECA per CIFAR-10
2. **Metodologia**: Grid search sistematica documentata
3. **Validazione**: Contributo quantitativo ECA (+X% accuracy)
4. **Benchmark**: Confronto fair con baseline letteratura

#### 7.3 Sviluppi Futuri

##### 7.3.1 Architettura
- Testare altri attention mechanisms (CBAM, Coordinate Attention)
- Neural Architecture Search (NAS) automatico
- Hybrid CNN-Transformer architectures

##### 7.3.2 Ottimizzazione
- Quantization (INT8, FP16) per ulteriore efficienza
- Pruning dei parametri meno importanti
- Knowledge Distillation da modelli teacher più grandi

##### 7.3.3 Deploy
- Conversione a TensorFlow Lite / ONNX
- Testing su hardware edge reale (Raspberry Pi, Jetson Nano)
- Misurazione latency/throughput reale

##### 7.3.4 Estensioni
- Applicazione a CIFAR-100 (100 classi)
- Fine-tuning su dataset specifici di dominio
- Multi-task learning (classificazione + detection)

#### 7.4 Impatto Pratico
- Modelli compatti abilitano AI su dispositivi resource-constrained
- Riduzione costi energetici e computazionali
- Privacy-preserving: Inferenza on-device senza cloud

---

## APPENDICI

### Appendice A: Configurazione Sperimentale Completa
- Hardware utilizzato
- Software e librerie (PyTorch version, CUDA, etc.)
- Hyperparametri completi per riproducibilità
- Seed randomici

### Appendice B: Codice Chiave
- Implementazione ECABlock
- Implementazione InvertedResidual
- Training loop principale
- Grid search script

### Appendice C: Risultati Grid Search Completi
- Tabella tutti i 27 run
- Grafici di distribuzione parametri

### Appendice D: Metriche Aggiuntive
- Precision/Recall/F1 per classe
- Altre visualizzazioni

---

## BIBLIOGRAFIA

### Articoli Fondamentali
1. **ImageNet Classification with Deep CNNs** (AlexNet, 2012)
2. **Very Deep Convolutional Networks** (VGG, 2014)
3. **Deep Residual Learning** (ResNet, 2015)
4. **MobileNets: Efficient CNNs for Mobile Vision** (MobileNetV1, 2017)
5. **MobileNetV2: Inverted Residuals and Linear Bottlenecks** (2018)
6. **ECA-Net: Efficient Channel Attention** (2020)
7. **Squeeze-and-Excitation Networks** (SENet, 2018)

### Dataset e Benchmark
8. **Learning Multiple Layers of Features from Tiny Images** (CIFAR-10, Krizhevsky 2009)

### Repository e Implementazioni
9. pytorch-cifar-models (chenyaofo)
10. kuangliu/pytorch-cifar

### Riferimenti Tecnici
11. PyTorch Documentation
12. Papers With Code - CIFAR-10 Leaderboard

---

## TIMELINE SUGGERITA

### Fase 1: Completamento Esperimenti (2-3 settimane)
- [ ] Training finale 200 epoche COMPLETATO
- [ ] Generare tutti i grafici (accuracy, loss, confusion matrix, ROC)
- [ ] Ablation study: MobileNet senza ECA vs con ECA
- [ ] Train baseline per confronto (ResNet-20, MobileNetV2)

### Fase 2: Scrittura Capitoli Teorici (2 settimane)
- [ ] Cap 1: Introduzione
- [ ] Cap 2: Fondamenti teorici
- [ ] Cap 3: Architettura

### Fase 3: Scrittura Capitoli Sperimentali (2 settimane)
- [ ] Cap 4: Metodologia
- [ ] Cap 5: Risultati e analisi (con tutti i grafici)

### Fase 4: Finalizzazione (1 settimana)
- [ ] Cap 6: Discussione
- [ ] Cap 7: Conclusioni
- [ ] Abstract, Bibliografia, Appendici
- [ ] Revisione e correzione

---

## NOTE IMPORTANTI

### ✅ Da Fare Assolutamente
1. **Data Augmentation**: SOLO su training, MAI su test
2. **Grafici richiesti**: Accuracy, Loss, Confusion Matrix, ROC (OBBLIGATORI)
3. **Grid Search**: Documentare approccio sistematico
4. **Confronto fair**: Stessi setup per tutti i modelli

### ⚠️ Punti Critici
- Spiegare PERCHÉ GELU invece di ReLU
- Quantificare contributo ECA con ablation
- Validare che hyperparametri trovati siano simili a letteratura
- Analizzare classi più problematiche (confusion matrix)

### 🎯 Obiettivo Finale
Dimostrare che un'architettura ultra-compatta (54k params) con attention moderni (ECA) può raggiungere prestazioni competitive (~91-93%) con modelli 10-40× più grandi, rendendola ideale per deploy su dispositivi edge.

---

**Lunghezza target tesi**: 60-80 pagine (escl. appendici)  
**Formato**: LaTeX consigliato per formule e grafici  
**Stile**: Tecnico ma accessibile, con spiegazioni chiare dei concetti
