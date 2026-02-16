# Piano Tesi: Ottimizzazione di MobileNetV2 con ECA e Reparameterization su CIFAR-10

## Introduzione
Questo documento delinea la struttura della tesi, i modelli analizzati e i risultati sperimentali da includere. L'obiettivo è dimostrare come tecniche moderne di ottimizzazione (Attention, Reparameterization) permettano di ottenere alte prestazioni (accuracy > 93%) con un numero estremamente ridotto di parametri (< 80k) su dataset a bassa risoluzione.

---

## Struttura della Tesi

### 1. Introduzione
*   **Contesto**: Importanza della classificazione di immagini su dispositivi edge (IoT, mobile).
*   **Problema**: I modelli SOTA (es. ViT, ResNet-50) sono troppo pesanti.
*   **Obiettivo**: Progettare una CNN ultraleggera basata su MobileNetV2 che superi il 93% di accuracy su CIFAR-10 con < 100k parametri.
*   **Dataset**: Breve descrizione di CIFAR-10 (32x32, 10 classi, bilanciato).

### 2. Stato dell'Arte e Backgound
*   **Convolutional Neural Networks (CNN)**: Concetti base.
*   **MobileNetV2**:
    *   Depthwise Separable Convolutions (efficienza).
    *   Inverted Residuals (bottleneck).
*   **Attention Mechanisms**:
    *   Spiegazione di SE (Squeeze-and-Excitation).
    *   **Focus su ECA (Efficient Channel Attention)**: Come migliora SE evitando la riduzione di dimensionalità e usando conv 1D.
*   **Struttural Reparameterization**:
    *   Concetto di RepVGG (Training multi-branch -> Inference single-branch).
    *   Vantaggi: Potenza espressiva in training, velocità in inferenza.
*   **Data Augmentation**: Cutout, AutoAugment, MixUp (cenni).

### 3. Metodologia: MobileNetECA-Rep
In questo capitolo descriviamo l'evoluzione del modello, partendo dalla baseline fino alla versione finale.

1.  **Baseline**: MobileNetV2 modificata per CIFAR-10 (width_mult ridotto a 0.5 per target < 100k params).
2.  **Integrazione ECA**: Inserimento del modulo ECA dopo la Depthwise Conv nel blocco Inverted Residual.
3.  **Reparameterization**: Sostituzione delle standard Conv2d con blocchi `RepConv` (3x3 + 1x1 + Identity) nei layer di espansione e proiezione.
4.  **Advanced Augmentation**: Aggiunta di AutoAugment e RandomErasing (Cutout) per prevenire l'overfitting dato l'aumento di capacità del modello ri-parametrizzato.

### 4. Implementazione e Setup Sperimentale
*   **Hardware**: Specificare GPU utilizzata.
*   **Iperparametri (da Grid Search)**:
    *   Learning Rate: 0.05
    *   Optimizer: SGD (momentum=0.9, weight_decay=5e-4) -> *Correction: Code uses 5e-4, Grid Search best was 5e-4*
    *   Scheduler: Cosine Annealing (200 epoche)
    *   Batch Size: 128
    *   Width Multiplier: 0.5
*   **Metriche**: Accuracy (Top-1), Loss, Parametri, FLOPs/MACs.

### 5. Analisi dei Risultati (Sperimentazione Incrementale)

Confronteremo 4 configurazioni per mostrare l'impatto di ogni componente.

| Modello | Configurazione | Best Val Acc (%) | Parametri | Note |
| :--- | :--- | :--- | :--- | :--- |
| **A** | MobileNetV2 (Baseline) | ~91.45% | TBD | Pure MobileNetV2 (0.5x) |
| **B** | + ECA Block | 92.40% | TBD | Aggiunta Attention |
| **C** | + Reparameterization | 92.76% | TBD | Capacità aumentata in training |
| **D** | + Advanced Aug (Final) | **93.50%** | **76.6k** | SOTA per <100k params |

*   **Analisi Tabella**:
    *   L'aggiunta di ECA porta un guadagno di ~1%.
    *   La ri-parametrizzazione aggiunge un ulteriore ~0.4%.
    *   L'augmentation avanzata permette di sfruttare appieno la struttura, guadagnando un altro ~0.7% e riducendo l'overfitting.

### 6. Analisi Approfondita (Analisi Mancanti da Svolgere)
Per completare la tesi, sono necessari i seguenti grafici e dati (come richiesto nelle note del prof):

*   **Curve di Apprendimento**:
    *   Grafico Loss Train vs Val (per dimostrare la convergenza e l'effetto dell'augmentation).
    *   Grafico Accuracy Train vs Val.
    *   *Mancante*: Generare i plot dai file `.json` salvati.
*   **Confusion Matrix**:
    *   Per il modello finale (D). Evidenziare quali classi vengono confuse (es. Gatto vs Cane).
    *   *Mancante*: Calcolare su Test Set con il modello `best_model_training.pth`.
*   **Curve ROC e AUC**:
    *   Per il modello finale.
    *   *Mancante*: Calcolare su Test Set.
*   **Analisi Computazionale**:
    *   Calcolo esatto dei FLOPs (MACs) per i 4 modelli.
    *   Confronto con altri modelli della letteratura (es. ResNet-20: 270k params vs Nostro: 76k) usando i dati dal file di ricerca.

### 7. Conclusioni
Riepilogo dei risultati: Abbiamo ottenuto un modello con **93.5%** di accuracy su CIFAR-10 usando solo **76.6k parametri**, dimostrando che l'architettura (ECA+Rep) conta quanto la dimensione.

---

## Prossimi Passi Operativi (To-Do List)

1.  [ ] **Python Script per Grafici**: Creare uno script che legge i file `_history.json` e genera i plot `loss.png` e `accuracy.png` comparativi.
2.  [ ] **Validation Script**: Creare uno script che carica il modello finale (`MobileNetECARep`), calcola e salva:
    *   Confusion Matrix (immagine).
    *   ROC Curve (immagine).
    *   Classification Report (Precision, Recall, F1 per classe).
3.  [ ] **FLOPs Counter**: Usare `thop` (se disponibile) o calcolo manuale per riempire la colonna "FLOPs" e "Parametri" della tabella comparativa per tutti e 4 i modelli.
