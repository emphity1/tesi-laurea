# Bozza Struttura Tesi (Target: 70+ Pagine)

## Titolo Provvisorio
**Ottimizzazione di Architetture Neurali Compatte con Meccanismi di Attenzione e Riparametrizzazione Strutturale per il Deep Learning su Edge**

---

## 1. Introduzione (8-10 Pagine)
*   **Contesto**: L'esplosione del Deep Learning e la necessità di portare l'IA su dispositivi mobili/IoT (Edge Computing).
*   **Il Problema**: Le reti SOTA (ResNet, DenseNet, ViT) sono troppo pesanti. Le reti leggere (MobileNet) perdono accuratezza.
*   **Obiettivo della Tesi**: Progettare un'architettura che bilanci perfettamente i parametri (<100k) con l'accuratezza (>92%) su CIFAR-10.
*   **Contributi**:
    1.  Integrazione di ECA (Efficient Channel Attention) in MobileNetV2.
    2.  Sostituzione di ReLU con GELU per gradienti più stabili in reti piccole.
    3.  Adozione della "Structural Reparameterization" (RepVGG style) per migliorare il training senza costi di inferenza.
    4.  Analisi critica di tecniche fallimentari (GhostNet, Knowledge Distillation) per questo specifico task.

## 2. Stato dell'Arte e Fondamenti (15-20 Pagine)
*   **CNN per Image Classification**: Dal Perceptron a ResNet.
*   **Architetture Efficienti**:
    *   **MobileNetV1/V2/V3**: Depthwise Separable Convolutions, Inverted Residuals.
    *   **ShuffleNet**: Channel cleaning.
    *   **GhostNet**: Feature redundancy.
*   **Meccanismi di Attenzione**:
    *   SE-Block (Squeeze-and-Excitation).
    *   CBAM.
    *   **ECA (Efficient Channel Attention)**: Focus su perché è migliore di SE per noi (1D conv vs FC layers = meno parametri).
*   **Tecniche Avanzate di Training**:
    *   **Reparameterization**: Spiegare matematicamente come $3 \times 3 + 1 \times 1 \to 3 \times 3$ finale.
    *   **Knowledge Distillation**: Teacher-Student paradigm.
    *   **Attivazioni**: ReLU vs GELU vs Swish.

## 3. Metodologia Proposta: MobileNetECA-Rep (15-20 Pagine)
*   **Evoluzione dell'Architettura (Il "Viaggio")**:
    *   *Step 1*: Baseline MobileNetV2 (Vanilla).
    *   *Step 2*: Integrazione ECA (Perché? Per recuperare capacità rappresentativa persa riducendo i canali).
    *   *Step 3*: Scelta di GELU (Per evitare "dead neurons" comuni in reti piccole con ReLU).
    *   *Step 4*: Reparameterization (Perché? Per avere i benefici di una multi-branch in training e la velocità di una single-branch in inferenza).
*   **Dettagli Implementativi**:
    *   Dataset CIFAR-10 e Data Augmentation (scelte fatte: Padding, Crop, Flip - no Cutout/Mixup per mantenere semplicità).
    *   Iperparametri: Scelta via Grid Search (citare Capitolo 4).

## 4. Esperimenti e Risultati (15-20 Pagine)
*   **Setup Sperimentale**: Hardware, PyTorch, metriche.
*   **Ablation Studies (Da fare/inserire)**:
    *   *Exp 1: Baseline vs Proposta*. MobileNetV2-ReLU vs MobileNetECA-GELU. (Serve per dire "ECA+GELU porta +X%").
    *   *Exp 2: Reparameterization*. Training con Rep vs Training senza Rep (ma stessa architettura inferenza). Dimostrare che Rep aiuta la convergenza.
*   **Confronto con SOTA**: Tabella con ResNet-20, MobileNetV2 originale. Mostrare che vinciamo in Efficienza (Acc/Params).
*   **Approcci "Scartati" (Analisi dei fallimenti - Molto importante per tesi metodologica)**:
    *   **Knowledge Distillation (KD)**: "Abbiamo provato a distillare da ResNet-18, MA..." (es. guadagno marginale non giustificava complessità training, oppure sbilanciamento capacità).
    *   **Ghost Modules**: "Abbiamo provato GhostNet per ridurre parametri, MA..." (es. perdita accuratezza eccessiva per il nostro target <100k).
*   **Metriche Avanzate**: Confusion Matrix, Curve ROC, F1 per classe (già fatti).

## 5. Conclusioni e Sviluppi Futuri (3-5 Pagine)
*   Sintesi.
*   Possibili sviluppi: Quantizzazione INT8, Deploy su Raspberry Pi vero, Test su CIFAR-100.

---

# Piano d'Azione Pratico (Cosa fare ora)
Per riempire le sezioni di "Ablation Study" e giustificare le scelte metodologiche, servono dati comparativi. Non basta dire "GELU è meglio", bisogna mostrarlo.

**Esperimenti da lanciare (50 epoche bastano per vedere il trend):**
1.  **Baseline Pura**: MobileNetV2 (stessa width 0.5) con **ReLU** e **senza ECA**. -> Questo ci dà il punto di partenza (es. 90.5%).
2.  **Solo ECA**: MobileNetV2 + ECA + ReLU. -> Vediamo quanto dà ECA (es. 91.2%).
3.  **Solo GELU**: MobileNetV2 + GELU (no ECA). -> Vediamo quanto dà GELU (es. 91.0%).
4.  **(Opzionale)** KD e Ghost: Se hai già dei log vecchi usiamo quelli, altrimenti lanciare una run rapida da 50 epoche per avere un numero da mettere in tabella "Tentativi Falliti".

In questo modo la tesi diventa: "Siamo partiti da A, abbiamo aggiunto B e C ottenendo D. Abbiamo provato E (KD) ma non conveniva. D è il risultato finale." -> Metodologia scientifica inattaccabile.
