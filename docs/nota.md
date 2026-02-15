# Riepilogo Attività Tesi

## Panoramica del Lavoro
L'obiettivo principale è stato finalizzare l'architettura **MobileNetECA Reparameterized** e generare tutti i risultati e grafici necessari per la tesi di laurea. Abbiamo ottimizzato il modello per ottenere un elevato trade-off tra accuratezza e numero di parametri su CIFAR-10.

## Modello Allenato
*   **Architettura**: MobileNetECA Reparameterized
*   **Dataset**: CIFAR-10
*   **Configurazione Training**:
    *   Epoche: 200
    *   Ottimizzatore: SGD con Momentum (0.9)
    *   Scheduler: Cosine Annealing Learning Rate
    *   Width Multiplier: 0.42
    *   Apprendimento LR Scale: 1.54
*   **Risultati Ottenuti**:
    *   **Accuratezza Migliore (Validazione)**: **92.47%** (superando la baseline del 91.44%)
    *   **Parametri (Inference)**: **76.6k** (target < 100k rispettato)
    *   **Loss Finale**: 0.0265
    *   **Efficienza**: 1.20 Acc/kParam (Stato dell'Arte per modelli < 100k params)

## Script e Codice Sorgente (`/src/scripts`)
Ecco i percorsi dei file principali utilizzati per ottenere questi risultati:

### 1. Definizione Modello
*   **`src/scripts/model.py`**: Contiene la classe `MobileNetECA` e la logica per la Reparameterization. È il cuore dell'architettura proposta.
*   **`src/legacy/MobileNetEca_Rep.py`**: Versione precedente/alternativa del modello (riferimento storico).

### 2. Training
*   **`src/scripts/train.py`**: Script principale per l'addestramento. Implementa il loop di training, la validazione, il salvataggio dei checkpoint e il logging delle metriche.
    *   Salva i modelli in: `../../models` (o percorso configurato)
    *   Salva i report in: `../../reports`

### 3. Valutazione e Metriche
*   **`src/scripts/evaluate_model_safe.py`**: Script robusto per caricare il modello addestrato e calcolare metriche dettagliate.
    *   Genera: Matrice di Confusione, Report di Classificazione.
    *   Analizza errori comuni (es. Gatto vs Cane).

### 4. Generazione Grafici per la Tesi
*   **`src/scripts/genera_grafici_tesi.py`**: Script dedicato alla creazione dei plot finali in alta qualità per il documento LaTeX.
    *   **`training_comparison.png`**: Confronto curve di apprendimento Baseline vs Reparameterized.
    *   **`accuracy_vs_params.png`**: Scatter plot efficienza rispetto allo Stato dell'Arte (SOTA).
    *   **`confusion_matrix.png`**: Visualizzazione errori di classificazione.
    *   **`roc_curves.png`**: Curve ROC One-vs-Rest.

### 5. Grid Search (Ottimizzazione)
*   **`src/scripts/grid_search_fast.py`**: Script utilizzato nelle fasi precedenti per trovare gli iperparametri ottimali (width multiplier, lr scale) usando l'early stopping per efficienza.

## Documentazione Tesi (`/docs/scrittura-tesi/tesi`)
*   **`main.tex`**: File principale LaTeX. Gestisce la struttura e l'inclusione dei capitoli.
*   **`capitolo5.tex`**: Capitolo "Risultati Sperimentali", aggiornato con i dati finali del modello e i grafici generati.
*   **`TesiDiaUniroma3.cls`**: Classe di stile per la tesi (corretto bug su `\generaIndiceTavole` -> `\generaIndiceTabelle`).

## Note Finali
Il sistema è ora configurato per compilare correttamente la tesi (`latexmk -pdf main.tex`) con tutti i riferimenti aggiornati. I grafici sono stati generati e salvati nella cartella `immagini/` (o percorso equivalente linkato nel tex).
