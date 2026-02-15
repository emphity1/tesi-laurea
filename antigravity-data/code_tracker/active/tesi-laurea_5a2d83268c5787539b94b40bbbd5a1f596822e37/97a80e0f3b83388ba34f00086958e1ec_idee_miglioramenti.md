�# Idee per Miglioramenti e Prossimi Passi

Basandoci sugli ottimi risultati attuali (MobileNetECA Reparameterized su CIFAR-10), ecco cosa potremmo aggiungere per arricchire la tesi, concentrandoci su analisi qualitative e diagrammi esplicativi.

## 1. Analisi Qualitativa degli Errori (Impatto Visivo Alto)
*   **Cosa fare**: Inserire una figura con una griglia 3x3 o 4x4 di immagini che il modello ha predetto in modo errato (es. un cane classificato come gatto, un aereo scuro confuso con un uccello).
*   **Perché**: Mostrare *perché* il modello sbaglia rende la discussione molto più concreta e meno astratta. Spesso gli errori sono su immagini ambigue o difficili anche per l'occhio umano, giustificando la non perfezione del modello.
*   **Azione**: Usare lo script di valutazione per salvare un collage di "Top Losses" o errori comuni.

## 2. Approfondimento Ablation Study (Componenti Chiave)
È fondamentale discutere il contributo dei singoli componenti per giustificare l'architettura scelta.
*   **Impatto ECA (Efficient Channel Attention)**:
    *   Discutere (o mostrare dati se presenti) quanto guadagna l'accuratezza inserendo *solo* il modulo ECA rispetto a una MobileNet base.
    *   Punto chiave: ECA porta benefici significativi (+1-2%) con un costo computazionale quasi nullo (pochi parametri in più).
*   **Impatto Reparameterization**:
    *   Confrontare il training standard vs training con Reparameterization.
    *   Spiegare che la struttura multi-branch (Rep) facilita la convergenza e l'esplorazione dello spazio dei gradienti durante il training, ma "sparisce" in inferenza, lasciando il modello leggero.
*   **Grafico Suggerito**: Un **Bar Plot incrementale** (o tabella) che mostra:
    1.  Baseline (MobileNetV2 ridotta)
    2.  + ECA (MobileNetECA base)
    3.  + Reparameterization (Il nostro modello finale: MobileNetECA Rep)

## 3. Nuovi Diagrammi Utili
Oltre ai grafici delle performance, mancano diagrammi che spieghino il "come" funziona il modello.
*   **Schema "Training vs Inference" (Cruciale)**:
    *   Un diagramma che mostra visivamente il blocco durante il *Training* (3 rami: Conv 3x3, Conv 1x1, Identità) e come vengono fusi matematicamente nel singolo ramo durante l'*Inference* (Conv 3x3 unificata).
    *   Questo chiarisce il concetto di "Reparameterization" nel Capitolo 3 meglio di mille parole.
*   **Curva del Learning Rate (Cosine Annealing)**:
    *   Inserire il grafico dell'andamento del Learning Rate durante le 200 epoche.
    *   Mostra visivamente come il LR decresce e (se usato con restart) risale, spiegando i miglioramenti costanti anche nelle fasi finali del training.
*   **Feature Importance (Grad-CAM - Opzionale)**:
    *   Una heatmap su un'immagine di test per vedere *dove* il modello sta guardando grazie all'ECA (es. si concentra sul muso del cane e non sull'erba).

## 4. Pulizia e Rilascio
*   Organizzare `src/legacy` e pulire gli script di test.
*   Assicurarsi che il repository sia pronto per essere "allegato" alla tesi o caricato su GitHub.
�"(5a2d83268c5787539b94b40bbbd5a1f596822e3728file:///workspace/tesi-laurea/docs/idee_miglioramenti.md:file:///workspace/tesi-laurea