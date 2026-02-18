Criticità tecniche
Il confronto con lo Stato dell'Arte (Tabella 6.2) ha un problema strutturale: si confrontano modelli addestrati su CIFAR-10 nativo (32×32) con modelli come VGG-16 o ResNet che sono stati progettati per ImageNet e adattati. Non è un confronto alla pari in termini di ottimizzazione del setup. Sarebbe stato più onesto confrontarsi solo con modelli specificamente progettati per CIFAR-10 a basso parametro, categoria dove la letteratura è ricca.
Il claim su ThriftyNet in nota [11] è problematico: la referenza citata ("arXiv:2102.04353") non corrisponde a ThriftyNet. Si tratta di un errore bibliografico potenzialmente serio — in sede di discussione un commissario attento lo noterebbe.
La latenza riportata (19.38 ms con σ = 28.5 ms) è curiosa: una deviazione standard maggiore della media su 1000 iterazioni indica una distribuzione fortemente asimmetrica (presumibilmente per il JIT warmup o per scheduling CPU). Questo andava discusso, non presentato come dato neutro.

Criticità metodologiche
La Grid Search degli iperparametri (Figura 6.8) mostra valori massimi intorno al 91.4%, mentre il modello finale raggiunge il 93.5%. Questo significa che la grid search è stata fatta su una configurazione diversa da quella finale (probabilmente senza augmentation avanzata), ma questo non viene chiarito esplicitamente nel testo, creando confusione.
Manca un validation set separato dal test set. Il "best model" viene selezionato sul test set attraverso le epoche (best_acc aggiornato ogni epoca sul test), il che introduce una forma di data leakage — il test set viene usato come criteri di selezione del modello. Per un lavoro rigoroso si dovrebbe usare un validation set separato per la selezione e il test set solo per la valutazione finale.

Criticità formali
Ci sono riferimenti incrociati non risolti nel testo: Tabella ?? compare due volte nel Capitolo 6 (sezioni 6.4 e 6.5), il che indica che il documento LaTeX non è stato compilato correttamente o ci sono label mancanti. Questo non dovrebbe essere presente nella versione finale.
Nel Capitolo 2 compare Capitolo ?? in un riferimento interno. Stesso problema.
Il claim che la ri-parametrizzazione "riduce la latenza del 30-40% su CPU" nel Capitolo 5 non ha una fonte citata né una misurazione propria a supporto — è un numero che fluttua nel testo senza ancoraggio.

Punti di debolezza minori
La sezione sulla complessità teorica confronta i FLOPs del modello in deploy (kernel fusi) con i FLOPs di MobileNetV2 in training. Sarebbe più corretto confrontare le stesse modalità operative.
La discussione sulle "Occlusion Sensitivity Maps" è qualitativa e si basa su 5 esempi scelti evidentemente favorevoli. Non c'è una quantificazione sistematica.