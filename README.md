# Struttura del Progetto Tesi

Questo documento spiega la riorganizzazione dei file del progetto.

## Directory Structure

*   **/src**: Contiene tutto il codice sorgente.
    *   **/src/scripts**: Script Python principali trovati nella root (es. `depetto.py`, `mobilenet_scaler.py`).
    *   **/src/experiments**: Codice relativo agli esperimenti di Neural Architecture Search (ex `31_codes`).
    *   **/src/legacy**: Codice vecchio o iterazioni precedenti (ex `Saved_codes`).
    *   **/src/sn31-v2**: Repository clonato/codice condiviso (probabilmente relativo al progetto NAS).
*   **/models**: Contiene i checkpoint dei modelli salvati (`.pt` files).
*   **/docs**: Documentazione, appunti e guide (PDF, MD, TXT).
*   **/external**: Materiale esterno o corsi (ex `Allenamento`).
*   **/mutex-wandb**: Cartella esclusa dalla riorganizzazione (WandB logs?).

## Note
*   I file originali non sono stati cancellati o modificati nel contenuto, solo spostati per ordine logico.
*   Per eseguire gli script, potresti dover aggiornare i percorsi (paths) se facevano riferimento a file nella stessa cartella che ora sono altrove (es. caricamento modelli).
