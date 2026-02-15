#!/bin/bash
# Script per configurare Antigravity ad ogni avvio del pod
# Salva TUTTO lo stato di Antigravity (logs, brain, knowledge, ecc.)

PERSISTENT_DIR="/workspace/tesi-laurea/antigravity-data"

echo "🔧 Configurazione Antigravity..."

# 1. Se la directory persistente non esiste (prima esecuzione), inizializzala
if [ ! -d "$PERSISTENT_DIR" ]; then
    echo "Inizializzazione storage persistente in $PERSISTENT_DIR..."
    mkdir -p "$PERSISTENT_DIR"
    
    # Copia i log delle conversazioni (CRUCIALE per la cronologia)
    if [ -d "/root/.gemini/antigravity/conversations" ]; then
        cp -r /root/.gemini/antigravity/conversations "$PERSISTENT_DIR/"
    fi
    
    # Gestione brain (media files)
    # Se brain è già un symlink alla vecchia location, copiamo da lì
    if [ -L "/root/.gemini/antigravity/brain" ]; then
        if [ -d "/workspace/tesi-laurea/antigravity-conversations" ]; then
            echo "Migrazione vecchi media..."
            cp -r /workspace/tesi-laurea/antigravity-conversations "$PERSISTENT_DIR/brain"
        fi
    elif [ -d "/root/.gemini/antigravity/brain" ]; then
        cp -r /root/.gemini/antigravity/brain "$PERSISTENT_DIR/"
    fi

    # Copia altre directory importanti se esistono
    for dir in knowledge annotations code_tracker context_state implicit; do
        if [ -d "/root/.gemini/antigravity/$dir" ]; then
            cp -r "/root/.gemini/antigravity/$dir" "$PERSISTENT_DIR/"
        fi
    done
    
    # Copia file sparsi
    cp /root/.gemini/antigravity/*.json "$PERSISTENT_DIR/" 2>/dev/null || true
    cp /root/.gemini/antigravity/installation_id "$PERSISTENT_DIR/" 2>/dev/null || true
fi

# 2. Configura il symlink all'avvio
# Se /root/.gemini/antigravity esiste ma NON è un symlink alla nostra dir persistente
if [ ! -L "/root/.gemini/antigravity" ] || [ "$(readlink /root/.gemini/antigravity)" != "$PERSISTENT_DIR" ]; then
    echo "Collegamento alla memoria persistente..."
    
    # Rimuovi la directory di default (o il vecchio symlink errato) e sostituiscila
    rm -rf /root/.gemini/antigravity
    ln -s "$PERSISTENT_DIR" /root/.gemini/antigravity
fi

echo "✅ Antigravity configurato! Tutto lo stato (inclusi i log) è ora persistente in $PERSISTENT_DIR"
