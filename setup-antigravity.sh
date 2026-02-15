#!/bin/bash
# Script per configurare Antigravity ad ogni avvio del pod
# Deve essere eseguito all'avvio per reindirizzare le conversazioni su /workspace

echo "🔧 Configurazione Antigravity..."

# Crea la directory delle conversazioni se non esiste
mkdir -p /workspace/tesi-laurea/antigravity-conversations

# Rimuovi la directory di default se esiste
if [ -d "/root/.gemini/antigravity/brain" ] && [ ! -L "/root/.gemini/antigravity/brain" ]; then
    echo "Rimuovo la directory di default..."
    rm -rf /root/.gemini/antigravity/brain
fi

# Crea la struttura delle directory se necessario
mkdir -p /root/.gemini/antigravity

# Crea il symlink
if [ ! -L "/root/.gemini/antigravity/brain" ]; then
    echo "Creo symlink per le conversazioni..."
    ln -s /workspace/tesi-laurea/antigravity-conversations /root/.gemini/antigravity/brain
fi

echo "✅ Antigravity configurato! Le conversazioni verranno salvate in /workspace/tesi-laurea/antigravity-conversations"
