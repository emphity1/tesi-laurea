#!/bin/bash
# ============================================================
# Avvia il training di tutti i modelli baseline SOTA su CIFAR-10.
# Ogni modello usa lo stesso identico protocollo della tesi:
#   200 epoche, SGD(lr=0.05, momentum=0.9), WD=5e-4, CosineAnnealing
#
# I risultati vengono salvati in src/baselines/results_*/
#
# Uso:
#   bash run_all_baselines.sh          # Tutti i modelli
#   bash run_all_baselines.sh mobilev2 # Solo MobileNetV2
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

run_mobilenetv2() {
    echo "============================================"
    echo "  Training MobileNetV2 (0.5x)"
    echo "============================================"
    python3 train_mobilenetv2.py --width 0.5

    echo ""
    echo "============================================"
    echo "  Training MobileNetV2 (1.0x)"
    echo "============================================"
    python3 train_mobilenetv2.py --width 1.0
}

run_shufflenetv2() {
    echo ""
    echo "============================================"
    echo "  Training ShuffleNetV2 (0.5x)"
    echo "============================================"
    python3 train_shufflenetv2.py
}

run_repvgg() {
    echo ""
    echo "============================================"
    echo "  Training RepVGG-A0"
    echo "============================================"
    python3 train_repvgg.py
}

if [ "$1" = "mobilev2" ]; then
    run_mobilenetv2
elif [ "$1" = "shuffle" ]; then
    run_shufflenetv2
elif [ "$1" = "repvgg" ]; then
    run_repvgg
else
    echo "=== Avvio training completo baselines SOTA ==="
    echo "Protocollo: 200 epoche, SGD+Momentum, CosineAnnealing"
    echo ""
    run_mobilenetv2
    run_shufflenetv2
    run_repvgg
    echo ""
    echo "============================================"
    echo "  TUTTI I BASELINE COMPLETATI!"
    echo "  Risultati in: $SCRIPT_DIR/results_*/"
    echo "============================================"
fi
