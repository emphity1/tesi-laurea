#!/bin/bash
# Quick test della fast grid search - solo 4 combinazioni per verificare funzionamento

echo "=================================="
echo "QUICK TEST - Fast Grid Search"
echo "=================================="
echo ""
echo "Questo test veloce verifica che lo script funzioni correttamente"
echo "Testa solo 4 combinazioni (2 lr × 2 width)"
echo "Tempo stimato: ~20-25 minuti"
echo ""

cd /workspace/tesi-laurea/src/scripts

python grid_search_fast.py \
    --lr 0.025 0.05 \
    --width_mult 0.42 0.5 \
    --weight_decay 0.0005 \
    --checkpoint_epoch 15 \
    --min_threshold 82.0 \
    --epochs 20 \
    --output_dir "../../reports/grid_search_fast_TEST" \
    --num_workers 2

echo ""
echo "=================================="
echo "Test completato!"
echo "Controlla i risultati in: /workspace/tesi-laurea/reports/grid_search_fast_TEST/"
echo "=================================="
