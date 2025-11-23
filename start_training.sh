#!/bin/bash
# Quick launch script for training with monitoring

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║         STARTING TRAINING WITH TRANSACTION COSTS (5 bps)              ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Steps to run:"
echo "  1. OLS-3 Baseline (~2 min)"
echo "  2. GBRT Model (~5 hours) "
echo "  3. Elastic Net & Fama-French (~20 min)"
echo "  4. Evaluation (~10 min)"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Run each step
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/4] OLS-3 Baseline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 src/02_baseline_benchmark.py
if [ $? -ne 0 ]; then
    echo "❌ Baseline failed!"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/4] GBRT Model (THIS WILL TAKE ~5 HOURS)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TIP: Open another terminal and run: python3 monitor_training.py"
echo ""
python3 src/03_gbrt_model.py
if [ $? -ne 0 ]; then
    echo "❌ GBRT failed!"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/4] Elastic Net & Fama-French"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 src/03_train_new_models.py
if [ $? -ne 0 ]; then
    echo "❌ New models failed!"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[4/4] Evaluation & Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 src/04_evaluation.py
python3 src/06_unified_evaluation.py

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                      ✅ TRAINING COMPLETE!                            ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to:"
echo "  • results/predictions/"
echo "  • results/tables/"
echo "  • results/figures/"
echo ""
echo "All results now include transaction costs (5 bps)!"
