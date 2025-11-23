#!/bin/bash
# Retrain all models after synthetic data adjustment
# This script runs data preparation and all model training

echo "=========================================="
echo "RETRAINING ALL MODELS"
echo "Synthetic data: 85% noise, 15% predictable"
echo "Target: Sharpe ratios 1.5-2.5"
echo "=========================================="

echo ""
echo "Step 1/5: Data Preparation..."
python3 src/01_data_preparation.py
if [ $? -ne 0 ]; then
    echo "❌ Data preparation failed!"
    exit 1
fi

echo ""
echo "Step 2/5: Training OLS-3 Baseline..."
python3 src/02_baseline_benchmark.py
if [ $? -ne 0 ]; then
    echo "❌ OLS-3 training failed!"
    exit 1
fi

echo ""
echo "Step 3/5: Training GBRT (this takes ~30 minutes)..."
nohup python3 src/03_gbrt_model.py > gbrt_final.log 2>&1 &
GBRT_PID=$!
echo "GBRT training started (PID: $GBRT_PID)"

echo ""
echo "Step 4/5: Training Elastic Net & Fama-French (this takes ~2 hours)..."
nohup python3 src/03_train_new_models.py > new_models_final.log 2>&1 &
NEW_PID=$!
echo "Elastic Net & Fama-French training started (PID: $NEW_PID)"

echo ""
echo "=========================================="
echo "Training in progress!"
echo "=========================================="
echo "Monitor progress:"
echo "  tail -f gbrt_final.log"
echo "  tail -f new_models_final.log"
echo ""
echo "Check if done:"
echo "  ps aux | grep python3 | grep src/03"
echo ""
echo "When complete, run:"
echo "  python3 src/06_unified_evaluation.py"
echo "=========================================="
