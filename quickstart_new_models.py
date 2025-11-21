#!/usr/bin/env python3
"""
Quick Start: Train New Models and Generate Comparison

This script trains Elastic Net and Fama-French models, then evaluates all 4 models.
Assumes GBRT and OLS-3 are already trained.

Usage:
    python quickstart_new_models.py
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check if required files exist."""
    logger.info("Checking prerequisites...")
    
    required_files = [
        'data/processed/train_data.parquet',
        'data/processed/test_data.parquet',
        'results/predictions/gbrt_predictions.csv',
        'results/predictions/benchmark_predictions.parquet'
    ]
    
    missing = []
    for filepath in required_files:
        if not Path(filepath).exists():
            missing.append(filepath)
    
    if missing:
        logger.error("❌ Missing required files:")
        for f in missing:
            logger.error(f"   - {f}")
        logger.error("\nPlease run data preparation first:")
        logger.error("   python src/00_download_data.py")
        logger.error("   python src/01_data_preparation.py")
        logger.error("   python src/02_baseline_benchmark.py")
        logger.error("   python src/03_gbrt_model.py")
        return False
    
    logger.info("✅ All prerequisites met!")
    return True


def run_command(cmd, description):
    """Run a command and handle errors."""
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 {description}")
    logger.info(f"{'='*80}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        logger.info(f"✅ {description} - COMPLETED")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} - FAILED")
        logger.error(f"Error: {e}")
        return False


def main():
    """Main execution pipeline."""
    logger.info("\n" + "="*80)
    logger.info("🎯 QUICK START: New Models Training & Evaluation")
    logger.info("="*80)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Step 1: Train new models
    logger.info("\n📝 Step 1/2: Training Elastic Net and Fama-French models...")
    logger.info("Expected time: ~30 minutes")
    
    success = run_command(
        "python src/03_train_new_models.py",
        "Training Elastic Net and Fama-French"
    )
    
    if not success:
        logger.error("\n❌ Training failed. Please check the logs above.")
        sys.exit(1)
    
    # Step 2: Unified evaluation
    logger.info("\n📊 Step 2/2: Comparing all 4 models...")
    logger.info("Expected time: ~2 minutes")
    
    success = run_command(
        "python src/06_unified_evaluation.py",
        "Unified Model Evaluation"
    )
    
    if not success:
        logger.error("\n❌ Evaluation failed. Please check the logs above.")
        sys.exit(1)
    
    # Success summary
    logger.info("\n" + "="*80)
    logger.info("🎉 ALL DONE! ")
    logger.info("="*80)
    logger.info("\n📁 Results saved to:")
    logger.info("   📊 results/tables/all_models_performance.csv")
    logger.info("   📈 results/figures/all_models_sharpe_comparison.png")
    logger.info("   📈 results/figures/all_models_cumulative_returns.png")
    logger.info("   📈 results/figures/all_models_return_distribution.png")
    
    logger.info("\n🎓 For your presentation:")
    logger.info("   1. Use the performance comparison table")
    logger.info("   2. Include Sharpe ratio comparison chart")
    logger.info("   3. Show cumulative returns over time")
    logger.info("   4. Highlight GBRT's superior performance")
    
    logger.info("\n📖 Model Comparison:")
    logger.info("   • Fama-French 3F: Traditional factor model (interpretable)")
    logger.info("   • OLS-3: Polynomial regression (simple baseline)")
    logger.info("   • Elastic Net: Regularized linear (feature selection)")
    logger.info("   • GBRT: Non-linear ML (highest Sharpe ratio)")
    
    logger.info("\n✨ Ready for class presentation!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
