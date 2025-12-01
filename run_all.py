"""
Master Pipeline Orchestration Script.

This script runs the complete empirical asset pricing pipeline:
1. Download raw data
2. Add synthetic returns
3. Prepare data (preprocessing, train/test split)
4. Train OLS-3 benchmark
5. Train GBRT model
6. Train Elastic Net & Fama-French models
7. Evaluate individual models
8. Unified evaluation (compare all 4 models)
9. Analyze feature importance
10. Generate analysis figures for each model

Usage:
    python run_all.py [--skip-download] [--skip-synthetic] [--skip-training] [--skip-figures]

Author: Replication of Gu, Kelly, and Xiu (2020)
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path at the beginning to prioritize our utils
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils import setup_logging

logger = setup_logging()

# Project directories
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_script(script_path: Path, description: str) -> bool:
    """
    Run a Python script and handle errors.
    
    Parameters
    ----------
    script_path : Path
        Full path to script
    description : str
        Description of what the script does
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    logger.info("="*80)
    logger.info(f"STEP: {description}")
    logger.info(f"Running: {script_path.name}")
    logger.info("="*80)
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False
    
    start_time = time.time()
    
    try:
        # Run script as subprocess from project root
        import subprocess
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✓ {description} completed successfully in {elapsed/60:.1f} minutes")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ {description} failed after {elapsed/60:.1f} minutes")
        logger.error(f"Error: {e}")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ Unexpected error after {elapsed/60:.1f} minutes: {e}")
        return False


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description='Run the complete empirical asset pricing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline from scratch
  python run_all.py
  
  # Skip data download (if already downloaded)
  python run_all.py --skip-download
  
  # Skip synthetic return generation (use existing)
  python run_all.py --skip-download --skip-synthetic
  
  # Skip all training (only run evaluation & figures)
  python run_all.py --skip-download --skip-synthetic --skip-training
  
  # Only regenerate figures (models already trained)
  python run_all.py --skip-download --skip-synthetic --skip-training --skip-evaluation
        """
    )
    
    # Data preparation flags
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip raw data download step')
    parser.add_argument('--skip-synthetic', action='store_true',
                       help='Skip synthetic returns generation')
    parser.add_argument('--skip-preparation', action='store_true',
                       help='Skip data preparation step')
    
    # Training flags
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip all model training')
    parser.add_argument('--skip-benchmark', action='store_true',
                       help='Skip OLS benchmark training')
    parser.add_argument('--skip-gbrt', action='store_true',
                       help='Skip GBRT model training')
    parser.add_argument('--skip-new-models', action='store_true',
                       help='Skip Elastic Net & Fama-French training')
    
    # Evaluation flags
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip model evaluation')
    parser.add_argument('--skip-importance', action='store_true',
                       help='Skip feature importance analysis')
    
    # Figure generation flags
    parser.add_argument('--skip-figures', action='store_true',
                       help='Skip figure generation')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("EMPIRICAL ASSET PRICING VIA MACHINE LEARNING")
    logger.info("Replication of Gu, Kelly, and Xiu (2020)")
    logger.info("="*80)
    
    pipeline_start = time.time()
    
    # Define pipeline steps: (script_path, description)
    steps = []
    
    # ========== PHASE 1: DATA PREPARATION ==========
    logger.info("\n📊 PHASE 1: DATA PREPARATION")
    
    if not args.skip_download:
        steps.append((SRC_DIR / '00_download_data.py', 'Download Raw Dataset'))
    
    if not args.skip_synthetic:
        steps.append((SCRIPTS_DIR / 'add_synthetic_returns.py', 'Generate Synthetic Returns'))
    
    if not args.skip_preparation:
        steps.append((SRC_DIR / '01_data_preparation.py', 'Prepare Data (preprocess, split)'))
    
    # ========== PHASE 2: MODEL TRAINING ==========
    if not args.skip_training:
        logger.info("\n🤖 PHASE 2: MODEL TRAINING")
        
        if not args.skip_benchmark:
            steps.append((SRC_DIR / '02_baseline_benchmark.py', 'Train OLS-3 Benchmark'))
        
        if not args.skip_gbrt:
            steps.append((SRC_DIR / '03_gbrt_model.py', 'Train GBRT Model'))
        
        if not args.skip_new_models:
            steps.append((SRC_DIR / '03_train_new_models.py', 'Train Elastic Net & Fama-French'))
    
    # ========== PHASE 3: MODEL EVALUATION ==========
    if not args.skip_evaluation:
        logger.info("\n📈 PHASE 3: MODEL EVALUATION")
        
        steps.append((SRC_DIR / '04_evaluation.py', 'Evaluate GBRT Model'))
        steps.append((SRC_DIR / '06_unified_evaluation.py', 'Compare All 4 Models'))
        
        if not args.skip_importance:
            steps.append((SRC_DIR / '05_feature_importance.py', 'Analyze Feature Importance'))
    
    # ========== PHASE 4: FIGURE GENERATION ==========
    if not args.skip_figures:
        logger.info("\n🎨 PHASE 4: FIGURE GENERATION")
        
        steps.append((SCRIPTS_DIR / 'analyze_gbrt.py', 'Generate GBRT Analysis Figures'))
        steps.append((SCRIPTS_DIR / 'analyze_elastic_net.py', 'Generate Elastic Net Figures'))
        steps.append((SCRIPTS_DIR / 'analyze_fama_french.py', 'Generate Fama-French Figures'))
        steps.append((SCRIPTS_DIR / 'visualize_architecture.py', 'Generate Architecture Diagram'))
    
    # Execute pipeline
    logger.info("\n" + "="*80)
    logger.info(f"PIPELINE: {len(steps)} steps to execute")
    logger.info("="*80)
    for i, (script, desc) in enumerate(steps, 1):
        logger.info(f"  {i:2d}. {desc}")
    logger.info("")
    
    results = []
    for script_path, description in steps:
        success = run_script(script_path, description)
        results.append((description, success))
        
        if not success:
            logger.error(f"\nPipeline stopped due to error in: {description}")
            logger.error("Please check the logs above for details")
            sys.exit(1)
        
        logger.info("")  # Blank line between steps
    
    # Final summary
    total_time = time.time() - pipeline_start
    
    logger.info("="*80)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*80)
    
    for description, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{status}: {description}")
    
    logger.info("="*80)
    logger.info(f"Total pipeline time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    logger.info("="*80)
    
    if all(success for _, success in results):
        logger.info("\n🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        logger.info("\nResults available in:")
        logger.info("  📁 Tables:      results/tables/")
        logger.info("  📁 Figures:     results/figures/")
        logger.info("  📁 Predictions: results/predictions/")
        logger.info("  📁 Models:      results/models/")
        logger.info("\nKey outputs:")
        logger.info("  📊 results/tables/all_models_performance.csv")
        logger.info("  📈 results/figures/model_comparison/")
        logger.info("  📈 results/figures/gbrt/")
        logger.info("  📈 results/figures/elastic_net/")
        logger.info("  📈 results/figures/fama_french/")
    else:
        logger.error("\n❌ PIPELINE FAILED")
        logger.error("Please review the logs and fix any errors before retrying")
        sys.exit(1)


if __name__ == "__main__":
    main()
