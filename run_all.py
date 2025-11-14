"""
Master Pipeline Orchestration Script.

This script runs the complete empirical asset pricing pipeline:
1. Download data
2. Prepare data
3. Train baseline benchmark
4. Train GBRT model
5. Evaluate models
6. Analyze feature importance

Usage:
    python run_all.py [--skip-download] [--skip-benchmark] [--skip-gbrt]

Author: Replication of Gu, Kelly, and Xiu (2020)
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from utils import setup_logging

logger = setup_logging()


def run_script(script_name: str, description: str) -> bool:
    """
    Run a Python script and handle errors.
    
    Parameters
    ----------
    script_name : str
        Name of script to run (e.g., '00_download_data.py')
    description : str
        Description of what the script does
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    logger.info("="*80)
    logger.info(f"STEP: {description}")
    logger.info(f"Running: {script_name}")
    logger.info("="*80)
    
    script_path = Path(__file__).parent / "src" / script_name
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False
    
    start_time = time.time()
    
    try:
        # Run script as subprocess
        import subprocess
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
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
  # Run full pipeline
  python run_all.py
  
  # Skip data download (if already downloaded)
  python run_all.py --skip-download
  
  # Run only evaluation (if models already trained)
  python run_all.py --skip-download --skip-benchmark --skip-gbrt
        """
    )
    
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip data download step')
    parser.add_argument('--skip-benchmark', action='store_true',
                       help='Skip OLS benchmark training')
    parser.add_argument('--skip-gbrt', action='store_true',
                       help='Skip GBRT model training')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip model evaluation')
    parser.add_argument('--skip-importance', action='store_true',
                       help='Skip feature importance analysis')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("EMPIRICAL ASSET PRICING VIA MACHINE LEARNING")
    logger.info("Replication of Gu, Kelly, and Xiu (2020)")
    logger.info("="*80)
    
    pipeline_start = time.time()
    
    # Define pipeline steps
    steps = []
    
    if not args.skip_download:
        steps.append(('00_download_data.py', 'Download Dataset'))
    
    steps.append(('01_data_preparation.py', 'Prepare Data'))
    
    if not args.skip_benchmark:
        steps.append(('02_baseline_benchmark.py', 'Train OLS Benchmark'))
    
    if not args.skip_gbrt:
        steps.append(('03_gbrt_model.py', 'Train GBRT Model'))
    
    if not args.skip_evaluation:
        steps.append(('04_evaluation.py', 'Evaluate Models'))
    
    if not args.skip_importance:
        steps.append(('05_feature_importance.py', 'Analyze Feature Importance'))
    
    # Execute pipeline
    logger.info(f"\nPipeline consists of {len(steps)} steps:")
    for i, (script, desc) in enumerate(steps, 1):
        logger.info(f"  {i}. {desc}")
    logger.info("")
    
    results = []
    for script, description in steps:
        success = run_script(script, description)
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
        logger.info(f"  - Tables: results/tables/")
        logger.info(f"  - Figures: results/figures/")
        logger.info(f"  - Predictions: results/predictions/")
        logger.info(f"  - Models: results/models/")
        logger.info("\nNext steps:")
        logger.info("  1. Review results/tables/performance_comparison.csv")
        logger.info("  2. Check figures in results/figures/")
        logger.info("  3. Explore notebooks/exploration.ipynb for interactive analysis")
    else:
        logger.error("\n❌ PIPELINE FAILED")
        logger.error("Please review the logs and fix any errors before retrying")
        sys.exit(1)


if __name__ == "__main__":
    main()
