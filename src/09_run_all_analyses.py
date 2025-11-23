"""
Run all model-specific analyses and reorganize figures.

This script:
1. Runs detailed analysis for Elastic Net
2. Runs detailed analysis for Fama-French  
3. Creates organized folder structure

Directory structure created:
results/figures/
├── elastic_net/
│   ├── ew_vs_vw_performance.png
│   ├── feature_importance.png
│   ├── monthly_returns_distribution.png
│   └── performance_summary.csv
├── fama_french/
│   ├── ew_vs_vw_performance.png
│   ├── factor_analysis.png
│   ├── monthly_returns_distribution.png
│   └── performance_summary.csv
└── model_comparison/
    ├── all_models_sharpe_comparison.png
    ├── all_models_cumulative_returns.png
    └── all_models_return_distribution.png

Author: Asset Pricing ML Project
Date: 2025-11-23
"""

import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run all analyses."""
    
    logger.info("="*80)
    logger.info("RUNNING ALL MODEL-SPECIFIC ANALYSES")
    logger.info("="*80)
    
    # 1. GBRT Analysis
    logger.info("\n1. Running GBRT detailed analysis...")
    result = subprocess.run(['python3', 'src/10_gbrt_analysis.py'], capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("✅ GBRT analysis completed")
    else:
        logger.error(f"❌ GBRT analysis failed:\n{result.stderr}")
    
    # 2. Elastic Net Analysis
    logger.info("\n2. Running Elastic Net detailed analysis...")
    result = subprocess.run(['python3', 'src/07_elastic_net_analysis.py'], capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("✅ Elastic Net analysis completed")
    else:
        logger.error(f"❌ Elastic Net analysis failed:\n{result.stderr}")
    
    # 3. Fama-French Analysis  
    logger.info("\n3. Running Fama-French detailed analysis...")
    result = subprocess.run(['python3', 'src/08_fama_french_analysis.py'], capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("✅ Fama-French analysis completed")
    else:
        logger.error(f"❌ Fama-French analysis failed:\n{result.stderr}")
    
    # 4. Unified Evaluation (already creates model_comparison folder)
    logger.info("\n4. Running unified evaluation...")
    result = subprocess.run(['python3', 'src/06_unified_evaluation.py'], capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("✅ Unified evaluation completed")
    else:
        logger.error(f"❌ Unified evaluation failed:\n{result.stderr}")
    
    # Display folder structure
    logger.info("\n" + "="*80)
    logger.info("FIGURES FOLDER STRUCTURE")
    logger.info("="*80)
    
    figures_dir = Path('results/figures')
    
    logger.info(f"\n{figures_dir}/")
    for folder in sorted(figures_dir.iterdir()):
        if folder.is_dir():
            logger.info(f"├── {folder.name}/")
            files = list(folder.glob('*'))
            for i, file in enumerate(sorted(files)):
                prefix = "│   ├──" if i < len(files) - 1 else "│   └──"
                logger.info(f"{prefix} {file.name}")
    
    logger.info("\n" + "="*80)
    logger.info("ALL ANALYSES COMPLETED!")
    logger.info("="*80)
    logger.info("\nGenerated folders:")
    logger.info("  - results/figures/gbrt/             (GBRT specific analysis)")
    logger.info("  - results/figures/elastic_net/      (Elastic Net specific analysis)")
    logger.info("  - results/figures/fama_french/      (Fama-French specific analysis)")
    logger.info("  - results/figures/model_comparison/ (Cross-model comparisons)")
    logger.info("  - results/figures/benchmarks/       (OLS-3 baseline)")
    logger.info("="*80)

if __name__ == '__main__':
    main()
