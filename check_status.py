#!/usr/bin/env python3
"""
Quick Status Check - View training progress at a glance
"""

import os
from pathlib import Path
from datetime import datetime

def check_status():
    base_dir = Path(__file__).parent
    
    print("=" * 70)
    print("TRAINING STATUS - Quick Check")
    print("=" * 70)
    print()
    
    files = [
        ('data/train_data.parquet', '✓ Training data'),
        ('data/test_data.parquet', '✓ Test data'),
        ('results/predictions/benchmark_predictions.parquet', '✓ OLS-3 predictions'),
        ('results/predictions/gbrt_predictions.parquet', '✓ GBRT predictions'),
        ('results/predictions/elastic_net_predictions.parquet', '✓ Elastic Net predictions'),
        ('results/predictions/fama_french_predictions.parquet', '✓ Fama-French predictions'),
        ('results/tables/performance_comparison.csv', '✓ Performance comparison'),
        ('results/figures/model_comparison/comparison_sharpe_ratios.png', '✓ Sharpe ratio chart'),
    ]
    
    completed = 0
    for filepath, description in files:
        full_path = base_dir / filepath
        if full_path.exists():
            size = full_path.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(full_path.stat().st_mtime)
            age = datetime.now() - mtime
            
            if age.days > 0:
                age_str = f"{age.days}d ago"
            elif age.seconds > 3600:
                age_str = f"{age.seconds//3600}h ago"
            else:
                age_str = f"{age.seconds//60}m ago"
            
            print(f"{description:40s} ({size:6.1f} MB, {age_str})")
            completed += 1
        else:
            print(f"○ {description[2:]:38s} (not found)")
    
    print()
    print(f"Progress: {completed}/{len(files)} files completed")
    print("=" * 70)

if __name__ == "__main__":
    check_status()
