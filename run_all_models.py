#!/usr/bin/env python3
"""
Run Complete Training Pipeline with Transaction Costs

This script runs all models in sequence with the new improvements:
- Transaction costs (5 bps)
- Reduced timeframe (1996-2016)
- Organized outputs

Usage: python run_all_models.py [--skip-data] [--skip-baseline] [--skip-gbrt]
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ANSI colors
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")

def print_step(step_num, total_steps, description):
    """Print step information."""
    print(f"\n{Colors.BOLD}{Colors.GREEN}[Step {step_num}/{total_steps}] {description}{Colors.END}")
    print(f"{Colors.YELLOW}{'─'*80}{Colors.END}\n")

def run_script(script_path, description, log_file=None):
    """Run a Python script and return success status."""
    print(f"{Colors.BOLD}Running:{Colors.END} {script_path}")
    print(f"{Colors.BOLD}Started:{Colors.END} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    try:
        if log_file:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    ['python3', script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                # Stream output
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)
                    f.flush()
                
                process.wait()
                returncode = process.returncode
        else:
            result = subprocess.run(
                ['python3', script_path],
                capture_output=False,
                text=True
            )
            returncode = result.returncode
        
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        if returncode == 0:
            print(f"\n{Colors.GREEN}✓ SUCCESS{Colors.END} - {description} completed in {minutes}m {seconds}s")
            return True
        else:
            print(f"\n{Colors.RED}✗ FAILED{Colors.END} - {description} failed with code {returncode}")
            return False
            
    except Exception as e:
        print(f"\n{Colors.RED}✗ ERROR{Colors.END} - {description} failed: {e}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if Path(filepath).exists():
        size = Path(filepath).stat().st_size / (1024 * 1024)  # MB
        print(f"{Colors.GREEN}✓{Colors.END} {description}: {filepath} ({size:.2f} MB)")
        return True
    else:
        print(f"{Colors.YELLOW}○{Colors.END} {description}: Not found")
        return False

def main():
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(description='Run complete model training pipeline')
    parser.add_argument('--skip-data', action='store_true', help='Skip data preparation')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip baseline model')
    parser.add_argument('--skip-gbrt', action='store_true', help='Skip GBRT model')
    parser.add_argument('--skip-new', action='store_true', help='Skip new models (Elastic Net, Fama-French)')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    os.chdir(base_dir)
    
    # Print header
    print_header("EMPIRICAL ASSET PRICING - FULL TRAINING PIPELINE")
    print(f"{Colors.BOLD}Configuration:{Colors.END}")
    print(f"  • Transaction costs: 5 bps (0.05%)")
    print(f"  • Test period: 1996-2016 (252 months)")
    print(f"  • Models: OLS-3, GBRT, Elastic Net, Fama-French")
    print(f"\n{Colors.BOLD}Estimated Total Time:{Colors.END} ~6 hours")
    
    # Check existing outputs
    print(f"\n{Colors.BOLD}Checking existing outputs...{Colors.END}\n")
    check_file_exists('data/train_data.parquet', 'Training data')
    check_file_exists('data/test_data.parquet', 'Test data')
    check_file_exists('results/predictions/benchmark_predictions.parquet', 'OLS-3 predictions')
    check_file_exists('results/predictions/gbrt_predictions.parquet', 'GBRT predictions')
    check_file_exists('results/predictions/elastic_net_predictions.parquet', 'Elastic Net predictions')
    check_file_exists('results/predictions/fama_french_predictions.parquet', 'Fama-French predictions')
    
    # Skip input prompt if not running in interactive terminal
    if sys.stdin.isatty():
        input(f"\n{Colors.BOLD}{Colors.YELLOW}Press Enter to start training or Ctrl+C to cancel...{Colors.END}")
    else:
        print(f"\n{Colors.BOLD}{Colors.GREEN}Starting training automatically (non-interactive mode)...{Colors.END}")
        time.sleep(2)
    
    total_steps = 6
    current_step = 0
    failed_steps = []
    
    # Step 1: Data Preparation
    if not args.skip_data:
        current_step += 1
        print_step(current_step, total_steps, "Data Preparation (1996-2016)")
        if not run_script('src/01_data_preparation.py', 'Data Preparation', 'data_prep.log'):
            failed_steps.append('Data Preparation')
    else:
        print(f"\n{Colors.YELLOW}Skipping data preparation{Colors.END}")
    
    # Step 2: OLS-3 Baseline
    if not args.skip_baseline:
        current_step += 1
        print_step(current_step, total_steps, "OLS-3 Baseline Model")
        if not run_script('src/02_baseline_benchmark.py', 'OLS-3 Baseline', 'baseline.log'):
            failed_steps.append('OLS-3 Baseline')
    else:
        print(f"\n{Colors.YELLOW}Skipping baseline model{Colors.END}")
    
    # Step 3: GBRT Model (longest step)
    if not args.skip_gbrt:
        current_step += 1
        print_step(current_step, total_steps, "GBRT Model (~5 hours)")
        print(f"{Colors.BOLD}{Colors.YELLOW}This will take approximately 5 hours...{Colors.END}")
        print(f"{Colors.BOLD}Tip:{Colors.END} Open another terminal and run: python monitor_training.py\n")
        
        if not run_script('src/03_gbrt_model.py', 'GBRT Model', 'gbrt_training.log'):
            failed_steps.append('GBRT Model')
    else:
        print(f"\n{Colors.YELLOW}Skipping GBRT model{Colors.END}")
    
    # Step 4: New Models (Elastic Net + Fama-French)
    if not args.skip_new:
        current_step += 1
        print_step(current_step, total_steps, "Elastic Net & Fama-French Models (~20 min)")
        if not run_script('src/03_train_new_models.py', 'New Models', 'new_models.log'):
            failed_steps.append('New Models')
    else:
        print(f"\n{Colors.YELLOW}Skipping new models{Colors.END}")
    
    # Step 5: Model Evaluation
    current_step += 1
    print_step(current_step, total_steps, "Model Evaluation")
    if not run_script('src/04_evaluation.py', 'Model Evaluation', 'evaluation.log'):
        failed_steps.append('Model Evaluation')
    
    # Step 6: Unified Comparison
    current_step += 1
    print_step(current_step, total_steps, "Unified Model Comparison")
    if not run_script('src/06_unified_evaluation.py', 'Unified Comparison', 'unified_eval.log'):
        failed_steps.append('Unified Comparison')
    
    # Final summary
    print_header("TRAINING PIPELINE COMPLETE")
    
    if not failed_steps:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All steps completed successfully!{Colors.END}\n")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some steps failed:{Colors.END}")
        for step in failed_steps:
            print(f"  • {step}")
        print()
    
    print(f"{Colors.BOLD}Results Location:{Colors.END}")
    print(f"  • Predictions: results/predictions/")
    print(f"  • Tables:      results/tables/")
    print(f"  • Figures:     results/figures/")
    print(f"\n{Colors.BOLD}Key Outputs:{Colors.END}")
    check_file_exists('results/tables/performance_comparison.csv', 'Performance comparison')
    check_file_exists('results/figures/model_comparison/comparison_sharpe_ratios.png', 'Sharpe ratio chart')
    
    print(f"\n{Colors.BOLD}All results include:{Colors.END}")
    print(f"  ✓ Transaction costs (5 bps)")
    print(f"  ✓ Realistic turnover tracking")
    print(f"  ✓ 1996-2016 timeframe")
    print(f"\n{Colors.GREEN}🎉 Training pipeline finished!{Colors.END}\n")

if __name__ == "__main__":
    main()
