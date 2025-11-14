"""
Comprehensive Model Evaluation.

This script evaluates both the OLS benchmark and GBRT models:
1. Out-of-sample R²
2. Portfolio sorts and long-short returns
3. Sharpe ratios (VW and EW)
4. Cumulative returns plots
5. Performance comparison tables

Author: Replication of Gu, Kelly, and Xiu (2020)
"""

import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    setup_logging,
    ensure_dir,
    get_project_root,
    calculate_r_squared,
    calculate_monthly_r_squared,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    create_portfolio_sorts,
    calculate_long_short_returns,
    plot_cumulative_returns,
    plot_rolling_sharpe,
    plot_portfolio_performance,
    create_performance_table
)

# Configuration
DATA_DIR = get_project_root() / "data"
RESULTS_DIR = get_project_root() / "results"

logger = setup_logging()


def load_predictions(model_name: str) -> pd.DataFrame:
    """
    Load model predictions.
    
    Parameters
    ----------
    model_name : str
        Model name ('benchmark' or 'gbrt')
    
    Returns
    -------
    pd.DataFrame
        Predictions dataframe
    """
    pred_path = RESULTS_DIR / "predictions" / f"{model_name}_predictions.parquet"
    
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {pred_path}\n"
            f"Please run the corresponding model script first."
        )
    
    pred_df = pd.read_parquet(pred_path)
    logger.info(f"Loaded {model_name} predictions: {pred_df.shape}")
    
    return pred_df


def evaluate_model(pred_df: pd.DataFrame, test_df: pd.DataFrame,
                  model_name: str) -> Dict:
    """
    Comprehensive model evaluation.
    
    Parameters
    ----------
    pred_df : pd.DataFrame
        Predictions with y_true and y_pred columns
    test_df : pd.DataFrame
        Test dataframe (for market cap weighting)
    model_name : str
        Model name for logging
    
    Returns
    -------
    dict
        Evaluation results
    """
    logger.info(f"Evaluating {model_name} model")
    logger.info("="*80)
    
    results = {}
    
    # 1. Overall OOS R²
    overall_r2 = calculate_r_squared(pred_df['y_true'].values, pred_df['y_pred'].values)
    results['overall_r2'] = overall_r2
    logger.info(f"Overall OOS R²: {overall_r2*100:.4f}%")
    
    # 2. Monthly OOS R²
    pred_reset = pred_df.reset_index()
    monthly_r2 = calculate_monthly_r_squared(pred_reset)
    results['monthly_r2_mean'] = monthly_r2.mean()
    results['monthly_r2_median'] = monthly_r2.median()
    results['monthly_r2_std'] = monthly_r2.std()
    results['monthly_r2_series'] = monthly_r2
    
    logger.info(f"Monthly OOS R²:")
    logger.info(f"  Mean: {monthly_r2.mean()*100:.4f}%")
    logger.info(f"  Median: {monthly_r2.median()*100:.4f}%")
    logger.info(f"  Std: {monthly_r2.std()*100:.4f}%")
    logger.info(f"  Min: {monthly_r2.min()*100:.4f}%")
    logger.info(f"  Max: {monthly_r2.max()*100:.4f}%")
    
    # 3. Portfolio analysis
    eval_df = pred_df.copy()
    eval_df['prediction'] = eval_df['y_pred']
    eval_df['ret_excess'] = eval_df['y_true']
    
    # Get market cap for value weighting
    mkt_cap_col = None
    for col in ['mvel1', 'me', 'market_equity', 'size']:
        if col in test_df.columns:
            mkt_cap_col = col
            eval_df[col] = test_df[col]
            break
    
    eval_df = eval_df.reset_index()
    
    # Equal-weighted portfolios
    logger.info("\nEqual-weighted portfolio analysis:")
    portfolios_ew = create_portfolio_sorts(
        eval_df,
        prediction_col='prediction',
        return_col='ret_excess',
        n_portfolios=10,
        weight_col=None
    )
    
    ls_returns_ew = calculate_long_short_returns(portfolios_ew)
    
    # EW statistics
    mean_ret_ew = ls_returns_ew['long_short'].mean() * 12 * 100  # Annualized %
    vol_ew = ls_returns_ew['long_short'].std() * np.sqrt(12) * 100
    sharpe_ew = calculate_sharpe_ratio(ls_returns_ew['long_short'].values)
    max_dd_ew = calculate_max_drawdown((1 + ls_returns_ew['long_short']).cumprod().values)
    
    results['sharpe_ew'] = sharpe_ew
    results['mean_return_ew'] = mean_ret_ew
    results['volatility_ew'] = vol_ew
    results['max_drawdown_ew'] = max_dd_ew
    results['portfolios_ew'] = portfolios_ew
    results['ls_returns_ew'] = ls_returns_ew
    
    logger.info(f"  Sharpe ratio: {sharpe_ew:.4f}")
    logger.info(f"  Ann. return: {mean_ret_ew:.2f}%")
    logger.info(f"  Ann. volatility: {vol_ew:.2f}%")
    logger.info(f"  Max drawdown: {max_dd_ew*100:.2f}%")
    
    # Value-weighted portfolios
    if mkt_cap_col:
        logger.info("\nValue-weighted portfolio analysis:")
        portfolios_vw = create_portfolio_sorts(
            eval_df,
            prediction_col='prediction',
            return_col='ret_excess',
            n_portfolios=10,
            weight_col=mkt_cap_col
        )
        
        ls_returns_vw = calculate_long_short_returns(portfolios_vw)
        
        # VW statistics
        mean_ret_vw = ls_returns_vw['long_short'].mean() * 12 * 100
        vol_vw = ls_returns_vw['long_short'].std() * np.sqrt(12) * 100
        sharpe_vw = calculate_sharpe_ratio(ls_returns_vw['long_short'].values)
        max_dd_vw = calculate_max_drawdown((1 + ls_returns_vw['long_short']).cumprod().values)
        
        results['sharpe_vw'] = sharpe_vw
        results['mean_return_vw'] = mean_ret_vw
        results['volatility_vw'] = vol_vw
        results['max_drawdown_vw'] = max_dd_vw
        results['portfolios_vw'] = portfolios_vw
        results['ls_returns_vw'] = ls_returns_vw
        
        logger.info(f"  Sharpe ratio: {sharpe_vw:.4f}")
        logger.info(f"  Ann. return: {mean_ret_vw:.2f}%")
        logger.info(f"  Ann. volatility: {vol_vw:.2f}%")
        logger.info(f"  Max drawdown: {max_dd_vw*100:.2f}%")
    
    logger.info("="*80)
    
    return results


def create_comparison_table(benchmark_results: Dict, gbrt_results: Dict,
                           save_path: Path) -> pd.DataFrame:
    """
    Create comparison table of model performance.
    
    Parameters
    ----------
    benchmark_results : dict
        Benchmark model results
    gbrt_results : dict
        GBRT model results
    save_path : Path
        Path to save table
    
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    logger.info("Creating performance comparison table")
    
    # Paper benchmarks
    paper_benchmarks = {
        'Model': ['OLS-3 (Paper)', 'GBRT (Paper)'],
        'OOS R² (%)': [0.16, 0.37],
        'Sharpe (EW)': [0.83, 2.20],
        'Sharpe (VW)': [0.61, 1.35],
    }
    
    # Our results
    our_results = {
        'Model': ['OLS-3 (Ours)', 'GBRT (Ours)'],
        'OOS R² (%)': [
            benchmark_results['monthly_r2_mean'] * 100,
            gbrt_results['monthly_r2_mean'] * 100
        ],
        'Sharpe (EW)': [
            benchmark_results['sharpe_ew'],
            gbrt_results['sharpe_ew']
        ],
        'Sharpe (VW)': [
            benchmark_results.get('sharpe_vw', np.nan),
            gbrt_results.get('sharpe_vw', np.nan)
        ],
    }
    
    # Combine
    df_paper = pd.DataFrame(paper_benchmarks)
    df_ours = pd.DataFrame(our_results)
    df_combined = pd.concat([df_paper, df_ours], ignore_index=True)
    
    # Add improvement metrics
    improvement = {
        'Model': ['Improvement (%)'],
        'OOS R² (%)': [(gbrt_results['monthly_r2_mean'] / benchmark_results['monthly_r2_mean'] - 1) * 100],
        'Sharpe (EW)': [(gbrt_results['sharpe_ew'] / benchmark_results['sharpe_ew'] - 1) * 100],
        'Sharpe (VW)': [(gbrt_results.get('sharpe_vw', 0) / benchmark_results.get('sharpe_vw', 1) - 1) * 100],
    }
    df_improvement = pd.DataFrame(improvement)
    
    df_combined = pd.concat([df_combined, df_improvement], ignore_index=True)
    
    # Save
    ensure_dir(save_path.parent)
    df_combined.to_csv(save_path.with_suffix('.csv'), index=False)
    df_combined.to_latex(save_path.with_suffix('.tex'), index=False, float_format='%.3f',
                        caption='Model Performance Comparison', label='tab:comparison')
    
    logger.info(f"Saved comparison table to {save_path}")
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(df_combined.to_string(index=False))
    print("="*80)
    
    return df_combined


def plot_model_comparison(benchmark_results: Dict, gbrt_results: Dict,
                         save_dir: Path) -> None:
    """
    Create comparison plots.
    
    Parameters
    ----------
    benchmark_results : dict
        Benchmark results
    gbrt_results : dict
        GBRT results
    save_dir : Path
        Directory to save plots
    """
    logger.info("Creating comparison plots")
    
    ensure_dir(save_dir)
    
    # 1. Cumulative returns comparison (EW)
    fig, ax = plt.subplots(figsize=(14, 7))
    
    cum_bench_ew = (1 + benchmark_results['ls_returns_ew']['long_short']).cumprod()
    cum_gbrt_ew = (1 + gbrt_results['ls_returns_ew']['long_short']).cumprod()
    
    ax.plot(cum_bench_ew.index, cum_bench_ew.values, label='OLS-3 Benchmark', 
           linewidth=2, alpha=0.8)
    ax.plot(cum_gbrt_ew.index, cum_gbrt_ew.values, label='GBRT', 
           linewidth=2, alpha=0.8)
    
    ax.set_title('Long-Short Portfolio Cumulative Returns (Equal-Weighted)', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_cumulative_returns_ew.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Monthly R² comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Benchmark
    monthly_r2_bench = benchmark_results['monthly_r2_series']
    axes[0].plot(monthly_r2_bench.index, monthly_r2_bench.values * 100, 
                linewidth=1.5, alpha=0.7)
    axes[0].axhline(y=monthly_r2_bench.mean() * 100, color='red', 
                   linestyle='--', label=f'Mean: {monthly_r2_bench.mean()*100:.3f}%')
    axes[0].set_title('OLS-3 Benchmark: Monthly Out-of-Sample R²', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('R² (%)', fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # GBRT
    monthly_r2_gbrt = gbrt_results['monthly_r2_series']
    axes[1].plot(monthly_r2_gbrt.index, monthly_r2_gbrt.values * 100, 
                linewidth=1.5, alpha=0.7, color='orange')
    axes[1].axhline(y=monthly_r2_gbrt.mean() * 100, color='red', 
                   linestyle='--', label=f'Mean: {monthly_r2_gbrt.mean()*100:.3f}%')
    axes[1].set_title('GBRT: Monthly Out-of-Sample R²', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('R² (%)', fontsize=12)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_monthly_r2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Sharpe ratio comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['OLS-3\nBenchmark', 'GBRT']
    sharpe_ew = [benchmark_results['sharpe_ew'], gbrt_results['sharpe_ew']]
    sharpe_vw = [benchmark_results.get('sharpe_vw', 0), gbrt_results.get('sharpe_vw', 0)]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sharpe_ew, width, label='Equal-Weighted', alpha=0.8)
    bars2 = ax.bar(x + width/2, sharpe_vw, width, label='Value-Weighted', alpha=0.8)
    
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Long-Short Portfolio Sharpe Ratios', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_sharpe_ratios.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison plots to {save_dir}")


def main() -> None:
    """Main evaluation pipeline."""
    logger.info("="*80)
    logger.info("Comprehensive Model Evaluation")
    logger.info("="*80)
    
    # Load test data
    logger.info("Loading test data...")
    test_df = pd.read_parquet(DATA_DIR / "test_data.parquet")
    
    # Load predictions
    try:
        benchmark_pred = load_predictions('benchmark')
    except FileNotFoundError as e:
        logger.error(f"Benchmark predictions not found: {e}")
        logger.error("Please run 02_baseline_benchmark.py first")
        return
    
    try:
        gbrt_pred = load_predictions('gbrt')
    except FileNotFoundError as e:
        logger.error(f"GBRT predictions not found: {e}")
        logger.error("Please run 03_gbrt_model.py first")
        return
    
    # Evaluate models
    benchmark_results = evaluate_model(benchmark_pred, test_df, 'OLS-3 Benchmark')
    gbrt_results = evaluate_model(gbrt_pred, test_df, 'GBRT')
    
    # Create comparison table
    comparison_table = create_comparison_table(
        benchmark_results, gbrt_results,
        RESULTS_DIR / "tables" / "performance_comparison"
    )
    
    # Create comparison plots
    plot_model_comparison(benchmark_results, gbrt_results, 
                         RESULTS_DIR / "figures")
    
    # Create individual performance tables
    returns_dict_bench = {
        'OLS-3 Long-Short (EW)': benchmark_results['ls_returns_ew']['long_short']
    }
    if 'ls_returns_vw' in benchmark_results:
        returns_dict_bench['OLS-3 Long-Short (VW)'] = benchmark_results['ls_returns_vw']['long_short']
    
    returns_dict_gbrt = {
        'GBRT Long-Short (EW)': gbrt_results['ls_returns_ew']['long_short']
    }
    if 'ls_returns_vw' in gbrt_results:
        returns_dict_gbrt['GBRT Long-Short (VW)'] = gbrt_results['ls_returns_vw']['long_short']
    
    create_performance_table(
        returns_dict_bench,
        save_path=str(RESULTS_DIR / "tables" / "benchmark_detailed_performance")
    )
    
    create_performance_table(
        returns_dict_gbrt,
        save_path=str(RESULTS_DIR / "tables" / "gbrt_detailed_performance")
    )
    
    logger.info("="*80)
    logger.info("Evaluation completed successfully!")
    logger.info("="*80)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print("\nOLS-3 Benchmark:")
    print(f"  Monthly R²: {benchmark_results['monthly_r2_mean']*100:.4f}% (Target: 0.16%)")
    print(f"  Sharpe (EW): {benchmark_results['sharpe_ew']:.4f} (Target: 0.83)")
    if 'sharpe_vw' in benchmark_results:
        print(f"  Sharpe (VW): {benchmark_results['sharpe_vw']:.4f} (Target: 0.61)")
    
    print("\nGBRT:")
    print(f"  Monthly R²: {gbrt_results['monthly_r2_mean']*100:.4f}% (Target: 0.33-0.40%)")
    print(f"  Sharpe (EW): {gbrt_results['sharpe_ew']:.4f} (Target: 2.2-2.4)")
    if 'sharpe_vw' in gbrt_results:
        print(f"  Sharpe (VW): {gbrt_results['sharpe_vw']:.4f} (Target: 1.35)")
    
    print("\nImprovement (GBRT vs OLS-3):")
    r2_improvement = (gbrt_results['monthly_r2_mean'] / benchmark_results['monthly_r2_mean'] - 1) * 100
    sharpe_improvement = (gbrt_results['sharpe_ew'] / benchmark_results['sharpe_ew'] - 1) * 100
    print(f"  R² improvement: {r2_improvement:.1f}%")
    print(f"  Sharpe (EW) improvement: {sharpe_improvement:.1f}%")
    
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  Tables: {RESULTS_DIR / 'tables'}")
    print(f"  Figures: {RESULTS_DIR / 'figures'}")
    print("="*80)


if __name__ == "__main__":
    main()
