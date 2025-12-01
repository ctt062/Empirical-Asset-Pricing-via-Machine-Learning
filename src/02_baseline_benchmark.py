"""
Baseline OLS Benchmark Model.

This script implements the 3-factor OLS benchmark from Gu et al. (2020):
- Size (log market cap)
- Book-to-Market ratio
- 12-month momentum

This serves as the baseline for comparing machine learning models.

Expected performance (from paper):
- Monthly OOS R² ≈ 0.16%
- Long-short decile Sharpe ≈ 0.61 (VW) / 0.83 (EW)

Author: Replication of Gu, Kelly, and Xiu (2020)
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    setup_logging,
    ensure_dir,
    get_project_root,
    calculate_r_squared,
    calculate_monthly_r_squared,
    calculate_sharpe_ratio,
    create_portfolio_sorts,
    calculate_long_short_returns,
    plot_cumulative_returns,
    create_performance_table,
    RANDOM_STATE
)

# Import configuration
from config import DATA_PROCESSED_DIR, RESULTS_DIR

logger = setup_logging()


def identify_benchmark_features(feature_cols: list) -> Dict[str, str]:
    """
    Identify the 3 benchmark features in the dataset.
    
    Looking for:
    - Size: mvel1, me, size, log_me, etc.
    - Book-to-Market: bm, bm_ia, book_to_market, etc.
    - Momentum: mom12m, mom1m, ret_12_1, etc.
    
    Parameters
    ----------
    feature_cols : list
        List of available feature columns
    
    Returns
    -------
    dict
        Dictionary mapping 'size', 'bm', 'momentum' to actual column names
    """
    logger.info("Identifying benchmark features")
    
    benchmark_features = {}
    
    # Size (market equity)
    size_candidates = ['mvel1', 'me', 'size', 'log_me', 'market_equity']
    for candidate in size_candidates:
        if candidate in feature_cols:
            benchmark_features['size'] = candidate
            break
    
    # Book-to-Market
    bm_candidates = ['bm', 'bm_ia', 'book_to_market', 'btm']
    for candidate in bm_candidates:
        if candidate in feature_cols:
            benchmark_features['bm'] = candidate
            break
    
    # Momentum (12-month)
    mom_candidates = ['mom12m', 'mom1m', 'ret_12_1', 'momentum', 'mom']
    for candidate in mom_candidates:
        if candidate in feature_cols:
            benchmark_features['momentum'] = candidate
            break
    
    # Log what was found
    logger.info(f"Benchmark features identified:")
    for key, val in benchmark_features.items():
        logger.info(f"  {key}: {val}")
    
    # Check if all features found
    if len(benchmark_features) < 3:
        missing = [k for k in ['size', 'bm', 'momentum'] if k not in benchmark_features]
        logger.warning(f"Missing benchmark features: {missing}")
        logger.warning(f"Available features (first 50): {feature_cols[:50]}")
        
        # Try more flexible matching
        logger.info("Attempting flexible feature matching...")
        for feat in feature_cols:
            feat_lower = feat.lower()
            if 'size' not in benchmark_features and any(x in feat_lower for x in ['mvel', 'me', 'size']):
                benchmark_features['size'] = feat
                logger.info(f"  Found size: {feat}")
            if 'bm' not in benchmark_features and any(x in feat_lower for x in ['bm', 'btm', 'book']):
                benchmark_features['bm'] = feat
                logger.info(f"  Found bm: {feat}")
            if 'momentum' not in benchmark_features and any(x in feat_lower for x in ['mom', 'ret_12']):
                benchmark_features['momentum'] = feat
                logger.info(f"  Found momentum: {feat}")
    
    if len(benchmark_features) < 3:
        raise ValueError(
            f"Could not identify all benchmark features. "
            f"Found: {benchmark_features}"
        )
    
    return benchmark_features


def prepare_benchmark_data(df: pd.DataFrame, benchmark_features: Dict[str, str],
                           target_col: str) -> pd.DataFrame:
    """
    Prepare data for benchmark model (select only relevant columns).
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    benchmark_features : dict
        Mapping of feature names to column names
    target_col : str
        Target column name
    
    Returns
    -------
    pd.DataFrame
        DataFrame with only target and 3 benchmark features
    """
    # Get benchmark feature columns
    feature_names = list(benchmark_features.values())
    
    # Select columns
    cols_to_keep = [target_col] + feature_names
    df_bench = df[cols_to_keep].copy()
    
    # Drop any rows with missing values
    df_bench = df_bench.dropna()
    
    logger.info(f"Benchmark data shape: {df_bench.shape}")
    
    return df_bench


def train_ols_expanding_window(train_df: pd.DataFrame, test_df: pd.DataFrame,
                               benchmark_features: Dict[str, str],
                               target_col: str) -> pd.DataFrame:
    """
    Train OLS model with expanding window and generate OOS predictions.
    
    For each test month:
    1. Use all training data up to that month
    2. Fit OLS model
    3. Predict for that month
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    benchmark_features : dict
        Benchmark feature mapping
    target_col : str
        Target column name
    
    Returns
    -------
    pd.DataFrame
        Test data with predictions
    """
    logger.info("Training OLS with expanding window")
    
    # Get feature columns
    feature_cols = list(benchmark_features.values())
    
    # Prepare benchmark data
    train_bench = prepare_benchmark_data(train_df, benchmark_features, target_col)
    test_bench = prepare_benchmark_data(test_df, benchmark_features, target_col)
    
    # Get unique test dates
    test_dates = test_bench.index.get_level_values('date').unique().sort_values()
    
    logger.info(f"Number of test months: {len(test_dates)}")
    
    # Store predictions
    predictions = []
    
    # Iterate through test dates with expanding window
    for test_date in tqdm(test_dates, desc="OLS expanding window"):
        # Get training data up to (but not including) test date
        train_dates = train_bench.index.get_level_values('date')
        train_window = train_bench[train_dates < test_date].copy()
        
        # Get test data for this date
        test_window = test_bench[test_bench.index.get_level_values('date') == test_date].copy()
        
        if len(train_window) < 100:  # Need minimum training samples
            continue
        
        if len(test_window) == 0:
            continue
        
        # Prepare X and y
        X_train = train_window[feature_cols].values
        y_train = train_window[target_col].values
        
        X_test = test_window[feature_cols].values
        y_test = test_window[target_col].values
        
        # Fit OLS model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Store predictions
        for idx, (permno, date) in enumerate(test_window.index):
            predictions.append({
                'permno': permno,
                'date': date,
                'y_true': y_test[idx],
                'y_pred': y_pred[idx]
            })
    
    # Create predictions dataframe
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.set_index(['permno', 'date'])
    
    logger.info(f"Generated predictions for {len(pred_df):,} observations")
    
    return pred_df


def evaluate_benchmark(pred_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """
    Evaluate OLS benchmark model performance.
    
    Metrics:
    1. Overall OOS R²
    2. Monthly OOS R²
    3. Portfolio sorts and long-short Sharpe ratio
    
    Parameters
    ----------
    pred_df : pd.DataFrame
        Predictions dataframe
    test_df : pd.DataFrame
        Test dataframe (for market cap weighting)
    
    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating benchmark model")
    
    results = {}
    
    # 1. Overall OOS R²
    overall_r2 = calculate_r_squared(pred_df['y_true'].values, pred_df['y_pred'].values)
    results['overall_r2'] = overall_r2
    logger.info(f"Overall OOS R²: {overall_r2*100:.4f}%")
    
    # 2. Monthly OOS R²
    monthly_r2 = calculate_monthly_r_squared(pred_df.reset_index())
    results['monthly_r2'] = monthly_r2
    logger.info(f"Average monthly OOS R²: {monthly_r2.mean()*100:.4f}%")
    logger.info(f"Median monthly OOS R²: {monthly_r2.median()*100:.4f}%")
    
    # 3. Portfolio analysis
    # Merge predictions with test data to get market cap for weighting
    eval_df = pred_df.copy()
    eval_df['prediction'] = eval_df['y_pred']
    eval_df['ret_excess'] = eval_df['y_true']
    
    # Try to get market cap column for value weighting
    mkt_cap_col = None
    for col in ['mvel1', 'me', 'market_equity', 'size']:
        if col in test_df.columns:
            mkt_cap_col = col
            eval_df[col] = test_df[col]
            break
    
    # Reset index for portfolio sorts
    eval_df = eval_df.reset_index()
    
    # Equal-weighted portfolios
    logger.info("Creating equal-weighted portfolios...")
    portfolios_ew = create_portfolio_sorts(
        eval_df, 
        prediction_col='prediction',
        return_col='ret_excess',
        n_portfolios=10,
        weight_col=None
    )
    
    # Long-short returns (EW)
    ls_returns_ew = calculate_long_short_returns(portfolios_ew)
    sharpe_ew = calculate_sharpe_ratio(ls_returns_ew['long_short'].values)
    results['sharpe_ew'] = sharpe_ew
    logger.info(f"Long-short Sharpe ratio (EW): {sharpe_ew:.4f}")
    
    # Value-weighted portfolios (if market cap available)
    if mkt_cap_col:
        logger.info("Creating value-weighted portfolios...")
        portfolios_vw = create_portfolio_sorts(
            eval_df,
            prediction_col='prediction',
            return_col='ret_excess',
            n_portfolios=10,
            weight_col=mkt_cap_col
        )
        
        # Long-short returns (VW)
        ls_returns_vw = calculate_long_short_returns(portfolios_vw)
        sharpe_vw = calculate_sharpe_ratio(ls_returns_vw['long_short'].values)
        results['sharpe_vw'] = sharpe_vw
        logger.info(f"Long-short Sharpe ratio (VW): {sharpe_vw:.4f}")
        
        results['portfolios_vw'] = portfolios_vw
        results['ls_returns_vw'] = ls_returns_vw
    else:
        logger.warning("Market cap column not found - skipping value-weighted portfolios")
    
    results['portfolios_ew'] = portfolios_ew
    results['ls_returns_ew'] = ls_returns_ew
    results['monthly_r2_series'] = monthly_r2
    
    return results


def save_benchmark_results(pred_df: pd.DataFrame, results: Dict) -> None:
    """
    Save benchmark results to disk.
    
    Parameters
    ----------
    pred_df : pd.DataFrame
        Predictions dataframe
    results : dict
        Evaluation results
    """
    logger.info("Saving benchmark results")
    
    ensure_dir(RESULTS_DIR / "predictions")
    ensure_dir(RESULTS_DIR / "tables")
    ensure_dir(RESULTS_DIR / "figures")
    
    # Save predictions
    pred_path = RESULTS_DIR / "predictions" / "benchmark_predictions.parquet"
    pred_df.to_parquet(pred_path)
    logger.info(f"Saved predictions to {pred_path}")
    
    # Save summary statistics
    summary = {
        'model': 'OLS Benchmark',
        'overall_r2_pct': results['overall_r2'] * 100,
        'mean_monthly_r2_pct': results['monthly_r2'].mean() * 100,
        'sharpe_ew': results['sharpe_ew'],
        'sharpe_vw': results.get('sharpe_vw', np.nan)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = RESULTS_DIR / "tables" / "benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")
    
    # Save to LaTeX
    latex_path = RESULTS_DIR / "tables" / "benchmark_summary.tex"
    summary_df.to_latex(latex_path, index=False, float_format='%.4f',
                       caption='OLS Benchmark Performance', label='tab:benchmark')
    logger.info(f"Saved LaTeX table to {latex_path}")
    
    # Plot long-short returns
    if 'ls_returns_ew' in results:
        plot_cumulative_returns(
            results['ls_returns_ew']['long_short'],
            title='OLS Benchmark: Long-Short Portfolio (Equal-Weighted)',
            save_path=str(RESULTS_DIR / "figures" / "benchmarks" / "benchmark_ls_ew.png")
        )
    
    if 'ls_returns_vw' in results:
        plot_cumulative_returns(
            results['ls_returns_vw']['long_short'],
            title='OLS Benchmark: Long-Short Portfolio (Value-Weighted)',
            save_path=str(RESULTS_DIR / "figures" / "benchmarks" / "benchmark_ls_vw.png")
        )


def main() -> None:
    """Main benchmark pipeline."""
    logger.info("="*80)
    logger.info("OLS Benchmark Model")
    logger.info("="*80)
    
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    train_df = pd.read_parquet(DATA_PROCESSED_DIR / "train_data.parquet")
    test_df = pd.read_parquet(DATA_PROCESSED_DIR / "test_data.parquet")
    
    # Load metadata
    import json
    with open(DATA_PROCESSED_DIR / "data_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    target_col = metadata['target_col']
    feature_cols = metadata['feature_cols']
    
    logger.info(f"Target: {target_col}")
    logger.info(f"Features: {len(feature_cols)}")
    
    # Identify benchmark features
    benchmark_features = identify_benchmark_features(feature_cols)
    
    # Train OLS with expanding window
    pred_df = train_ols_expanding_window(
        train_df, test_df, benchmark_features, target_col
    )
    
    # Evaluate
    results = evaluate_benchmark(pred_df, test_df)
    
    # Save results
    save_benchmark_results(pred_df, results)
    
    logger.info("="*80)
    logger.info("Benchmark model completed successfully!")
    logger.info("="*80)
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK PERFORMANCE SUMMARY")
    logger.info("="*80)
    logger.info(f"Overall OOS R²: {results['overall_r2']*100:.4f}%")
    logger.info(f"Average monthly OOS R²: {results['monthly_r2'].mean()*100:.4f}%")
    logger.info(f"Long-short Sharpe (EW): {results['sharpe_ew']:.4f}")
    if 'sharpe_vw' in results:
        logger.info(f"Long-short Sharpe (VW): {results['sharpe_vw']:.4f}")
    logger.info("="*80)
    
    # Compare with paper
    logger.info("\nComparison with paper targets:")
    logger.info(f"  Monthly R²: {results['monthly_r2'].mean()*100:.4f}% vs. 0.16% (target)")
    logger.info(f"  Sharpe (EW): {results['sharpe_ew']:.4f} vs. 0.83 (target)")
    if 'sharpe_vw' in results:
        logger.info(f"  Sharpe (VW): {results['sharpe_vw']:.4f} vs. 0.61 (target)")


if __name__ == "__main__":
    main()
