"""
Utility functions for Empirical Asset Pricing via Machine Learning replication.

This module provides helper functions for:
- Logging and configuration
- Data preprocessing and validation
- Portfolio construction and backtesting
- Performance metrics (R², Sharpe ratio, etc.)
- Visualization utilities

Author: Replication of Gu, Kelly, and Xiu (2020)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configure matplotlib style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging for the project.
    
    Parameters
    ----------
    log_file : str, optional
        Path to log file. If None, logs only to console.
    level : int
        Logging level (default: logging.INFO)
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger('asset_pricing_ml')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def ensure_dir(directory: Union[str, Path]) -> None:
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


# ============================================================================
# Data Preprocessing Functions
# ============================================================================

def rank_transform(series: pd.Series) -> pd.Series:
    """
    Rank transform values to [-1, 1] range (cross-sectionally).
    
    This replicates the data preprocessing in Gu et al. (2020).
    
    Parameters
    ----------
    series : pd.Series
        Series to transform
    
    Returns
    -------
    pd.Series
        Rank-transformed series in [-1, 1]
    """
    # Remove NaNs for ranking
    valid_mask = series.notna()
    if valid_mask.sum() == 0:
        return series
    
    # Rank and normalize to [-1, 1]
    ranked = series[valid_mask].rank(method='average')
    n_valid = valid_mask.sum()
    normalized = 2 * (ranked - 1) / (n_valid - 1) - 1 if n_valid > 1 else pd.Series(0, index=ranked.index)
    
    result = pd.Series(np.nan, index=series.index)
    result[valid_mask] = normalized
    
    return result


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Winsorize series at specified percentiles.
    
    Parameters
    ----------
    series : pd.Series
        Series to winsorize
    lower : float
        Lower percentile (default: 0.01)
    upper : float
        Upper percentile (default: 0.99)
    
    Returns
    -------
    pd.Series
        Winsorized series
    """
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def handle_missing_values(df: pd.DataFrame, method: str = 'forward_fill', max_fill: int = 3) -> pd.DataFrame:
    """
    Handle missing values in panel data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel data with MultiIndex (permno, date)
    method : str
        Method for handling missing values: 'forward_fill', 'median', 'drop'
    max_fill : int
        Maximum number of periods to forward fill
    
    Returns
    -------
    pd.DataFrame
        DataFrame with handled missing values
    """
    if method == 'forward_fill':
        df = df.groupby(level=0).fillna(method='ffill', limit=max_fill)
    elif method == 'median':
        df = df.fillna(df.median())
    elif method == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df


# ============================================================================
# Performance Metrics
# ============================================================================

def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate out-of-sample R-squared.
    
    R² = 1 - SS_res / SS_tot
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    float
        R-squared value
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan


def calculate_monthly_r_squared(df: pd.DataFrame, y_col: str = 'y_true', 
                                 pred_col: str = 'y_pred') -> pd.Series:
    """
    Calculate monthly R-squared values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date index and true/predicted columns
    y_col : str
        Column name for true values
    pred_col : str
        Column name for predictions
    
    Returns
    -------
    pd.Series
        Monthly R-squared values
    """
    def monthly_r2(group):
        return calculate_r_squared(group[y_col].values, group[pred_col].values)
    
    return df.groupby('date').apply(monthly_r2)


def calculate_sharpe_ratio(returns: np.ndarray, periods_per_year: int = 12) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    periods_per_year : int
        Number of periods per year (default: 12 for monthly)
    
    Returns
    -------
    float
        Annualized Sharpe ratio
    """
    mask = np.isfinite(returns)
    returns = returns[mask]
    
    if len(returns) == 0 or np.std(returns) == 0:
        return np.nan
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    
    return sharpe


def calculate_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray,
                                 periods_per_year: int = 12) -> float:
    """
    Calculate Information Ratio (IR).
    
    IR = E[R_p - R_b] / σ(R_p - R_b)
    
    Parameters
    ----------
    returns : np.ndarray
        Portfolio returns
    benchmark_returns : np.ndarray
        Benchmark returns
    periods_per_year : int
        Periods per year
    
    Returns
    -------
    float
        Annualized Information Ratio
    """
    active_returns = returns - benchmark_returns
    mask = np.isfinite(active_returns)
    active_returns = active_returns[mask]
    
    if len(active_returns) == 0 or np.std(active_returns) == 0:
        return np.nan
    
    mean_active = np.mean(active_returns)
    std_active = np.std(active_returns, ddof=1)
    
    ir = (mean_active / std_active) * np.sqrt(periods_per_year)
    
    return ir


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Parameters
    ----------
    cumulative_returns : np.ndarray
        Cumulative returns series
    
    Returns
    -------
    float
        Maximum drawdown (positive value)
    """
    cumulative_returns = np.array(cumulative_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    return abs(np.min(drawdown))


# ============================================================================
# Portfolio Construction
# ============================================================================

def create_portfolio_sorts(df: pd.DataFrame, prediction_col: str = 'prediction',
                           return_col: str = 'ret_excess', n_portfolios: int = 10,
                           weight_col: Optional[str] = None) -> pd.DataFrame:
    """
    Create portfolio sorts based on predictions.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date, predictions, and returns
    prediction_col : str
        Column name for predictions
    return_col : str
        Column name for returns
    n_portfolios : int
        Number of portfolios (default: 10 for deciles)
    weight_col : str, optional
        Column for value weighting (e.g., 'mvel1' for market cap).
        If None, uses equal weighting.
    
    Returns
    -------
    pd.DataFrame
        Portfolio returns by date and portfolio number
    """
    results = []
    
    for date, group in df.groupby('date'):
        # Remove missing predictions or returns
        valid = group[[prediction_col, return_col]].notna().all(axis=1)
        group = group[valid].copy()
        
        if len(group) < n_portfolios:
            continue
        
        # Assign to portfolios based on prediction quantiles
        group['portfolio'] = pd.qcut(group[prediction_col], q=n_portfolios, 
                                      labels=False, duplicates='drop') + 1
        
        # Calculate portfolio returns
        for port in range(1, n_portfolios + 1):
            port_data = group[group['portfolio'] == port]
            
            if len(port_data) == 0:
                continue
            
            if weight_col and weight_col in port_data.columns:
                # Value-weighted
                weights = port_data[weight_col].fillna(0)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    port_return = (port_data[return_col] * weights).sum()
                else:
                    port_return = port_data[return_col].mean()
            else:
                # Equal-weighted
                port_return = port_data[return_col].mean()
            
            results.append({
                'date': date,
                'portfolio': port,
                'return': port_return,
                'n_stocks': len(port_data)
            })
    
    portfolio_df = pd.DataFrame(results)
    
    return portfolio_df


def calculate_long_short_returns(portfolio_df: pd.DataFrame, 
                                  long_port: int = 10, 
                                  short_port: int = 1) -> pd.DataFrame:
    """
    Calculate long-short portfolio returns.
    
    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Portfolio returns from create_portfolio_sorts
    long_port : int
        Portfolio number for long leg (default: 10)
    short_port : int
        Portfolio number for short leg (default: 1)
    
    Returns
    -------
    pd.DataFrame
        Long-short returns by date
    """
    long = portfolio_df[portfolio_df['portfolio'] == long_port].set_index('date')['return']
    short = portfolio_df[portfolio_df['portfolio'] == short_port].set_index('date')['return']
    
    ls_returns = pd.DataFrame({
        'long': long,
        'short': short,
        'long_short': long - short
    })
    
    return ls_returns


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_cumulative_returns(returns: pd.Series, title: str = 'Cumulative Returns',
                            save_path: Optional[str] = None) -> None:
    """
    Plot cumulative returns over time.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns indexed by date
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    cumulative = (1 + returns).cumprod()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    cumulative.plot(ax=ax, linewidth=2)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_rolling_sharpe(returns: pd.Series, window: int = 36,
                        title: str = 'Rolling Sharpe Ratio',
                        save_path: Optional[str] = None) -> None:
    """
    Plot rolling Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns indexed by date
    window : int
        Rolling window size in periods (default: 36 months)
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(12)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    rolling_sharpe.plot(ax=ax, linewidth=2)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{window}-Month Rolling Sharpe Ratio', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_portfolio_performance(portfolio_df: pd.DataFrame, 
                               title: str = 'Portfolio Performance',
                               save_path: Optional[str] = None) -> None:
    """
    Plot performance of all portfolios.
    
    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Portfolio returns from create_portfolio_sorts
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    # Pivot to wide format
    portfolio_wide = portfolio_df.pivot(index='date', columns='portfolio', values='return')
    
    # Calculate cumulative returns
    cumulative = (1 + portfolio_wide).cumprod()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot each portfolio
    for col in cumulative.columns:
        if col == 1:
            ax.plot(cumulative.index, cumulative[col], label=f'P{col} (Low)', 
                   linewidth=2, linestyle='--', alpha=0.7)
        elif col == cumulative.columns.max():
            ax.plot(cumulative.index, cumulative[col], label=f'P{col} (High)', 
                   linewidth=2.5, alpha=0.9)
        else:
            ax.plot(cumulative.index, cumulative[col], label=f'P{col}', 
                   linewidth=1, alpha=0.5)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def create_performance_table(returns_dict: Dict[str, pd.Series],
                            save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a comprehensive performance statistics table.
    
    Parameters
    ----------
    returns_dict : dict
        Dictionary of strategy name -> returns series
    save_path : str, optional
        Path to save table as CSV and LaTeX
    
    Returns
    -------
    pd.DataFrame
        Performance statistics table
    """
    stats = []
    
    for name, returns in returns_dict.items():
        returns = returns.dropna()
        
        if len(returns) == 0:
            continue
        
        cumulative = (1 + returns).prod()
        total_return = cumulative - 1
        n_years = len(returns) / 12
        annualized_return = (cumulative ** (1 / n_years) - 1) if n_years > 0 else 0
        
        stats.append({
            'Strategy': name,
            'Mean Return (%)': returns.mean() * 100,
            'Std Dev (%)': returns.std() * 100,
            'Sharpe Ratio': calculate_sharpe_ratio(returns.values),
            'Ann. Return (%)': annualized_return * 100,
            'Total Return (%)': total_return * 100,
            'Max Drawdown (%)': calculate_max_drawdown((1 + returns).cumprod().values) * 100,
            'Skewness': stats.skew(returns),
            'Kurtosis': stats.kurtosis(returns),
            'N Months': len(returns)
        })
    
    df_stats = pd.DataFrame(stats)
    df_stats = df_stats.round(3)
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        
        # Save as CSV
        csv_path = save_path.replace('.tex', '.csv') if save_path.endswith('.tex') else save_path + '.csv'
        df_stats.to_csv(csv_path, index=False)
        print(f"Table saved to {csv_path}")
        
        # Save as LaTeX
        tex_path = save_path if save_path.endswith('.tex') else save_path + '.tex'
        df_stats.to_latex(tex_path, index=False, float_format='%.3f', 
                         caption='Performance Statistics', label='tab:performance')
        print(f"LaTeX table saved to {tex_path}")
    
    return df_stats


def print_summary_statistics(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print summary statistics for a dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to summarize
    name : str
        Name of the dataset
    """
    print(f"\n{'='*80}")
    print(f"{name} Summary Statistics")
    print(f"{'='*80}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
    print(f"Number of unique stocks: {df.index.get_level_values('permno').nunique()}")
    print(f"Number of months: {df.index.get_level_values('date').nunique()}")
    print(f"\nMissing values by column:")
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
    print(missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Test functions
    logger = setup_logging()
    logger.info("Utils module loaded successfully")
    
    # Test rank transform
    test_series = pd.Series([1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10])
    ranked = rank_transform(test_series)
    print("Rank transform test:")
    print(ranked)
