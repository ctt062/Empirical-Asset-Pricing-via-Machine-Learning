"""
Detailed Fama-French Model Analysis and Visualization.

This script:
1. Loads Fama-French predictions
2. Calculates detailed performance metrics (EW and VW)
3. Creates comprehensive visualizations
4. Analyzes factor loadings (beta, SMB, HML)
5. Saves all results to results/figures/fama_french/

Author: Asset Pricing ML Project
Date: 2025-11-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from config import TRANSACTION_COST

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
RESULTS_DIR = Path('results')
FIGURES_DIR = RESULTS_DIR / 'figures' / 'fama_french'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_predictions():
    """Load Fama-French predictions and test data."""
    logger.info("Loading Fama-French predictions...")
    
    # Load predictions
    pred_df = pd.read_parquet(RESULTS_DIR / 'predictions' / 'fama_french_predictions.parquet')
    
    # Load test data for characteristics
    test_df = pd.read_parquet('data/test_data.parquet')
    
    # Merge
    merged = pred_df.merge(
        test_df.reset_index(),
        on=['permno', 'date'],
        how='left'
    )
    
    logger.info(f"Loaded {len(merged):,} predictions")
    return merged

def calculate_portfolio_metrics(df, weight_col='equal'):
    """Calculate portfolio metrics with transaction costs."""
    
    # Sort by prediction
    df = df.sort_values(['date', 'prediction'], ascending=[True, False])
    
    # Create decile portfolios using ranking (handles duplicate values better than qcut)
    df['rank'] = df.groupby('date')['prediction'].rank(method='first', ascending=False)
    df['decile'] = df.groupby('date')['rank'].transform(
        lambda x: pd.cut(x, bins=10, labels=False, include_lowest=True)
    )
    
    # Long (top decile) and Short (bottom decile)
    long_df = df[df['decile'] == 0].copy()  # Changed from 9 to 0 since rank is descending
    short_df = df[df['decile'] == 9].copy()  # Changed from 0 to 9
    
    # Calculate actual turnover and returns month by month
    monthly_returns_ew = []
    monthly_returns_vw = []
    turnover_ew_list = []
    turnover_vw_list = []
    prev_long_stocks = set()
    prev_short_stocks = set()
    prev_long_weights_vw = {}
    prev_short_weights_vw = {}
    
    for date in sorted(df['date'].unique()):
        long_month = long_df[long_df['date'] == date]
        short_month = short_df[short_df['date'] == date]
        
        # Current portfolio stocks
        curr_long_stocks = set(long_month['permno'])
        curr_short_stocks = set(short_month['permno'])
        
        # Calculate turnover
        if len(prev_long_stocks) > 0:
            # Equal-weighted turnover
            long_turnover = len(curr_long_stocks - prev_long_stocks) / len(curr_long_stocks) if len(curr_long_stocks) > 0 else 0
            short_turnover = len(curr_short_stocks - prev_short_stocks) / len(curr_short_stocks) if len(curr_short_stocks) > 0 else 0
            turnover_ew = (long_turnover + short_turnover) / 2
            turnover_ew_list.append(turnover_ew)
            
            # Value-weighted turnover
            if weight_col == 'value':
                long_weights = long_month['mvel1'] / long_month['mvel1'].sum() if long_month['mvel1'].sum() > 0 else pd.Series([1/len(long_month)]*len(long_month), index=long_month.index)
                short_weights = short_month['mvel1'] / short_month['mvel1'].sum() if short_month['mvel1'].sum() > 0 else pd.Series([1/len(short_month)]*len(short_month), index=short_month.index)
                
                long_weights_vw = dict(zip(long_month['permno'], long_weights))
                short_weights_vw = dict(zip(short_month['permno'], short_weights))
                
                long_vw_turnover = sum(abs(long_weights_vw.get(p, 0) - prev_long_weights_vw.get(p, 0)) 
                                      for p in curr_long_stocks | set(prev_long_weights_vw.keys()))
                short_vw_turnover = sum(abs(short_weights_vw.get(p, 0) - prev_short_weights_vw.get(p, 0))
                                       for p in curr_short_stocks | set(prev_short_weights_vw.keys()))
                turnover_vw = (long_vw_turnover + short_vw_turnover) / 2
                turnover_vw_list.append(turnover_vw)
                
                prev_long_weights_vw = long_weights_vw
                prev_short_weights_vw = short_weights_vw
        else:
            # First month: 100% turnover
            turnover_ew_list.append(1.0)
            if weight_col == 'value':
                turnover_vw_list.append(1.0)
                long_weights = long_month['mvel1'] / long_month['mvel1'].sum() if long_month['mvel1'].sum() > 0 else pd.Series([1/len(long_month)]*len(long_month), index=long_month.index)
                short_weights = short_month['mvel1'] / short_month['mvel1'].sum() if short_month['mvel1'].sum() > 0 else pd.Series([1/len(short_month)]*len(short_month), index=short_month.index)
                prev_long_weights_vw = dict(zip(long_month['permno'], long_weights))
                prev_short_weights_vw = dict(zip(short_month['permno'], short_weights))
        
        prev_long_stocks = curr_long_stocks
        prev_short_stocks = curr_short_stocks
        
        # Calculate gross returns
        if weight_col == 'equal':
            ret_long = long_month['actual'].mean()
            ret_short = short_month['actual'].mean()
            tc = TRANSACTION_COST * turnover_ew_list[-1] * 2
            monthly_returns_ew.append(ret_long - ret_short - tc)
        else:  # value-weighted
            long_weights = long_month['mvel1'] / long_month['mvel1'].sum() if long_month['mvel1'].sum() > 0 else pd.Series([1/len(long_month)]*len(long_month), index=long_month.index)
            short_weights = short_month['mvel1'] / short_month['mvel1'].sum() if short_month['mvel1'].sum() > 0 else pd.Series([1/len(short_month)]*len(short_month), index=short_month.index)
            ret_long = (long_month['actual'] * long_weights).sum()
            ret_short = (short_month['actual'] * short_weights).sum()
            tc = TRANSACTION_COST * turnover_vw_list[-1] * 2
            monthly_returns_vw.append(ret_long - ret_short - tc)
    
    # Convert to series
    if weight_col == 'equal':
        ls_returns = pd.Series(monthly_returns_ew, index=sorted(df['date'].unique()))
        long_ret = long_df.groupby('date')['actual'].mean()
        short_ret = short_df.groupby('date')['actual'].mean()
    else:
        ls_returns = pd.Series(monthly_returns_vw, index=sorted(df['date'].unique()))
        # For plotting, calculate without TC
        long_df['weight'] = long_df.groupby('date')['mvel1'].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 1/len(x)
        )
        short_df['weight'] = short_df.groupby('date')['mvel1'].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 1/len(x)
        )
        long_ret_raw = long_df.groupby('date').apply(lambda x: (x['actual'] * x['weight']).sum())
        short_ret_raw = short_df.groupby('date').apply(lambda x: (x['actual'] * x['weight']).sum())
        
        if isinstance(long_ret_raw, pd.DataFrame):
            long_ret = pd.Series(long_ret_raw.iloc[:, 0].to_numpy(), dtype=np.float64, index=pd.to_datetime(long_ret_raw.index))
        else:
            long_ret = pd.Series(long_ret_raw.to_numpy(), dtype=np.float64, index=pd.to_datetime(long_ret_raw.index))
            
        if isinstance(short_ret_raw, pd.DataFrame):
            short_ret = pd.Series(short_ret_raw.iloc[:, 0].to_numpy(), dtype=np.float64, index=pd.to_datetime(short_ret_raw.index))
        else:
            short_ret = pd.Series(short_ret_raw.to_numpy(), dtype=np.float64, index=pd.to_datetime(short_ret_raw.index))
    
    # Calculate metrics
    mean_ret = ls_returns.mean() * 12  # Annualized
    std_ret = ls_returns.std() * np.sqrt(12)  # Annualized
    sharpe = mean_ret / std_ret if std_ret > 0 else 0
    cumulative = (1 + ls_returns).cumprod() - 1
    
    return {
        'returns': ls_returns,
        'long_returns': long_ret,
        'short_returns': short_ret,
        'cumulative': cumulative,
        'mean_annual': mean_ret,
        'std_annual': std_ret,
        'sharpe': sharpe,
        'long_df': long_df,
        'short_df': short_df
    }

def plot_performance_comparison(ew_metrics, vw_metrics):
    """Plot EW vs VW performance comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fama-French 3-Factor: Equal-Weighted vs Value-Weighted Performance', fontsize=16, fontweight='bold')
    
    # 1. Cumulative returns
    ax = axes[0, 0]
    ew_metrics['cumulative'].plot(ax=ax, label='Equal-Weighted', linewidth=2, color='blue')
    vw_metrics['cumulative'].plot(ax=ax, label='Value-Weighted', linewidth=2, color='red')
    ax.set_title('Cumulative Returns (Long-Short Portfolio)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 2. Sharpe ratio comparison
    ax = axes[0, 1]
    sharpes = [ew_metrics['sharpe'], vw_metrics['sharpe']]
    colors = ['blue', 'red']
    bars = ax.bar(['Equal-Weighted', 'Value-Weighted'], sharpes, color=colors, alpha=0.7)
    ax.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, sharpes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3. Rolling Sharpe (12-month)
    ax = axes[1, 0]
    ew_rolling_sharpe = ew_metrics['returns'].rolling(12).mean() / ew_metrics['returns'].rolling(12).std() * np.sqrt(12)
    vw_rolling_sharpe = vw_metrics['returns'].rolling(12).mean() / vw_metrics['returns'].rolling(12).std() * np.sqrt(12)
    ew_rolling_sharpe.plot(ax=ax, label='Equal-Weighted', linewidth=2, color='blue', alpha=0.7)
    vw_rolling_sharpe.plot(ax=ax, label='Value-Weighted', linewidth=2, color='red', alpha=0.7)
    ax.set_title('Rolling 12-Month Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. Annual return and volatility
    ax = axes[1, 1]
    x = np.arange(2)
    width = 0.35
    
    returns = [ew_metrics['mean_annual']*100, vw_metrics['mean_annual']*100]
    vols = [ew_metrics['std_annual']*100, vw_metrics['std_annual']*100]
    
    bars1 = ax.bar(x - width/2, returns, width, label='Annual Return (%)', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, vols, width, label='Annual Volatility (%)', color='orange', alpha=0.7)
    
    ax.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Equal-Weighted', 'Value-Weighted'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ew_vs_vw_performance.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {FIGURES_DIR / 'ew_vs_vw_performance.png'}")
    plt.close()

def analyze_factor_loadings(df):
    """Analyze factor loadings distribution."""
    
    logger.info("Analyzing factor loadings...")
    
    # The prediction from Fama-French is based on factor loadings
    # We'll analyze the relationship between characteristics and predictions
    
    # Key characteristics for Fama-French factors
    size_col = 'mvel1' if 'mvel1' in df.columns else None
    bm_col = 'bm' if 'bm' in df.columns else None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fama-French: Factor-Related Analysis', fontsize=16, fontweight='bold')
    
    # 1. Prediction distribution
    ax = axes[0, 0]
    df['prediction'].hist(ax=ax, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Return')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # 2. Size x B/M Matrix - Predictions (if available)
    ax = axes[0, 1]
    if size_col and bm_col:
        # Create quintiles
        df['size_quintile'] = pd.qcut(df[size_col], 5, labels=['Small', 'Size-2', 'Size-3', 'Size-4', 'Large'], duplicates='drop')
        df['bm_quintile'] = pd.qcut(df[bm_col], 5, labels=['Growth', 'BM-2', 'BM-3', 'BM-4', 'Value'], duplicates='drop')
        
        # Create pivot table for heatmap (predictions)
        pivot_pred = df.groupby(['size_quintile', 'bm_quintile'])['prediction'].mean().unstack()
        
        # Plot heatmap
        import seaborn as sns
        sns.heatmap(pivot_pred * 100, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
                    ax=ax, cbar_kws={'label': 'Avg Predicted Return (%)'})
        ax.set_title('Avg Prediction by Size × B/M', fontsize=12, fontweight='bold')
        ax.set_xlabel('Book-to-Market Quintile')
        ax.set_ylabel('Size Quintile')
    else:
        ax.text(0.5, 0.5, 'Size/B/M data not available', ha='center', va='center', transform=ax.transAxes)
    
    # 3. Size x B/M Matrix - Actual Returns (if available)
    ax = axes[1, 0]
    if size_col and bm_col and 'size_quintile' in df.columns and 'bm_quintile' in df.columns:
        # Create pivot table for actual returns
        pivot_actual = df.groupby(['size_quintile', 'bm_quintile'])['actual'].mean().unstack()
        
        # Plot heatmap
        sns.heatmap(pivot_actual * 100, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
                    ax=ax, cbar_kws={'label': 'Avg Actual Return (%)'})
        ax.set_title('Avg Actual Return by Size × B/M', fontsize=12, fontweight='bold')
        ax.set_xlabel('Book-to-Market Quintile')
        ax.set_ylabel('Size Quintile')
    else:
        ax.text(0.5, 0.5, 'Size/B/M data not available', ha='center', va='center', transform=ax.transAxes)
    
    # 4. Prediction vs Actual scatter
    ax = axes[1, 1]
    sample = df.sample(min(10000, len(df)))  # Sample for visualization
    ax.scatter(sample['prediction'], sample['actual'], alpha=0.3, s=10, color='blue')
    ax.set_title('Predictions vs Actual Returns', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Return')
    ax.set_ylabel('Actual Return')
    ax.grid(True, alpha=0.3)
    
    # Add regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(sample['prediction'], sample['actual'])
    line_x = np.array([sample['prediction'].min(), sample['prediction'].max()])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, 'r-', linewidth=2, label=f'R²={r_value**2:.3f}')
    
    # Add diagonal reference line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='y=x')
    ax.legend(loc='upper left')
    
    # Add correlation in text box
    corr = df[['prediction', 'actual']].corr().iloc[0, 1]
    ax.text(0.95, 0.05, f'Corr: {corr:.4f}', 
            transform=ax.transAxes, fontsize=11, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'factor_analysis.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {FIGURES_DIR / 'factor_analysis.png'}")
    plt.close()

def plot_portfolio_value_overtime(ew_metrics, vw_metrics):
    """Plot cumulative portfolio value over time."""
    
    logger.info("Creating portfolio value over time plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Fama-French: Portfolio Value Over Time', fontsize=16, fontweight='bold')
    
    # Calculate cumulative returns (portfolio value starting at $1)
    ew_cum_returns = (1 + ew_metrics['returns']).cumprod()
    vw_cum_returns = (1 + vw_metrics['returns']).cumprod()
    
    # 1. Equal-Weighted Portfolio Value
    ax = axes[0]
    ew_cum_returns.plot(ax=ax, linewidth=2, color='blue', label='Fama-French EW')
    ax.set_title('Equal-Weighted Portfolio Value', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add final value annotation
    final_value = ew_cum_returns.iloc[-1]
    ax.text(0.98, 0.02, f'Final Value: ${final_value:.2f}', 
            transform=ax.transAxes, fontsize=11, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 2. Value-Weighted Portfolio Value
    ax = axes[1]
    vw_cum_returns.plot(ax=ax, linewidth=2, color='red', label='Fama-French VW')
    ax.set_title('Value-Weighted Portfolio Value', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add final value annotation
    final_value = vw_cum_returns.iloc[-1]
    ax.text(0.98, 0.02, f'Final Value: ${final_value:.2f}', 
            transform=ax.transAxes, fontsize=11, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'portfolio_value_overtime.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {FIGURES_DIR / 'portfolio_value_overtime.png'}")
    plt.close()

def plot_monthly_returns(ew_metrics, vw_metrics):
    """Plot monthly return distributions."""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Fama-French: Monthly Return Distributions', fontsize=16, fontweight='bold')
    
    # Equal-weighted
    ax = axes[0]
    ew_metrics['returns'].hist(ax=ax, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(ew_metrics['returns'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ew_metrics["returns"].mean()*100:.2f}%')
    ax.set_title('Equal-Weighted Monthly Returns', fontsize=12, fontweight='bold')
    ax.set_xlabel('Monthly Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Value-weighted
    ax = axes[1]
    vw_metrics['returns'].hist(ax=ax, bins=30, color='red', alpha=0.7, edgecolor='black')
    ax.axvline(vw_metrics['returns'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {vw_metrics["returns"].mean()*100:.2f}%')
    ax.set_title('Value-Weighted Monthly Returns', fontsize=12, fontweight='bold')
    ax.set_xlabel('Monthly Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'monthly_returns_distribution.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {FIGURES_DIR / 'monthly_returns_distribution.png'}")
    plt.close()

def save_summary_statistics(ew_metrics, vw_metrics):
    """Save detailed summary statistics."""
    
    summary = {
        'Metric': [
            'Annual Return (%)',
            'Annual Volatility (%)',
            'Sharpe Ratio',
            'Cumulative Return (%)',
            'Monthly Win Rate (%)',
            'Max Drawdown (%)',
            'Avg Monthly Return (%)',
            'Median Monthly Return (%)'
        ],
        'Equal-Weighted': [
            f"{ew_metrics['mean_annual']*100:.2f}",
            f"{ew_metrics['std_annual']*100:.2f}",
            f"{ew_metrics['sharpe']:.2f}",
            f"{ew_metrics['cumulative'].iloc[-1]*100:.2f}",
            f"{(ew_metrics['returns'] > 0).sum() / len(ew_metrics['returns']) * 100:.1f}",
            f"{(ew_metrics['cumulative'] / (1 + ew_metrics['cumulative']).cummax() - 1).min()*100:.2f}",
            f"{ew_metrics['returns'].mean()*100:.2f}",
            f"{ew_metrics['returns'].median()*100:.2f}"
        ],
        'Value-Weighted': [
            f"{vw_metrics['mean_annual']*100:.2f}",
            f"{vw_metrics['std_annual']*100:.2f}",
            f"{vw_metrics['sharpe']:.2f}",
            f"{vw_metrics['cumulative'].iloc[-1]*100:.2f}",
            f"{(vw_metrics['returns'] > 0).sum() / len(vw_metrics['returns']) * 100:.1f}",
            f"{(vw_metrics['cumulative'] / (1 + vw_metrics['cumulative']).cummax() - 1).min()*100:.2f}",
            f"{vw_metrics['returns'].mean()*100:.2f}",
            f"{vw_metrics['returns'].median()*100:.2f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(FIGURES_DIR / 'performance_summary.csv', index=False)
    logger.info(f"Saved: {FIGURES_DIR / 'performance_summary.csv'}")
    
    return summary_df

def main():
    """Main execution."""
    
    logger.info("="*80)
    logger.info("FAMA-FRENCH MODEL - DETAILED ANALYSIS")
    logger.info("="*80)
    
    # Load data
    df = load_predictions()
    
    # Calculate metrics
    logger.info("Calculating Equal-Weighted metrics...")
    ew_metrics = calculate_portfolio_metrics(df, weight_col='equal')
    
    logger.info("Calculating Value-Weighted metrics...")
    vw_metrics = calculate_portfolio_metrics(df, weight_col='value')
    
    # Create visualizations
    logger.info("Creating visualizations...")
    plot_performance_comparison(ew_metrics, vw_metrics)
    plot_monthly_returns(ew_metrics, vw_metrics)
    plot_portfolio_value_overtime(ew_metrics, vw_metrics)
    analyze_factor_loadings(df)
    
    # Save summary
    summary_df = save_summary_statistics(ew_metrics, vw_metrics)
    
    logger.info("="*80)
    logger.info("FAMA-FRENCH PERFORMANCE SUMMARY")
    logger.info("="*80)
    print(summary_df.to_string(index=False))
    logger.info("="*80)
    logger.info(f"All results saved to: {FIGURES_DIR}")
    logger.info("="*80)

if __name__ == '__main__':
    main()
