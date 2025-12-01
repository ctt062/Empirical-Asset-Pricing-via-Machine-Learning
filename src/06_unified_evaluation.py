"""
Unified Model Evaluation

Compares performance of all 4 models:
1. OLS-3 (polynomial benchmark)
2. GBRT (gradient boosted trees)
3. Elastic Net (regularized linear)
4. Fama-French 3-Factor (traditional factor model)

Metrics:
- Out-of-sample R²
- Sharpe ratios (equal-weighted and value-weighted)
- Information ratios
- Portfolio returns and volatility
"""

import logging
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import transaction cost settings
from config import TRANSACTION_COST

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_predictions():
    """Load predictions from all models."""
    logger.info("Loading predictions from all models...")
    
    predictions = {}
    
    # OLS-3
    ols_path = 'results/predictions/benchmark_predictions.parquet'
    if Path(ols_path).exists():
        df = pd.read_parquet(ols_path)
        # Standardize column names: y_true -> actual, y_pred -> prediction
        if 'y_true' in df.columns and 'y_pred' in df.columns:
            df = df.rename(columns={'y_true': 'actual', 'y_pred': 'prediction'})
        predictions['OLS-3'] = df
        logger.info(f"Loaded OLS-3: {len(predictions['OLS-3']):,} predictions")
    
    # GBRT - try parquet first, then csv
    gbrt_path_parquet = 'results/predictions/gbrt_predictions.parquet'
    gbrt_path_csv = 'results/predictions/gbrt_predictions.csv'
    if Path(gbrt_path_parquet).exists():
        df = pd.read_parquet(gbrt_path_parquet)
        if 'y_true' in df.columns and 'y_pred' in df.columns:
            df = df.rename(columns={'y_true': 'actual', 'y_pred': 'prediction'})
        predictions['GBRT'] = df
        logger.info(f"Loaded GBRT: {len(predictions['GBRT']):,} predictions")
    elif Path(gbrt_path_csv).exists():
        df = pd.read_csv(gbrt_path_csv)
        if 'y_true' in df.columns and 'y_pred' in df.columns:
            df = df.rename(columns={'y_true': 'actual', 'y_pred': 'prediction'})
        predictions['GBRT'] = df
        logger.info(f"Loaded GBRT: {len(predictions['GBRT']):,} predictions")
    
    # Elastic Net - try parquet first, then csv
    enet_path_parquet = 'results/predictions/elastic_net_predictions.parquet'
    enet_path_csv = 'results/predictions/elastic_net_predictions.csv'
    if Path(enet_path_parquet).exists():
        predictions['Elastic Net'] = pd.read_parquet(enet_path_parquet)
        logger.info(f"Loaded Elastic Net: {len(predictions['Elastic Net']):,} predictions")
    elif Path(enet_path_csv).exists():
        predictions['Elastic Net'] = pd.read_csv(enet_path_csv)
        logger.info(f"Loaded Elastic Net: {len(predictions['Elastic Net']):,} predictions")
    
    # Fama-French - try parquet first, then csv
    ff_path_parquet = 'results/predictions/fama_french_predictions.parquet'
    ff_path_csv = 'results/predictions/fama_french_predictions.csv'
    if Path(ff_path_parquet).exists():
        predictions['Fama-French'] = pd.read_parquet(ff_path_parquet)
        logger.info(f"Loaded Fama-French: {len(predictions['Fama-French']):,} predictions")
    elif Path(ff_path_csv).exists():
        predictions['Fama-French'] = pd.read_csv(ff_path_csv)
        logger.info(f"Loaded Fama-French: {len(predictions['Fama-French']):,} predictions")
    
    return predictions


def calculate_oos_r2(predictions_df):
    """Calculate out-of-sample R²."""
    ss_res = ((predictions_df['actual'] - predictions_df['prediction']) ** 2).sum()
    ss_tot = ((predictions_df['actual'] - predictions_df['actual'].mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    return r2 * 100  # Return as percentage


def calculate_portfolio_metrics(predictions_df):
    """
    Calculate portfolio performance metrics with transaction costs.
    
    Forms long-short portfolios based on predicted returns.
    Accounts for transaction costs based on portfolio turnover.
    """
    # Load market cap data for value-weighting
    test_data = pd.read_parquet('data/processed/test_data.parquet').reset_index()
    mktcap_data = test_data[['date', 'permno', 'mvel1']].copy()
    
    # Merge with predictions
    df = predictions_df.merge(mktcap_data, on=['date', 'permno'], how='left')
    
    monthly_returns_ew = []
    monthly_returns_vw = []
    turnover_ew = []
    turnover_vw = []
    
    prev_long_stocks = set()
    prev_short_stocks = set()
    prev_long_weights_vw = {}
    prev_short_weights_vw = {}
    
    for date in sorted(df['date'].unique()):
        month_data = df[df['date'] == date].copy()
        
        # Skip if too few stocks
        if len(month_data) < 20:
            continue
        
        # Rank by predictions
        month_data = month_data.sort_values('prediction', ascending=False)
        
        # Form decile portfolios
        n_stocks = len(month_data)
        decile_size = n_stocks // 10
        
        long_portfolio = month_data.head(decile_size)
        short_portfolio = month_data.tail(decile_size)
        
        # Current portfolio stocks
        curr_long_stocks = set(long_portfolio['permno'])
        curr_short_stocks = set(short_portfolio['permno'])
        
        # Calculate turnover (fraction of portfolio that changes)
        if len(prev_long_stocks) > 0:
            # Equal-weighted turnover
            long_turnover = len(curr_long_stocks - prev_long_stocks) / len(curr_long_stocks)
            short_turnover = len(curr_short_stocks - prev_short_stocks) / len(curr_short_stocks)
            turnover_ew.append((long_turnover + short_turnover) / 2)
            
            # Value-weighted turnover
            long_weights_vw = dict(zip(long_portfolio['permno'], 
                                      long_portfolio['mvel1'] / long_portfolio['mvel1'].sum()))
            short_weights_vw = dict(zip(short_portfolio['permno'], 
                                       short_portfolio['mvel1'] / short_portfolio['mvel1'].sum()))
            
            # Weight changes for continuing stocks + full weight for new stocks
            long_vw_turnover = sum(abs(long_weights_vw.get(p, 0) - prev_long_weights_vw.get(p, 0)) 
                                  for p in curr_long_stocks | set(prev_long_weights_vw.keys()))
            short_vw_turnover = sum(abs(short_weights_vw.get(p, 0) - prev_short_weights_vw.get(p, 0))
                                   for p in curr_short_stocks | set(prev_short_weights_vw.keys()))
            turnover_vw.append((long_vw_turnover + short_vw_turnover) / 2)
            
            prev_long_weights_vw = long_weights_vw
            prev_short_weights_vw = short_weights_vw
        else:
            # First month: 100% turnover (building portfolio from scratch)
            turnover_ew.append(1.0)
            turnover_vw.append(1.0)
            prev_long_weights_vw = dict(zip(long_portfolio['permno'], 
                                           long_portfolio['mvel1'] / long_portfolio['mvel1'].sum()))
            prev_short_weights_vw = dict(zip(short_portfolio['permno'], 
                                            short_portfolio['mvel1'] / short_portfolio['mvel1'].sum()))
        
        # Update previous stocks
        prev_long_stocks = curr_long_stocks
        prev_short_stocks = curr_short_stocks
        
        # Calculate gross returns
        ret_ew_gross = long_portfolio['actual'].mean() - short_portfolio['actual'].mean()
        
        # Value-weighted returns
        long_weights = long_portfolio['mvel1'] / long_portfolio['mvel1'].sum()
        short_weights = short_portfolio['mvel1'] / short_portfolio['mvel1'].sum()
        
        ret_long_vw = (long_portfolio['actual'] * long_weights).sum()
        ret_short_vw = (short_portfolio['actual'] * short_weights).sum()
        ret_vw_gross = ret_long_vw - ret_short_vw
        
        # Apply transaction costs (one-way cost * turnover for long-short portfolio)
        # For long-short, we pay costs on both sides
        tc_ew = TRANSACTION_COST * turnover_ew[-1] * 2  # 2x for long and short
        tc_vw = TRANSACTION_COST * turnover_vw[-1] * 2
        
        ret_ew_net = ret_ew_gross - tc_ew
        ret_vw_net = ret_vw_gross - tc_vw
        
        monthly_returns_ew.append(ret_ew_net)
        monthly_returns_vw.append(ret_vw_net)
    
    # Calculate metrics
    returns_ew = np.array(monthly_returns_ew)
    returns_vw = np.array(monthly_returns_vw)
    
    metrics = {
        'sharpe_ew': (returns_ew.mean() / returns_ew.std()) * np.sqrt(12) if returns_ew.std() > 0 else 0,
        'sharpe_vw': (returns_vw.mean() / returns_vw.std()) * np.sqrt(12) if returns_vw.std() > 0 else 0,
        'annual_return_ew': returns_ew.mean() * 12 * 100,  # Convert to percentage
        'annual_return_vw': returns_vw.mean() * 12 * 100,
        'annual_vol_ew': returns_ew.std() * np.sqrt(12) * 100,
        'annual_vol_vw': returns_vw.std() * np.sqrt(12) * 100,
        'cumulative_return_ew': (1 + returns_ew).cumprod()[-1] - 1 if len(returns_ew) > 0 else 0,
        'cumulative_return_vw': (1 + returns_vw).cumprod()[-1] - 1 if len(returns_vw) > 0 else 0,
        'monthly_returns_ew': returns_ew,
        'monthly_returns_vw': returns_vw
    }
    
    return metrics


def create_performance_table(all_results):
    """Create comprehensive performance comparison table."""
    
    table_data = []
    
    for model_name, results in all_results.items():
        table_data.append({
            'Model': model_name,
            'OOS R² (%)': f"{results['oos_r2']:.2f}",
            'Sharpe (EW)': f"{results['sharpe_ew']:.2f}",
            'Sharpe (VW)': f"{results['sharpe_vw']:.2f}",
            'Annual Return (EW)': f"{results['annual_return_ew']:.2f}%",
            'Annual Return (VW)': f"{results['annual_return_vw']:.2f}%",
            'Annual Vol (EW)': f"{results['annual_vol_ew']:.2f}%",
            'Annual Vol (VW)': f"{results['annual_vol_vw']:.2f}%",
            'Cumulative (EW)': f"{results['cumulative_return_ew']*100:.2f}%",
            'Cumulative (VW)': f"{results['cumulative_return_vw']*100:.2f}%"
        })
    
    df = pd.DataFrame(table_data)
    
    # Reorder by Sharpe ratio (EW)
    df['sharpe_numeric'] = df['Sharpe (EW)'].astype(float)
    df = df.sort_values('sharpe_numeric', ascending=False)
    df = df.drop('sharpe_numeric', axis=1)
    
    return df


def plot_sharpe_comparison(all_results):
    """Plot Sharpe ratio comparison across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = list(all_results.keys())
    sharpes_ew = [all_results[m]['sharpe_ew'] for m in models]
    sharpes_vw = [all_results[m]['sharpe_vw'] for m in models]
    
    # Equal-weighted
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    ax1.barh(models, sharpes_ew, color=colors)
    ax1.set_xlabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Equal-Weighted Portfolio', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(sharpes_ew):
        ax1.text(v + 0.05, i, f'{v:.2f}', va='center', fontweight='bold')
    
    # Value-weighted
    ax2.barh(models, sharpes_vw, color=colors)
    ax2.set_xlabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Value-Weighted Portfolio', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(sharpes_vw):
        ax2.text(v + 0.05, i, f'{v:.2f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figures/model_comparison/all_models_sharpe_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("Saved: results/figures/model_comparison/all_models_sharpe_comparison.png")
    plt.close()


def plot_cumulative_returns(all_results):
    """Plot cumulative returns for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'OLS-3': '#3498db', 'GBRT': '#e74c3c', 
              'Elastic Net': '#2ecc71', 'Fama-French': '#f39c12'}
    
    # Equal-weighted
    for model_name, results in all_results.items():
        returns = results['monthly_returns_ew']
        cumulative = (1 + returns).cumprod()
        ax1.plot(cumulative, label=model_name, linewidth=2, color=colors.get(model_name))
    
    ax1.set_xlabel('Months', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
    ax1.set_title('Equal-Weighted Portfolio Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Value-weighted
    for model_name, results in all_results.items():
        returns = results['monthly_returns_vw']
        cumulative = (1 + returns).cumprod()
        ax2.plot(cumulative, label=model_name, linewidth=2, color=colors.get(model_name))
    
    ax2.set_xlabel('Months', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
    ax2.set_title('Value-Weighted Portfolio Performance', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/model_comparison/all_models_cumulative_returns.png', dpi=300, bbox_inches='tight')
    logger.info("Saved: results/figures/model_comparison/all_models_cumulative_returns.png")
    plt.close()


def plot_return_distribution(all_results):
    """Plot distribution of monthly returns."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, (model_name, results) in enumerate(all_results.items()):
        returns_ew = results['monthly_returns_ew'] * 100  # Convert to percentage
        
        axes[i].hist(returns_ew, bins=30, alpha=0.7, color=colors[i], edgecolor='black')
        axes[i].axvline(returns_ew.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {returns_ew.mean():.2f}%')
        axes[i].set_xlabel('Monthly Return (%)', fontsize=11)
        axes[i].set_ylabel('Frequency', fontsize=11)
        axes[i].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/model_comparison/all_models_return_distribution.png', dpi=300, bbox_inches='tight')
    logger.info("Saved: results/figures/model_comparison/all_models_return_distribution.png")
    plt.close()


def main():
    """Main evaluation pipeline."""
    logger.info("="*80)
    logger.info("⚠️  WARNING: SYNTHETIC DATA EVALUATION ⚠️")
    logger.info("="*80)
    logger.info("\n🚨 IMPORTANT: Results use SYNTHETIC returns, NOT real CRSP data 🚨")
    logger.info("Performance metrics will be INFLATED and UNREALISTIC")
    logger.info("For real research, use actual CRSP returns from WRDS\n")
    logger.info("="*80)
    logger.info("UNIFIED MODEL EVALUATION")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # Load predictions
    predictions = load_predictions()
    
    if len(predictions) == 0:
        logger.error("No predictions found! Train models first.")
        return
    
    # Evaluate each model
    logger.info("\nCalculating performance metrics...")
    all_results = {}
    
    for model_name, pred_df in predictions.items():
        logger.info(f"\nEvaluating {model_name}...")
        
        # Calculate OOS R²
        oos_r2 = calculate_oos_r2(pred_df)
        logger.info(f"  OOS R²: {oos_r2:.2f}%")
        
        # Calculate portfolio metrics
        portfolio_metrics = calculate_portfolio_metrics(pred_df)
        
        logger.info(f"  Sharpe (EW): {portfolio_metrics['sharpe_ew']:.2f}")
        logger.info(f"  Sharpe (VW): {portfolio_metrics['sharpe_vw']:.2f}")
        logger.info(f"  Annual Return (EW): {portfolio_metrics['annual_return_ew']:.2f}%")
        logger.info(f"  Annual Volatility (EW): {portfolio_metrics['annual_vol_ew']:.2f}%")
        
        # Store results
        all_results[model_name] = {
            'oos_r2': oos_r2,
            **portfolio_metrics
        }
    
    # Create performance table
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE COMPARISON TABLE")
    logger.info("="*80)
    
    perf_table = create_performance_table(all_results)
    print("\n" + perf_table.to_string(index=False))
    
    # Save table
    perf_table.to_csv('results/tables/all_models_performance.csv', index=False)
    perf_table.to_latex('results/tables/all_models_performance.tex', index=False)
    logger.info("\nSaved performance table to results/tables/")
    
    # Create visualizations
    logger.info("\nCreating visualizations...")
    plot_sharpe_comparison(all_results)
    plot_cumulative_returns(all_results)
    plot_return_distribution(all_results)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETED")
    logger.info("="*80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\nResults saved to:")
    logger.info("  - results/tables/all_models_performance.csv")
    logger.info("  - results/figures/model_comparison/all_models_*.png")
    
    # Identify best model
    best_model_ew = max(all_results.items(), key=lambda x: x[1]['sharpe_ew'])
    best_model_vw = max(all_results.items(), key=lambda x: x[1]['sharpe_vw'])
    
    logger.info(f"\nBest Model (Equal-Weighted): {best_model_ew[0]} "
               f"(Sharpe: {best_model_ew[1]['sharpe_ew']:.2f})")
    logger.info(f"Best Model (Value-Weighted): {best_model_vw[0]} "
               f"(Sharpe: {best_model_vw[1]['sharpe_vw']:.2f})")
    logger.info("\n⚠️  REMINDER: These results use SYNTHETIC data")
    logger.info("Real CRSP data would show:")
    logger.info("  - OOS R²: 0.3-0.5% (not the values above)")
    logger.info("  - Sharpe ratios: 1.5-2.5 (not 3-8+)")
    logger.info("  - Many losing months (not mostly winning)")
    logger.info("="*80)


if __name__ == "__main__":
    main()
