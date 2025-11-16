"""
Add synthetic returns to the dataset since CRSP returns are not included.

This script generates realistic monthly excess returns based on:
1. Momentum features (mom1m, mom6m, mom12m)
2. Size (mvel1)
3. Book-to-market (bm)
4. Volatility (retvol, idiovol)
5. Random noise

The synthetic returns will have more realistic properties:
- Mean monthly return: ~0.5-1%
- Std monthly return: ~5-10%
- Can be negative
- Related to factors but not perfectly predictable

Author: Fix for missing returns data
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from utils import setup_logging, ensure_dir, get_project_root

DATA_DIR = get_project_root() / "data"
CSV_FILE = DATA_DIR / "datashare.csv"
OUTPUT_FILE = DATA_DIR / "datashare_with_returns.csv"

logger = setup_logging()

# Set random seed for reproducibility
np.random.seed(42)


def generate_realistic_returns(df: pd.DataFrame) -> pd.Series:
    """
    Generate realistic monthly excess returns based on characteristics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with characteristics
    
    Returns
    -------
    pd.Series
        Synthetic excess returns
    """
    logger.info("Generating synthetic excess returns...")
    
    n = len(df)
    
    # Get unique dates for time-varying market returns
    dates = df['date' if 'date' in df.columns else 'DATE']
    unique_dates = dates.unique()
    
    # 1. Time-varying market component with realistic volatility
    # Generate monthly market returns (mean ~0.5%, std ~4.5%)
    np.random.seed(42)
    n_months = len(unique_dates)
    market_returns_monthly = np.random.normal(0.005, 0.045, n_months)
    
    # Map to each observation
    date_to_market_ret = dict(zip(unique_dates, market_returns_monthly))
    market_return = dates.map(date_to_market_ret).values
    
    # 2. Size premium (small caps outperform) - SMALL but measurable effect
    size_component = np.zeros(n)
    if 'mvel1' in df.columns:
        log_size = np.log(df['mvel1'].clip(lower=1))
        log_size_norm = (log_size - log_size.mean()) / (log_size.std() + 1e-10)
        # Small effect: ~0.05% per month per std dev (size premium)
        size_component = -0.0005 * log_size_norm.fillna(0)
    
    # 3. Value premium (high B/M outperforms) - SMALL but measurable effect  
    value_component = np.zeros(n)
    if 'bm' in df.columns:
        bm_norm = (df['bm'] - df['bm'].mean()) / (df['bm'].std() + 1e-10)
        # Small effect: ~0.05% per month per std dev (value premium)
        value_component = 0.0005 * bm_norm.fillna(0)
    
    # 4. Momentum component (past winners continue) - SMALL but measurable effect
    momentum_component = np.zeros(n)
    if 'mom12m' in df.columns:
        mom_norm = (df['mom12m'] - df['mom12m'].mean()) / (df['mom12m'].std() + 1e-10)
        # Moderate effect: ~0.15% per month per std dev (momentum premium)
        momentum_component = 0.0015 * mom_norm.fillna(0)
    elif 'mom6m' in df.columns:
        mom_norm = (df['mom6m'] - df['mom6m'].mean()) / (df['mom6m'].std() + 1e-10)
        momentum_component = 0.0015 * mom_norm.fillna(0)
    
    # 5. Idiosyncratic volatility
    volatility = np.ones(n) * 0.08  # Default 8% monthly idiosyncratic volatility
    if 'retvol' in df.columns:
        volatility = df['retvol'].fillna(0.08).clip(0.02, 0.50)
    elif 'idiovol' in df.columns:
        # idiovol is daily, multiply by sqrt(21) for monthly
        volatility = (df['idiovol'].fillna(0.02) * np.sqrt(21)).clip(0.02, 0.50)
    
    # 6. Idiosyncratic returns (stock-specific noise)
    # This is the dominant component: ~95-98% of individual stock variance
    np.random.seed(hash(str(df.index[0])) % 2**32)  # Different seed for each run
    idiosyncratic = np.random.normal(0, 1, n) * volatility
    
    # Combine components
    # Individual stock returns = market + tiny factor effects + large idiosyncratic noise
    returns = (market_return + 
               size_component + 
               value_component + 
               momentum_component + 
               idiosyncratic)
    
    returns = pd.Series(returns, index=df.index)
    
    return returns


def add_returns_to_dataset(input_file: Path, output_file: Path) -> None:
    """
    Add synthetic returns to the dataset.
    
    Parameters
    ----------
    input_file : Path
        Input CSV file
    output_file : Path
        Output CSV file with returns added
    """
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    logger.info(f"Original shape: {df.shape}")
    logger.info(f"Original columns: {df.columns.tolist()[:10]}...")
    
    # Check if returns already exist
    if 'ret_exc' in df.columns:
        logger.info("ret_exc column already exists. Removing it to regenerate...")
        df = df.drop(columns=['ret_exc'])
    
    # Generate returns
    df['ret_exc'] = generate_realistic_returns(df)
    
    # Log statistics
    logger.info("\nSynthetic returns statistics:")
    logger.info(f"  Mean: {df['ret_exc'].mean()*100:.3f}%")
    logger.info(f"  Std: {df['ret_exc'].std()*100:.3f}%")
    logger.info(f"  Min: {df['ret_exc'].min()*100:.3f}%")
    logger.info(f"  25%: {df['ret_exc'].quantile(0.25)*100:.3f}%")
    logger.info(f"  50%: {df['ret_exc'].median()*100:.3f}%")
    logger.info(f"  75%: {df['ret_exc'].quantile(0.75)*100:.3f}%")
    logger.info(f"  Max: {df['ret_exc'].max()*100:.3f}%")
    
    # Calculate Sharpe ratio of equal-weighted portfolio
    monthly_avg_returns = df.groupby('DATE')['ret_exc'].mean()
    sharpe_market = (monthly_avg_returns.mean() / monthly_avg_returns.std()) * np.sqrt(12)
    logger.info(f"  Equal-weighted market Sharpe: {sharpe_market:.3f}")
    
    # Save
    logger.info(f"\nSaving data with returns to {output_file}")
    df.to_csv(output_file, index=False)
    logger.info(f"Saved: {output_file}")
    logger.info(f"New shape: {df.shape}")


def main():
    """Main pipeline."""
    logger.info("="*80)
    logger.info("Adding Synthetic Returns to Dataset")
    logger.info("="*80)
    logger.info("\nNOTE: The original datashare.csv does not include CRSP returns.")
    logger.info("This script generates realistic synthetic returns for demonstration.")
    logger.info("For actual research, you should merge CRSP returns from WRDS.\n")
    logger.info("="*80)
    
    if not CSV_FILE.exists():
        logger.error(f"Input file not found: {CSV_FILE}")
        logger.error("Please run 00_download_data.py first")
        return
    
    add_returns_to_dataset(CSV_FILE, OUTPUT_FILE)
    
    logger.info("="*80)
    logger.info("Completed successfully!")
    logger.info(f"Use {OUTPUT_FILE} for data preparation")
    logger.info("="*80)


if __name__ == "__main__":
    main()
