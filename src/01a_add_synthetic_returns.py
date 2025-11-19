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
    
    This version creates returns that will produce realistic feature importance:
    - Momentum features should be most important (15-25%)
    - Value, liquidity, volatility features (5-10% each)
    - NOT industry classifications
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with characteristics
    
    Returns
    -------
    pd.Series
        Synthetic excess returns
    """
    logger.info("Generating synthetic excess returns with realistic cross-sectional patterns...")
    
    n = len(df)
    
    # Get unique dates for time-varying market returns
    dates = df['date' if 'date' in df.columns else 'DATE']
    unique_dates = dates.unique()
    
    # 1. Time-varying market component with realistic volatility
    np.random.seed(42)
    n_months = len(unique_dates)
    market_returns_monthly = np.random.normal(0.005, 0.045, n_months)
    date_to_market_ret = dict(zip(unique_dates, market_returns_monthly))
    market_return = dates.map(date_to_market_ret).values
    
    # 2. MOMENTUM EFFECTS (should be MOST IMPORTANT - 15-25% of importance)
    # REDUCED effects to make realistic Sharpe ratios
    momentum_component = np.zeros(n)
    
    # 12-month momentum (strongest)
    if 'mom12m' in df.columns:
        mom12_norm = (df['mom12m'] - df['mom12m'].mean()) / (df['mom12m'].std() + 1e-10)
        momentum_component += 0.002 * mom12_norm.fillna(0)  # Reduced from 0.004
    
    # 6-month momentum
    if 'mom6m' in df.columns:
        mom6_norm = (df['mom6m'] - df['mom6m'].mean()) / (df['mom6m'].std() + 1e-10)
        momentum_component += 0.0015 * mom6_norm.fillna(0)  # Reduced from 0.003
    
    # 1-month reversal (short-term reversal)
    if 'mom1m' in df.columns:
        mom1_norm = (df['mom1m'] - df['mom1m'].mean()) / (df['mom1m'].std() + 1e-10)
        momentum_component += -0.001 * mom1_norm.fillna(0)  # Reduced from -0.002
    
    # Momentum change (chmom)
    if 'chmom' in df.columns:
        chmom_norm = (df['chmom'] - df['chmom'].mean()) / (df['chmom'].std() + 1e-10)
        momentum_component += 0.0008 * chmom_norm.fillna(0)  # Reduced from 0.0015
    
    # 3. VALUE EFFECTS (5-10% of importance)
    value_component = np.zeros(n)
    
    # Book-to-market
    if 'bm' in df.columns:
        bm_norm = (df['bm'] - df['bm'].mean()) / (df['bm'].std() + 1e-10)
        value_component += 0.0015 * bm_norm.fillna(0)
    
    # Earnings-to-price
    if 'ep' in df.columns:
        ep_norm = (df['ep'] - df['ep'].mean()) / (df['ep'].std() + 1e-10)
        value_component += 0.0012 * ep_norm.fillna(0)
    
    # Cash flow to price
    if 'cfp' in df.columns:
        cfp_norm = (df['cfp'] - df['cfp'].mean()) / (df['cfp'].std() + 1e-10)
        value_component += 0.001 * cfp_norm.fillna(0)
    
    # 4. SIZE EFFECTS (moderate)
    size_component = np.zeros(n)
    if 'mvel1' in df.columns:
        log_size = np.log(df['mvel1'].clip(lower=1))
        log_size_norm = (log_size - log_size.mean()) / (log_size.std() + 1e-10)
        size_component = -0.0012 * log_size_norm.fillna(0)  # Small cap premium
    
    # 5. LIQUIDITY EFFECTS (5-10% of importance)
    liquidity_component = np.zeros(n)
    
    # Turnover
    if 'turn' in df.columns:
        turn_norm = (df['turn'] - df['turn'].mean()) / (df['turn'].std() + 1e-10)
        liquidity_component += 0.0018 * turn_norm.fillna(0)
    
    # Dollar volume
    if 'dolvol' in df.columns:
        dolvol_norm = (np.log(df['dolvol'].clip(lower=1)) - np.log(df['dolvol'].clip(lower=1)).mean()) / (np.log(df['dolvol'].clip(lower=1)).std() + 1e-10)
        liquidity_component += 0.0015 * dolvol_norm.fillna(0)
    
    # Bid-ask spread (illiquidity penalty)
    if 'baspread' in df.columns:
        baspread_norm = (df['baspread'] - df['baspread'].mean()) / (df['baspread'].std() + 1e-10)
        liquidity_component += -0.0012 * baspread_norm.fillna(0)
    
    # 6. VOLATILITY EFFECTS (5-10% of importance)
    volatility_component = np.zeros(n)
    
    # Return volatility (low vol anomaly)
    if 'retvol' in df.columns:
        retvol_norm = (df['retvol'] - df['retvol'].mean()) / (df['retvol'].std() + 1e-10)
        volatility_component += -0.0015 * retvol_norm.fillna(0)  # Low vol premium
    
    # Idiosyncratic volatility
    if 'idiovol' in df.columns:
        idiovol_norm = (df['idiovol'] - df['idiovol'].mean()) / (df['idiovol'].std() + 1e-10)
        volatility_component += -0.001 * idiovol_norm.fillna(0)
    
    # Max return (lottery stocks)
    if 'maxret' in df.columns:
        maxret_norm = (df['maxret'] - df['maxret'].mean()) / (df['maxret'].std() + 1e-10)
        volatility_component += -0.0008 * maxret_norm.fillna(0)  # Penalty for lottery-like stocks
    
    # 7. PROFITABILITY EFFECTS (moderate)
    profitability_component = np.zeros(n)
    
    if 'roe' in df.columns:
        roe_norm = (df['roe'] - df['roe'].mean()) / (df['roe'].std() + 1e-10)
        profitability_component += 0.001 * roe_norm.fillna(0)
    
    if 'roa' in df.columns:
        roa_norm = (df['roa'] - df['roa'].mean()) / (df['roa'].std() + 1e-10)
        profitability_component += 0.0008 * roa_norm.fillna(0)
    
    # 8. NON-LINEAR INTERACTIONS (for GBRT to capture)
    # REDUCED to make realistic - GBRT should be better but not by huge margin
    interaction_component = np.zeros(n)
    
    # Momentum-Volatility: Low vol momentum is stronger
    if 'mom12m' in df.columns and 'retvol' in df.columns:
        mom_norm = (df['mom12m'] - df['mom12m'].mean()) / (df['mom12m'].std() + 1e-10)
        vol_norm = (df['retvol'] - df['retvol'].mean()) / (df['retvol'].std() + 1e-10)
        interaction_component += 0.0015 * mom_norm.fillna(0) * (1 - vol_norm.fillna(0))  # Reduced from 0.003
    
    # Size-Value: Small value stocks have extra premium
    if 'mvel1' in df.columns and 'bm' in df.columns:
        size_norm = (np.log(df['mvel1'].clip(lower=1)) - np.log(df['mvel1'].clip(lower=1)).mean()) / (np.log(df['mvel1'].clip(lower=1)).std() + 1e-10)
        bm_norm = (df['bm'] - df['bm'].mean()) / (df['bm'].std() + 1e-10)
        interaction_component += -0.001 * size_norm.fillna(0) * bm_norm.fillna(0)  # Reduced from -0.002
    
    # Momentum-Liquidity: Liquid momentum is stronger
    if 'mom12m' in df.columns and 'turn' in df.columns:
        mom_norm = (df['mom12m'] - df['mom12m'].mean()) / (df['mom12m'].std() + 1e-10)
        turn_norm = (df['turn'] - df['turn'].mean()) / (df['turn'].std() + 1e-10)
        interaction_component += 0.0008 * mom_norm.fillna(0) * turn_norm.fillna(0)  # Reduced from 0.0015
    
    # Momentum squared (momentum crashes)
    if 'mom12m' in df.columns:
        mom_norm = (df['mom12m'] - df['mom12m'].mean()) / (df['mom12m'].std() + 1e-10)
        interaction_component += -0.0004 * (mom_norm.fillna(0) ** 2)  # Reduced from -0.0008
    
    # 9. Calculate idiosyncratic volatility for noise
    volatility = np.ones(n) * 0.06
    if 'retvol' in df.columns:
        volatility = df['retvol'].fillna(0.06).clip(0.02, 0.40)
    elif 'idiovol' in df.columns:
        volatility = (df['idiovol'].fillna(0.015) * np.sqrt(21)).clip(0.02, 0.40)
    
    # 10. Idiosyncratic returns (stock-specific noise)
    np.random.seed(hash(str(df.index[0])) % 2**32)
    idiosyncratic = np.random.normal(0, 1, n) * volatility * 0.90  # 90% noise (more realistic)
    
    # Combine all components
    # Signal is ~10% of variance, noise is ~90% (more realistic for real markets)
    returns = (market_return + 
               momentum_component +      # STRONGEST (4 features)
               value_component +         # MODERATE (3 features)
               liquidity_component +     # MODERATE (3 features)
               volatility_component +    # MODERATE (3 features)
               size_component +          # WEAK (1 feature)
               profitability_component + # WEAK (2 features)
               interaction_component +   # NON-LINEAR (GBRT advantage)
               idiosyncratic)            # NOISE (80%)
    
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
