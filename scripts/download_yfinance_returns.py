"""
Download real stock returns from Yahoo Finance to replace synthetic returns.

This script:
1. Reads the stock identifiers from datashare.csv
2. Downloads historical price data from Yahoo Finance
3. Calculates monthly returns
4. Merges with the characteristics data
5. Handles missing data appropriately

Author: Asset Pricing ML Project
Date: 2025-11-23
"""

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import logging
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_stock_returns(tickers, start_date='1986-01-01', end_date='2016-12-31'):
    """
    Download monthly returns for a list of tickers from Yahoo Finance.
    
    Parameters:
    -----------
    tickers : list
        List of stock tickers
    start_date : str
        Start date for data download
    end_date : str
        End date for data download
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with monthly returns indexed by (date, ticker)
    """
    all_returns = []
    
    logger.info(f"Downloading returns for {len(tickers)} unique tickers...")
    logger.info("This may take 15-30 minutes depending on your internet connection.")
    
    # Process in batches to avoid rate limits
    batch_size = 50
    for i in tqdm(range(0, len(tickers), batch_size), desc="Downloading"):
        batch_tickers = tickers[i:i+batch_size]
        
        for ticker in batch_tickers:
            try:
                # Download data
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date, interval='1mo')
                
                if len(hist) > 0:
                    # Calculate returns
                    hist['return'] = hist['Close'].pct_change()
                    
                    # Create DataFrame with proper structure
                    returns_df = pd.DataFrame({
                        'date': hist.index,
                        'permno': ticker,  # Using ticker as identifier
                        'ret': hist['return'].values
                    })
                    
                    all_returns.append(returns_df)
                    
            except Exception as e:
                logger.debug(f"Failed to download {ticker}: {str(e)}")
                continue
        
        # Rate limiting
        time.sleep(0.5)
    
    if len(all_returns) == 0:
        logger.error("No returns downloaded!")
        return None
    
    # Combine all returns
    returns_df = pd.concat(all_returns, ignore_index=True)
    returns_df['date'] = pd.to_datetime(returns_df['date']).dt.to_period('M')
    
    logger.info(f"Successfully downloaded returns for {returns_df['permno'].nunique()} stocks")
    logger.info(f"Total observations: {len(returns_df):,}")
    
    return returns_df


def calculate_excess_returns(returns_df):
    """
    Calculate excess returns by subtracting risk-free rate.
    For simplicity, we'll use the monthly average market return as a proxy.
    """
    # Calculate equal-weighted market return by month
    market_returns = returns_df.groupby('date')['ret'].mean()
    
    # Simple risk-free rate approximation (3% annual = 0.25% monthly)
    rf = 0.0025
    
    # Calculate excess returns
    returns_df['ret_exc'] = returns_df['ret'] - rf
    
    logger.info(f"Calculated excess returns (rf = {rf*12*100:.1f}% annual)")
    
    return returns_df


def match_permnos_to_tickers(datashare_df):
    """
    Extract unique stock identifiers from datashare.csv.
    The permno column contains stock identifiers.
    """
    unique_permnos = datashare_df['permno'].unique()
    logger.info(f"Found {len(unique_permnos)} unique stocks in datashare.csv")
    
    # For yfinance, we need ticker symbols, not permnos
    # Since datashare uses permnos, we'll need to map them
    # For this implementation, we'll use the permno as-is and let yfinance try
    
    return unique_permnos.tolist()


def create_ticker_mapping(permnos):
    """
    Create mapping from PERMNO to ticker symbols.
    This is a simplified version - in practice, you'd use CRSP or manual mapping.
    """
    # For demonstration, we'll try to use permnos directly as tickers
    # In production, you'd want a proper PERMNO-to-ticker mapping
    logger.warning("Using simplified permno-to-ticker mapping. For production, use proper CRSP mapping.")
    
    return {p: str(p) for p in permnos}


def main():
    logger.info("="*80)
    logger.info("DOWNLOADING REAL RETURNS FROM YAHOO FINANCE")
    logger.info("="*80)
    
    # Load datashare
    logger.info("Loading datashare.csv...")
    datashare = pd.read_csv('data/datashare.csv')
    logger.info(f"Loaded {len(datashare):,} observations")
    
    # Check if ticker column exists
    if 'ticker' in datashare.columns:
        logger.info("Found 'ticker' column in datashare.csv")
        unique_tickers = datashare['ticker'].dropna().unique().tolist()
    else:
        logger.warning("No 'ticker' column found. Attempting to use permno...")
        unique_tickers = datashare['permno'].astype(str).unique().tolist()
    
    logger.info(f"Will attempt to download {len(unique_tickers)} stocks")
    
    # Download returns
    returns_df = download_stock_returns(
        unique_tickers,
        start_date='1986-01-01',
        end_date='2016-12-31'
    )
    
    if returns_df is None:
        logger.error("Failed to download any returns!")
        return
    
    # Calculate excess returns
    returns_df = calculate_excess_returns(returns_df)
    
    # Merge with datashare
    logger.info("Merging returns with characteristics...")
    
    # Convert date formats to match
    datashare['date'] = pd.to_datetime(datashare['date']).dt.to_period('M')
    
    # Merge
    if 'ticker' in datashare.columns:
        merged_df = datashare.merge(
            returns_df[['date', 'permno', 'ret_exc']],
            left_on=['date', 'ticker'],
            right_on=['date', 'permno'],
            how='left'
        )
        merged_df = merged_df.drop('permno_y', axis=1).rename(columns={'permno_x': 'permno'})
    else:
        datashare['permno_str'] = datashare['permno'].astype(str)
        merged_df = datashare.merge(
            returns_df[['date', 'permno', 'ret_exc']],
            left_on=['date', 'permno_str'],
            right_on=['date', 'permno'],
            how='left',
            suffixes=('', '_yf')
        )
    
    # Check coverage
    coverage = merged_df['ret_exc'].notna().sum() / len(merged_df) * 100
    logger.info(f"Return coverage: {coverage:.2f}%")
    
    if coverage < 20:
        logger.error("Very low coverage! Yahoo Finance may not have data for these stocks.")
        logger.error("This likely means the stock identifiers don't match Yahoo Finance tickers.")
        logger.error("\nRECOMMENDATION: We need a proper PERMNO-to-ticker mapping file.")
        logger.error("Alternative: Use slightly more predictable synthetic returns instead.")
        return
    
    # Save
    logger.info("Saving data with real returns...")
    merged_df['date'] = merged_df['date'].astype(str)
    merged_df.to_csv('data/datashare_with_returns.csv', index=False)
    
    logger.info("="*80)
    logger.info("DOWNLOAD COMPLETED")
    logger.info("="*80)
    logger.info(f"Saved: data/datashare_with_returns.csv")
    logger.info(f"Coverage: {coverage:.2f}%")
    logger.info(f"Observations with returns: {merged_df['ret_exc'].notna().sum():,}")
    
    # Statistics
    logger.info("\nReturn Statistics:")
    logger.info(f"  Mean: {merged_df['ret_exc'].mean()*100:.3f}%")
    logger.info(f"  Std:  {merged_df['ret_exc'].std()*100:.2f}%")
    logger.info(f"  Min:  {merged_df['ret_exc'].min()*100:.2f}%")
    logger.info(f"  Max:  {merged_df['ret_exc'].max()*100:.2f}%")
    
    logger.info("\nNext step: Run python3 src/01_data_preparation.py")


if __name__ == '__main__':
    main()
