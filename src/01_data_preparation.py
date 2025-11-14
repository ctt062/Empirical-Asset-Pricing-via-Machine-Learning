"""
Data preparation and preprocessing for machine learning models.

This script:
1. Loads the datashare.csv file
2. Parses dates and creates proper panel structure
3. Handles missing values
4. Creates train/test splits with expanding window
5. Saves preprocessed data for modeling

Author: Replication of Gu, Kelly, and Xiu (2020)
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    setup_logging, 
    ensure_dir, 
    get_project_root,
    print_summary_statistics,
    winsorize_series
)

# Configuration
DATA_DIR = get_project_root() / "data"
RESULTS_DIR = get_project_root() / "results"
CSV_FILE = DATA_DIR / "datashare.csv"

# Train/test split date
TRAIN_END_DATE = "1995-12-31"
TEST_START_DATE = "1996-01-01"
TEST_END_DATE = "2016-12-31"

logger = setup_logging()


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """
    Load the raw datashare.csv file.
    
    Parameters
    ----------
    file_path : Path
        Path to CSV file
    
    Returns
    -------
    pd.DataFrame
        Raw data
    """
    logger.info(f"Loading data from {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            "Please run 00_download_data.py first"
        )
    
    # Load data
    df = pd.read_csv(file_path)
    
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    
    return df


def parse_dates(df: pd.DataFrame, date_col: str = 'DATE') -> pd.DataFrame:
    """
    Parse date column and set as index.
    
    The DATE column is in YYYYMM format (e.g., 195701 for January 1957).
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe
    date_col : str
        Name of date column
    
    Returns
    -------
    pd.DataFrame
        DataFrame with parsed dates
    """
    logger.info("Parsing dates")
    
    # Convert YYYYMMDD to datetime
    df['date'] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d')
    
    # Convert to month-end dates for consistency
    df['date'] = df['date'].dt.to_period('M').dt.to_timestamp('M')
    
    # Drop original date column
    df = df.drop(columns=[date_col])
    
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def create_panel_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create proper panel structure with MultiIndex (permno, date).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with permno and date columns
    
    Returns
    -------
    pd.DataFrame
        Panel data with MultiIndex
    """
    logger.info("Creating panel structure")
    
    # Ensure permno is integer
    df['permno'] = df['permno'].astype(int)
    
    # Set MultiIndex
    df = df.set_index(['permno', 'date'])
    
    # Sort index
    df = df.sort_index()
    
    logger.info(f"Panel shape: {df.shape}")
    logger.info(f"Unique stocks: {df.index.get_level_values('permno').nunique():,}")
    logger.info(f"Unique months: {df.index.get_level_values('date').nunique():,}")
    
    return df


def identify_columns(df: pd.DataFrame) -> Tuple[str, list]:
    """
    Identify target variable and feature columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel dataframe
    
    Returns
    -------
    tuple
        (target_col, feature_cols)
    """
    logger.info("Identifying columns")
    
    # Target variable (excess returns)
    # Could be 'ret_exc', 'ret_excess', or similar
    possible_targets = ['ret_exc', 'ret_excess', 'exret', 'excess_ret']
    target_col = None
    
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        logger.warning(f"Target column not found. Looking for columns with 'ret'...")
        ret_cols = [col for col in df.columns if 'ret' in col.lower()]
        logger.info(f"Columns with 'ret': {ret_cols}")
        if ret_cols:
            target_col = ret_cols[0]
            logger.info(f"Using {target_col} as target")
    
    if target_col is None:
        raise ValueError("Could not identify target column")
    
    # Feature columns (exclude target and any identifiers)
    exclude_cols = [target_col, 'permno', 'date']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"Target column: {target_col}")
    logger.info(f"Number of features: {len(feature_cols)}")
    
    # Print first 20 features
    logger.info(f"First 20 features: {feature_cols[:20]}")
    
    return target_col, feature_cols


def handle_missing_data(df: pd.DataFrame, target_col: str, 
                        feature_cols: list, max_missing_pct: float = 0.5) -> pd.DataFrame:
    """
    Handle missing values in panel data.
    
    Strategy:
    1. Drop rows with missing target
    2. Drop features with >50% missing
    3. Forward fill remaining missing values (within each stock)
    4. Fill remaining with cross-sectional median
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel dataframe
    target_col : str
        Target column name
    feature_cols : list
        List of feature columns
    max_missing_pct : float
        Maximum allowed missing percentage for features
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe
    """
    logger.info("Handling missing data")
    
    initial_shape = df.shape
    
    # 1. Drop rows with missing target
    df = df[df[target_col].notna()].copy()
    logger.info(f"After dropping missing targets: {df.shape} (dropped {initial_shape[0] - df.shape[0]:,} rows)")
    
    # 2. Check missing percentages for features
    missing_pct = df[feature_cols].isnull().sum() / len(df)
    high_missing = missing_pct[missing_pct > max_missing_pct]
    
    if len(high_missing) > 0:
        logger.info(f"Dropping {len(high_missing)} features with >{max_missing_pct*100}% missing:")
        logger.info(f"{high_missing.to_dict()}")
        feature_cols = [col for col in feature_cols if col not in high_missing.index]
    
    # 3. Forward fill within each stock (up to 3 periods)
    logger.info("Forward filling missing values within each stock...")
    for col in tqdm(feature_cols, desc="Forward filling"):
        df[col] = df.groupby(level='permno')[col].fillna(method='ffill', limit=3)
    
    # 4. Fill remaining with cross-sectional median
    logger.info("Filling remaining missing with cross-sectional median...")
    for col in tqdm(feature_cols, desc="Median filling"):
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    
    # Verify no missing values remain in features
    remaining_missing = df[feature_cols].isnull().sum().sum()
    logger.info(f"Remaining missing values in features: {remaining_missing}")
    
    return df


def winsorize_data(df: pd.DataFrame, feature_cols: list, 
                   lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Winsorize feature values to handle outliers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel dataframe
    feature_cols : list
        List of feature columns
    lower : float
        Lower percentile
    upper : float
        Upper percentile
    
    Returns
    -------
    pd.DataFrame
        Winsorized dataframe
    """
    logger.info(f"Winsorizing features at {lower*100}% and {upper*100}% percentiles")
    
    for col in tqdm(feature_cols, desc="Winsorizing"):
        df[col] = winsorize_series(df[col], lower=lower, upper=upper)
    
    return df


def create_train_test_split(df: pd.DataFrame, train_end: str = TRAIN_END_DATE,
                            test_start: str = TEST_START_DATE,
                            test_end: str = TEST_END_DATE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test split based on dates.
    
    Train: All data up to and including train_end
    Test: Data from test_start to test_end (OOS period)
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel dataframe
    train_end : str
        End date for training period
    test_start : str
        Start date for test period
    test_end : str
        End date for test period
    
    Returns
    -------
    tuple
        (train_df, test_df)
    """
    logger.info("Creating train/test split")
    
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end)
    
    # Get date index
    dates = df.index.get_level_values('date')
    
    # Create splits
    train_df = df[dates <= train_end_dt].copy()
    test_df = df[(dates >= test_start_dt) & (dates <= test_end_dt)].copy()
    
    logger.info(f"Train period: {train_df.index.get_level_values('date').min()} to "
               f"{train_df.index.get_level_values('date').max()}")
    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Train stocks: {train_df.index.get_level_values('permno').nunique():,}")
    logger.info(f"Train months: {train_df.index.get_level_values('date').nunique()}")
    
    logger.info(f"\nTest period: {test_df.index.get_level_values('date').min()} to "
               f"{test_df.index.get_level_values('date').max()}")
    logger.info(f"Test shape: {test_df.shape}")
    logger.info(f"Test stocks: {test_df.index.get_level_values('permno').nunique():,}")
    logger.info(f"Test months: {test_df.index.get_level_values('date').nunique()}")
    
    return train_df, test_df


def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       target_col: str, feature_cols: list) -> None:
    """
    Save processed data and metadata.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    target_col : str
        Target column name
    feature_cols : list
        List of feature columns
    """
    logger.info("Saving processed data")
    
    ensure_dir(DATA_DIR)
    
    # Save dataframes
    train_path = DATA_DIR / "train_data.parquet"
    test_path = DATA_DIR / "test_data.parquet"
    
    train_df.to_parquet(train_path, compression='snappy')
    test_df.to_parquet(test_path, compression='snappy')
    
    logger.info(f"Saved training data to {train_path}")
    logger.info(f"Saved test data to {test_path}")
    
    # Save metadata
    metadata = {
        'target_col': target_col,
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'train_shape': train_df.shape,
        'test_shape': test_df.shape,
        'train_date_range': (
            str(train_df.index.get_level_values('date').min()),
            str(train_df.index.get_level_values('date').max())
        ),
        'test_date_range': (
            str(test_df.index.get_level_values('date').min()),
            str(test_df.index.get_level_values('date').max())
        )
    }
    
    import json
    metadata_path = DATA_DIR / "data_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_path}")


def main() -> None:
    """Main data preparation pipeline."""
    logger.info("="*80)
    logger.info("Data Preparation Pipeline")
    logger.info("="*80)
    
    # 1. Load raw data
    df = load_raw_data(CSV_FILE)
    
    # 2. Parse dates
    df = parse_dates(df)
    
    # 3. Create panel structure
    df = create_panel_structure(df)
    
    # 4. Identify columns
    target_col, feature_cols = identify_columns(df)
    
    # Print initial statistics
    print_summary_statistics(df, "Raw Data")
    
    # 5. Handle missing data
    df = handle_missing_data(df, target_col, feature_cols)
    
    # 6. Winsorize to handle outliers
    df = winsorize_data(df, feature_cols)
    
    # Print statistics after cleaning
    print_summary_statistics(df, "Cleaned Data")
    
    # 7. Create train/test split
    train_df, test_df = create_train_test_split(df)
    
    # 8. Save processed data
    save_processed_data(train_df, test_df, target_col, feature_cols)
    
    logger.info("="*80)
    logger.info("Data preparation completed successfully!")
    logger.info("="*80)
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Target variable: {target_col}")
    logger.info(f"Number of features: {len(feature_cols)}")
    logger.info(f"Training samples: {len(train_df):,}")
    logger.info(f"Test samples: {len(test_df):,}")
    logger.info(f"Training period: {train_df.index.get_level_values('date').min()} to "
               f"{train_df.index.get_level_values('date').max()}")
    logger.info(f"Test period: {test_df.index.get_level_values('date').min()} to "
               f"{test_df.index.get_level_values('date').max()}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
