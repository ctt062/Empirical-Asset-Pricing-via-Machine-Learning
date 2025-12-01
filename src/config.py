"""
Configuration settings for Empirical Asset Pricing project.

This module contains all configurable parameters including:
- Data paths and date ranges
- Model hyperparameters
- Transaction cost assumptions
- Evaluation settings
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# ============================================================================
# DATA SETTINGS
# ============================================================================
# Training period: 1957-1995 (38 years for initial training)
TRAIN_START_DATE = "1957-01-01"
TRAIN_END_DATE = "1995-12-31"

# Test period: 1996-2016 (21 years = 252 months)
# Reduced from 1986-2016 for faster computation
TEST_START_DATE = "1996-01-01"
TEST_END_DATE = "2016-12-31"

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# TRANSACTION COST SETTINGS
# ============================================================================
# Realistic transaction costs for institutional investors
# Based on academic literature (Frazzini et al. 2018, etc.)

# One-way transaction cost (5 basis points = 0.05%)
TRANSACTION_COST_BPS = 5
TRANSACTION_COST = TRANSACTION_COST_BPS / 10000  # 0.0005

# Round-trip cost = 2 * one-way cost (10 bps)
ROUND_TRIP_COST = 2 * TRANSACTION_COST

# Notes on transaction cost assumptions:
# - 5 bps is reasonable for liquid large-cap stocks
# - Includes bid-ask spread, market impact, commissions
# - More realistic than zero-cost assumption in original paper
# - Can be adjusted based on:
#   * Small-cap vs large-cap: 10-50 bps for small-cap
#   * Market conditions: higher during crises
#   * Trading style: higher for high-frequency strategies

# ============================================================================
# PORTFOLIO CONSTRUCTION SETTINGS
# ============================================================================
N_PORTFOLIOS = 10  # Number of decile portfolios
LONG_PORTFOLIO = 10  # Top decile (highest predictions)
SHORT_PORTFOLIO = 1  # Bottom decile (lowest predictions)

# Rebalancing frequency
REBALANCE_FREQUENCY = "monthly"  # Options: 'monthly', 'quarterly', 'annual'

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# GBRT Hyperparameters (LightGBM) - Optimized for Sharpe > 1
GBRT_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 5,
    'num_leaves': 32,
    'min_child_samples': 100,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'random_state': RANDOM_SEED,
    'verbose': -1,
    'n_jobs': -1
}

# Elastic Net Hyperparameters
ELASTIC_NET_PARAMS = {
    'l1_ratio': [0.1, 0.5, 0.9],  # Balance between L1 and L2
    'n_alphas': 100,
    'cv': 5,
    'random_state': RANDOM_SEED,
    'max_iter': 10000
}

# OLS-3 Settings
OLS3_FEATURES = ['mvel1', 'mom12m', 'bm']  # Size, momentum, value

# Fama-French Settings
FF_FACTORS = ['MKT', 'SMB', 'HML']  # Market, size, value factors

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Performance metrics
ANNUALIZATION_FACTOR = 12  # Monthly returns
RISK_FREE_RATE = 0.0  # Already using excess returns

# Plotting settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# Table formatting
TABLE_FLOAT_FORMAT = '%.4f'
LATEX_ESCAPE = False

# ============================================================================
# FEATURE IMPORTANCE SETTINGS
# ============================================================================
TOP_N_FEATURES = 50  # Number of top features to analyze
SHAP_SAMPLE_SIZE = 10000  # Sample size for SHAP analysis (full dataset = slow)

# ============================================================================
# LOGGING SETTINGS
# ============================================================================
LOG_LEVEL = 'INFO'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# COMPUTATIONAL SETTINGS
# ============================================================================
N_JOBS = -1  # Use all available cores (-1)
USE_GPU = False  # Set to True if CUDA-enabled GPU available

# Memory optimization
CHUNK_SIZE = 10000  # For processing large datasets in chunks
LOW_MEMORY_MODE = False  # Set True if RAM < 16GB
