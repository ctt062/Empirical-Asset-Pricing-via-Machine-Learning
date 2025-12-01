"""
Gradient Boosted Regression Trees (GBRT) Model using LightGBM.

This is the main implementation replicating the GBRT results from 
Gu, Kelly, and Xiu (2020). GBRT was identified as one of the top-performing
methods, nearly matching neural networks while being more interpretable.

Expected performance (from paper):
- Monthly OOS R² ≈ 0.33-0.40%
- Bottom-up S&P 500 R² ≈ 1.5%
- Long-short decile Sharpe ≈ 1.35 (VW) / 2.2-2.4 (EW)

Author: Replication of Gu, Kelly, and Xiu (2020)
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    setup_logging,
    ensure_dir,
    get_project_root,
    calculate_r_squared,
    RANDOM_STATE
)

# Import configuration
from config import DATA_PROCESSED_DIR, RESULTS_DIR
MODELS_DIR = RESULTS_DIR / "models"

# LightGBM hyperparameters (optimized for stronger predictions)
DEFAULT_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,  # Slower learning for better generalization
    'num_leaves': 32,  # Reduced to prevent overfitting
    'max_depth': 5,  # Shallower trees
    'min_child_samples': 100,  # More samples per leaf
    'subsample': 0.7,  # More aggressive subsampling
    'subsample_freq': 1,
    'colsample_bytree': 0.7,  # Feature subsampling
    'reg_alpha': 0.5,  # L1 regularization
    'reg_lambda': 0.5,  # L2 regularization
    'min_split_gain': 0.01,  # Minimum gain to split
    'random_state': RANDOM_STATE,
    'verbose': -1,
    'n_jobs': -1,
    'force_col_wise': True  # Better for wide datasets
}

logger = setup_logging()


def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                         n_splits: int = 5) -> Dict:
    """
    Tune LightGBM hyperparameters using time-series cross-validation.
    
    This performs a grid search over key parameters:
    - learning_rate: [0.01, 0.05, 0.1]
    - num_leaves: [32, 64, 128]
    - max_depth: [4, 6, 8]
    - subsample: [0.7, 0.8, 0.9]
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training target
    n_splits : int
        Number of CV splits
    
    Returns
    -------
    dict
        Best hyperparameters
    """
    logger.info("Tuning hyperparameters with time-series CV")
    
    # Parameter grid (optimized for better generalization)
    param_grid = {
        'learning_rate': [0.01, 0.03, 0.05],
        'num_leaves': [16, 32, 64],
        'max_depth': [4, 5, 6],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
    }
    
    best_score = float('inf')
    best_params = DEFAULT_PARAMS.copy()
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Grid search
    from itertools import product
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in product(*param_grid.values())]
    
    logger.info(f"Testing {len(param_combinations)} parameter combinations")
    
    for params in tqdm(param_combinations, desc="Hyperparameter tuning"):
        scores = []
        
        # Combine with default params
        test_params = DEFAULT_PARAMS.copy()
        test_params.update(params)
        
        # Cross-validation
        for train_idx, val_idx in tscv.split(X_train):
            X_tr = X_train[train_idx]
            y_tr = y_train[train_idx]
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]
            
            # Train model
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                test_params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
            )
            
            # Predict and score
            y_pred = model.predict(X_val)
            mse = np.mean((y_val - y_pred) ** 2)
            scores.append(mse)
        
        # Average CV score
        mean_score = np.mean(scores)
        
        if mean_score < best_score:
            best_score = mean_score
            best_params = test_params.copy()
            logger.info(f"New best CV MSE: {best_score:.6f} with params: {params}")
    
    logger.info(f"Best hyperparameters found:")
    for key, val in best_params.items():
        if key in param_grid:
            logger.info(f"  {key}: {val}")
    
    return best_params


def train_gbrt_expanding_window(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                feature_cols: list, target_col: str,
                                params: Optional[Dict] = None,
                                tune_params: bool = False,
                                save_models: bool = True) -> pd.DataFrame:
    """
    Train GBRT with expanding window and generate OOS predictions.
    
    For each test month:
    1. Use all training data up to that month
    2. Split into train/validation (last 5 years as validation)
    3. Train LightGBM with early stopping
    4. Predict for that month
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    feature_cols : list
        List of feature columns
    target_col : str
        Target column name
    params : dict, optional
        LightGBM parameters. If None, uses DEFAULT_PARAMS.
    tune_params : bool
        Whether to tune hyperparameters (expensive!)
    save_models : bool
        Whether to save trained models
    
    Returns
    -------
    pd.DataFrame
        Test data with predictions
    """
    logger.info("Training GBRT with expanding window")
    logger.info(f"Features: {len(feature_cols)}")
    
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    # Get unique test dates
    test_dates = test_df.index.get_level_values('date').unique().sort_values()
    logger.info(f"Number of test months: {len(test_dates)}")
    
    # Store predictions
    predictions = []
    
    # Create models directory
    if save_models:
        ensure_dir(MODELS_DIR)
    
    # Hyperparameter tuning (one-time on first window)
    if tune_params:
        logger.info("Performing hyperparameter tuning on initial training data...")
        X_tune = train_df[feature_cols].values
        y_tune = train_df[target_col].values
        params = tune_hyperparameters(X_tune, y_tune)
    
    # Track time for progress estimation
    import time
    start_time = time.time()
    
    # Iterate through test dates with expanding window
    for i, test_date in enumerate(tqdm(test_dates, desc="GBRT expanding window")):
        # Get training data up to (but not including) test date
        train_dates = train_df.index.get_level_values('date')
        train_window = train_df[train_dates < test_date].copy()
        
        # Get test data for this date
        test_window = test_df[test_df.index.get_level_values('date') == test_date].copy()
        
        if len(train_window) < 1000:  # Need minimum training samples
            continue
        
        if len(test_window) == 0:
            continue
        
        # Create validation set (last 5 years of training data)
        val_start_date = test_date - pd.DateOffset(years=5)
        val_mask = (train_window.index.get_level_values('date') >= val_start_date)
        
        train_split = train_window[~val_mask].copy()
        val_split = train_window[val_mask].copy()
        
        # Skip if validation set is too small
        if len(val_split) < 100 or len(train_split) < 100:
            # Fall back to simple 80/20 split
            train_size = int(0.8 * len(train_window))
            train_split = train_window.iloc[:train_size].copy()
            val_split = train_window.iloc[train_size:].copy()
            
            # If still too small, skip this month
            if len(val_split) < 100 or len(train_split) < 100:
                continue
        
        # Prepare data
        X_train = train_split[feature_cols].values
        y_train = train_split[target_col].values
        
        X_val = val_split[feature_cols].values
        y_val = val_split[target_col].values
        
        X_test = test_window[feature_cols].values
        y_test = test_window[target_col].values
        
        # Handle any remaining NaNs or infinities
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_cols)
        
        # Train model with early stopping
        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(0)  # Suppress output
            ]
        )
        
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
        
        # Save model periodically (e.g., every 12 months)
        if save_models and (i % 12 == 0 or i == len(test_dates) - 1):
            model_path = MODELS_DIR / f"gbrt_model_{test_date.strftime('%Y%m')}.txt"
            model.save_model(str(model_path))
            
        # Log progress every 24 months
        if (i + 1) % 24 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(test_dates) - i - 1)
            logger.info(f"Processed {i+1}/{len(test_dates)} months. "
                       f"Est. time remaining: {remaining/60:.1f} minutes")
    
    # Create predictions dataframe
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.set_index(['permno', 'date'])
    
    logger.info(f"Generated predictions for {len(pred_df):,} observations")
    
    # Calculate OOS R² for sample months to track quality
    sample_r2 = calculate_r_squared(pred_df['y_true'].values, pred_df['y_pred'].values)
    logger.info(f"Overall OOS R²: {sample_r2*100:.4f}%")
    
    return pred_df


def train_single_model(train_df: pd.DataFrame, feature_cols: list, target_col: str,
                      params: Optional[Dict] = None) -> lgb.Booster:
    """
    Train a single GBRT model on all training data.
    
    This is useful for feature importance analysis.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    feature_cols : list
        List of feature columns
    target_col : str
        Target column name
    params : dict, optional
        LightGBM parameters
    
    Returns
    -------
    lgb.Booster
        Trained model
    """
    logger.info("Training single GBRT model on all training data")
    
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    
    # Handle NaNs
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Use last 20% as validation
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
    
    # Create datasets
    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_cols)
    
    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100)
        ]
    )
    
    logger.info(f"Model trained. Best iteration: {model.best_iteration}")
    
    return model


def save_gbrt_predictions(pred_df: pd.DataFrame) -> None:
    """
    Save GBRT predictions.
    
    Parameters
    ----------
    pred_df : pd.DataFrame
        Predictions dataframe
    """
    logger.info("Saving GBRT predictions")
    
    ensure_dir(RESULTS_DIR / "predictions")
    
    pred_path = RESULTS_DIR / "predictions" / "gbrt_predictions.parquet"
    pred_df.to_parquet(pred_path)
    logger.info(f"Saved predictions to {pred_path}")
    
    # Also save as CSV for easier inspection
    csv_path = RESULTS_DIR / "predictions" / "gbrt_predictions.csv"
    pred_df.head(1000).to_csv(csv_path)
    logger.info(f"Saved first 1000 predictions to {csv_path}")


def main() -> None:
    """Main GBRT training pipeline."""
    logger.info("="*80)
    logger.info("Gradient Boosted Regression Trees (GBRT) Model")
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
    logger.info(f"Training samples: {len(train_df):,}")
    logger.info(f"Test samples: {len(test_df):,}")
    
    # Option: Tune hyperparameters (expensive - set to False for quick run)
    tune_hyperparams = False  # Set to True for full replication
    
    # Train GBRT with expanding window
    logger.info("\nStarting GBRT training with expanding window...")
    logger.info("This may take 30-60 minutes depending on your hardware...")
    
    pred_df = train_gbrt_expanding_window(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        params=None,  # Use default params
        tune_params=tune_hyperparams,
        save_models=True
    )
    
    # Save predictions
    save_gbrt_predictions(pred_df)
    
    # Train single model for feature importance
    logger.info("\nTraining single model for feature importance analysis...")
    model = train_single_model(train_df, feature_cols, target_col)
    
    # Save single model
    ensure_dir(MODELS_DIR)
    model_path = MODELS_DIR / "gbrt_full_model.txt"
    model.save_model(str(model_path))
    logger.info(f"Saved full model to {model_path}")
    
    # Quick evaluation
    overall_r2 = calculate_r_squared(pred_df['y_true'].values, pred_df['y_pred'].values)
    
    logger.info("="*80)
    logger.info("GBRT model training completed successfully!")
    logger.info("="*80)
    logger.info(f"\nOverall OOS R²: {overall_r2*100:.4f}%")
    logger.info(f"Target from paper: 0.33-0.40%")
    logger.info("\nRun 04_evaluation.py for detailed performance analysis")
    logger.info("Run 05_feature_importance.py for interpretability analysis")
    logger.info("="*80)


if __name__ == "__main__":
    main()
