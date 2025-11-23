"""
Unified Model Training Script

Trains all models with expanding window approach:
1. OLS-3 (existing baseline)
2. GBRT (existing, keep unchanged)
3. Elastic Net (new)
4. Fama-French 3-Factor (new)

Each model is trained on data up to time t and predicts returns at t+1.
"""

import logging
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models.elastic_net import ElasticNetModel
from models.fama_french import FamaFrenchModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """Load training and test data."""
    logger.info("Loading data...")
    
    train_data = pd.read_parquet('data/train_data.parquet')
    test_data = pd.read_parquet('data/test_data.parquet')
    
    logger.info(f"Train: {len(train_data):,} samples")
    logger.info(f"Test: {len(test_data):,} samples")
    logger.info(f"Features: {len(train_data.columns)}")
    
    return train_data, test_data


def train_elastic_net(train_data, test_data):
    """
    Train Elastic Net model with expanding window.
    
    Returns predictions for the test period.
    """
    logger.info("="*80)
    logger.info("TRAINING ELASTIC NET MODEL")
    logger.info("="*80)
    
    # Reset index to access date as column
    train_data = train_data.reset_index()
    test_data = test_data.reset_index()
    
    # Get unique test dates
    test_dates = sorted(test_data['date'].unique())
    logger.info(f"Predicting {len(test_dates)} months from {test_dates[0]} to {test_dates[-1]}")
    
    # Prepare columns
    feature_cols = [col for col in train_data.columns 
                   if col not in ['date', 'permno', 'ret_exc', 'sic2']]
    
    all_predictions = []
    
    # Expanding window training
    for i, test_date in enumerate(test_dates, 1):
        # Training data: all months before test_date
        train_window = train_data[train_data['date'] < test_date].copy()
        test_window = test_data[test_data['date'] == test_date].copy()
        
        if len(train_window) < 10000:  # Need minimum training data
            logger.warning(f"Skipping {test_date}: insufficient training data")
            continue
        
        # Prepare features and target
        X_train = train_window[feature_cols]
        y_train = train_window['ret_exc']
        X_test = test_window[feature_cols]
        
        # Initialize and train model
        model = ElasticNetModel(
            l1_ratio=0.5,
            use_cv=True,
            n_alphas=50,
            cv_folds=3,
            max_iter=5000
        )
        
        logger.info(f"[{i}/{len(test_dates)}] Training on {len(train_window):,} samples, "
                   f"predicting {len(test_window):,} stocks for {test_date}")
        
        model.train(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_test)
        
        # Store predictions
        pred_df = pd.DataFrame({
            'date': test_date,
            'permno': test_window['permno'].values,
            'prediction': predictions,
            'actual': test_window['ret_exc'].values
        })
        
        all_predictions.append(pred_df)
        
        # Log progress
        if i % 12 == 0:
            logger.info(f"Progress: {i}/{len(test_dates)} months completed")
    
    # Combine all predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Save predictions
    os.makedirs('results/predictions', exist_ok=True)
    output_path = 'results/predictions/elastic_net_predictions.parquet'
    predictions_df.to_parquet(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")
    
    # Calculate overall OOS R²
    ss_res = ((predictions_df['actual'] - predictions_df['prediction']) ** 2).sum()
    ss_tot = ((predictions_df['actual'] - predictions_df['actual'].mean()) ** 2).sum()
    oos_r2 = 1 - (ss_res / ss_tot)
    
    logger.info(f"Elastic Net Overall OOS R²: {oos_r2*100:.4f}%")
    logger.info("="*80)
    
    return predictions_df


def train_fama_french(train_data, test_data):
    """
    Train Fama-French 3-Factor model with expanding window.
    
    Returns predictions for the test period.
    """
    logger.info("="*80)
    logger.info("TRAINING FAMA-FRENCH 3-FACTOR MODEL")
    logger.info("="*80)
    
    # Reset index to access date as column
    train_data = train_data.reset_index()
    test_data = test_data.reset_index()
    
    # Get unique test dates
    test_dates = sorted(test_data['date'].unique())
    logger.info(f"Predicting {len(test_dates)} months from {test_dates[0]} to {test_dates[-1]}")
    
    # Need columns: date, permno, ret_exc, mvel1, bm
    required_cols = ['date', 'permno', 'ret_exc', 'mvel1', 'bm']
    
    all_predictions = []
    
    # Expanding window training
    for i, test_date in enumerate(test_dates, 1):
        # Training data: all months before test_date
        train_window = train_data[train_data['date'] < test_date][required_cols].copy()
        test_window = test_data[test_data['date'] == test_date][required_cols].copy()
        
        if len(train_window) < 10000:
            logger.warning(f"Skipping {test_date}: insufficient training data")
            continue
        
        # Prepare features and target
        X_train = train_window[['date', 'permno', 'mvel1', 'bm']]
        y_train = train_window['ret_exc']
        X_test = test_window[['date', 'permno', 'mvel1', 'bm']]
        
        # Initialize and train model
        model = FamaFrenchModel(lookback_months=60)
        
        logger.info(f"[{i}/{len(test_dates)}] Training on {len(train_window):,} samples, "
                   f"predicting {len(test_window):,} stocks for {test_date}")
        
        model.train(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_test)
        
        # Store predictions
        pred_df = pd.DataFrame({
            'date': test_date,
            'permno': test_window['permno'].values,
            'prediction': predictions,
            'actual': test_window['ret_exc'].values
        })
        
        all_predictions.append(pred_df)
        
        # Log progress
        if i % 12 == 0:
            logger.info(f"Progress: {i}/{len(test_dates)} months completed")
    
    # Combine all predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Save predictions
    os.makedirs('results/predictions', exist_ok=True)
    output_path = 'results/predictions/fama_french_predictions.parquet'
    predictions_df.to_parquet(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")
    
    # Calculate overall OOS R²
    ss_res = ((predictions_df['actual'] - predictions_df['prediction']) ** 2).sum()
    ss_tot = ((predictions_df['actual'] - predictions_df['actual'].mean()) ** 2).sum()
    oos_r2 = 1 - (ss_res / ss_tot)
    
    logger.info(f"Fama-French Overall OOS R²: {oos_r2*100:.4f}%")
    logger.info("="*80)
    
    return predictions_df


def main():
    """Main training pipeline."""
    logger.info("="*80)
    logger.info("UNIFIED MODEL TRAINING PIPELINE")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # Load data
    train_data, test_data = load_data()
    
    # Train Elastic Net
    elastic_net_path = 'results/predictions/elastic_net_predictions.parquet'
    if os.path.exists(elastic_net_path):
        logger.info("\n" + "="*80)
        logger.info("STEP 1: ELASTIC NET (SKIPPING - predictions already exist)")
        logger.info("="*80)
        logger.info(f"Found existing predictions at: {elastic_net_path}")
    else:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: ELASTIC NET")
        logger.info("="*80)
        elastic_net_preds = train_elastic_net(train_data, test_data)
    
    # Train Fama-French
    logger.info("\n" + "="*80)
    logger.info("STEP 2: FAMA-FRENCH 3-FACTOR")
    logger.info("="*80)
    fama_french_preds = train_fama_french(train_data, test_data)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\nPredictions saved:")
    logger.info("  - results/predictions/elastic_net_predictions.parquet")
    logger.info("  - results/predictions/fama_french_predictions.parquet")
    logger.info("\nNote: GBRT and OLS-3 predictions already exist (not retrained)")
    logger.info("\nNext steps:")
    logger.info("  1. Run: python src/06_unified_evaluation.py")
    logger.info("  2. Compare all 4 models")
    logger.info("="*80)


if __name__ == "__main__":
    main()
