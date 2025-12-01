"""
Fama-French 3-Factor Model (Characteristic-Based)

Implementation of a characteristic-based Fama-French style model.
Instead of estimating factor loadings (which requires long time series),
this version directly uses size and value characteristics to predict returns.

This approach:
1. Ranks stocks by size (small outperforms) and value (high B/M outperforms)
2. Combines rankings into a composite score
3. Predicts higher returns for small-cap value stocks

This is more practical for cross-sectional prediction and aligns with the
economic intuition behind the Fama-French factors.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.linear_model import LinearRegression
from .base_model import AssetPricingModel

logger = logging.getLogger(__name__)


class FamaFrenchModel(AssetPricingModel):
    """
    Fama-French Characteristic-Based Model
    
    Uses size (market cap) and value (book-to-market) characteristics
    directly to predict returns, based on the empirical regularities
    that small-cap and value stocks tend to outperform.
    
    This is a simpler but more robust approach than estimating factor
    loadings, especially for cross-sectional prediction.
    """
    
    def __init__(self, lookback_months: int = 60, **kwargs):
        """
        Initialize Fama-French model.
        
        Args:
            lookback_months: Number of months for estimating factor loadings (default: 60)
        """
        super().__init__("Fama-French-3F", lookback_months=lookback_months, **kwargs)
        self.lookback_months = lookback_months
        self.size_premium = None  # Estimated size premium
        self.value_premium = None  # Estimated value premium
        self.momentum_premium = None  # Estimated momentum premium
        
    def estimate_factor_premia(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate factor premia from historical data using portfolio sorts.
        
        Args:
            data: DataFrame with columns [date, permno, ret_exc, mvel1, bm]
            
        Returns:
            Dictionary with SMB and HML premia
        """
        logger.info("Estimating factor premia from historical returns...")
        
        smb_returns = []
        hml_returns = []
        
        for date in data['date'].unique():
            month_data = data[data['date'] == date].copy()
            
            if len(month_data) < 20:  # Need minimum stocks
                continue
            
            # Size factor: Small - Big
            size_median = month_data['mvel1'].median()
            small_ret = month_data[month_data['mvel1'] <= size_median]['ret_exc'].mean()
            big_ret = month_data[month_data['mvel1'] > size_median]['ret_exc'].mean()
            
            if not pd.isna(small_ret) and not pd.isna(big_ret):
                smb_returns.append(small_ret - big_ret)
            
            # Value factor: High B/M - Low B/M
            bm_median = month_data['bm'].median()
            high_bm_ret = month_data[month_data['bm'] >= bm_median]['ret_exc'].mean()
            low_bm_ret = month_data[month_data['bm'] < bm_median]['ret_exc'].mean()
            
            if not pd.isna(high_bm_ret) and not pd.isna(low_bm_ret):
                hml_returns.append(high_bm_ret - low_bm_ret)
        
        smb_premium = np.mean(smb_returns) if smb_returns else 0.002
        hml_premium = np.mean(hml_returns) if hml_returns else 0.002
        
        logger.info(f"Estimated SMB premium: {smb_premium*100:.3f}% monthly")
        logger.info(f"Estimated HML premium: {hml_premium*100:.3f}% monthly")
        
        return {'SMB': smb_premium, 'HML': hml_premium}
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train Fama-French model by estimating factor premia.
        
        Args:
            X_train: Features including [date, permno, mvel1, bm]
            y_train: Excess returns
        """
        logger.info("Training Fama-French Characteristic-Based model...")
        
        # Combine features and target
        train_data = X_train.copy()
        train_data['ret_exc'] = y_train.values
        
        # Estimate factor premia from historical data
        premia = self.estimate_factor_premia(train_data)
        self.size_premium = premia['SMB']
        self.value_premium = premia['HML']
        
        # Also estimate momentum premium if available
        if 'mom12m' in X_train.columns:
            mom_returns = []
            for date in train_data['date'].unique():
                month_data = train_data[train_data['date'] == date].copy()
                if len(month_data) < 20:
                    continue
                mom_median = month_data['mom12m'].median()
                high_mom = month_data[month_data['mom12m'] >= mom_median]['ret_exc'].mean()
                low_mom = month_data[month_data['mom12m'] < mom_median]['ret_exc'].mean()
                if not pd.isna(high_mom) and not pd.isna(low_mom):
                    mom_returns.append(high_mom - low_mom)
            self.momentum_premium = np.mean(mom_returns) if mom_returns else 0.005
            logger.info(f"Estimated momentum premium: {self.momentum_premium*100:.3f}% monthly")
        else:
            self.momentum_premium = 0.005  # Default momentum premium
        
        self.is_trained = True
        logger.info("Fama-French model training completed")
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict returns using characteristic-based Fama-French approach.
        
        Stocks are ranked by:
        1. Size (smaller = higher expected return)
        2. Value (higher B/M = higher expected return)
        3. Momentum (higher past returns = higher expected return)
        
        Args:
            X_test: Test features with [permno, mvel1, bm]
            
        Returns:
            Array of predicted returns
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = np.zeros(len(X_test))
        
        # Rank by size (small = high rank = higher predicted return)
        if 'mvel1' in X_test.columns:
            size_vals = X_test['mvel1'].values.astype(float)
            # Normalize and invert (smaller = higher score)
            size_norm = (size_vals - np.nanmean(size_vals)) / (np.nanstd(size_vals) + 1e-10)
            size_score = -size_norm  # Negative because small is good
            predictions += size_score * abs(self.size_premium) * 3  # Scale factor
        
        # Rank by value (high B/M = high rank = higher predicted return)
        if 'bm' in X_test.columns:
            bm_vals = X_test['bm'].values.astype(float)
            bm_norm = (bm_vals - np.nanmean(bm_vals)) / (np.nanstd(bm_vals) + 1e-10)
            predictions += bm_norm * abs(self.value_premium) * 3  # Scale factor
        
        # Rank by momentum (high momentum = higher predicted return)
        if 'mom12m' in X_test.columns:
            mom_vals = X_test['mom12m'].values.astype(float)
            mom_norm = (mom_vals - np.nanmean(mom_vals)) / (np.nanstd(mom_vals) + 1e-10)
            predictions += mom_norm * abs(self.momentum_premium) * 2  # Scale factor
        
        # Handle NaN values
        predictions = np.nan_to_num(predictions, nan=0.0)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            'model_name': self.model_name,
            'lookback_months': self.lookback_months,
            'size_premium': self.size_premium,
            'value_premium': self.value_premium,
            'momentum_premium': self.momentum_premium,
            'is_trained': self.is_trained
        }
    
    def get_factor_summary(self) -> pd.DataFrame:
        """Get summary of estimated factor premia."""
        if not self.is_trained:
            return pd.DataFrame()
        
        summary = pd.DataFrame({
            'Factor': ['SMB (Size)', 'HML (Value)', 'MOM (Momentum)'],
            'Monthly Premium (%)': [
                self.size_premium * 100 if self.size_premium else 0,
                self.value_premium * 100 if self.value_premium else 0,
                self.momentum_premium * 100 if self.momentum_premium else 0
            ],
            'Annualized Premium (%)': [
                self.size_premium * 12 * 100 if self.size_premium else 0,
                self.value_premium * 12 * 100 if self.value_premium else 0,
                self.momentum_premium * 12 * 100 if self.momentum_premium else 0
            ]
        })
        
        return summary
