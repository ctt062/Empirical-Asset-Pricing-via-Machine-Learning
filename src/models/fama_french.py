"""
Fama-French 3-Factor Model

Implementation of the Fama-French (1993) 3-factor model:
    E[R_i - R_f] = alpha + beta_MKT * (R_m - R_f) + beta_SMB * SMB + beta_HML * HML

Where:
- MKT (Market): Excess return on the market portfolio
- SMB (Small Minus Big): Size factor (small cap - large cap)
- HML (High Minus Low): Value factor (high B/M - low B/M)
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
    Fama-French 3-Factor Model
    
    Estimates factor loadings (betas) using rolling historical returns,
    then predicts returns based on expected factor premia.
    """
    
    def __init__(self, lookback_months: int = 60, **kwargs):
        """
        Initialize Fama-French 3-Factor model.
        
        Args:
            lookback_months: Number of months for estimating factor loadings (default: 60)
        """
        super().__init__("Fama-French-3F", lookback_months=lookback_months, **kwargs)
        self.lookback_months = lookback_months
        self.factor_loadings = {}  # Store betas for each stock
        self.factor_premia = None  # Expected factor returns
        
    def construct_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Construct Fama-French factors from cross-section of stocks.
        
        Args:
            data: DataFrame with columns [date, permno, ret_exc, mvel1, bm]
            
        Returns:
            DataFrame with factors [MKT, SMB, HML] by date
        """
        logger.info("Constructing Fama-French factors...")
        
        factors_list = []
        
        for date in data['date'].unique():
            month_data = data[data['date'] == date].copy()
            
            if len(month_data) < 10:  # Need minimum stocks
                continue
            
            # MKT: Equal-weighted market excess return
            mkt = month_data['ret_exc'].mean()
            
            # Size breakpoint: Median market cap
            size_median = month_data['mvel1'].median()
            
            # B/M breakpoints: 30th and 70th percentiles
            bm_30 = month_data['bm'].quantile(0.30)
            bm_70 = month_data['bm'].quantile(0.70)
            
            # Classify stocks
            month_data['size_group'] = np.where(month_data['mvel1'] <= size_median, 'S', 'B')
            month_data['bm_group'] = pd.cut(month_data['bm'], 
                                           bins=[-np.inf, bm_30, bm_70, np.inf],
                                           labels=['L', 'M', 'H'])
            
            # Calculate portfolio returns
            portfolios = month_data.groupby(['size_group', 'bm_group'])['ret_exc'].mean()
            
            # SMB: Average(Small portfolios) - Average(Big portfolios)
            try:
                small_avg = portfolios.loc['S'].mean()
                big_avg = portfolios.loc['B'].mean()
                smb = small_avg - big_avg
            except:
                smb = 0
            
            # HML: Average(High B/M) - Average(Low B/M)
            try:
                high_avg = month_data[month_data['bm_group'] == 'H']['ret_exc'].mean()
                low_avg = month_data[month_data['bm_group'] == 'L']['ret_exc'].mean()
                hml = high_avg - low_avg
            except:
                hml = 0
            
            factors_list.append({
                'date': date,
                'MKT': mkt,
                'SMB': smb,
                'HML': hml
            })
        
        factors_df = pd.DataFrame(factors_list)
        logger.info(f"Constructed factors for {len(factors_df)} periods")
        
        return factors_df
    
    def estimate_factor_loadings(self, returns: pd.Series, factors: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate factor loadings (betas) for a single stock using OLS.
        
        Args:
            returns: Historical returns for the stock
            factors: Factor returns (MKT, SMB, HML)
            
        Returns:
            Dictionary with beta_MKT, beta_SMB, beta_HML, alpha
        """
        # Align returns and factors
        merged = pd.merge(
            returns.reset_index(),
            factors,
            on='date',
            how='inner'
        )
        
        if len(merged) < 24:  # Need at least 24 months
            return {'beta_MKT': 1.0, 'beta_SMB': 0.0, 'beta_HML': 0.0, 'alpha': 0.0}
        
        # Prepare X (factors) and y (returns)
        X = merged[['MKT', 'SMB', 'HML']].values
        y = merged['ret_exc'].values
        
        # OLS regression
        model = LinearRegression()
        model.fit(X, y)
        
        return {
            'beta_MKT': model.coef_[0],
            'beta_SMB': model.coef_[1],
            'beta_HML': model.coef_[2],
            'alpha': model.intercept_
        }
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train Fama-French model by estimating factor loadings.
        
        Args:
            X_train: Features including [date, permno, mvel1, bm]
            y_train: Excess returns
        """
        logger.info("Training Fama-French 3-Factor model...")
        
        # Combine features and target
        train_data = X_train.copy()
        train_data['ret_exc'] = y_train
        
        # Construct factors
        self.factors = self.construct_factors(train_data)
        
        # Estimate expected factor premia (historical average)
        self.factor_premia = {
            'MKT': self.factors['MKT'].mean(),
            'SMB': self.factors['SMB'].mean(),
            'HML': self.factors['HML'].mean()
        }
        
        logger.info(f"Factor premia - MKT: {self.factor_premia['MKT']:.4f}, "
                   f"SMB: {self.factor_premia['SMB']:.4f}, "
                   f"HML: {self.factor_premia['HML']:.4f}")
        
        # Estimate factor loadings for each stock
        unique_stocks = train_data['permno'].unique()
        logger.info(f"Estimating factor loadings for {len(unique_stocks)} stocks...")
        
        for permno in unique_stocks:
            stock_data = train_data[train_data['permno'] == permno]
            returns = stock_data.set_index('date')['ret_exc']
            
            loadings = self.estimate_factor_loadings(returns, self.factors)
            self.factor_loadings[permno] = loadings
        
        self.is_trained = True
        logger.info("Fama-French model training completed")
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict returns using Fama-French factor model.
        
        Expected return = alpha + beta_MKT * E[MKT] + beta_SMB * E[SMB] + beta_HML * E[HML]
        
        Args:
            X_test: Test features with [permno]
            
        Returns:
            Array of predicted returns
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = []
        
        for idx, row in X_test.iterrows():
            permno = row['permno']
            
            # Get factor loadings (use market beta of 1.0 if not available)
            if permno in self.factor_loadings:
                loadings = self.factor_loadings[permno]
            else:
                loadings = {'beta_MKT': 1.0, 'beta_SMB': 0.0, 'beta_HML': 0.0, 'alpha': 0.0}
            
            # Predict return using factor model
            pred = (loadings['alpha'] +
                   loadings['beta_MKT'] * self.factor_premia['MKT'] +
                   loadings['beta_SMB'] * self.factor_premia['SMB'] +
                   loadings['beta_HML'] * self.factor_premia['HML'])
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            'model_name': self.model_name,
            'lookback_months': self.lookback_months,
            'n_stocks': len(self.factor_loadings),
            'factor_premia': self.factor_premia,
            'is_trained': self.is_trained
        }
    
    def get_factor_summary(self) -> pd.DataFrame:
        """Get summary statistics of factor loadings across all stocks."""
        if not self.factor_loadings:
            return pd.DataFrame()
        
        loadings_df = pd.DataFrame(self.factor_loadings).T
        summary = loadings_df.describe()
        
        return summary
