"""
Elastic Net Regularized Regression Model

Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization:
    Loss = MSE + alpha * (l1_ratio * ||w||_1 + (1 - l1_ratio) * ||w||_2^2)

Advantages:
- Feature selection via L1 penalty
- Handles multicollinearity via L2 penalty
- Prevents overfitting with high-dimensional data
"""

import logging
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler
from .base_model import AssetPricingModel

logger = logging.getLogger(__name__)


class ElasticNetModel(AssetPricingModel):
    """
    Elastic Net Regularized Regression for Return Prediction
    
    Uses cross-validation to select optimal regularization parameters.
    """
    
    def __init__(self, 
                 alpha: float = None,
                 l1_ratio: float = 0.5,
                 use_cv: bool = True,
                 n_alphas: int = 100,
                 cv_folds: int = 5,
                 max_iter: int = 10000,
                 eps: float = 1e-4,
                 **kwargs):
        """
        Initialize Elastic Net model.
        
        Args:
            alpha: Regularization strength (if None, use CV to select)
            l1_ratio: Mix of L1 and L2 (0=Ridge, 1=Lasso, 0.5=equal mix)
            use_cv: Use cross-validation to select alpha
            n_alphas: Number of alphas to try in CV
            cv_folds: Number of CV folds
            max_iter: Maximum iterations for convergence
            eps: Length of the path (smaller = less regularization options)
        """
        super().__init__("Elastic Net", 
                        alpha=alpha, 
                        l1_ratio=l1_ratio,
                        use_cv=use_cv,
                        n_alphas=n_alphas,
                        cv_folds=cv_folds,
                        max_iter=max_iter,
                        eps=eps,
                        **kwargs)
        
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.use_cv = use_cv
        self.n_alphas = n_alphas
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.eps = eps
        
        self.scaler = StandardScaler()
        self.feature_names = None
        self.n_features_used = 0
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train Elastic Net model with optional cross-validation.
        
        Args:
            X_train: Training features (firm characteristics)
            y_train: Training target (excess returns)
        """
        logger.info(f"Training Elastic Net model (use_cv={self.use_cv})...")
        
        # Validate data
        self.validate_data(X_train, y_train)
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Remove non-numeric columns (date, permno, etc.)
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_numeric = X_train[numeric_cols].copy()
        
        # Handle missing values
        X_numeric = X_numeric.fillna(X_numeric.mean())
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        # Train model
        if self.use_cv:
            logger.info(f"Using CV with {self.n_alphas} alphas and {self.cv_folds} folds...")
            # Use multiple l1_ratios to find best mix of L1/L2
            l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9] if self.l1_ratio == 0.5 else [self.l1_ratio]
            self.model = ElasticNetCV(
                l1_ratio=l1_ratios,
                n_alphas=self.n_alphas,
                cv=self.cv_folds,
                max_iter=self.max_iter,
                eps=self.eps,  # Controls regularization path
                random_state=42,
                n_jobs=-1,
                selection='random'  # Faster convergence
            )
        else:
            alpha_value = self.alpha if self.alpha is not None else 0.001  # Lower default alpha
            logger.info(f"Using fixed alpha={alpha_value}")
            self.model = ElasticNet(
                alpha=alpha_value,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                random_state=42,
                selection='random'
            )
        
        # Fit model
        self.model.fit(X_scaled, y_train)
        
        # Count non-zero coefficients (selected features)
        self.n_features_used = np.sum(self.model.coef_ != 0)
        
        if self.use_cv:
            logger.info(f"Optimal alpha selected: {self.model.alpha_:.6f}")
        
        logger.info(f"Features selected: {self.n_features_used}/{len(numeric_cols)}")
        logger.info(f"Training R²: {self.model.score(X_scaled, y_train):.4f}")
        
        self.is_trained = True
        self.numeric_cols = numeric_cols  # Store for prediction
        
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict returns using trained Elastic Net model.
        
        Args:
            X_test: Test features
            
        Returns:
            Array of predicted returns
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Select numeric columns used in training
        X_numeric = X_test[self.numeric_cols].copy()
        
        # Handle missing values
        X_numeric = X_numeric.fillna(X_numeric.mean())
        
        # Standardize features using training scaler
        X_scaled = self.scaler.transform(X_numeric)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata and statistics."""
        info = {
            'model_name': self.model_name,
            'l1_ratio': self.l1_ratio,
            'use_cv': self.use_cv,
            'n_features_total': len(self.feature_names) if self.feature_names else 0,
            'n_features_used': self.n_features_used,
            'is_trained': self.is_trained
        }
        
        if self.is_trained:
            info['alpha'] = self.model.alpha_ if self.use_cv else self.alpha
            info['intercept'] = self.model.intercept_
        
        return info
    
    def get_feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        """
        Get most important features (largest absolute coefficients).
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and coefficients
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Get non-zero coefficients
        coef_df = pd.DataFrame({
            'feature': self.numeric_cols,
            'coefficient': self.model.coef_
        })
        
        # Filter non-zero and sort by absolute value
        coef_df = coef_df[coef_df['coefficient'] != 0].copy()
        coef_df['abs_coefficient'] = np.abs(coef_df['coefficient'])
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        
        return coef_df.head(top_n)[['feature', 'coefficient', 'abs_coefficient']]
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'numeric_cols': self.numeric_cols,
            'params': self.params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.numeric_cols = model_data['numeric_cols']
        self.params = model_data['params']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
