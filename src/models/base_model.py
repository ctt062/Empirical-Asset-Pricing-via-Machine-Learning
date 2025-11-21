"""
Base Model Class for Asset Pricing Models

This module provides an abstract base class that all asset pricing models inherit from.
Ensures consistent interface across different models (OLS-3, GBRT, Elastic Net, Fama-French).
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class AssetPricingModel(ABC):
    """
    Abstract base class for asset pricing models.
    
    All models must implement:
    - train(): Train the model on historical data
    - predict(): Generate return predictions
    - get_model_info(): Return model metadata
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the asset pricing model.
        
        Args:
            model_name: Name of the model (e.g., 'GBRT', 'Elastic Net')
            **kwargs: Model-specific hyperparameters
        """
        self.model_name = model_name
        self.params = kwargs
        self.is_trained = False
        self.model = None
        
        logger.info(f"Initialized {model_name} model with params: {kwargs}")
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model on historical data.
        
        Args:
            X_train: Training features (firm characteristics)
            y_train: Training target (excess returns)
        """
        pass
    
    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generate return predictions.
        
        Args:
            X_test: Test features
            
        Returns:
            Array of predicted returns
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model metadata and parameters.
        
        Returns:
            Dictionary with model information
        """
        pass
    
    def validate_data(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Validate input data for common issues.
        
        Args:
            X: Feature dataframe
            y: Target series (optional)
        """
        # Check for NaN values
        if X.isnull().any().any():
            n_missing = X.isnull().sum().sum()
            logger.warning(f"Found {n_missing} missing values in features")
            
        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            logger.warning("Found infinite values in features")
            
        if y is not None:
            if y.isnull().any():
                logger.warning(f"Found {y.isnull().sum()} missing values in target")
            if np.isinf(y).any():
                logger.warning("Found infinite values in target")
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        raise NotImplementedError("Subclass must implement save_model()")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        raise NotImplementedError("Subclass must implement load_model()")
    
    def __repr__(self) -> str:
        """String representation of the model."""
        param_str = ', '.join([f"{k}={v}" for k, v in self.params.items()])
        return f"{self.model_name}({param_str})"
