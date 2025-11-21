"""
Models Package

Contains implementations of various asset pricing models:
- OLS-3: Polynomial regression benchmark
- GBRT: Gradient Boosted Regression Trees
- Elastic Net: Regularized linear model with L1+L2 penalties
- Fama-French: Traditional 3-factor model
"""

from .base_model import AssetPricingModel
from .fama_french import FamaFrenchModel
from .elastic_net import ElasticNetModel

__all__ = [
    'AssetPricingModel',
    'FamaFrenchModel',
    'ElasticNetModel'
]
