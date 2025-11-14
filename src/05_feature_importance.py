"""
Feature Importance and Interpretability Analysis.

This script performs comprehensive interpretability analysis for GBRT:
1. Global feature importance (gain, split count)
2. SHAP values for local explanations
3. Dependence plots for top features
4. Summary visualizations

Key findings from Gu et al. (2020):
- Top predictors: momentum, liquidity (volume, turnover), volatility
- Market microstructure variables highly important
- Interactions between features matter

Author: Replication of Gu, Kelly, and Xiu (2020)
"""

import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils import setup_logging, ensure_dir, get_project_root

# Configuration
DATA_DIR = get_project_root() / "data"
RESULTS_DIR = get_project_root() / "results"
MODELS_DIR = RESULTS_DIR / "models"

logger = setup_logging()


def load_model() -> lgb.Booster:
    """
    Load the trained GBRT model.
    
    Returns
    -------
    lgb.Booster
        Trained model
    """
    model_path = MODELS_DIR / "gbrt_full_model.txt"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Please run 03_gbrt_model.py first"
        )
    
    logger.info(f"Loading model from {model_path}")
    model = lgb.Booster(model_file=str(model_path))
    
    return model


def extract_global_importance(model: lgb.Booster, feature_cols: list,
                              top_n: int = 30) -> pd.DataFrame:
    """
    Extract global feature importance from model.
    
    Parameters
    ----------
    model : lgb.Booster
        Trained model
    feature_cols : list
        List of feature names
    top_n : int
        Number of top features to return
    
    Returns
    -------
    pd.DataFrame
        Feature importance dataframe
    """
    logger.info("Extracting global feature importance")
    
    # Get importance by gain
    importance_gain = model.feature_importance(importance_type='gain')
    
    # Get importance by split
    importance_split = model.feature_importance(importance_type='split')
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'gain': importance_gain,
        'split': importance_split
    })
    
    # Normalize to percentages
    importance_df['gain_pct'] = 100 * importance_df['gain'] / importance_df['gain'].sum()
    importance_df['split_pct'] = 100 * importance_df['split'] / importance_df['split'].sum()
    
    # Sort by gain
    importance_df = importance_df.sort_values('gain', ascending=False).reset_index(drop=True)
    
    logger.info(f"Top {top_n} most important features:")
    for idx, row in importance_df.head(top_n).iterrows():
        logger.info(f"  {idx+1}. {row['feature']}: {row['gain_pct']:.2f}% (gain), "
                   f"{row['split_pct']:.2f}% (splits)")
    
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 30,
                           save_path: Optional[Path] = None) -> None:
    """
    Plot feature importance.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance dataframe
    top_n : int
        Number of top features to plot
    save_path : Path, optional
        Path to save figure
    """
    logger.info(f"Plotting top {top_n} features")
    
    # Get top features
    top_features = importance_df.head(top_n).copy()
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # Plot by gain
    axes[0].barh(range(top_n), top_features['gain_pct'].values, alpha=0.8, color='steelblue')
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top_features['feature'].values)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Importance (%)', fontsize=12)
    axes[0].set_title('Feature Importance by Gain', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Plot by split
    axes[1].barh(range(top_n), top_features['split_pct'].values, alpha=0.8, color='coral')
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(top_features['feature'].values)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Importance (%)', fontsize=12)
    axes[1].set_title('Feature Importance by Split Count', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    
    plt.show()


def calculate_shap_values(model: lgb.Booster, X: np.ndarray,
                         sample_size: int = 1000) -> Tuple[shap.Explainer, np.ndarray]:
    """
    Calculate SHAP values for model interpretability.
    
    Parameters
    ----------
    model : lgb.Booster
        Trained model
    X : np.ndarray
        Feature matrix
    sample_size : int
        Sample size for SHAP calculation (smaller = faster)
    
    Returns
    -------
    tuple
        (explainer, shap_values)
    """
    logger.info(f"Calculating SHAP values (sample size: {sample_size})")
    
    # Sample data for efficiency
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    logger.info(f"Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    logger.info(f"Computing SHAP values...")
    shap_values = explainer.shap_values(X_sample)
    
    logger.info(f"SHAP values shape: {shap_values.shape}")
    
    return explainer, shap_values, X_sample


def plot_shap_summary(shap_values: np.ndarray, X: np.ndarray, feature_names: list,
                     save_path: Optional[Path] = None, max_display: int = 30) -> None:
    """
    Create SHAP summary plot.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values
    X : np.ndarray
        Feature matrix
    feature_names : list
        Feature names
    save_path : Path, optional
        Path to save figure
    max_display : int
        Maximum features to display
    """
    logger.info("Creating SHAP summary plot")
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, feature_names=feature_names, 
                     max_display=max_display, show=False)
    plt.title('SHAP Feature Importance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved SHAP summary plot to {save_path}")
    
    plt.show()


def plot_shap_dependence(shap_values: np.ndarray, X: np.ndarray, feature_names: list,
                        top_features: list, save_dir: Path) -> None:
    """
    Create SHAP dependence plots for top features.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values
    X : np.ndarray
        Feature matrix
    feature_names : list
        Feature names
    top_features : list
        List of top feature names to plot
    save_dir : Path
        Directory to save plots
    """
    logger.info(f"Creating SHAP dependence plots for top {len(top_features)} features")
    
    ensure_dir(save_dir)
    
    # Get feature indices
    feature_indices = [feature_names.index(feat) for feat in top_features if feat in feature_names]
    
    for idx in feature_indices[:5]:  # Plot top 5
        feature_name = feature_names[idx]
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(idx, shap_values, X, feature_names=feature_names, show=False)
        plt.title(f'SHAP Dependence Plot: {feature_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = save_dir / f"shap_dependence_{feature_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved dependence plot for {feature_name}")
        plt.close()


def analyze_feature_groups(importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze feature importance by groups (momentum, value, liquidity, etc.).
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance dataframe
    
    Returns
    -------
    pd.DataFrame
        Group-level importance
    """
    logger.info("Analyzing feature groups")
    
    # Define feature groups (keywords in feature names)
    feature_groups = {
        'Momentum': ['mom', 'ret_', 'turn', 'prc_high'],
        'Value': ['bm', 'ep', 'cfp', 'sp', 'dp'],
        'Size': ['mvel', 'me', 'size'],
        'Liquidity': ['vol', 'turn', 'illiq', 'zerotrade', 'bid_ask'],
        'Volatility': ['retvol', 'std', 'beta', 'idio'],
        'Profitability': ['roe', 'roa', 'profit', 'gp', 'cashpr'],
        'Investment': ['inv', 'noa', 'dpi2a', 'ag'],
        'Accruals': ['acc', 'pctacc'],
    }
    
    # Assign features to groups
    importance_df['group'] = 'Other'
    
    for group_name, keywords in feature_groups.items():
        for keyword in keywords:
            mask = importance_df['feature'].str.contains(keyword, case=False, na=False)
            importance_df.loc[mask, 'group'] = group_name
    
    # Aggregate by group
    group_importance = importance_df.groupby('group').agg({
        'gain_pct': 'sum',
        'split_pct': 'sum',
        'feature': 'count'
    }).rename(columns={'feature': 'n_features'})
    
    group_importance = group_importance.sort_values('gain_pct', ascending=False)
    
    logger.info("\nFeature group importance:")
    print(group_importance.to_string())
    
    return group_importance


def plot_group_importance(group_importance: pd.DataFrame, 
                         save_path: Optional[Path] = None) -> None:
    """
    Plot feature group importance.
    
    Parameters
    ----------
    group_importance : pd.DataFrame
        Group-level importance
    save_path : Path, optional
        Path to save figure
    """
    logger.info("Plotting feature group importance")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    groups = group_importance.index.tolist()
    importance = group_importance['gain_pct'].values
    
    bars = ax.barh(range(len(groups)), importance, alpha=0.8, color='teal')
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.invert_yaxis()
    ax.set_xlabel('Total Importance (%)', fontsize=12)
    ax.set_title('Feature Importance by Group', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for idx, (bar, val, n_feat) in enumerate(zip(bars, importance, group_importance['n_features'])):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}% ({int(n_feat)} features)',
               va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved group importance plot to {save_path}")
    
    plt.show()


def save_importance_tables(importance_df: pd.DataFrame, 
                          group_importance: pd.DataFrame) -> None:
    """
    Save feature importance tables.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature-level importance
    group_importance : pd.DataFrame
        Group-level importance
    """
    logger.info("Saving feature importance tables")
    
    tables_dir = RESULTS_DIR / "tables"
    ensure_dir(tables_dir)
    
    # Top 50 features
    top_features = importance_df.head(50)
    
    # Save as CSV
    top_features.to_csv(tables_dir / "feature_importance_top50.csv", index=False)
    
    # Save as LaTeX
    latex_df = top_features[['feature', 'gain_pct', 'split_pct']].copy()
    latex_df.columns = ['Feature', 'Gain (\%)', 'Splits (\%)']
    latex_df.to_latex(tables_dir / "feature_importance_top50.tex", 
                     index=False, float_format='%.2f',
                     caption='Top 50 Most Important Features',
                     label='tab:feature_importance')
    
    # Save group importance
    group_importance.to_csv(tables_dir / "feature_group_importance.csv")
    group_importance.to_latex(tables_dir / "feature_group_importance.tex",
                             float_format='%.2f',
                             caption='Feature Importance by Group',
                             label='tab:group_importance')
    
    logger.info(f"Saved importance tables to {tables_dir}")


def main() -> None:
    """Main feature importance analysis pipeline."""
    logger.info("="*80)
    logger.info("Feature Importance and Interpretability Analysis")
    logger.info("="*80)
    
    # Load model
    model = load_model()
    
    # Load data for SHAP analysis
    logger.info("Loading data...")
    train_df = pd.read_parquet(DATA_DIR / "train_data.parquet")
    
    # Load metadata
    import json
    with open(DATA_DIR / "data_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    feature_cols = metadata['feature_cols']
    target_col = metadata['target_col']
    
    logger.info(f"Number of features: {len(feature_cols)}")
    
    # 1. Global feature importance
    importance_df = extract_global_importance(model, feature_cols, top_n=30)
    
    # Plot feature importance
    plot_feature_importance(
        importance_df, 
        top_n=30,
        save_path=RESULTS_DIR / "figures" / "feature_importance.png"
    )
    
    # 2. Feature group analysis
    group_importance = analyze_feature_groups(importance_df)
    
    # Plot group importance
    plot_group_importance(
        group_importance,
        save_path=RESULTS_DIR / "figures" / "feature_group_importance.png"
    )
    
    # 3. SHAP analysis
    logger.info("\n" + "="*80)
    logger.info("SHAP Analysis")
    logger.info("="*80)
    
    # Sample data for SHAP (use recent data)
    recent_data = train_df.tail(10000).copy()
    X = recent_data[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate SHAP values
    try:
        explainer, shap_values, X_sample = calculate_shap_values(
            model, X, sample_size=1000
        )
        
        # SHAP summary plot
        plot_shap_summary(
            shap_values, X_sample, feature_cols,
            save_path=RESULTS_DIR / "figures" / "shap_summary.png",
            max_display=30
        )
        
        # SHAP dependence plots for top features
        top_feature_names = importance_df.head(10)['feature'].tolist()
        plot_shap_dependence(
            shap_values, X_sample, feature_cols, top_feature_names,
            save_dir=RESULTS_DIR / "figures" / "shap_dependence"
        )
        
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")
        logger.warning("Continuing without SHAP analysis...")
    
    # 4. Save tables
    save_importance_tables(importance_df, group_importance)
    
    logger.info("="*80)
    logger.info("Feature importance analysis completed successfully!")
    logger.info("="*80)
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print("\nTop 10 Most Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {idx+1}. {row['feature']}: {row['gain_pct']:.2f}%")
    
    print("\nTop Feature Groups:")
    for group, row in group_importance.head(5).iterrows():
        print(f"  {group}: {row['gain_pct']:.1f}% ({int(row['n_features'])} features)")
    
    print("\nComparison with Paper (Gu et al. 2020):")
    print("  Expected top predictors: Momentum, Liquidity, Volatility")
    print("  Our findings:", ", ".join(group_importance.head(3).index.tolist()))
    print("="*80)


if __name__ == "__main__":
    main()
