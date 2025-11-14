# Quick Start Guide

## First Time Setup (5 minutes)

### 1. Install Dependencies
```bash
# Option A: Using conda (recommended)
conda env create -f environment.yml
conda activate asset-pricing-ml

# Option B: Using pip
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
python run_all.py
```

**That's it!** The pipeline will:
1. Download the dataset (~200MB)
2. Preprocess the data
3. Train the OLS benchmark
4. Train the GBRT model
5. Evaluate both models
6. Generate interpretability analysis

**Expected runtime:** 30-90 minutes

---

## Understanding the Output

### Console Output
You'll see progress messages like:
```
================================================================================
STEP: Download Dataset
Running: 00_download_data.py
================================================================================
Downloading from https://dachxiu.chicagobooth.edu/download/datashare.zip
...
✓ Download Dataset completed successfully in 2.3 minutes
```

### Generated Files

After completion, check:

**`results/tables/`**
- `performance_comparison.csv` - Main results table
- `feature_importance_top50.csv` - Top predictive features
- `benchmark_summary.csv` - OLS benchmark stats
- `gbrt_detailed_performance.csv` - GBRT detailed stats

**`results/figures/`**
- `comparison_cumulative_returns_ew.png` - Long-short portfolio returns
- `comparison_monthly_r2.png` - Predictive accuracy over time
- `feature_importance.png` - Top 30 features
- `shap_summary.png` - SHAP interpretability plot

**`results/predictions/`**
- `benchmark_predictions.parquet` - OLS predictions
- `gbrt_predictions.parquet` - GBRT predictions

**`results/models/`**
- `gbrt_full_model.txt` - Trained GBRT model
- Monthly model checkpoints

---

## Quick Commands

### Run specific parts only
```bash
# Skip download if data exists
python run_all.py --skip-download

# Run only evaluation (if models exist)
python run_all.py --skip-download --skip-benchmark --skip-gbrt

# Run individual scripts
python src/00_download_data.py
python src/01_data_preparation.py
python src/02_baseline_benchmark.py
python src/03_gbrt_model.py
python src/04_evaluation.py
python src/05_feature_importance.py
```

### Interactive exploration
```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## Expected Results

### Model Performance (Out-of-Sample 1996-2016)

| Metric | OLS-3 Target | GBRT Target | Your Results |
|--------|--------------|-------------|--------------|
| Monthly R² | 0.16% | 0.37% | Check `results/tables/` |
| Sharpe (EW) | 0.83 | 2.20 | Check `results/tables/` |
| Sharpe (VW) | 0.61 | 1.35 | Check `results/tables/` |

Your results should be close to these targets (within ±10%).

### Key Findings

1. **GBRT outperforms OLS by 130%+** in predictive accuracy
2. **Top predictors:** Momentum, Liquidity, Volatility
3. **Long-short Sharpe ratio:** 2.2+ (equal-weighted)
4. **Annualized returns:** 15-20% for long-short portfolio

---

## Troubleshooting

### Problem: Out of memory
**Solution:** Reduce sample size in `src/05_feature_importance.py`:
```python
sample_size = 500  # Instead of 1000
```

### Problem: Download fails
**Solution:** Manually download from:
https://dachxiu.chicagobooth.edu/download/datashare.zip

Extract `datashare.csv` to `data/` folder.

### Problem: Models take too long
**Solution:** Reduce hyperparameter search:
```python
# In src/03_gbrt_model.py, line 290
tune_hyperparams = False
```

### Problem: Import errors
**Solution:** Make sure you're in the conda environment:
```bash
conda activate asset-pricing-ml
```

---

## Next Steps

1. **Review Results**
   - Open `results/tables/performance_comparison.csv`
   - Compare your results with paper benchmarks

2. **Visualize**
   - Check all plots in `results/figures/`
   - Look for cumulative returns, feature importance

3. **Explore**
   - Open `notebooks/exploration.ipynb`
   - Run interactive analysis

4. **Customize**
   - Modify hyperparameters in `src/03_gbrt_model.py`
   - Add features in `src/01_data_preparation.py`
   - Try different portfolio strategies in `utils.py`

---

## Getting Help

- **Issues?** Check the troubleshooting section above
- **Questions?** Open a GitHub issue
- **Want to contribute?** Submit a pull request

---

## Citation

If you use this code, please cite:

```bibtex
@article{gu2020empirical,
  title={Empirical asset pricing via machine learning},
  author={Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},
  journal={The Review of Financial Studies},
  volume={33},
  number={5},
  pages={2223--2273},
  year={2020}
}
```

---

**Happy researching! 🚀📈**
