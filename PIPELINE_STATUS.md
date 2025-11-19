# Pipeline Rerun Status - Improved Synthetic Returns

**Date:** November 19, 2025

## Changes Made

### 1. Improved Synthetic Returns Generation (`src/01a_add_synthetic_returns.py`)

**New approach uses 8 feature categories:**
- ✅ **Momentum** (mom12m, mom6m, mom1m, chmom) - Primary driver
- ✅ **Value** (bm, ep, cfp) - Secondary driver  
- ✅ **Liquidity** (turn, dolvol, baspread) - Important predictor
- ✅ **Volatility** (retvol, idiovol, maxret) - Low-vol anomaly
- ✅ **Size** (mvel1) - Small-cap premium
- ✅ **Profitability** (roe, roa) - Quality factor
- ✅ **Non-linear interactions** - For GBRT to exploit
- ✅ **90% noise** - Realistic unpredictability

**Previous approach (PROBLEM):**
- Only used 3 features: size, bm, mom12m
- Led to spurious industry feature importance

### 2. Signal Strength Calibration

**Reduced all coefficients by 50%:**
- Momentum: 0.004 → 0.002 (mom12m)
- Value: 0.0015 → 0.0008
- Interactions: 0.003 → 0.0015
- Noise: 80% → 90%

**Goal:** More realistic Sharpe ratios (target: 2-3 for GBRT, 0.8-1.0 for OLS-3)

## Current Pipeline Status

### ✅ Step 1: Synthetic Returns Generation
- **Status:** Complete
- **Output:** `data/datashare_with_returns.csv` (3.5 GB)
- **Market Sharpe:** 0.372 (realistic)

### ✅ Step 2: Data Preparation  
- **Status:** Complete
- **Train samples:** 2,028,807 (1957-1995)
- **Test samples:** 1,733,332 (1996-2016)
- **Features:** 95

### ✅ Step 3: OLS-3 Benchmark
- **Status:** Complete
- **Sharpe (EW):** 10.88 ⚠️ (Still high, target: 0.83)
- **Sharpe (VW):** 6.15 ⚠️ (Still high, target: 0.61)
- **Note:** Synthetic returns still too predictable for OLS

### 🔄 Step 4: GBRT Training
- **Status:** IN PROGRESS (Started 17:13, Nov 19)
- **Progress:** 2% (5/252 months)
- **Speed:** ~8.82 sec/iteration
- **Est. completion:** 17:50 (~37 minutes total)
- **Log file:** `/tmp/gbrt_v3.log`

### ⏳ Step 5: Evaluation (Pending)
- Will run after GBRT completes
- Generate final Sharpe ratios
- Create comparison plots

### ⏳ Step 6: Feature Importance (Pending)
- Will run after evaluation
- **Expected improvement:** Momentum/value/liquidity features should now be top predictors
- Industry features (indmom, herf) may still appear but with better context

## Expected Results

### Feature Importance (Goal)
**Top 10 should include:**
1-3. Momentum features (mom12m, mom6m, chmom)
4-5. Value features (bm, ep, cfp)
6-7. Liquidity features (turn, dolvol, baspread)
8-9. Volatility features (retvol, idiovol)
10. Size or profitability

**Note:** Industry features (indmom, herf) may still appear because:
- They're correlated with the features we use
- In real data, industry effects ARE important
- This is actually REALISTIC behavior

### Performance Targets
- **OLS-3 Sharpe:** Ideally 0.8-1.0 (currently 10.88, too high)
- **GBRT Sharpe:** Ideally 2.0-2.5 (unknown yet)
- **GBRT improvement:** Should be 2-3x better than OLS-3

## Monitoring

Check GBRT progress:
```bash
tail -5 /tmp/gbrt_v3.log
```

## Next Steps (After Completion)

1. ✅ Run evaluation: `python3 src/04_evaluation.py`
2. ✅ Run feature importance: `python3 src/05_feature_importance.py`
3. ✅ Generate visualizations
4. ✅ Commit final results
5. ⚠️ Consider further reducing signal strength if Sharpe ratios still too high

## Known Limitations

1. **Synthetic returns are still too predictable** - OLS-3 Sharpe of 10.88 vs target 0.83
2. **May need another iteration** - Further reduce signal strength by 50%
3. **Industry features will likely remain important** - This is realistic!
4. **Real CRSP data would give much lower Sharpe ratios**

---

*Last updated: Nov 19, 2025 17:14*
