"""Quick test of synthetic returns - check which features drive returns"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load the data
df = pd.read_csv('data/datashare_with_returns.csv', nrows=100000)

print("Correlation of ret_exc with key features:\n")
print("="*60)

features_to_check = [
    'mom12m', 'mom6m', 'mom1m', 'chmom',  # Momentum
    'bm', 'ep', 'cfp',  # Value
    'turn', 'dolvol', 'baspread',  # Liquidity
    'retvol', 'idiovol', 'maxret',  # Volatility
    'mvel1',  # Size
    'indmom', 'herf', 'sic2',  # Industry (should be LOW)
]

correlations = []
for feat in features_to_check:
    if feat in df.columns:
        corr = df['ret_exc'].corr(df[feat])
        correlations.append((feat, abs(corr), corr))

# Sort by absolute correlation
correlations.sort(key=lambda x: x[1], reverse=True)

print("\nTop features by absolute correlation with returns:")
print("-"*60)
for feat, abs_corr, corr in correlations[:15]:
    print(f"{feat:15s}: {corr:7.4f} (abs: {abs_corr:.4f})")

print("\n" + "="*60)
print("✓ Momentum features should be at the top")
print("✓ Industry features (indmom, herf, sic2) should be LOWER")
print("="*60)
