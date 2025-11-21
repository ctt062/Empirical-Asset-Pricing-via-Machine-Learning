"""
Create Architecture Diagram for 4-Model Comparison

Visualizes the restructured project architecture with all 4 models.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Colors
color_data = '#3498db'
color_model = '#e74c3c'
color_eval = '#2ecc71'
color_output = '#f39c12'
color_traditional = '#9b59b6'

def add_box(ax, x, y, width, height, text, color, fontsize=10, fontweight='bold'):
    """Add a fancy box with text."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor=color,
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center',
           fontsize=fontsize, fontweight=fontweight,
           color='white')

def add_arrow(ax, x1, y1, x2, y2, color='black'):
    """Add an arrow between boxes."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.8',
        color=color,
        linewidth=2.5,
        alpha=0.7
    )
    ax.add_patch(arrow)

# Title
ax.text(5, 11.5, 'Empirical Asset Pricing: 4-Model Comparison Framework',
       ha='center', fontsize=18, fontweight='bold')
ax.text(5, 11, 'Professional Architecture for Academic Research',
       ha='center', fontsize=12, style='italic', color='gray')

# ===== LAYER 1: Data =====
ax.text(0.5, 10, 'LAYER 1: DATA', fontsize=12, fontweight='bold', color=color_data)

add_box(ax, 0.5, 8.8, 1.8, 0.8, 'Raw Data\n(CRSP)', color_data)
add_box(ax, 2.5, 8.8, 1.8, 0.8, 'Preprocessing\n94 features', color_data)
add_box(ax, 4.5, 8.8, 1.8, 0.8, 'Train\n(1957-1995)', color_data)
add_box(ax, 6.5, 8.8, 1.8, 0.8, 'Test\n(1996-2016)', color_data)

add_arrow(ax, 1.4, 9.2, 2.5, 9.2, color_data)
add_arrow(ax, 3.4, 9.2, 4.5, 9.2, color_data)
add_arrow(ax, 5.4, 9.2, 6.5, 9.2, color_data)

# ===== LAYER 2: Models =====
ax.text(0.5, 7.8, 'LAYER 2: MODELS', fontsize=12, fontweight='bold', color=color_model)

# Model 1: Fama-French
add_box(ax, 0.5, 6.2, 1.8, 1.2, 'Fama-French\n3-Factor\n(Traditional)', color_traditional, fontsize=9)
ax.text(1.4, 5.9, 'α + β·Factors', ha='center', fontsize=7, style='italic')
ax.text(1.4, 5.6, 'Sharpe: ~1.5', ha='center', fontsize=7, fontweight='bold', color=color_traditional)

# Model 2: OLS-3
add_box(ax, 2.5, 6.2, 1.8, 1.2, 'OLS-3\nPolynomial\n(Baseline)', color_model, fontsize=9)
ax.text(3.4, 5.9, 'Linear + x² + x³', ha='center', fontsize=7, style='italic')
ax.text(3.4, 5.6, 'Sharpe: 2.31', ha='center', fontsize=7, fontweight='bold', color=color_model)

# Model 3: Elastic Net
add_box(ax, 4.5, 6.2, 1.8, 1.2, 'Elastic Net\nRegularized\n(L1+L2)', color=color_model, fontsize=9)
ax.text(5.4, 5.9, 'LASSO + Ridge', ha='center', fontsize=7, style='italic')
ax.text(5.4, 5.6, 'Sharpe: ~2.5', ha='center', fontsize=7, fontweight='bold', color=color_model)

# Model 4: GBRT
add_box(ax, 6.5, 6.2, 1.8, 1.2, 'GBRT\nMachine Learning\n(Non-linear)', color='#c0392b', fontsize=9)
ax.text(7.4, 5.9, '2000 trees', ha='center', fontsize=7, style='italic')
ax.text(7.4, 5.6, 'Sharpe: 3.09 ✓', ha='center', fontsize=7, fontweight='bold', color='#c0392b')

# Arrows from data to models
add_arrow(ax, 5.4, 8.8, 1.4, 7.4, color_data)
add_arrow(ax, 5.4, 8.8, 3.4, 7.4, color_data)
add_arrow(ax, 5.4, 8.8, 5.4, 7.4, color_data)
add_arrow(ax, 7.4, 8.8, 7.4, 7.4, color_data)

# ===== LAYER 3: Evaluation =====
ax.text(0.5, 5.0, 'LAYER 3: EVALUATION', fontsize=12, fontweight='bold', color=color_eval)

# Predictions
add_box(ax, 0.5, 3.8, 1.8, 0.8, 'Predictions\n252 months', color_eval, fontsize=9)
add_box(ax, 2.5, 3.8, 1.8, 0.8, 'Portfolio\nConstruction', color_eval, fontsize=9)
add_box(ax, 4.5, 3.8, 1.8, 0.8, 'Performance\nMetrics', color_eval, fontsize=9)
add_box(ax, 6.5, 3.8, 1.8, 0.8, 'Model\nComparison', color_eval, fontsize=9)

# Arrows from models to evaluation
for x in [1.4, 3.4, 5.4, 7.4]:
    add_arrow(ax, x, 6.2, 1.4, 4.6, color_eval)

add_arrow(ax, 1.4, 4.2, 2.5, 4.2, color_eval)
add_arrow(ax, 3.4, 4.2, 4.5, 4.2, color_eval)
add_arrow(ax, 5.4, 4.2, 6.5, 4.2, color_eval)

# ===== LAYER 4: Results =====
ax.text(0.5, 3.0, 'LAYER 4: RESULTS', fontsize=12, fontweight='bold', color=color_output)

# Results boxes
add_box(ax, 1.0, 1.5, 1.5, 0.8, 'Tables\n(CSV, LaTeX)', color_output, fontsize=9)
add_box(ax, 2.8, 1.5, 1.5, 0.8, 'Figures\n(PNG, PDF)', color_output, fontsize=9)
add_box(ax, 4.6, 1.5, 1.5, 0.8, 'Feature\nImportance', color_output, fontsize=9)
add_box(ax, 6.4, 1.5, 1.5, 0.8, 'Presentation\nSlides', color_output, fontsize=9)

# Arrows to results
add_arrow(ax, 7.4, 3.8, 1.8, 2.3, color_output)
add_arrow(ax, 7.4, 3.8, 3.6, 2.3, color_output)
add_arrow(ax, 7.4, 3.8, 5.4, 2.3, color_output)
add_arrow(ax, 7.4, 3.8, 7.2, 2.3, color_output)

# ===== Key Metrics Box =====
metrics_box = FancyBboxPatch(
    (8.8, 6), 1.1, 3,
    boxstyle="round,pad=0.1",
    edgecolor='black',
    facecolor='#ecf0f1',
    linewidth=2
)
ax.add_patch(metrics_box)
ax.text(9.35, 8.7, 'Key Metrics', ha='center', fontsize=11, fontweight='bold')
ax.text(9.35, 8.3, '━━━━━━━', ha='center', fontsize=8)
ax.text(9.35, 7.9, 'OOS R²', ha='center', fontsize=9)
ax.text(9.35, 7.5, 'Sharpe Ratio', ha='center', fontsize=9)
ax.text(9.35, 7.1, 'Returns', ha='center', fontsize=9)
ax.text(9.35, 6.7, 'Volatility', ha='center', fontsize=9)
ax.text(9.35, 6.3, 'Max Drawdown', ha='center', fontsize=9)

# ===== Features Box =====
features_box = FancyBboxPatch(
    (8.8, 2), 1.1, 3.5,
    boxstyle="round,pad=0.1",
    edgecolor='black',
    facecolor='#ecf0f1',
    linewidth=2
)
ax.add_patch(features_box)
ax.text(9.35, 5.2, 'Design', ha='center', fontsize=11, fontweight='bold')
ax.text(9.35, 4.9, 'Principles', ha='center', fontsize=11, fontweight='bold')
ax.text(9.35, 4.5, '━━━━━━━', ha='center', fontsize=8)
ax.text(9.35, 4.2, '✓ Modular', ha='center', fontsize=8)
ax.text(9.35, 3.9, '✓ Expandable', ha='center', fontsize=8)
ax.text(9.35, 3.6, '✓ OOP Design', ha='center', fontsize=8)
ax.text(9.35, 3.3, '✓ Consistent', ha='center', fontsize=8)
ax.text(9.35, 3.0, '✓ Reproducible', ha='center', fontsize=8)
ax.text(9.35, 2.7, '✓ Documented', ha='center', fontsize=8)
ax.text(9.35, 2.4, '✓ Tested', ha='center', fontsize=8)

# ===== Legend =====
ax.text(1, 0.8, 'COLOR LEGEND:', fontsize=10, fontweight='bold')
legend_items = [
    (color_data, 'Data Processing'),
    (color_traditional, 'Traditional Model'),
    (color_model, 'Modern Models'),
    (color_eval, 'Evaluation'),
    (color_output, 'Outputs')
]

x_start = 1
for i, (color, label) in enumerate(legend_items):
    rect = mpatches.Rectangle((x_start + i*1.3, 0.5), 0.2, 0.2, facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(x_start + i*1.3 + 0.3, 0.6, label, fontsize=8, va='center')

# Footer
ax.text(5, 0.1, '© 2025 | Professional Asset Pricing Research Framework',
       ha='center', fontsize=8, style='italic', color='gray')

plt.tight_layout()
plt.savefig('results/figures/architecture/architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Architecture diagram saved to: results/figures/architecture/architecture_diagram.png")
plt.close()

print("\n🎨 Diagram includes:")
print("   • 4-layer architecture (Data → Models → Evaluation → Results)")
print("   • 4 different models with characteristics")
print("   • Performance metrics and expected Sharpe ratios")
print("   • Design principles")
print("   • Color-coded components")
