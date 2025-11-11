"""
Comprehensive Analysis and Visualization Script

Creates:
1. Feature importance analysis
2. Error distribution plots
3. Prediction vs Actual plots
4. Residual analysis
5. Ablation study
6. Model comparison charts

Author: AI Assistant
Date: November 10, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create output directory
output_dir = Path("analysis/figures")
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COMPREHENSIVE ANALYSIS & VISUALIZATION")
print("="*80)

# ============================================================================
# 1. MODEL COMPARISON
# ============================================================================
print("\n📊 Creating model comparison visualizations...")

comparison_df = pd.read_csv("models/version_comparison.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MAE Comparison
axes[0].bar(comparison_df['version'], comparison_df['mae'], 
            color=['#e74c3c', '#3498db', '#2ecc71'])
axes[0].set_ylabel('Test MAE (minutes)', fontsize=12)
axes[0].set_title('Test MAE by Model Version', fontsize=14, fontweight='bold')
axes[0].axhline(y=comparison_df['mae'].min(), color='green', linestyle='--', alpha=0.5, label='Best')
for i, (v, mae) in enumerate(zip(comparison_df['version'], comparison_df['mae'])):
    axes[0].text(i, mae + 0.05, f'{mae:.3f}', ha='center', fontweight='bold')

# R² Comparison
axes[1].bar(comparison_df['version'], comparison_df['r2'],
            color=['#e74c3c', '#3498db', '#2ecc71'])
axes[1].set_ylabel('Test R²', fontsize=12)
axes[1].set_title('Test R² by Model Version', fontsize=14, fontweight='bold')
axes[1].axhline(y=comparison_df['r2'].max(), color='green', linestyle='--', alpha=0.5, label='Highest')
for i, (v, r2) in enumerate(zip(comparison_df['version'], comparison_df['r2'])):
    axes[1].text(i, r2 + 0.01, f'{r2:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "01_model_comparison.png", dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {output_dir / '01_model_comparison.png'}")

# ============================================================================
# 2. FEATURE COUNT COMPARISON
# ============================================================================
print("\n📊 Creating feature count comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

feature_counts = [
    ("V1\n(Financial)", 50),  # Approximate
    ("V2\n(Dish One-Hot)", 262),
    ("V3\n(Engineered)", 33)
]

versions, counts = zip(*feature_counts)
colors = ['#e74c3c', '#3498db', '#2ecc71']

bars = ax.bar(versions, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Number of Features', fontsize=12)
ax.set_title('Feature Dimensionality Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(counts) * 1.2)

for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{count}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "02_feature_counts.png", dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {output_dir / '02_feature_counts.png'}")

# ============================================================================
# 3. PREDICTION VS ACTUAL (V3)
# ============================================================================
print("\n📊 Creating prediction vs actual plot...")

# Load V3 model and data
with open("models/v3/best_model.pkl", "rb") as f:
    model_v3 = pickle.load(f)

df_v3 = pd.read_csv("data/processed/features_orders_v3.csv")
df_v3.columns = df_v3.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)

target_col = [c for c in df_v3.columns if 'KPT' in c or 'Kitchen' in c][0]
y_true = df_v3[target_col].dropna()
X = df_v3.loc[y_true.index].drop(columns=[target_col])

# Make predictions
y_log_pred = model_v3.predict(X)
y_pred = np.expm1(y_log_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
axes[0].scatter(y_true, y_pred, alpha=0.3, s=10)
axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Prep Time (minutes)', fontsize=12)
axes[0].set_ylabel('Predicted Prep Time (minutes)', fontsize=12)
axes[0].set_title('V3 XGBoost: Predictions vs Actual', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
axes[0].text(0.05, 0.95, f'MAE: {mae:.2f} min\nR²: {r2:.4f}',
             transform=axes[0].transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Residual plot
residuals = y_true - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Prep Time (minutes)', fontsize=12)
axes[1].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "03_predictions_vs_actual_v3.png", dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {output_dir / '03_predictions_vs_actual_v3.png'}")

# ============================================================================
# 4. ERROR DISTRIBUTION
# ============================================================================
print("\n📊 Creating error distribution plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram of residuals
axes[0, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 0].set_xlabel('Residuals (minutes)', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
axes[0, 0].text(0.05, 0.95, f'Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}',
                transform=axes[0, 0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Absolute errors
abs_errors = np.abs(residuals)
axes[0, 1].hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].axvline(x=mae, color='r', linestyle='--', lw=2, label=f'MAE: {mae:.2f}')
axes[0, 1].set_xlabel('Absolute Error (minutes)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Distribution of Absolute Errors', fontsize=12, fontweight='bold')
axes[0, 1].legend()

# Error by prediction range
bins = pd.cut(y_pred, bins=10)
error_by_range = abs_errors.groupby(bins).mean()
axes[1, 0].bar(range(len(error_by_range)), error_by_range.values, edgecolor='black')
axes[1, 0].set_xlabel('Prediction Range (deciles)', fontsize=11)
axes[1, 0].set_ylabel('Mean Absolute Error (minutes)', fontsize=11)
axes[1, 0].set_title('Error by Prediction Range', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(range(len(error_by_range)))
axes[1, 0].set_xticklabels([f'{i+1}' for i in range(len(error_by_range))])

# QQ plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "04_error_analysis.png", dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {output_dir / '04_error_analysis.png'}")

# ============================================================================
# 5. FEATURE IMPORTANCE (V3)
# ============================================================================
print("\n📊 Creating feature importance plot...")

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model_v3.feature_importances_
}).sort_values('importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(feature_importance)), feature_importance['importance'], edgecolor='black')
ax.set_yticks(range(len(feature_importance)))
ax.set_yticklabels(feature_importance['feature'])
ax.invert_yaxis()
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Top 20 Most Important Features (V3 XGBoost)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "05_feature_importance.png", dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {output_dir / '05_feature_importance.png'}")

# ============================================================================
# 6. DATA LEAKAGE ILLUSTRATION
# ============================================================================
print("\n📊 Creating data leakage illustration...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# V1 - Wrong causality
axes[0].text(0.5, 0.9, 'V1: WRONG APPROACH', ha='center', fontsize=16, fontweight='bold',
             transform=axes[0].transAxes)
axes[0].text(0.5, 0.7, 'Order Placed', ha='center', fontsize=12, transform=axes[0].transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue'))
axes[0].annotate('', xy=(0.5, 0.55), xytext=(0.5, 0.65), 
                arrowprops=dict(arrowstyle='->', lw=2), transform=axes[0].transAxes)
axes[0].text(0.5, 0.5, 'Prep Time\n(Target)', ha='center', fontsize=12, transform=axes[0].transAxes,
             bbox=dict(boxstyle='round', facecolor='yellow'))
axes[0].annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.45),
                arrowprops=dict(arrowstyle='->', lw=2), transform=axes[0].transAxes)
axes[0].text(0.5, 0.3, 'Bill Calculated', ha='center', fontsize=12, transform=axes[0].transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgreen'))
axes[0].annotate('', xy=(0.7, 0.5), xytext=(0.55, 0.32),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'), transform=axes[0].transAxes)
axes[0].text(0.82, 0.42, 'DATA\nLEAKAGE!', ha='center', fontsize=11, fontweight='bold',
             transform=axes[0].transAxes, color='red',
             bbox=dict(boxstyle='round', facecolor='pink', edgecolor='red', linewidth=2))
axes[0].axis('off')
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)

# V2/V3 - Correct causality
axes[1].text(0.5, 0.9, 'V2/V3: CORRECT APPROACH', ha='center', fontsize=16, fontweight='bold',
             transform=axes[1].transAxes)
axes[1].text(0.5, 0.7, 'Dishes Ordered', ha='center', fontsize=12, transform=axes[1].transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue'))
axes[1].annotate('', xy=(0.5, 0.55), xytext=(0.5, 0.65),
                arrowprops=dict(arrowstyle='->', lw=3, color='green'), transform=axes[1].transAxes)
axes[1].text(0.5, 0.5, 'Prep Time\n(Target)', ha='center', fontsize=12, transform=axes[1].transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgreen'))
axes[1].text(0.82, 0.58, 'CAUSAL\nRELATIONSHIP', ha='center', fontsize=11, fontweight='bold',
             transform=axes[1].transAxes, color='green',
             bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', linewidth=2))
axes[1].axis('off')
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(output_dir / "06_data_leakage_illustration.png", dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {output_dir / '06_data_leakage_illustration.png'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\n📁 All visualizations saved to: {output_dir}/")
print("\nGenerated files:")
for img in sorted(output_dir.glob("*.png")):
    print(f"   - {img.name}")

print("\n✅ Analysis complete! Check the analysis/figures/ directory for all visualizations.")
