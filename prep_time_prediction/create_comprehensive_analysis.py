"""
Comprehensive Analysis for Prep Time Prediction
Creates visualizations and ablation study
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pickle
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Load data
print("="*80)
print("📊 COMPREHENSIVE ANALYSIS")
print("="*80)

df = pd.read_csv("data/processed/features_final.csv")
target_col = [c for c in df.columns if 'KPT' in c or 'duration' in c][0]

# Load best model
with open("models/production/best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

with open("models/production/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Prepare data
feature_cols = metadata['feature_names']
X = df[feature_cols].fillna(0)
y = df[target_col]
y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# ============================================================================
# 1. MODEL COMPARISON VISUALIZATION
# ============================================================================

print("\n1. Creating model comparison charts...")

model_results = pd.read_csv("models/production/model_comparison.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MAE comparison
axes[0].barh(model_results['Model'], model_results['MAE_Test'], color='steelblue')
axes[0].set_xlabel('Mean Absolute Error (minutes)', fontsize=12)
axes[0].set_title('Model Comparison - MAE (Lower is Better)', fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)
for i, v in enumerate(model_results['MAE_Test']):
    axes[0].text(v + 0.02, i, f'{v:.3f}', va='center')

# R² comparison
axes[1].barh(model_results['Model'], model_results['R2_Test'], color='coral')
axes[1].set_xlabel('R² Score', fontsize=12)
axes[1].set_title('Model Comparison - R² (Higher is Better)', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)
for i, v in enumerate(model_results['R2_Test']):
    axes[1].text(v + 0.005, i, f'{v:.4f}', va='center')

plt.tight_layout()
plt.savefig('analysis/figures/model_comparison.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: analysis/figures/model_comparison.png")

# ============================================================================
# 2. PREDICTION QUALITY VISUALIZATION
# ============================================================================

print("\n2. Creating prediction quality plots...")

y_pred_test = np.expm1(best_model.predict(X_test))
y_true_test = np.expm1(y_test)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Predictions vs Actual
axes[0, 0].scatter(y_true_test, y_pred_test, alpha=0.3, s=10)
axes[0, 0].plot([y_true_test.min(), y_true_test.max()], 
                [y_true_test.min(), y_true_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Prep Time (minutes)', fontsize=12)
axes[0, 0].set_ylabel('Predicted Prep Time (minutes)', fontsize=12)
axes[0, 0].set_title(f'{metadata["model_name"]} - Predictions vs Actual\nMAE: {metadata["mae"]:.3f} min, R²: {metadata["r2"]:.4f}', 
                     fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Residual plot
residuals = y_pred_test - y_true_test
axes[0, 1].scatter(y_pred_test, residuals, alpha=0.3, s=10)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Prep Time (minutes)', fontsize=12)
axes[0, 1].set_ylabel('Residuals (minutes)', fontsize=12)
axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Error distribution
axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Prediction Error (minutes)', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title(f'Error Distribution\nMean: {residuals.mean():.3f}, Std: {residuals.std():.3f}', 
                     fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Error by actual value
error_bins = pd.cut(y_true_test, bins=10)
error_by_range = pd.DataFrame({
    'actual': y_true_test,
    'error': np.abs(residuals),
    'bin': error_bins
})
error_summary = error_by_range.groupby('bin')['error'].mean()
bin_centers = [interval.mid for interval in error_summary.index]

axes[1, 1].plot(bin_centers, error_summary.values, marker='o', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Actual Prep Time (minutes)', fontsize=12)
axes[1, 1].set_ylabel('Mean Absolute Error (minutes)', fontsize=12)
axes[1, 1].set_title('Error vs Actual Prep Time', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/figures/prediction_quality.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: analysis/figures/prediction_quality.png")

# ============================================================================
# 3. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n3. Creating feature importance analysis...")

# Get feature importance
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Top 20 features
    top_features = importance_df.head(20)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'], color='teal')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top 20 Most Important Features - {metadata["model_name"]}', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis/figures/feature_importance_top20.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: analysis/figures/feature_importance_top20.png")
    
    # Feature group importance
    feature_groups = {
        'Dishes': [f for f in feature_cols if f.startswith('dish_')],
        'Order Complexity': ['num_items', 'num_unique_dishes', 'max_dish_quantity', 
                            'order_complexity', 'dish_diversity'],
        'Temporal': ['hour', 'day_of_week', 'is_weekend', 'day_of_month', 'month',
                    'is_lunch_peak', 'is_dinner_peak', 'is_late_night', 'is_early_morning',
                    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'],
        'Kitchen Load': ['orders_last_30min', 'items_last_30min', 'is_high_load'],
        'Dish Popularity': ['avg_dish_popularity'],
        'External': ['temperature_normalized', 'has_precipitation', 'is_holiday']
    }
    
    group_importance = {}
    for group, features in feature_groups.items():
        group_features = [f for f in features if f in feature_cols]
        if group_features:
            group_imp = importance_df[importance_df['feature'].isin(group_features)]['importance'].sum()
            group_importance[group] = group_imp
    
    group_df = pd.DataFrame(list(group_importance.items()), 
                           columns=['Feature Group', 'Total Importance']).sort_values('Total Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set3(range(len(group_df)))
    plt.barh(group_df['Feature Group'], group_df['Total Importance'], color=colors)
    plt.xlabel('Total Importance Score', fontsize=12)
    plt.title('Feature Group Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(group_df['Total Importance']):
        plt.text(v + 0.1, i, f'{v:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig('analysis/figures/feature_group_importance.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: analysis/figures/feature_group_importance.png")
    
    # Save importance data
    importance_df.to_csv('analysis/feature_importance.csv', index=False)
    group_df.to_csv('analysis/feature_group_importance.csv', index=False)

# ============================================================================
# 4. ABLATION STUDY
# ============================================================================

print("\n4. Running ablation study...")

ablation_results = []

# Baseline (all features)
y_pred_baseline = np.expm1(best_model.predict(X_test))
baseline_mae = mean_absolute_error(y_true_test, y_pred_baseline)
baseline_r2 = r2_score(y_true_test, y_pred_baseline)

ablation_results.append({
    'Features Removed': 'None (Baseline)',
    'Num Features': len(feature_cols),
    'MAE': baseline_mae,
    'R2': baseline_r2,
    'MAE_Delta': 0,
    'R2_Delta': 0
})

print(f"\n   Baseline: MAE={baseline_mae:.3f}, R²={baseline_r2:.4f}")

# Test each feature group
for group_name, group_features in feature_groups.items():
    group_features = [f for f in group_features if f in feature_cols]
    if not group_features:
        continue
    
    print(f"\n   Testing without {group_name} ({len(group_features)} features)...")
    
    # Remove feature group
    remaining_features = [f for f in feature_cols if f not in group_features]
    X_train_ablation = X_train[remaining_features]
    X_test_ablation = X_test[remaining_features]
    
    # Train model
    model_ablation = XGBRegressor(
        learning_rate=0.05, max_depth=7, n_estimators=300,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    )
    model_ablation.fit(X_train_ablation, y_train)
    
    # Evaluate
    y_pred_ablation = np.expm1(model_ablation.predict(X_test_ablation))
    mae_ablation = mean_absolute_error(y_true_test, y_pred_ablation)
    r2_ablation = r2_score(y_true_test, y_pred_ablation)
    
    ablation_results.append({
        'Features Removed': group_name,
        'Num Features': len(remaining_features),
        'MAE': mae_ablation,
        'R2': r2_ablation,
        'MAE_Delta': mae_ablation - baseline_mae,
        'R2_Delta': r2_ablation - baseline_r2
    })
    
    print(f"      MAE={mae_ablation:.3f} (+{mae_ablation - baseline_mae:.3f}), R²={r2_ablation:.4f} ({r2_ablation - baseline_r2:+.4f})")

ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_csv('analysis/ablation_study.csv', index=False)

# Visualize ablation study
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MAE impact
ablation_sorted = ablation_df[ablation_df['Features Removed'] != 'None (Baseline)'].sort_values('MAE_Delta', ascending=False)
colors_mae = ['red' if x > 0 else 'green' for x in ablation_sorted['MAE_Delta']]
axes[0].barh(ablation_sorted['Features Removed'], ablation_sorted['MAE_Delta'], color=colors_mae)
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1)
axes[0].set_xlabel('Change in MAE (minutes)', fontsize=12)
axes[0].set_title('Ablation Study - MAE Impact\n(Positive = Performance Degradation)', 
                  fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# R² impact
colors_r2 = ['red' if x < 0 else 'green' for x in ablation_sorted['R2_Delta']]
axes[1].barh(ablation_sorted['Features Removed'], ablation_sorted['R2_Delta'], color=colors_r2)
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
axes[1].set_xlabel('Change in R²', fontsize=12)
axes[1].set_title('Ablation Study - R² Impact\n(Negative = Performance Degradation)', 
                  fontsize=12, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/figures/ablation_study.png', dpi=300, bbox_inches='tight')
print("\n   ✅ Saved: analysis/figures/ablation_study.png")

# ============================================================================
# 5. DATA OVERVIEW
# ============================================================================

print("\n5. Creating data overview...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Target distribution
axes[0, 0].hist(y, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].axvline(y.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {y.mean():.1f}')
axes[0, 0].axvline(y.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {y.median():.1f}')
axes[0, 0].set_xlabel('Prep Time (minutes)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Preparation Time Distribution', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Order size distribution
if 'num_items' in df.columns:
    axes[0, 1].hist(df['num_items'], bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
    axes[0, 1].set_xlabel('Number of Items', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Order Size Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

# Prep time by hour
if 'hour' in df.columns:
    hourly_prep = df.groupby('hour')[target_col].agg(['mean', 'std'])
    axes[1, 0].plot(hourly_prep.index, hourly_prep['mean'], marker='o', linewidth=2, markersize=6)
    axes[1, 0].fill_between(hourly_prep.index, 
                            hourly_prep['mean'] - hourly_prep['std'],
                            hourly_prep['mean'] + hourly_prep['std'],
                            alpha=0.3)
    axes[1, 0].set_xlabel('Hour of Day', fontsize=12)
    axes[1, 0].set_ylabel('Prep Time (minutes)', fontsize=12)
    axes[1, 0].set_title('Prep Time by Hour (with std)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

# Prep time by day
if 'day_of_week' in df.columns:
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily_prep = df.groupby('day_of_week')[target_col].mean()
    axes[1, 1].bar(range(7), daily_prep.values, color='mediumseagreen', edgecolor='black')
    axes[1, 1].set_xticks(range(7))
    axes[1, 1].set_xticklabels(day_names)
    axes[1, 1].set_ylabel('Avg Prep Time (minutes)', fontsize=12)
    axes[1, 1].set_title('Prep Time by Day of Week', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/figures/data_overview.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: analysis/figures/data_overview.png")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  📊 analysis/figures/model_comparison.png")
print("  📊 analysis/figures/prediction_quality.png")
print("  📊 analysis/figures/feature_importance_top20.png")
print("  📊 analysis/figures/feature_group_importance.png")
print("  📊 analysis/figures/ablation_study.png")
print("  📊 analysis/figures/data_overview.png")
print("  📄 analysis/feature_importance.csv")
print("  📄 analysis/feature_group_importance.csv")
print("  📄 analysis/ablation_study.csv")
