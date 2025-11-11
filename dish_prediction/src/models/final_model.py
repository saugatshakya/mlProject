"""
FINAL DEMAND PREDICTION MODEL
Based on the working approach from notebooks_backup/final.ipynb

Features:
- Temporal: hour, day_of_week, is_weekend, cyclical encoding
- Lag features: 1h, 2h, 3h for each dish  
- Smoothed dish history: 3-hour rolling mean
- Weather: temp, humidity, precipitation, wind
- Pollution: AQI (if available)
- Events: holidays, major events

Model: Multi-output CatBoost/XGBoost (predicts all dishes together)
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FINAL DEMAND PREDICTION MODEL")
print("="*80)

# Load the base hourly data with weather/pollution/events
print("\nLoading data...")
df = pd.read_csv('data/processed/hourly_data_with_features.csv')
df['hour'] = pd.to_datetime(df['hour'])
print(f"Loaded {len(df)} hours")

# Get dish columns
exclude_cols = ['hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'week_of_year', 
               'month', 'date', 'is_weekend', 'is_friday', 'is_peak_hour', 
               'is_lunch_rush', 'is_dinner_rush', 'is_late_night', 'meal_period',
               'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
               'env_temp', 'env_rhum', 'env_precip', 'env_wspd', 'env_condition',
               'aqi', 'pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2', 'nh3',
               'event', 'holiday', 'has_event']

dish_cols = [col for col in df.columns if col not in exclude_cols]
print(f"Found {len(dish_cols)} dishes")

# Select top 10 dishes by volume
dish_volumes = df[dish_cols].sum().sort_values(ascending=False)
top_dishes = dish_volumes.head(10).index.tolist()
print(f"\nTop 10 dishes:")
for i, dish in enumerate(top_dishes, 1):
    print(f"  {i}. {dish}: {dish_volumes[dish]:.0f} orders")

# Keep only top dishes
dish_cols = top_dishes
df_dishes = df[dish_cols].copy()

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\nEngineering features...")

features_df = pd.DataFrame(index=df.index)

# 1. TEMPORAL FEATURES
features_df['hour'] = df['hour'].dt.hour
features_df['day_of_week'] = df['hour'].dt.dayofweek
features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
features_df['sin_hour'] = np.sin(2 * np.pi * features_df['hour'] / 24)
features_df['cos_hour'] = np.cos(2 * np.pi * features_df['hour'] / 24)

# 2. LAG FEATURES (1, 2, 3 hours)
for dish in dish_cols:
    for lag in [1, 2, 3]:
        features_df[f'{dish}_lag{lag}'] = df[dish].shift(lag)

# 3. SMOOTHED DISH HISTORY (3-hour rolling mean)
for dish in dish_cols:
    features_df[f'{dish}_smooth'] = df[dish].rolling(window=3, min_periods=1).mean()

# 4. WEATHER FEATURES
if 'env_temp' in df.columns:
    features_df['env_temp'] = df['env_temp']
    features_df['env_rhum'] = df['env_rhum']
    features_df['env_precip'] = df['env_precip']
    features_df['env_wspd'] = df['env_wspd']

# 5. POLLUTION (if available)
if 'aqi' in df.columns:
    features_df['aqi'] = df['aqi']

# 6. EVENTS
if 'has_event' in df.columns:
    features_df['has_event'] = df['has_event'].astype(int)
if 'holiday' in df.columns:
    features_df['holiday'] = df['holiday'].astype(int)

# Fill NaN from lags with 0
features_df = features_df.fillna(0)

print(f"Created {len(features_df.columns)} features")

# ============================================================================
# TRAIN/TEST SPLIT (80-20 temporal)
# ============================================================================
split_idx = int(len(df) * 0.8)
X_train = features_df.iloc[:split_idx]
X_test = features_df.iloc[split_idx:]
y_train = df_dishes.iloc[:split_idx]
y_test = df_dishes.iloc[split_idx:]

print(f"\nTrain: {len(X_train)} hours | Test: {len(X_test)} hours")

# ============================================================================
# MODEL TRAINING
# ============================================================================
print("\n" + "="*80)
print("TRAINING MODELS")
print("="*80)

models = {
    'CatBoost': MultiOutputRegressor(
        CatBoostRegressor(
            iterations=300,
            depth=5,
            learning_rate=0.1,
            verbose=0,
            random_state=42
        )
    ),
    'XGBoost': MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Overall metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"  Overall Train R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
    print(f"  Overall Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}")
    
    # Per-dish metrics
    dish_metrics = []
    for i, dish in enumerate(dish_cols):
        dish_r2 = r2_score(y_test[dish], y_test_pred[:, i])
        dish_mae = mean_absolute_error(y_test[dish], y_test_pred[:, i])
        dish_metrics.append({
            'model': name,
            'dish': dish,
            'test_r2': dish_r2,
            'test_mae': dish_mae
        })
        print(f"    {dish}: R² = {dish_r2:.4f}, MAE = {dish_mae:.4f}")
    
    results.extend(dish_metrics)
    
    # Save best model
    if name == 'CatBoost':  # CatBoost usually performs best
        output_dir = Path('models/final')
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_dir / 'catboost_multi_output.pkl')
        print(f"\n  ✓ Model saved to {output_dir / 'catboost_multi_output.pkl'}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(['model', 'test_r2'], ascending=[True, False])

output_dir = Path('reports')
output_dir.mkdir(exist_ok=True)
results_df.to_csv(output_dir / 'final_model_results.csv', index=False)
print(f"\n✓ Results saved to {output_dir / 'final_model_results.csv'}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\nCreating visualizations...")
fig_dir = Path('reports/figures/final_model')
fig_dir.mkdir(parents=True, exist_ok=True)

# 1. Model comparison
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=results_df, x='dish', y='test_r2', hue='model', ax=ax)
ax.set_xlabel('Dish', fontsize=12)
ax.set_ylabel('Test R²', fontsize=12)
ax.set_title('Model Performance by Dish', fontsize=14, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(fig_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Top 3 dishes - predictions vs actuals
best_model_name = 'CatBoost'
best_model = models[best_model_name]
y_pred = best_model.predict(X_test)

top_3_dishes = results_df[results_df['model'] == best_model_name].head(3)['dish'].tolist()
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, dish in enumerate(top_3_dishes):
    ax = axes[idx]
    dish_idx = dish_cols.index(dish)
    ax.scatter(y_test[dish], y_pred[:, dish_idx], alpha=0.5)
    ax.plot([y_test[dish].min(), y_test[dish].max()], 
            [y_test[dish].min(), y_test[dish].max()], 
            'r--', lw=2)
    r2 = results_df[(results_df['model'] == best_model_name) & (results_df['dish'] == dish)]['test_r2'].values[0]
    ax.set_xlabel('Actual', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title(f'{dish}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / 'predictions_vs_actuals.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Figures saved to {fig_dir}")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\nSaving models...")
model_dir = Path('models/final')
model_dir.mkdir(parents=True, exist_ok=True)

# Save CatBoost model (best performer)
joblib.dump(models['CatBoost'], model_dir / 'catboost_model.pkl')
print(f"✓ CatBoost model saved to {model_dir / 'catboost_model.pkl'}")

# Save XGBoost model  
joblib.dump(models['XGBoost'], model_dir / 'xgboost_model.pkl')
print(f"✓ XGBoost model saved to {model_dir / 'xgboost_model.pkl'}")

# Save feature names for reference
feature_names = list(features_df.columns)
with open(model_dir / 'feature_names.txt', 'w') as f:
    for feat in feature_names:
        f.write(f"{feat}\n")
print(f"✓ Feature names saved to {model_dir / 'feature_names.txt'}")

# Save dish names
with open(model_dir / 'dish_names.txt', 'w') as f:
    for dish in dish_cols:
        f.write(f"{dish}\n")
print(f"✓ Dish names saved to {model_dir / 'dish_names.txt'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

for model_name in ['CatBoost', 'XGBoost']:
    model_results = results_df[results_df['model'] == model_name]
    print(f"\n{model_name}:")
    print(f"  Mean R²: {model_results['test_r2'].mean():.4f}")
    print(f"  Mean MAE: {model_results['test_mae'].mean():.4f}")
    print(f"  Best dish: {model_results.iloc[0]['dish']} (R² = {model_results.iloc[0]['test_r2']:.4f})")

print("\n" + "="*80)
print("✅ DONE!")
print("="*80)
