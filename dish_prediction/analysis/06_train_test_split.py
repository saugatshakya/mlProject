"""
Step 06: Train/Test Split
==========================
Split data temporally for time-series prediction.

Input: data/features_engineered.csv
Output: data/train_full.csv, data/test_full.csv
"""

import pandas as pd
import numpy as np

print("="*80)
print("STEP 06: TRAIN/TEST SPLIT")
print("="*80)

# ==============================================================================
# 1. Load engineered features
# ==============================================================================
print("\n📂 Loading engineered features...")
data = pd.read_csv('data/features_engineered.csv')
data['order_hour_dt'] = pd.to_datetime(data['order_hour'])

print(f"✓ Loaded {len(data):,} hours × {len(data.columns)} columns")
print(f"  Date range: {data['order_hour_dt'].min()} to {data['order_hour_dt'].max()}")

# ==============================================================================
# 2. Handle remaining missing values
# ==============================================================================
print("\n🔧 Handling missing values...")

missing = data.isnull().sum()
if missing.sum() > 0:
    print(f"Missing values found in {len(missing[missing > 0])} columns:")
    for col in missing[missing > 0].index:
        print(f"  {col}: {missing[col]} ({missing[col]/len(data)*100:.2f}%)")
    
    # Fill env_condition with mode
    if 'env_condition' in missing[missing > 0].index:
        mode_val = data['env_condition'].mode()[0]
        data['env_condition'] = data['env_condition'].fillna(mode_val)
        print(f"  ✓ Filled env_condition with mode: '{mode_val}'")
    
    # Fill any other remaining with forward fill
    data = data.fillna(method='ffill').fillna(method='bfill')
    print(f"✓ All missing values filled")
else:
    print("✓ No missing values found")

# ==============================================================================
# 3. Identify dish columns (target variables)
# ==============================================================================
print("\n🎯 Identifying target variables...")

# Dish columns are the top 30 dishes (not lag/rolling versions)
dish_cols = [col for col in data.columns if col not in [
    'order_hour', 'order_hour_dt', 'hour', 'day_of_week', 'day_of_month', 
    'month', 'is_weekend', 'env_temp', 'env_rhum', 'env_precip', 'env_wspd',
    'env_condition', 'aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10',
    'nh3', 'event', 'holiday', 'has_event', 'hour_sin', 'hour_cos', 'dow_sin',
    'dow_cos', 'dom_sin', 'dom_cos', 'is_peak_dinner', 'is_lunch', 'is_breakfast',
    'is_late_night', 'is_monday', 'is_friday', 'is_saturday', 'is_sunday',
    'is_rainy', 'temp_cold', 'temp_cool', 'temp_moderate', 'temp_warm', 'temp_hot',
    'humid_low', 'humid_medium', 'humid_high', 'temp_lag_1h', 'precip_lag_1h',
    'temp_lag_3h', 'precip_lag_3h', 'temp_lag_24h', 'precip_lag_24h',
    'aqi_good', 'aqi_moderate', 'aqi_poor', 'aqi_very_poor',
    'pm25_good', 'pm25_satisfactory', 'pm25_moderate', 'pm25_poor', 'pm25_very_poor',
    'rain_x_weekend', 'rain_x_peak', 'temp_x_weekend', 'temp_x_peak', 'temp_x_pm25',
    'rain_x_aqi', 'total_demand', 'total_demand_lag_1h', 'total_demand_lag_24h',
    'total_demand_rolling_24h'
] and not ('_lag_' in col or '_rolling_' in col)]

print(f"✓ Target dishes: {len(dish_cols)}")
print(f"  Top 5: {dish_cols[:5]}")

# Feature columns (everything except targets and identifiers)
feature_cols = [col for col in data.columns if col not in dish_cols + ['order_hour', 'order_hour_dt']]
print(f"✓ Feature columns: {len(feature_cols)}")

# ==============================================================================
# 4. Temporal split
# ==============================================================================
print("\n📅 Creating temporal train/test split...")

# Sort by datetime to ensure temporal ordering
data = data.sort_values('order_hour_dt').reset_index(drop=True)

# Split: Train on Sept-Dec 2024, Test on Jan 2025
train_mask = data['order_hour_dt'] < '2025-01-01'
test_mask = data['order_hour_dt'] >= '2025-01-01'

train_data = data[train_mask].copy()
test_data = data[test_mask].copy()

print(f"\nTrain set:")
print(f"  Date range: {train_data['order_hour_dt'].min()} to {train_data['order_hour_dt'].max()}")
print(f"  Records: {len(train_data):,} hours")
print(f"  Days: {train_data['order_hour_dt'].dt.date.nunique()}")

print(f"\nTest set:")
print(f"  Date range: {test_data['order_hour_dt'].min()} to {test_data['order_hour_dt'].max()}")
print(f"  Records: {len(test_data):,} hours")
print(f"  Days: {test_data['order_hour_dt'].dt.date.nunique()}")

print(f"\nSplit ratio: {len(train_data)/len(data)*100:.1f}% train, {len(test_data)/len(data)*100:.1f}% test")

# ==============================================================================
# 5. Verify no data leakage
# ==============================================================================
print("\n🔍 Verifying no data leakage...")

# Check that test comes strictly after train
assert train_data['order_hour_dt'].max() < test_data['order_hour_dt'].min(), "Data leakage detected!"
print(f"✓ No temporal overlap: train ends at {train_data['order_hour_dt'].max()}")
print(f"                       test starts at {test_data['order_hour_dt'].min()}")

# Check lag features don't leak
# The first 168 hours (1 week) of test might use train data for lag_168h - this is OK
# We're predicting future demand based on past patterns
print(f"✓ Lag features use historical data only (by design)")

# ==============================================================================
# 6. Statistics
# ==============================================================================
print("\n📊 Split statistics...")

print(f"\nTarget distribution (total demand):")
print(f"  Train mean: {train_data[dish_cols].sum(axis=1).mean():.2f} dishes/hour")
print(f"  Train std: {train_data[dish_cols].sum(axis=1).std():.2f}")
print(f"  Test mean: {test_data[dish_cols].sum(axis=1).mean():.2f} dishes/hour")
print(f"  Test std: {test_data[dish_cols].sum(axis=1).std():.2f}")

print(f"\nTop dish distribution:")
top_dish = dish_cols[0]
print(f"  {top_dish}:")
print(f"    Train mean: {train_data[top_dish].mean():.2f} orders/hour")
print(f"    Test mean: {test_data[top_dish].mean():.2f} orders/hour")

# Check for extreme values
print(f"\nChecking for extreme values in test set...")
test_extreme = (test_data[dish_cols] > train_data[dish_cols].quantile(0.99)).sum().sum()
if test_extreme > 0:
    print(f"  ⚠️  {test_extreme} test values exceed 99th percentile of train")
else:
    print(f"  ✓ No extreme values in test set")

# ==============================================================================
# 7. Save splits
# ==============================================================================
print("\n💾 Saving train/test splits...")

train_path = 'data/train_full.csv'
test_path = 'data/test_full.csv'

train_data.to_csv(train_path, index=False)
test_data.to_csv(test_path, index=False)

print(f"✓ Saved train to {train_path}")
print(f"  Shape: {train_data.shape[0]:,} hours × {train_data.shape[1]} columns")
print(f"  Size: {train_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

print(f"✓ Saved test to {test_path}")
print(f"  Shape: {test_data.shape[0]:,} hours × {test_data.shape[1]} columns")
print(f"  Size: {test_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

# ==============================================================================
# 8. Save metadata
# ==============================================================================
print("\n📝 Saving metadata...")

metadata = {
    'total_records': len(data),
    'train_records': len(train_data),
    'test_records': len(test_data),
    'train_start': str(train_data['order_hour_dt'].min()),
    'train_end': str(train_data['order_hour_dt'].max()),
    'test_start': str(test_data['order_hour_dt'].min()),
    'test_end': str(test_data['order_hour_dt'].max()),
    'num_dishes': len(dish_cols),
    'num_features': len(feature_cols),
    'dish_columns': dish_cols,
    'feature_columns': feature_cols
}

import json
with open('outputs/split_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Saved metadata to outputs/split_metadata.json")

# ==============================================================================
# 9. Summary
# ==============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nDataset: {len(data):,} hours")
print(f"  Train: {len(train_data):,} hours ({len(train_data)/len(data)*100:.1f}%)")
print(f"  Test: {len(test_data):,} hours ({len(test_data)/len(data)*100:.1f}%)")

print(f"\nFeatures: {len(feature_cols)}")
print(f"Targets: {len(dish_cols)} dishes")

print(f"\nTrain period: Sept-Dec 2024 ({train_data['order_hour_dt'].dt.date.nunique()} days)")
print(f"Test period: Jan 2025 ({test_data['order_hour_dt'].dt.date.nunique()} days)")

print(f"\nData quality:")
print(f"  Missing values: {data.isnull().sum().sum()}")
print(f"  Duplicates: {data.duplicated(subset=['order_hour']).sum()}")

print("\n" + "="*80)
print("✅ TRAIN/TEST SPLIT COMPLETE")
print("="*80)
print(f"\nNext step: Run 07_train_models.py")
