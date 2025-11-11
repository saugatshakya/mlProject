
"""
Step 05: Hypothesis-driven Feature Engineering
===============================================
This script creates a compact, hypothesis-driven feature set tailored to the
dataset and the earlier statistical findings. It keeps the useful lag/rolling
features but avoids exploding the feature space unnecessarily.

Inputs:
  - data/dish_pivot.csv

Outputs:
  - data/features_engineered_v2.csv
  - outputs/feature_list.json

Key focuses (driven by hypotheses):
  - Rain: intensity, day counts, recent rain lags
  - Weekend & peak interactions
  - Temperature: bins + deviation from optimal (12.5°C)
  - Pollution: AQI one-hot, pm2.5 rolling, pollution lags
  - Event / holiday aggregation
  - Chicken flag per target dish (meta saved in feature list)

"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# 1) Load pivot data
# -----------------------------
print('\n📂 Loading pivot data...')
data = pd.read_csv('data/dish_pivot.csv')
data['order_hour_dt'] = pd.to_datetime(data['order_hour'])

print(f'✓ Loaded {len(data):,} hours × {len(data.columns)} columns')
print(f'  Date range: {data.order_hour_dt.min()} to {data.order_hour_dt.max()}')

# identify dish columns (top targets)
base_cols = ['order_hour', 'order_hour_dt', 'hour', 'day_of_week', 'day_of_month',
             'month', 'is_weekend', 'env_temp', 'env_rhum', 'env_precip', 'env_wspd',
             'env_condition', 'aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10',
             'nh3', 'event', 'holiday', 'has_event']

dish_cols = [c for c in data.columns if c not in base_cols]
print(f'✓ Found {len(dish_cols)} target dishes')

# Ensure hour is numeric and create common temporal flags used by hypotheses
data['hour'] = data['hour'].astype(int)
# Dinner peak as used in H1 (19-21), midday off-peak (11-16) used for comparisons
data['is_peak_dinner'] = data['hour'].isin([19,20,21]).astype(int)
data['is_offpeak_midday'] = data['hour'].between(11,16).astype(int)

# -----------------------------
 
# -----------------------------
# 4) Rain features (hypothesis H3)
# -----------------------------
print('\n🌧️  Creating rain features aligned to hypothesis H3...')
data['rain_amount'] = data['env_precip'].fillna(0)
data['is_rainy'] = (data['rain_amount'] > 0).astype(int)

# Daily rain count (hours per day with rain)
data['date'] = data['order_hour_dt'].dt.date
rain_hours = data.groupby('date')['is_rainy'].sum().rename('rain_hours_per_day')
data = data.merge(rain_hours.reset_index(), on='date', how='left')

# Recent rain features
data['rain_lag_1h'] = data['is_rainy'].shift(1).fillna(0)
data['rain_lag_24h'] = data['is_rainy'].shift(24).fillna(0)
data['rain_prev_day_hours'] = data['rain_hours_per_day'].shift(1).fillna(0)

print('✓ Rain features created')


# -----------------------------
# 5) Temperature features (hypothesis H4)
# -----------------------------
print('\n🌡️ Creating temperature features aligned to hypothesis H4...')
data['temp'] = data['env_temp']
# deviation from optimal center (midpoint ~12.5°C)
data['temp_dev_opt'] = (data['temp'] - 12.5).abs()

# bins (consistent with earlier analysis)
data['temp_bin'] = pd.cut(data['temp'], bins=[-np.inf,10,15,20,25,30,np.inf],
                          labels=['<10','10-15','15-20','20-25','25-30','>30'])
for b in data['temp_bin'].cat.categories:
    data[f'temp_bin_{b}'] = (data['temp_bin'] == b).astype(int)

print('✓ Temperature features created')


# -----------------------------
# 6) Pollution features (hypothesis H5)
# -----------------------------
print('\n🏭 Creating pollution features aligned to hypothesis H5...')
data['aqi'] = data['aqi'].fillna(5)  # default to Very Poor if missing (conservative)

# one-hot AQI categories (as used earlier)
data['aqi_1_2'] = (data['aqi'] <= 2).astype(int)
data['aqi_3'] = (data['aqi'] == 3).astype(int)
data['aqi_4'] = (data['aqi'] == 4).astype(int)
data['aqi_5plus'] = (data['aqi'] >= 5).astype(int)

# PM2.5 rolling (recent exposure)
if 'pm2_5' in data.columns:
    data['pm25_rolling_24h'] = data['pm2_5'].rolling(window=24, min_periods=1).mean()
    data['pm25_lag_24h'] = data['pm2_5'].shift(24).fillna(data['pm2_5'].mean())

print('✓ Pollution features ready')


# -----------------------------
# 7) Event features
# -----------------------------
print('\n🎉 Creating event/holiday features...')
data['holiday'] = data['holiday'].fillna(0).astype(int)
data['has_event'] = data['has_event'].fillna(0).astype(int)

# events per day
events_per_day = data.groupby('date')['has_event'].sum().rename('events_per_day')
data = data.merge(events_per_day.reset_index(), on='date', how='left')

print('✓ Event features created')


# -----------------------------
# 8) Chicken flags for targets (H6)
# -----------------------------
print('\n🍗 Creating chicken flags for target dishes (H6)...')
chicken_keywords = ['Chicken', 'chicken', 'Murgh', 'Tender', 'Tangdi','Grilled']
dish_meta = {}
for dish in dish_cols:
    is_chicken = any(kw.lower() in dish.lower() for kw in chicken_keywords)
    dish_meta[dish] = {'is_chicken': bool(is_chicken)}

print(f'✓ Chicken flags created for {len(dish_cols)} dishes')


# -----------------------------
# 9) Controlled lag & rolling features (avoid explosion)
# -----------------------------
print('\n⏮️ Creating controlled lag and rolling features (selective)')

# We'll create lags/rolling for total_demand and top-K dishes only (K=10)
K = min(10, len(dish_cols))
top_k_dishes = dish_cols[:K]

# aggregate demand
data['total_demand'] = data[top_k_dishes].sum(axis=1)
data['total_demand_lag_1h'] = data['total_demand'].shift(1).fillna(0)
data['total_demand_lag_24h'] = data['total_demand'].shift(24).fillna(0)
data['total_demand_rolling_24h'] = data['total_demand'].rolling(window=24, min_periods=1).mean()

# per-dish selective lags
for dish in top_k_dishes:
    data[f'{dish}_lag_1h'] = data[dish].shift(1).fillna(0)
    data[f'{dish}_lag_24h'] = data[dish].shift(24).fillna(0)
    data[f'{dish}_rolling_24h_mean'] = data[dish].rolling(window=24, min_periods=1).mean()

print(f'✓ Created selective lag/rolling for top {K} dishes')


# -----------------------------
# 10) Interaction features (hypothesis-driven)
# -----------------------------
print('\n🔀 Creating interaction features...')
data['rain_x_weekend'] = data['is_rainy'] * data['is_weekend']
data['rain_x_peak'] = data['is_rainy'] * data['is_peak_dinner']
data['temp_dev_x_rain'] = data['temp_dev_opt'] * data['is_rainy']
data['aqi_x_rain'] = data['aqi_5plus'] * data['is_rainy']

print('✓ Interaction features done')


# -----------------------------
# 11) Final housekeeping: drop helpers, save
# -----------------------------
print('\n🧾 Finalizing features and saving list...')

# define final feature list (exclude raw target dish columns)
exclude = ['order_hour', 'order_hour_dt', 'date'] + dish_cols
feature_cols = [c for c in data.columns if c not in exclude]

# Save features and metadata
out_path = Path('data/features_engineered_v2.csv')
data.to_csv(out_path, index=False)

feature_list = {
    'feature_cols': feature_cols,
    'dish_cols': dish_cols,
    'dish_meta': dish_meta,
    'notes': 'Hypothesis-driven feature set: rain, temp, pollution, events, selective lags.'
}
Path('outputs').mkdir(parents=True, exist_ok=True)
with open('outputs/feature_list.json', 'w') as f:
    json.dump(feature_list, f, indent=2)

print(f'✓ Saved engineered features to {out_path} ({data.shape[0]} rows × {len(feature_cols)} features)')
print('✓ Saved feature list to outputs/feature_list.json')

missing = data[feature_cols].isnull().sum().sum()
print(f'\nMissing values in final feature columns: {missing}')

print('\n' + '=' * 80)
print('✅ FEATURE ENGINEERING (v2) COMPLETE')
print('=' * 80)
print('\nNext step: Run 06_train_test_split.py to refresh splits (if wanted)')
