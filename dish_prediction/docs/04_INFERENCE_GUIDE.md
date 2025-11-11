# Inference Guide - Making Predictions

This guide explains how to make predictions with the trained model using your existing data.

---

## 🚀 Quick Start

### Simple Inference (Recommended)

```bash
cd /Users/saugatshakya/Projects/ML2025/project/dish_prediction
python inference_simple.py
```

**Output**:

```
PREDICTIONS FOR NEXT HOUR
================================================================================
Timestamp: 2025-11-09 18:05
Weather: 11.9°C, 100.0% humidity
Air Quality: AQI 5
Data Source: Existing CSV files ✓
--------------------------------------------------------------------------------
Bageecha Pizza                          :    0.2 orders
Chilli Cheese Garlic Bread              :    0.2 orders
Bone in Jamaican Grilled Chicken        :    3.2 orders
All About Chicken Pizza                 :    1.0 orders
Makhani Paneer Pizza                    :    1.0 orders
Margherita Pizza                        :    1.0 orders
Cheesy Garlic Bread                     :    0.1 orders
Jamaican Chicken Melt                   :    1.0 orders
Herbed Potato                           :    0.0 orders
Tripple Cheese Pizza                    :    1.2 orders
--------------------------------------------------------------------------------
TOTAL                                   :    8.9 orders
```

---

## 📋 Features Required for Inference

### Total: 52 Features

#### 1. Temporal Features (5)

- `hour_of_day` (0-23)
- `day_of_week` (0-6, Monday=0)
- `is_weekend` (0 or 1)
- `sin_hour` = sin(2π × hour / 24)
- `cos_hour` = cos(2π × hour / 24)

#### 2. Historical Features (40)

For each of the 10 dishes, need:

- `{dish}_lag1`: Orders 1 hour ago
- `{dish}_lag2`: Orders 2 hours ago
- `{dish}_lag3`: Orders 3 hours ago
- `{dish}_smooth`: Rolling 3-hour average

**10 dishes × 4 features = 40 features**

#### 3. Weather Features (4)

- `env_temp`: Temperature in °C
- `env_rhum`: Relative humidity (%)
- `env_precip`: Precipitation (mm)
- `env_wspd`: Wind speed (km/h)

#### 4. Pollution Features (2)

- `aqi`: Air Quality Index
- (Note: Full model uses 6, but only AQI is critical)

#### 5. Event Features (2)

- `holiday`: Is it a holiday? (0 or 1)
- `has_event`: Is there an event? (0 or 1)

---

## 📊 Data Sources

### Your Existing Data (Already Available!)

✅ **Order History**: `data/processed/hourly_data_with_features.csv`

- Contains historical orders for all dishes
- Used to create lag features

✅ **Weather Data**: `data/hourly_orders_weather.csv`

- Columns: `env_temp`, `env_rhum`, `env_precip`, `env_wspd`, `env_condition`
- Latest row provides current weather

✅ **Pollution Data**: `data/pollution.csv`

- Columns: `aqi`, `pm2_5`, `pm10`, `no2`, `o3`, `co`, `so2`, `nh3`
- Latest row provides current pollution levels

✅ **Events**: `data/events.csv`

- Contains holidays and special events
- Check if current date matches

---

## 🔧 Inference Methods

### Method 1: Using Existing CSV Data (Current Implementation)

```python
from inference_simple import predict_next_hour_from_existing_data

# Make prediction using latest data from CSV files
predictions = predict_next_hour_from_existing_data()
```

**How it works**:

1. Loads latest weather from `hourly_orders_weather.csv`
2. Loads latest pollution from `pollution.csv`
3. Extracts last 3 hours of orders from processed data
4. Creates all 52 features automatically
5. Makes prediction with trained model

**Advantages**:

- ✅ No external APIs needed
- ✅ Uses your existing data pipeline
- ✅ Simple and reliable

---

### Method 2: Manual Feature Creation

```python
import pandas as pd
import joblib
from datetime import datetime
import numpy as np

# Load model
model = joblib.load('models/final/catboost_model.pkl')

# Create features manually
features = {}

# 1. Temporal
now = datetime.now()
features['hour_of_day'] = now.hour
features['day_of_week'] = now.weekday()
features['is_weekend'] = 1 if now.weekday() >= 5 else 0
features['sin_hour'] = np.sin(2 * np.pi * now.hour / 24)
features['cos_hour'] = np.cos(2 * np.pi * now.hour / 24)

# 2. Historical (get from your database/logs)
recent_orders = get_last_3_hours_orders()  # Your function
for dish in TOP_DISHES:
    dish_orders = recent_orders[recent_orders['dish'] == dish]
    features[f'{dish}_lag1'] = dish_orders.iloc[0]['quantity']
    features[f'{dish}_lag2'] = dish_orders.iloc[1]['quantity']
    features[f'{dish}_lag3'] = dish_orders.iloc[2]['quantity']
    features[f'{dish}_smooth'] = dish_orders['quantity'].mean()

# 3. Weather (get from your API/database)
current_weather = get_current_weather()  # Your function
features['env_temp'] = current_weather['temperature']
features['env_rhum'] = current_weather['humidity']
features['env_precip'] = current_weather['precipitation']
features['env_wspd'] = current_weather['wind_speed']

# 4. Pollution (get from your API/database)
current_pollution = get_current_pollution()  # Your function
features['aqi'] = current_pollution['aqi']

# 5. Events (check calendar)
features['holiday'] = is_holiday(now)  # Your function
features['has_event'] = has_special_event(now)  # Your function

# Make prediction
X = pd.DataFrame([features])
predictions = model.predict(X)[0]  # Returns array of 10 predictions

# Format results
for dish, pred in zip(TOP_DISHES, predictions):
    print(f"{dish}: {max(0, pred):.1f} orders")
```

---

### Method 3: Simplified Model (Recommended based on Ablation Study)

**Based on ablation study findings**, you can use **ONLY HISTORICAL** features:

```python
# Only need 40 features instead of 52!
features = {}

recent_orders = get_last_3_hours_orders()
for dish in TOP_DISHES:
    dish_orders = recent_orders[recent_orders['dish'] == dish]
    features[f'{dish}_lag1'] = dish_orders.iloc[0]['quantity']
    features[f'{dish}_lag2'] = dish_orders.iloc[1]['quantity']
    features[f'{dish}_lag3'] = dish_orders.iloc[2]['quantity']
    features[f'{dish}_smooth'] = dish_orders['quantity'].mean()

# Make prediction (retrain model with only historical features first)
X = pd.DataFrame([features])
predictions = simplified_model.predict(X)[0]
```

**Advantages**:

- ✅ **Better performance** (R² = 0.9545 vs 0.9417)
- ✅ **No weather/pollution needed**
- ✅ **Simpler infrastructure**
- ✅ **Faster predictions**

---

## 🎯 Top 10 Dishes (in order)

```python
TOP_DISHES = [
    'Bageecha Pizza',
    'Chilli Cheese Garlic Bread',
    'Bone in Jamaican Grilled Chicken',
    'All About Chicken Pizza',
    'Makhani Paneer Pizza',
    'Margherita Pizza',
    'Cheesy Garlic Bread',
    'Jamaican Chicken Melt',
    'Herbed Potato',
    'Tripple Cheese Pizza'
]
```

**Important**: Predictions are returned in this exact order!

---

## ⚠️ Important Notes

### Feature Order Matters!

The model expects features in a **specific order**. Use:

```python
feature_names = model.estimators_[0].feature_names_
```

to get the exact order.

### Handling Missing Data

If any historical data is missing:

```python
# Use 0 for missing lag values
features[f'{dish}_lag1'] = orders.get(hour_ago_1, 0)

# Use available data for smooth
available_hours = [h for h in last_3_hours if h in orders]
features[f'{dish}_smooth'] = np.mean([orders[h] for h in available_hours])
```

### Default Values

For missing weather/pollution:

```python
# Safe defaults based on median values
defaults = {
    'env_temp': 25.0,      # °C
    'env_rhum': 60.0,      # %
    'env_precip': 0.0,     # mm
    'env_wspd': 10.0,      # km/h
    'aqi': 100,            # AQI
    'holiday': 0,
    'has_event': 0
}
```

---

## 🔄 Real-Time Inference Pipeline

### Recommended Architecture

```
┌─────────────────┐
│  New Order      │
│  Received       │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Update Order   │
│  History DB     │
└────────┬────────┘
         │
         v
┌─────────────────┐      ┌──────────────┐
│  Every Hour     │─────>│ Get Last     │
│  Trigger        │      │ 3 Hours      │
└─────────────────┘      └──────┬───────┘
                                │
                                v
                         ┌──────────────┐
                         │ Create       │
                         │ Features     │
                         └──────┬───────┘
                                │
                                v
                         ┌──────────────┐
                         │ Model        │
                         │ Prediction   │
                         └──────┬───────┘
                                │
                                v
                         ┌──────────────┐
                         │ Save to DB/  │
                         │ Send Alert   │
                         └──────────────┘
```

---

## 📦 Dependencies

```python
# Required packages
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
```

**Versions** (from project):

- pandas >= 1.3.0
- numpy >= 1.21.0
- joblib >= 1.0.0
- catboost >= 1.0.0 (if loading model)

---

## 🐛 Troubleshooting

### Error: "Feature X not found"

**Solution**: Check feature names match exactly:

```python
print(model.estimators_[0].feature_names_)
```

### Error: "Expected 52 features, got 40"

**Solution**: Using wrong model version. Either:

1. Use full model with all 52 features, OR
2. Retrain with only historical features (recommended)

### Predictions are all zeros

**Causes**:

1. Model file corrupted → Retrain
2. Wrong feature order → Check feature_names
3. All lag features are 0 → Need actual historical data

### Predictions are negative

**Solution**: Clip predictions:

```python
predictions = np.maximum(0, model.predict(X)[0])
```

---

## 🎓 Best Practices

### 1. Validate Input Data

```python
# Check for missing values
assert not features.isnull().any(), "Missing values in features!"

# Check feature count
assert len(features) == 52, f"Expected 52 features, got {len(features)}"

# Check value ranges
assert 0 <= features['hour_of_day'] <= 23
assert 0 <= features['day_of_week'] <= 6
```

### 2. Log Predictions

```python
import logging

logging.info(f"Prediction at {datetime.now()}: {predictions}")
```

### 3. Monitor Performance

```python
# Track actual vs predicted
actual_orders = get_actual_orders(hour)
error = abs(actual_orders - predictions)
mae = error.mean()

logging.info(f"MAE this hour: {mae:.2f}")
```

### 4. Handle Edge Cases

```python
# Very first prediction (no history)
if not has_historical_data():
    # Use average orders from training data
    predictions = AVERAGE_ORDERS_PER_DISH

# Extreme weather
if weather['temp'] < -10 or weather['temp'] > 50:
    logging.warning(f"Extreme temperature: {weather['temp']}")
    # Use safe defaults or flag for review
```

---

## 📊 Expected Performance

Based on test data evaluation:

| Metric        | Expected Value |
| ------------- | -------------- |
| Mean R²       | 0.9494         |
| Mean MAE      | 0.657 orders   |
| Best Dish R²  | 0.9913         |
| Worst Dish R² | 0.8107         |

**Per-dish accuracy varies**. Some dishes more predictable than others.

---

## 🔗 Related Files

- `inference_simple.py` - Simple inference using CSV data
- `inference_production.py` - Production inference with API integration
- `api_integration.py` - Weather/pollution API functions
- `src/models/final_model.py` - Model training script

---

## ✅ Recommended Workflow

### For Production Deployment

1. **Simplify the Model** (based on ablation study)

   ```bash
   # Retrain with only historical features
   python src/models/train_historical_only.py
   ```

2. **Remove External Dependencies**

   - No weather API needed
   - No pollution API needed
   - Only need order history

3. **Set Up Hourly Cron Job**

   ```bash
   0 * * * * cd /path/to/project && python inference_simple.py >> predictions.log
   ```

4. **Monitor and Alert**
   - Track prediction accuracy
   - Alert on anomalies
   - Retrain monthly with new data

---

_Last Updated: November 9, 2025_  
_Model Version: CatBoost Multi-Output v1.0_  
_Based on: Ablation Study Findings_
