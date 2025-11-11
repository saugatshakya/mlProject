# 🔮 INFERENCE GUIDE - Dish Demand Prediction

## 📋 FEATURES REQUIRED FOR INFERENCE

To make predictions for the **next hour**, you need the following data:

### 1️⃣ **CURRENT TIME INFORMATION** (Required)

- `current_hour`: Current hour (0-23)
- `current_day_of_week`: Day of week (0=Monday, 6=Sunday)
- `is_weekend`: Boolean (True if Saturday/Sunday)

These are automatically derived from the current timestamp.

---

### 2️⃣ **HISTORICAL ORDER DATA** (Required)

You need the **last 3 hours** of order data for each dish you're predicting.

For **each of the 10 dishes**, you need:

- Orders from **1 hour ago**
- Orders from **2 hours ago**
- Orders from **3 hours ago**

**Example for "Bageecha Pizza":**

```python
{
    "Bageecha Pizza_lag1": 5,  # orders 1 hour ago
    "Bageecha Pizza_lag2": 3,  # orders 2 hours ago
    "Bageecha Pizza_lag3": 4,  # orders 3 hours ago
    "Bageecha Pizza_smooth": 4.0  # 3-hour average: (5+3+4)/3
}
```

**Top 10 Dishes to Track:**

1. Bageecha Pizza
2. Chilli Cheese Garlic Bread
3. Bone in Jamaican Grilled Chicken
4. All About Chicken Pizza
5. Makhani Paneer Pizza
6. Margherita Pizza
7. Cheesy Garlic Bread
8. Jamaican Chicken Melt
9. Herbed Potato
10. Tripple Cheese Pizza

---

### 3️⃣ **WEATHER DATA** (Required)

Get current weather from API (e.g., OpenWeatherMap, WeatherAPI):

- `env_temp`: Temperature in Celsius (e.g., 25.5)
- `env_rhum`: Relative humidity percentage (e.g., 65)
- `env_precip`: Precipitation in mm (e.g., 0.0)
- `env_wspd`: Wind speed in km/h (e.g., 12.5)

**🔌 API Integration Options:**

**OPTION 1: WeatherAPI.com** (Recommended - Easiest)

- **Free Tier**: 1 million calls/month
- **Sign up**: https://www.weatherapi.com/signup.aspx
- **Includes**: Weather + Air Quality in one call
- **Setup**:
  ```bash
  export WEATHERAPI_KEY="your_api_key_here"
  ```

**OPTION 2: OpenWeatherMap**

- **Free Tier**: 60 calls/min, 1 million calls/month
- **Sign up**: https://home.openweathermap.org/users/sign_up
- **Setup**:
  ```bash
  export OPENWEATHER_API_KEY="your_api_key_here"
  ```

**Example Code** (see `api_integration.py`):

```python
from api_integration import get_weather_weatherapi

weather, pollution = get_weather_weatherapi("Delhi")
# Returns: {'temp': 28.5, 'humidity': 60, 'precipitation': 0.0, 'wind_speed': 8.2}
```

**If API unavailable**, the model uses defaults:

```python
{
    "env_temp": 25.0,    # Default temperature
    "env_rhum": 60,      # Default humidity
    "env_precip": 0.0,   # No rain
    "env_wspd": 10.0     # Moderate wind
}
```

---

### 4️⃣ **POLLUTION DATA** (Optional but improves accuracy)

Get air quality from API:

- `aqi`: Air Quality Index (0-500)

**🔌 API Options:**

- **WeatherAPI.com**: Includes AQI in weather call (easiest)
- **OpenWeatherMap Air Pollution API**: Separate endpoint
- **AQI API**: https://aqicn.org/api/ (free for non-commercial)

**Example Code** (see `api_integration.py`):

```python
from api_integration import get_pollution_openweathermap

pollution = get_pollution_openweathermap(lat=28.6139, lon=77.2090)
# Returns: {'aqi': 150}
```

**If API unavailable**, set to default:

```python
{
    "aqi": 100  # Moderate pollution (typical for Delhi)
}
```

Model will still work, but predictions may be slightly less accurate.

---

### 5️⃣ **EVENT/HOLIDAY FLAGS** (Optional but improves accuracy)

Check if current date is special:

- `has_event`: 1 if major event happening in Delhi, 0 otherwise
- `holiday`: 1 if public holiday, 0 otherwise

**Example:**

```python
{
    "has_event": 0,  # No special event
    "holiday": 0     # Not a holiday
}
```

If not available, set both to `0`.

---

## 📊 COMPLETE FEATURE LIST

Here's the exact list of features the model expects (in order):

### Temporal Features (5)

```python
['hour', 'day_of_week', 'is_weekend', 'sin_hour', 'cos_hour']
```

### Lag Features for Each Dish (30 features = 10 dishes × 3 lags)

```python
[
    'Bageecha Pizza_lag1', 'Bageecha Pizza_lag2', 'Bageecha Pizza_lag3',
    'Chilli Cheese Garlic Bread_lag1', 'Chilli Cheese Garlic Bread_lag2', 'Chilli Cheese Garlic Bread_lag3',
    'Bone in Jamaican Grilled Chicken_lag1', 'Bone in Jamaican Grilled Chicken_lag2', 'Bone in Jamaican Grilled Chicken_lag3',
    'All About Chicken Pizza_lag1', 'All About Chicken Pizza_lag2', 'All About Chicken Pizza_lag3',
    'Makhani Paneer Pizza_lag1', 'Makhani Paneer Pizza_lag2', 'Makhani Paneer Pizza_lag3',
    'Margherita Pizza_lag1', 'Margherita Pizza_lag2', 'Margherita Pizza_lag3',
    'Cheesy Garlic Bread_lag1', 'Cheesy Garlic Bread_lag2', 'Cheesy Garlic Bread_lag3',
    'Jamaican Chicken Melt_lag1', 'Jamaican Chicken Melt_lag2', 'Jamaican Chicken Melt_lag3',
    'Herbed Potato_lag1', 'Herbed Potato_lag2', 'Herbed Potato_lag3',
    'Tripple Cheese Pizza_lag1', 'Tripple Cheese Pizza_lag2', 'Tripple Cheese Pizza_lag3'
]
```

### Smoothed History Features (10 features = 10 dishes)

```python
[
    'Bageecha Pizza_smooth',
    'Chilli Cheese Garlic Bread_smooth',
    'Bone in Jamaican Grilled Chicken_smooth',
    'All About Chicken Pizza_smooth',
    'Makhani Paneer Pizza_smooth',
    'Margherita Pizza_smooth',
    'Cheesy Garlic Bread_smooth',
    'Jamaican Chicken Melt_smooth',
    'Herbed Potato_smooth',
    'Tripple Cheese Pizza_smooth'
]
```

### Weather Features (4)

```python
['env_temp', 'env_rhum', 'env_precip', 'env_wspd']
```

### Pollution Features (1)

```python
['aqi']
```

### Event Features (2)

```python
['has_event', 'holiday']
```

**TOTAL: 52 features**

---

## 💻 PYTHON INFERENCE EXAMPLE

```python
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load the trained model
model = joblib.load('models/final/catboost_model.pkl')

# Current time
now = datetime.now()
current_hour = now.hour
current_dow = now.weekday()
is_weekend = 1 if current_dow >= 5 else 0

# Temporal features
temporal_features = {
    'hour': current_hour,
    'day_of_week': current_dow,
    'is_weekend': is_weekend,
    'sin_hour': np.sin(2 * np.pi * current_hour / 24),
    'cos_hour': np.cos(2 * np.pi * current_hour / 24)
}

# Historical data (last 3 hours for each dish)
# You would fetch this from your database
historical_data = {
    'Bageecha Pizza_lag1': 5,
    'Bageecha Pizza_lag2': 3,
    'Bageecha Pizza_lag3': 4,
    'Bageecha Pizza_smooth': 4.0,
    # ... repeat for all 10 dishes
}

# Weather data (from API)
weather_data = {
    'env_temp': 28.5,
    'env_rhum': 60,
    'env_precip': 0.0,
    'env_wspd': 8.2
}

# Pollution data (from API)
pollution_data = {
    'aqi': 150
}

# Event flags
event_data = {
    'has_event': 0,
    'holiday': 0
}

# Combine all features
features = {
    **temporal_features,
    **historical_data,
    **weather_data,
    **pollution_data,
    **event_data
}

# Create DataFrame with single row
X = pd.DataFrame([features])

# Make prediction
predictions = model.predict(X)[0]

# Map predictions to dishes
dishes = [
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

print("Predictions for next hour:")
for dish, pred in zip(dishes, predictions):
    print(f"{dish}: {pred:.1f} orders")
```

---

## 🗄️ MINIMUM DATA STORAGE REQUIREMENTS

To run inference continuously, you need to store:

### 1. **Order History Database**

- Keep rolling window of **last 3 hours** of orders
- Store: `dish_name`, `quantity`, `timestamp`
- Update every hour

### 2. **Weather API Integration**

- Call weather API before each prediction
- No need to store (unless API has rate limits)

### 3. **Pollution API Integration** (optional)

- Call AQI API before each prediction
- Can cache for 1 hour (pollution doesn't change rapidly)

### 4. **Holiday/Event Calendar**

- Static list of holidays (update yearly)
- Event list (update as events are planned)

---

## 📡 REAL-TIME INFERENCE WORKFLOW

```
1. Current Time → Extract temporal features
                  ↓
2. Database Query → Get last 3 hours orders for 10 dishes
                  ↓
3. Weather API → Get current temperature, humidity, precipitation, wind
                  ↓
4. Air Quality API → Get current AQI (optional)
                  ↓
5. Calendar Check → Is today a holiday? Any events?
                  ↓
6. Combine Features → Create feature vector (52 features)
                  ↓
7. Model.predict() → Get predictions for 10 dishes
                  ↓
8. Return Predictions → Expected orders for next hour
```

**Processing Time:** < 100ms (model inference is very fast)

---

## 🔑 KEY POINTS

1. **Most Important Features:**

   - ✅ Last 3 hours of order history (lag features)
   - ✅ Current time (hour, day of week)
   - ✅ Weather (especially temperature)

2. **Nice to Have (but not critical):**

   - Pollution data (AQI)
   - Event/holiday flags

3. **If Data is Missing:**

   - Historical orders: **CANNOT PREDICT** (critical)
   - Weather: Use **default values** (temp=25, humidity=60)
   - Pollution: Set **aqi=0**
   - Events: Set **both to 0**

4. **Feature Engineering is Simple:**
   - Just fetch raw data
   - Calculate lags (shift by 1,2,3 hours)
   - Calculate rolling mean (last 3 hours)
   - No complex transformations needed!

---

## 📁 SAVED MODEL LOCATION

The trained model is saved at:

```
dish_prediction/models/final/catboost_model.pkl
```

Load it with:

```python
import joblib
model = joblib.load('models/final/catboost_model.pkl')
```

---

## ⚠️ IMPORTANT NOTES

1. **Feature Order Matters:** The model expects features in the exact order it was trained. Use the same feature names.

2. **Missing Lags:** If you're making the first prediction and don't have historical data, initialize with:

   - All lag features = 0
   - All smooth features = 0
   - This will give rough predictions until you have real history

3. **Time Zones:** Ensure all timestamps are in the same timezone (IST for Delhi)

4. **Data Quality:** Garbage in, garbage out. Ensure:
   - Order counts are accurate
   - Weather data is current (< 1 hour old)
   - Timestamps are correct

---

_For questions or issues, check the main README.md or FINAL_RESULTS_SUMMARY.md_
