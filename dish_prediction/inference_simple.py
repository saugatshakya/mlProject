"""
INFERENCE WITH EXISTING DATA SOURCES
Uses your existing weather/pollution data (no external APIs needed!)
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Top 10 dishes (in order)
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

def load_model():
    """Load the trained CatBoost model"""
    model_path = Path('models/final/catboost_model.pkl')
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run: python src/models/final_model.py")
    return joblib.load(model_path)

def load_latest_weather_pollution():
    """
    Load the most recent weather and pollution data from your existing CSV files
    
    Returns:
        dict: Weather and pollution data
    """
    print("Loading latest environmental data from existing sources...")
    
    # Load weather data
    weather_path = Path('../data/hourly_orders_weather.csv')
    if weather_path.exists():
        weather_df = pd.read_csv(weather_path)
        weather_df['order_hour'] = pd.to_datetime(weather_df['order_hour'])
        
        # Get the most recent entry
        latest_weather = weather_df.sort_values('order_hour', ascending=False).iloc[0]
        
        weather = {
            'temp': latest_weather['env_temp'],
            'humidity': latest_weather['env_rhum'],
            'precipitation': latest_weather['env_precip'],
            'wind_speed': latest_weather['env_wspd']
        }
        print(f"  ✓ Weather from: {latest_weather['order_hour']}")
        print(f"    Temp: {weather['temp']}°C, Humidity: {weather['humidity']}%")
    else:
        print("  ⚠️  Weather file not found, using defaults")
        weather = {'temp': 25.0, 'humidity': 60.0, 'precipitation': 0.0, 'wind_speed': 10.0}
    
    # Load pollution data
    pollution_path = Path('../data/pollution.csv')
    if pollution_path.exists():
        pollution_df = pd.read_csv(pollution_path)
        pollution_df['pollution_time_utc'] = pd.to_datetime(pollution_df['pollution_time_utc'])
        
        # Get the most recent entry
        latest_pollution = pollution_df.sort_values('pollution_time_utc', ascending=False).iloc[0]
        
        pollution = {
            'aqi': latest_pollution['aqi']
        }
        print(f"  ✓ Pollution from: {latest_pollution['pollution_time_utc']}")
        print(f"    AQI: {pollution['aqi']}")
    else:
        print("  ⚠️  Pollution file not found, using defaults")
        pollution = {'aqi': 100}
    
    return weather, pollution

def get_temporal_features(timestamp=None):
    """Extract temporal features from timestamp"""
    if timestamp is None:
        timestamp = datetime.now()
    
    hour = timestamp.hour
    dow = timestamp.weekday()
    
    return {
        'hour': hour,
        'day_of_week': dow,
        'is_weekend': 1 if dow >= 5 else 0,
        'sin_hour': np.sin(2 * np.pi * hour / 24),
        'cos_hour': np.cos(2 * np.pi * hour / 24)
    }

def get_historical_features(order_history):
    """Create lag and smoothed features from order history"""
    features = {}
    
    for dish in TOP_DISHES:
        dish_history = order_history[order_history['dish_name'] == dish].sort_values('timestamp', ascending=False)
        quantities = dish_history.head(3)['quantity'].values
        
        while len(quantities) < 3:
            quantities = np.append(quantities, 0)
        
        features[f'{dish}_lag1'] = quantities[0] if len(quantities) > 0 else 0
        features[f'{dish}_lag2'] = quantities[1] if len(quantities) > 1 else 0
        features[f'{dish}_lag3'] = quantities[2] if len(quantities) > 2 else 0
        features[f'{dish}_smooth'] = np.mean(quantities)
    
    return features

def load_order_history_from_csv(hours_back=3):
    """
    Load order history from your existing processed data
    
    Args:
        hours_back: Number of hours to look back
        
    Returns:
        DataFrame with order history
    """
    print(f"\nLoading last {hours_back} hours from processed data...")
    
    # Load the processed data with all features
    data_path = Path('data/processed/hourly_data_with_features.csv')
    if not data_path.exists():
        print("  ⚠️  Processed data not found. Using dummy data.")
        return create_dummy_history(hours_back)
    
    df = pd.read_csv(data_path)
    df['hour'] = pd.to_datetime(df['hour'])
    
    # Get the last N hours
    latest_time = df['hour'].max()
    cutoff_time = latest_time - timedelta(hours=hours_back-1)
    recent_data = df[df['hour'] >= cutoff_time].sort_values('hour', ascending=False)
    
    # Convert to order history format
    order_history = []
    for _, row in recent_data.iterrows():
        for dish in TOP_DISHES:
            if dish in df.columns:
                order_history.append({
                    'timestamp': row['hour'],
                    'dish_name': dish,
                    'quantity': row[dish]
                })
    
    history_df = pd.DataFrame(order_history)
    print(f"  ✓ Loaded {len(recent_data)} hours of data")
    print(f"    Date range: {recent_data['hour'].min()} to {recent_data['hour'].max()}")
    
    return history_df

def create_dummy_history(hours_back=3):
    """Create dummy history for testing"""
    now = datetime.now()
    dummy_history = []
    
    for hours_ago in range(1, hours_back + 1):
        timestamp = now - timedelta(hours=hours_ago)
        for dish in TOP_DISHES:
            quantity = np.random.randint(0, 10)
            dummy_history.append({
                'timestamp': timestamp,
                'dish_name': dish,
                'quantity': quantity
            })
    
    return pd.DataFrame(dummy_history)

def predict_next_hour_from_existing_data():
    """
    Make predictions using your existing data sources
    No external APIs needed!
    """
    print("="*80)
    print("PREDICTION USING EXISTING DATA SOURCES")
    print("="*80)
    
    # Load model
    print("\n[1/5] Loading model...")
    model = load_model()
    print("✓ Model loaded")
    
    # Get current time
    now = datetime.now()
    print(f"\n[2/5] Current time: {now.strftime('%Y-%m-%d %H:%M')}")
    
    # Load order history from your existing processed data
    print("\n[3/5] Loading order history...")
    order_history_df = load_order_history_from_csv(hours_back=3)
    
    # Extract features
    print("\n[4/5] Extracting features...")
    features = {}
    
    # Temporal
    features.update(get_temporal_features(now))
    print(f"  ✓ Temporal: Hour {features['hour']}, DoW {features['day_of_week']}")
    
    # Historical
    features.update(get_historical_features(order_history_df))
    print(f"  ✓ Historical: Last 3 hours of orders")
    
    # Weather & Pollution from your existing data
    weather, pollution = load_latest_weather_pollution()
    features.update({
        'env_temp': weather['temp'],
        'env_rhum': weather['humidity'],
        'env_precip': weather['precipitation'],
        'env_wspd': weather['wind_speed'],
        'aqi': pollution['aqi']
    })
    
    # Event features (could load from your events.csv)
    features.update({'has_event': 0, 'holiday': 0})
    
    # Make prediction
    print("\n[5/5] Making predictions...")
    X = pd.DataFrame([features])
    predictions = model.predict(X)[0]
    
    # Format results
    results = {}
    total = 0
    for dish, pred in zip(TOP_DISHES, predictions):
        quantity = max(0, round(pred, 1))
        results[dish] = quantity
        total += quantity
    
    print("✓ Predictions complete")
    
    # Display
    print("\n" + "="*80)
    print("PREDICTIONS FOR NEXT HOUR")
    print("="*80)
    print(f"Timestamp: {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"Weather: {weather['temp']}°C, {weather['humidity']}% humidity")
    print(f"Air Quality: AQI {pollution['aqi']}")
    print(f"Data Source: Existing CSV files ✓")
    print("-" * 80)
    
    for dish, qty in results.items():
        print(f"{dish:40s}: {qty:6.1f} orders")
    
    print("-" * 80)
    print(f"{'TOTAL':40s}: {total:6.1f} orders")
    print("="*80)
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 Starting inference with existing data sources...")
    print("   (No external APIs needed - using your CSV files!)\n")
    
    try:
        predictions = predict_next_hour_from_existing_data()
        
        print("\n✅ Inference complete!")
        
        print("\n" + "="*80)
        print("DATA SOURCES USED")
        print("="*80)
        print("""
✓ Order History: data/processed/hourly_data_with_features.csv
✓ Weather Data: ../data/hourly_orders_weather.csv  
✓ Pollution Data: ../data/pollution.csv

NO EXTERNAL API CALLS NEEDED!

Your existing data files already contain:
- Historical orders (last 3 hours)
- Weather (temperature, humidity, precipitation, wind)
- Pollution (AQI, PM2.5, PM10, NO2, O3, CO)

For REAL-TIME updates:
- You can keep updating these CSV files from your data pipeline
- Or integrate external APIs using api_integration.py
        """)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
