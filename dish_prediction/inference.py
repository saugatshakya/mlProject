"""
INFERENCE SCRIPT - Predict Dish Demand for Next Hour

Usage:
    python inference.py
    
Output:
    Predictions for each of the top 10 dishes
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path

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
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)

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
    """
    Create lag and smoothed features from order history
    
    Args:
        order_history: DataFrame with columns ['timestamp', 'dish_name', 'quantity']
                      Must contain last 3 hours of data
    
    Returns:
        dict: Features dictionary
    """
    features = {}
    
    # For each dish, calculate lags and smooth
    for dish in TOP_DISHES:
        dish_history = order_history[order_history['dish_name'] == dish].sort_values('timestamp', ascending=False)
        
        # Get quantities for last 3 hours (most recent first)
        quantities = dish_history.head(3)['quantity'].values
        
        # If we have less than 3 hours, pad with zeros
        while len(quantities) < 3:
            quantities = np.append(quantities, 0)
        
        # Lag features
        features[f'{dish}_lag1'] = quantities[0] if len(quantities) > 0 else 0
        features[f'{dish}_lag2'] = quantities[1] if len(quantities) > 1 else 0
        features[f'{dish}_lag3'] = quantities[2] if len(quantities) > 2 else 0
        
        # Smooth (3-hour average)
        features[f'{dish}_smooth'] = np.mean(quantities)
    
    return features

def get_weather_features(api_data=None):
    """
    Get weather features from API or use defaults
    
    Args:
        api_data: dict with keys ['temp', 'humidity', 'precipitation', 'wind_speed']
                 If None, uses default values
    
    Returns:
        dict: Weather features
    """
    if api_data is None:
        # Default values for Delhi (reasonable estimates)
        return {
            'env_temp': 25.0,
            'env_rhum': 60.0,
            'env_precip': 0.0,
            'env_wspd': 10.0
        }
    
    return {
        'env_temp': api_data.get('temp', 25.0),
        'env_rhum': api_data.get('humidity', 60.0),
        'env_precip': api_data.get('precipitation', 0.0),
        'env_wspd': api_data.get('wind_speed', 10.0)
    }

def get_pollution_features(api_data=None):
    """Get pollution features from API or use defaults"""
    if api_data is None:
        return {'aqi': 100}  # Moderate AQI as default
    
    return {'aqi': api_data.get('aqi', 100)}

def get_event_features(date=None):
    """
    Check if date is a holiday or has events
    
    Args:
        date: datetime object, if None uses today
    
    Returns:
        dict: Event features
    """
    if date is None:
        date = datetime.now()
    
    # TODO: Replace with actual holiday/event checking logic
    # For now, returns defaults
    return {
        'has_event': 0,
        'holiday': 0
    }

def predict_next_hour(order_history, weather_data=None, pollution_data=None, timestamp=None):
    """
    Predict dish demand for the next hour
    
    Args:
        order_history: DataFrame with order history (last 3 hours minimum)
        weather_data: dict with weather data (optional)
        pollution_data: dict with pollution data (optional)
        timestamp: datetime for prediction (default: now)
    
    Returns:
        dict: Predictions for each dish
    """
    # Load model
    print("Loading model...")
    model = load_model()
    
    # Get all features
    print("Extracting features...")
    features = {}
    features.update(get_temporal_features(timestamp))
    features.update(get_historical_features(order_history))
    features.update(get_weather_features(weather_data))
    features.update(get_pollution_features(pollution_data))
    features.update(get_event_features(timestamp))
    
    # Create feature DataFrame
    X = pd.DataFrame([features])
    
    # Make prediction
    print("Making predictions...")
    predictions = model.predict(X)[0]
    
    # Map to dishes
    results = {}
    for dish, pred in zip(TOP_DISHES, predictions):
        results[dish] = max(0, round(pred, 1))  # Round and ensure non-negative
    
    return results

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("DISH DEMAND PREDICTION - NEXT HOUR")
    print("="*80)
    
    # Example 1: Using dummy historical data
    print("\nExample 1: Using dummy data")
    print("-" * 80)
    
    # Create dummy order history (last 3 hours)
    now = datetime.now()
    dummy_history = []
    
    for hours_ago in [1, 2, 3]:
        timestamp = now - timedelta(hours=hours_ago)
        for dish in TOP_DISHES:
            # Random orders between 0-10
            quantity = np.random.randint(0, 10)
            dummy_history.append({
                'timestamp': timestamp,
                'dish_name': dish,
                'quantity': quantity
            })
    
    order_history_df = pd.DataFrame(dummy_history)
    
    # Dummy weather data
    weather = {
        'temp': 28.5,
        'humidity': 65,
        'precipitation': 0.0,
        'wind_speed': 12.0
    }
    
    # Dummy pollution data
    pollution = {
        'aqi': 150
    }
    
    # Make prediction
    try:
        predictions = predict_next_hour(
            order_history=order_history_df,
            weather_data=weather,
            pollution_data=pollution
        )
        
        print("\nPREDICTIONS FOR NEXT HOUR:")
        print("="*80)
        print(f"Timestamp: {now.strftime('%Y-%m-%d %H:%M')}")
        print(f"Weather: {weather['temp']}°C, {weather['humidity']}% humidity")
        print(f"Air Quality: AQI {pollution['aqi']}")
        print("-" * 80)
        
        total_predicted = 0
        for dish, quantity in predictions.items():
            print(f"{dish:40s}: {quantity:6.1f} orders")
            total_predicted += quantity
        
        print("-" * 80)
        print(f"{'TOTAL':40s}: {total_predicted:6.1f} orders")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease train the model first by running:")
        print("  python src/models/final_model.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETE")
    print("="*80)
    
    print("\nTo use with real data:")
    print("1. Fetch last 3 hours of orders from your database")
    print("2. Get current weather from weather API")
    print("3. Get current AQI from pollution API")
    print("4. Call predict_next_hour() with real data")
