"""
PRODUCTION INFERENCE SCRIPT
Real-time dish demand prediction with API integration

This version integrates with real weather/pollution APIs
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Import API integration functions
try:
    from api_integration import get_weather_weatherapi, get_weather_openweathermap, get_pollution_openweathermap
    API_AVAILABLE = True
except ImportError:
    print("⚠️  api_integration.py not found. Using default values.")
    API_AVAILABLE = False

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
    """
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

def get_real_weather_data(use_api='weatherapi', city='Delhi'):
    """
    Fetch real weather data from API
    
    Args:
        use_api: 'weatherapi' or 'openweathermap'
        city: City name
    
    Returns:
        weather_dict, pollution_dict (or defaults if API fails)
    """
    if not API_AVAILABLE:
        print("⚠️  API integration not available. Using defaults.")
        return None, None
    
    print(f"Fetching real-time data for {city}...")
    
    if use_api == 'weatherapi':
        weather, pollution = get_weather_weatherapi(city)
        return weather, pollution
    
    elif use_api == 'openweathermap':
        weather = get_weather_openweathermap(city)
        pollution = get_pollution_openweathermap()
        return weather, pollution
    
    return None, None

def predict_production(order_history_df, city='Delhi', use_api='weatherapi'):
    """
    Production inference with real API data
    
    Args:
        order_history_df: DataFrame with last 3 hours of orders
        city: City for weather/pollution lookup
        use_api: Which API to use ('weatherapi' or 'openweathermap')
    
    Returns:
        dict: Predictions with metadata
    """
    print("="*80)
    print("PRODUCTION DISH DEMAND PREDICTION")
    print("="*80)
    
    # Load model
    print("\n[1/5] Loading model...")
    model = load_model()
    print("✓ Model loaded")
    
    # Get current time
    now = datetime.now()
    print(f"\n[2/5] Current time: {now.strftime('%Y-%m-%d %H:%M')}")
    
    # Extract features
    print("\n[3/5] Extracting features...")
    features = {}
    
    # Temporal
    features.update(get_temporal_features(now))
    print(f"  ✓ Temporal features: Hour {features['hour']}, DoW {features['day_of_week']}")
    
    # Historical
    features.update(get_historical_features(order_history_df))
    print(f"  ✓ Historical features: Last 3 hours of orders")
    
    # Weather & Pollution (real-time)
    print("\n[4/5] Fetching real-time environmental data...")
    weather, pollution = get_real_weather_data(use_api, city)
    
    if weather:
        print(f"  ✓ Weather: {weather['temp']}°C, {weather['humidity']}% humidity, {weather['precipitation']}mm rain")
        features.update({
            'env_temp': weather['temp'],
            'env_rhum': weather['humidity'],
            'env_precip': weather['precipitation'],
            'env_wspd': weather['wind_speed']
        })
    else:
        print("  ⚠️  Using default weather values")
        features.update({
            'env_temp': 25.0,
            'env_rhum': 60.0,
            'env_precip': 0.0,
            'env_wspd': 10.0
        })
    
    if pollution:
        print(f"  ✓ Air Quality: AQI {pollution['aqi']}")
        features.update({'aqi': pollution['aqi']})
    else:
        print("  ⚠️  Using default AQI value")
        features.update({'aqi': 100})
    
    # Event features (placeholder - implement your calendar logic)
    features.update({'has_event': 0, 'holiday': 0})
    
    # Make prediction
    print("\n[5/5] Making predictions...")
    X = pd.DataFrame([features])
    predictions = model.predict(X)[0]
    
    # Format results
    results = {
        'timestamp': now,
        'predictions': {},
        'metadata': {
            'weather': weather if weather else 'default',
            'pollution': pollution if pollution else 'default',
            'api_used': use_api if (weather or pollution) else 'none'
        }
    }
    
    total = 0
    for dish, pred in zip(TOP_DISHES, predictions):
        quantity = max(0, round(pred, 1))
        results['predictions'][dish] = quantity
        total += quantity
    
    results['total_predicted'] = round(total, 1)
    
    print("✓ Predictions complete")
    return results

def display_results(results):
    """Pretty print prediction results"""
    print("\n" + "="*80)
    print("PREDICTIONS FOR NEXT HOUR")
    print("="*80)
    print(f"Timestamp: {results['timestamp'].strftime('%Y-%m-%d %H:%M')}")
    
    if results['metadata']['weather'] != 'default':
        w = results['metadata']['weather']
        print(f"Weather: {w['temp']}°C, {w['humidity']}% humidity, {w['precipitation']}mm precip")
    else:
        print("Weather: Using default values")
    
    if results['metadata']['pollution'] != 'default':
        p = results['metadata']['pollution']
        print(f"Air Quality: AQI {p['aqi']}")
    else:
        print("Air Quality: Using default AQI")
    
    print(f"Data Source: {results['metadata']['api_used']}")
    print("-" * 80)
    
    for dish, qty in results['predictions'].items():
        print(f"{dish:40s}: {qty:6.1f} orders")
    
    print("-" * 80)
    print(f"{'TOTAL':40s}: {results['total_predicted']:6.1f} orders")
    print("="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 Starting production inference...")
    
    # Create dummy order history (replace with real database query)
    print("\n⚠️  Using dummy order history. Replace with real database query!")
    now = datetime.now()
    dummy_history = []
    
    for hours_ago in [1, 2, 3]:
        timestamp = now - timedelta(hours=hours_ago)
        for dish in TOP_DISHES:
            quantity = np.random.randint(0, 10)
            dummy_history.append({
                'timestamp': timestamp,
                'dish_name': dish,
                'quantity': quantity
            })
    
    order_history_df = pd.DataFrame(dummy_history)
    
    # Run production inference
    try:
        # Try WeatherAPI first (recommended)
        results = predict_production(
            order_history_df,
            city='Delhi',
            use_api='weatherapi'  # or 'openweathermap'
        )
        
        display_results(results)
        
        # Optional: Save predictions to database/file
        # save_predictions_to_db(results)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✅ Inference complete!")
    print("\n" + "="*80)
    print("PRODUCTION DEPLOYMENT NOTES")
    print("="*80)
    print("""
For production deployment:

1. DATABASE INTEGRATION
   - Replace dummy_history with real database query
   - Query last 3 hours of orders grouped by dish
   
2. API SETUP
   - Get API key from WeatherAPI.com or OpenWeatherMap
   - Set as environment variable:
     export WEATHERAPI_KEY="your_key"
   
3. SCHEDULING
   - Run this script every hour using cron:
     0 * * * * cd /path/to/project && python inference_production.py
   
4. OUTPUT HANDLING
   - Save predictions to database for inventory planning
   - Send alerts if predictions exceed capacity
   - Log predictions for monitoring
   
5. ERROR HANDLING
   - Implement retry logic for API failures
   - Fallback to default values if APIs down
   - Alert ops team if model fails
    """)
