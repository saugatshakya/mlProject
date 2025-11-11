"""
REAL-TIME DATA INTEGRATION FOR INFERENCE

This script shows how to fetch weather and pollution data from real APIs
"""

import requests
from datetime import datetime
import os

# ============================================================================
# WEATHER DATA - OpenWeatherMap API
# ============================================================================

def get_weather_openweathermap(city="Delhi", api_key=None):
    """
    Fetch current weather from OpenWeatherMap API
    
    Free API: https://openweathermap.org/api
    Sign up for free API key at: https://home.openweathermap.org/users/sign_up
    
    Args:
        city: City name (default: Delhi)
        api_key: Your OpenWeatherMap API key
    
    Returns:
        dict: Weather data with keys ['temp', 'humidity', 'precipitation', 'wind_speed']
    """
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            print("⚠️  No OpenWeatherMap API key found. Using default values.")
            return None
    
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'  # Celsius
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        weather = {
            'temp': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'precipitation': data.get('rain', {}).get('1h', 0.0),  # Rain in last 1h (mm)
            'wind_speed': data['wind']['speed'] * 3.6  # Convert m/s to km/h
        }
        
        print(f"✓ Weather data fetched: {weather['temp']}°C, {weather['humidity']}% humidity")
        return weather
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching weather data: {e}")
        return None

# ============================================================================
# POLLUTION DATA - OpenWeatherMap Air Pollution API
# ============================================================================

def get_pollution_openweathermap(lat=28.6139, lon=77.2090, api_key=None):
    """
    Fetch air quality data from OpenWeatherMap Air Pollution API
    
    Free API: https://openweathermap.org/api/air-pollution
    
    Args:
        lat: Latitude (default: Delhi = 28.6139)
        lon: Longitude (default: Delhi = 77.2090)
        api_key: Your OpenWeatherMap API key
    
    Returns:
        dict: Pollution data with key 'aqi'
    """
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            print("⚠️  No OpenWeatherMap API key found. Using default AQI.")
            return None
    
    url = f"http://api.openweathermap.org/data/2.5/air_pollution"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # OpenWeatherMap returns AQI from 1-5, convert to 0-500 scale
        # 1=Good, 2=Fair, 3=Moderate, 4=Poor, 5=Very Poor
        aqi_mapping = {1: 50, 2: 100, 3: 150, 4: 200, 5: 300}
        aqi_level = data['list'][0]['main']['aqi']
        aqi = aqi_mapping.get(aqi_level, 100)
        
        pollution = {'aqi': aqi}
        
        print(f"✓ Air quality data fetched: AQI {aqi}")
        return pollution
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching pollution data: {e}")
        return None

# ============================================================================
# ALTERNATIVE: WeatherAPI.com (Easier, more generous free tier)
# ============================================================================

def get_weather_weatherapi(city="Delhi", api_key=None):
    """
    Fetch weather from WeatherAPI.com
    
    Free API: https://www.weatherapi.com/
    Free tier: 1 million calls/month
    Sign up at: https://www.weatherapi.com/signup.aspx
    
    Args:
        city: City name
        api_key: Your WeatherAPI.com API key
    
    Returns:
        dict: Combined weather and air quality data
    """
    if api_key is None:
        api_key = os.getenv('WEATHERAPI_KEY')
        if not api_key:
            print("⚠️  No WeatherAPI key found. Using default values.")
            return None, None
    
    url = f"http://api.weatherapi.com/v1/current.json"
    params = {
        'key': api_key,
        'q': city,
        'aqi': 'yes'  # Include air quality
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        weather = {
            'temp': data['current']['temp_c'],
            'humidity': data['current']['humidity'],
            'precipitation': data['current']['precip_mm'],
            'wind_speed': data['current']['wind_kph']
        }
        
        # WeatherAPI provides detailed air quality
        if 'air_quality' in data['current']:
            aqi_us = data['current']['air_quality'].get('us-epa-index', 3)
            # US EPA index: 1=Good, 2=Moderate, 3=Unhealthy for sensitive, 4=Unhealthy, 5=Very Unhealthy, 6=Hazardous
            aqi_mapping = {1: 50, 2: 100, 3: 150, 4: 200, 5: 300, 6: 400}
            pollution = {'aqi': aqi_mapping.get(aqi_us, 100)}
        else:
            pollution = {'aqi': 100}
        
        print(f"✓ Data fetched: {weather['temp']}°C, {weather['humidity']}% humidity, AQI {pollution['aqi']}")
        return weather, pollution
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data from WeatherAPI: {e}")
        return None, None

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("TESTING WEATHER & POLLUTION API INTEGRATION")
    print("="*80)
    
    print("\n1. Testing OpenWeatherMap API")
    print("-" * 80)
    # Set your API key as environment variable:
    # export OPENWEATHER_API_KEY="your_api_key_here"
    weather = get_weather_openweathermap("Delhi")
    pollution = get_pollution_openweathermap()
    
    if weather:
        print(f"Weather: {weather}")
    if pollution:
        print(f"Pollution: {pollution}")
    
    print("\n2. Testing WeatherAPI.com (Recommended - easier & more generous)")
    print("-" * 80)
    # Set your API key as environment variable:
    # export WEATHERAPI_KEY="your_api_key_here"
    weather, pollution = get_weather_weatherapi("Delhi")
    
    if weather and pollution:
        print(f"Weather: {weather}")
        print(f"Pollution: {pollution}")
    
    print("\n" + "="*80)
    print("SETUP INSTRUCTIONS")
    print("="*80)
    print("""
To use real weather/pollution data:

OPTION 1: WeatherAPI.com (Recommended - Free, 1M calls/month)
1. Sign up at: https://www.weatherapi.com/signup.aspx
2. Get your API key from dashboard
3. Set environment variable:
   export WEATHERAPI_KEY="your_key_here"
   
OPTION 2: OpenWeatherMap (Free, 60 calls/min, 1M calls/month)
1. Sign up at: https://home.openweathermap.org/users/sign_up
2. Get your API key from dashboard
3. Set environment variable:
   export OPENWEATHER_API_KEY="your_key_here"

Then modify inference.py to use these functions:
    from api_integration import get_weather_weatherapi
    
    # In predict_next_hour():
    weather, pollution = get_weather_weatherapi("Delhi")
    predictions = predict_next_hour(
        order_history=order_history_df,
        weather_data=weather,
        pollution_data=pollution
    )
    """)
