"""
EDA Phase 2: External Factor Analysis
Exploring how weather, pollution, and events impact restaurant orders in Delhi

As a restaurant domain expert, I hypothesize:
1. Bad weather (rain, cold) → More orders (people stay home)
2. High pollution → Indoor activity → More orders
3. Festivals/holidays → Different dish preferences
4. Weekend events → Spike in specific areas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EDA PHASE 2: EXTERNAL FACTORS CORRELATION ANALYSIS")
print("="*80)

# ============================================================================
# LOAD PROCESSED DATA
# ============================================================================

print("\n📊 Loading processed data...")
df = pd.read_csv('data/interim/orders_with_temporal.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = pd.to_datetime(df['date'])
print(f"   Orders loaded: {len(df):,}")

# Aggregate to hourly for external data merge
hourly_agg = df.groupby(df['datetime'].dt.floor('H')).agg({
    'Order ID': 'count',
    'Total': 'sum',
    'KPT duration (minutes)': 'mean',
    'Rider wait time (minutes)': 'mean'
}).rename(columns={'Order ID': 'order_count', 'Total': 'revenue'})

hourly_agg.index.name = 'datetime'
hourly_agg = hourly_agg.reset_index()
print(f"   Hourly data points: {len(hourly_agg):,}")

# ============================================================================
# CHECK FOR EXTERNAL DATA FILES
# ============================================================================

print("\n" + "="*80)
print("EXTERNAL DATA AVAILABILITY CHECK")
print("="*80)

import os

external_files = {
    'Weather': '../data/delhi_weather_hourly.csv',
    'Pollution': '../data/delhi_pollution_hourly.csv',
    'Events': '../data/delhi_major_events.csv'
}

available_data = {}
for name, path in external_files.items():
    if os.path.exists(path):
        print(f"   ✅ {name:15s} | {path}")
        available_data[name] = path
    else:
        print(f"   ❌ {name:15s} | NOT FOUND: {path}")

# ============================================================================
# ANALYZE AVAILABLE EXTERNAL DATA
# ============================================================================

if available_data:
    print("\n" + "="*80)
    print("ANALYZING EXTERNAL DATA")
    print("="*80)
    
    # ========== WEATHER DATA ==========
    if 'Weather' in available_data:
        print("\n🌤️  WEATHER DATA ANALYSIS:")
        weather = pd.read_csv(available_data['Weather'])
        print(f"\n   Shape: {weather.shape}")
        print(f"   Columns: {list(weather.columns)}")
        
        if 'datetime' in weather.columns or 'date' in weather.columns:
            datetime_col = 'datetime' if 'datetime' in weather.columns else 'date'
            weather[datetime_col] = pd.to_datetime(weather[datetime_col])
            print(f"   Date range: {weather[datetime_col].min()} to {weather[datetime_col].max()}")
            
            # Try to merge with hourly data
            print("\n   Attempting merge with order data...")
            hourly_agg['date_hour'] = hourly_agg['datetime'].dt.strftime('%Y-%m-%d %H')
            weather['date_hour'] = pd.to_datetime(weather[datetime_col]).dt.strftime('%Y-%m-%d %H')
            
            merged = hourly_agg.merge(weather, on='date_hour', how='inner')
            print(f"   ✅ Merged records: {len(merged):,} / {len(hourly_agg):,} ({len(merged)/len(hourly_agg)*100:.1f}%)")
            
            if len(merged) > 0:
                # Analyze weather impact
                print("\n   📊 Weather Impact Analysis:")
                
                # Temperature correlation
                if 'temperature' in merged.columns or 'temp' in merged.columns:
                    temp_col = 'temperature' if 'temperature' in merged.columns else 'temp'
                    corr = merged['order_count'].corr(merged[temp_col])
                    print(f"      Temperature vs Orders: r={corr:.3f}")
                
                # Rain impact
                if 'precipitation' in merged.columns or 'rain' in merged.columns:
                    rain_col = 'precipitation' if 'precipitation' in merged.columns else 'rain'
                    rainy = merged[merged[rain_col] > 0]
                    no_rain = merged[merged[rain_col] == 0]
                    print(f"      Orders when raining: {rainy['order_count'].mean():.1f} avg")
                    print(f"      Orders when dry: {no_rain['order_count'].mean():.1f} avg")
                    print(f"      Rain impact: {((rainy['order_count'].mean() / no_rain['order_count'].mean()) - 1) * 100:+.1f}%")
        else:
            print("   ⚠️  No datetime column found")
    
    # ========== POLLUTION DATA ==========
    if 'Pollution' in available_data:
        print("\n🏭 POLLUTION DATA ANALYSIS:")
        pollution = pd.read_csv(available_data['Pollution'])
        print(f"\n   Shape: {pollution.shape}")
        print(f"   Columns: {list(pollution.columns)}")
        
        if 'datetime' in pollution.columns or 'date' in pollution.columns:
            datetime_col = 'datetime' if 'datetime' in pollution.columns else 'date'
            pollution[datetime_col] = pd.to_datetime(pollution[datetime_col])
            print(f"   Date range: {pollution[datetime_col].min()} to {pollution[datetime_col].max()}")
            
            # Try to merge
            print("\n   Attempting merge with order data...")
            pollution['date_hour'] = pd.to_datetime(pollution[datetime_col]).dt.strftime('%Y-%m-%d %H')
            
            merged = hourly_agg.merge(pollution, on='date_hour', how='inner')
            print(f"   ✅ Merged records: {len(merged):,} / {len(hourly_agg):,} ({len(merged)/len(hourly_agg)*100:.1f}%)")
            
            if len(merged) > 0:
                print("\n   📊 Pollution Impact Analysis:")
                
                # AQI correlation
                if 'AQI' in merged.columns or 'aqi' in merged.columns:
                    aqi_col = 'AQI' if 'AQI' in merged.columns else 'aqi'
                    corr = merged['order_count'].corr(merged[aqi_col])
                    print(f"      AQI vs Orders: r={corr:.3f}")
                    
                    # High vs low pollution
                    high_aqi = merged[merged[aqi_col] > merged[aqi_col].median()]
                    low_aqi = merged[merged[aqi_col] <= merged[aqi_col].median()]
                    print(f"      Orders on high AQI days: {high_aqi['order_count'].mean():.1f} avg")
                    print(f"      Orders on low AQI days: {low_aqi['order_count'].mean():.1f} avg")
                    print(f"      High AQI impact: {((high_aqi['order_count'].mean() / low_aqi['order_count'].mean()) - 1) * 100:+.1f}%")
        else:
            print("   ⚠️  No datetime column found")
    
    # ========== EVENTS DATA ==========
    if 'Events' in available_data:
        print("\n🎉 EVENTS DATA ANALYSIS:")
        events = pd.read_csv(available_data['Events'])
        print(f"\n   Shape: {events.shape}")
        print(f"   Columns: {list(events.columns)}")
        print(f"\n   Event types:")
        
        if 'event_type' in events.columns or 'type' in events.columns:
            type_col = 'event_type' if 'event_type' in events.columns else 'type'
            for event_type, count in events[type_col].value_counts().items():
                print(f"      {event_type:20s} | {count:3d} events")
        
        if 'date' in events.columns:
            events['date'] = pd.to_datetime(events['date'])
            print(f"   Date range: {events['date'].min()} to {events['date'].max()}")
            
            # Daily aggregation for events
            daily_orders = df.groupby('date').agg({
                'Order ID': 'count',
                'Total': 'sum'
            }).reset_index()
            daily_orders.columns = ['date', 'order_count', 'revenue']
            daily_orders['date'] = pd.to_datetime(daily_orders['date'])
            
            # Add event flag
            daily_orders['has_event'] = daily_orders['date'].isin(events['date'])
            
            event_days = daily_orders[daily_orders['has_event']]
            normal_days = daily_orders[~daily_orders['has_event']]
            
            print(f"\n   📊 Event Impact:")
            print(f"      Days with events: {len(event_days)}")
            print(f"      Orders on event days: {event_days['order_count'].mean():.1f} avg")
            print(f"      Orders on normal days: {normal_days['order_count'].mean():.1f} avg")
            print(f"      Event impact: {((event_days['order_count'].mean() / normal_days['order_count'].mean()) - 1) * 100:+.1f}%")

else:
    print("\n⚠️  No external data files found.")
    print("   Expected locations:")
    for name, path in external_files.items():
        print(f"      {name}: {path}")

# ============================================================================
# DOMAIN-SPECIFIC PATTERNS (Delhi Restaurant Insights)
# ============================================================================

print("\n" + "="*80)
print("DELHI RESTAURANT DOMAIN INSIGHTS")
print("="*80)

print("\n🍕 DISH PREFERENCE BY TIME:")
# Load dish data
dish_orders = []
import re

def parse_items(item_str):
    if pd.isna(item_str):
        return []
    items = []
    parts = str(item_str).split(',')
    for part in parts:
        match = re.search(r'(\d+)\s*x\s*(.+)', part.strip())
        if match:
            qty = int(match.group(1))
            dish = match.group(2).strip()
            items.append({'dish': dish, 'quantity': qty})
    return items

for idx, row in df.iterrows():
    dishes = parse_items(row['Items in order'])
    for dish_info in dishes:
        dish_orders.append({
            'datetime': row['datetime'],
            'hour': row['hour'],
            'day_of_week': row['day_of_week'],
            'dish': dish_info['dish'],
            'quantity': dish_info['quantity']
        })

dish_df = pd.DataFrame(dish_orders)

# Top dishes by time of day
print("\n   Peak Lunch (12-14h) vs Peak Dinner (19-22h):")
lunch_dishes = dish_df[dish_df['hour'].between(12, 14)].groupby('dish')['quantity'].sum().nlargest(5)
dinner_dishes = dish_df[dish_df['hour'].between(19, 22)].groupby('dish')['quantity'].sum().nlargest(5)

print("\n   Top 5 Lunch Dishes:")
for i, (dish, qty) in enumerate(lunch_dishes.items(), 1):
    print(f"      {i}. {dish[:40]:40s} | {qty:4,} units")

print("\n   Top 5 Dinner Dishes:")
for i, (dish, qty) in enumerate(dinner_dishes.items(), 1):
    print(f"      {i}. {dish[:40]:40s} | {qty:4,} units")

# Weekend vs Weekday preferences
print("\n   Weekend (Fri-Sun) vs Weekday Preferences:")
weekend_dishes = dish_df[dish_df['day_of_week'].isin([4, 5, 6])].groupby('dish')['quantity'].sum().nlargest(5)
weekday_dishes = dish_df[~dish_df['day_of_week'].isin([4, 5, 6])].groupby('dish')['quantity'].sum().nlargest(5)

print("\n   Top 5 Weekend Dishes:")
for i, (dish, qty) in enumerate(weekend_dishes.items(), 1):
    print(f"      {i}. {dish[:40]:40s} | {qty:4,} units")

print("\n   Top 5 Weekday Dishes:")
for i, (dish, qty) in enumerate(weekday_dishes.items(), 1):
    print(f"      {i}. {dish[:40]:40s} | {qty:4,} units")

# ============================================================================
# KEY INSIGHTS FOR FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("🎯 KEY INSIGHTS FOR FEATURE ENGINEERING")
print("="*80)

print("""
Based on this EDA, I will engineer the following feature categories:

1. TEMPORAL FEATURES (Critical for restaurants):
   ✓ Hour of day (peak hours: 19-21)
   ✓ Day of week (weekend effect exists)
   ✓ Weekend flag
   ✓ Meal period (breakfast/lunch/dinner/late-night)
   ✓ Peak hour flag
   ✓ Week of year (seasonal trends)
   
2. LAG FEATURES (Order history):
   ✓ Previous 1h, 2h, 3h, 6h, 12h, 24h orders
   ✓ Same hour yesterday, last week
   ✓ Trend indicators (increasing/decreasing)
   
3. ROLLING STATISTICS (Recent patterns):
   ✓ 3h, 6h, 12h, 24h rolling mean/std/min/max
   ✓ 7-day rolling average (weekly pattern)
   ✓ Rolling coefficient of variation
   
4. WEATHER FEATURES (if available):
   ✓ Temperature (continuous + binned)
   ✓ Precipitation/rain flag
   ✓ Weather condition categories
   ✓ Humidity, wind speed
   
5. POLLUTION FEATURES (if available):
   ✓ AQI level (Good/Moderate/Poor/Severe)
   ✓ PM2.5, PM10 levels
   ✓ High pollution flag
   
6. EVENT FEATURES (if available):
   ✓ Holiday/festival flag
   ✓ Days until/since major event
   ✓ Event type (festival/holiday/sporting)
   
7. DELHI-SPECIFIC FEATURES (Domain expertise):
   ✓ Delhi season (summer/monsoon/winter/spring)
   ✓ Smog season (Oct-Feb → outdoor avoidance)
   ✓ Major festivals (Diwali, Holi impact)
   ✓ Exam season patterns
   
8. DISH-SPECIFIC FEATURES:
   ✓ Dish category (pizza/chicken/paneer/bread)
   ✓ Price segment
   ✓ Dish popularity rank
   ✓ Category trends

TOTAL: ~200-300 features expected
""")

print("\n" + "="*80)
print("✅ EDA PHASE 2 COMPLETE")
print("="*80)
print("\nNext: Create visualization suite for comprehensive understanding")
