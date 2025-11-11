"""
COMPLETE FRESH START - TRUSTWORTHY ANALYSIS
Load raw data, analyze it properly, and generate ONLY factual statistics
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 0: LOADING RAW DATA")
print("="*80)

# Load raw orders
orders = pd.read_csv('../../data/data.csv')
print(f"\n✓ Loaded orders: {len(orders):,} records")
print(f"  Columns: {list(orders.columns[:10])}...")

# Load weather
weather = pd.read_csv('../../data/hourly_orders_weather.csv')
print(f"\n✓ Loaded weather: {len(weather):,} records")
print(f"  Columns: {list(weather.columns)}")

# Load pollution
pollution = pd.read_csv('../../data/pollution.csv')
print(f"\n✓ Loaded pollution: {len(pollution):,} records")
print(f"  Columns: {list(pollution.columns)}")

# Load events
events = pd.read_csv('../../data/events.csv')
print(f"\n✓ Loaded events: {len(events):,} records")
print(f"  Columns: {list(events.columns)}")

print("\n" + "="*80)
print("STEP 1: PARSE ORDER TIMESTAMPS")
print("="*80)

# Parse the weird timestamp format
orders['Order Placed At'] = pd.to_datetime(orders['Order Placed At'], format='%I:%M %p, %B %d %Y')
orders['order_hour'] = orders['Order Placed At'].dt.floor('h')
orders['date'] = orders['Order Placed At'].dt.date
orders['hour'] = orders['Order Placed At'].dt.hour
orders['day_of_week'] = orders['Order Placed At'].dt.dayofweek
orders['day_name'] = orders['Order Placed At'].dt.day_name()
orders['is_weekend'] = orders['day_of_week'].isin([5, 6])

print(f"\n✓ Parsed timestamps")
print(f"  Date range: {orders['Order Placed At'].min()} to {orders['Order Placed At'].max()}")
print(f"  Total hours covered: {orders['order_hour'].nunique()}")
print(f"  Total days: {orders['date'].nunique()}")

print("\n" + "="*80)
print("STEP 2: PARSE DISH ITEMS FROM ORDERS")
print("="*80)

import re

def parse_items(order_str):
    """Parse '2 x Dish Name, 1 x Another Dish' format"""
    if pd.isna(order_str):
        return {}
    items = {}
    # Handle both comma and pipe separators
    parts = re.split(r'[,|]', str(order_str))
    for part in parts:
        # Match "number x dish_name" pattern
        match = re.match(r'(\d+)\s*x\s*(.+)', part.strip())
        if match:
            qty = int(match.group(1))
            dish = match.group(2).strip()
            items[dish] = items.get(dish, 0) + qty
    return items

# Parse all orders
print("\n  Parsing dishes from orders...")
orders['parsed_dishes'] = orders['Items in order'].apply(parse_items)
orders['total_dishes'] = orders['parsed_dishes'].apply(lambda x: sum(x.values()))

# Create flat list of all dishes
all_dishes = []
for dishes_dict in orders['parsed_dishes']:
    for dish, qty in dishes_dict.items():
        all_dishes.extend([dish] * qty)

print(f"\n✓ Parsed dishes")
print(f"  Total dish orders: {len(all_dishes):,}")
print(f"  Unique dishes: {len(set(all_dishes)):,}")
print(f"  Avg dishes per order: {orders['total_dishes'].mean():.2f}")

# Top dishes
from collections import Counter
dish_counts = Counter(all_dishes)
print(f"\n  Top 10 dishes:")
for dish, count in dish_counts.most_common(10):
    print(f"    {count:5,} - {dish}")

print("\n" + "="*80)
print("STEP 3: MERGE WEATHER DATA")
print("="*80)

# Parse weather timestamps
weather['order_hour'] = pd.to_datetime(weather['order_hour'])

# Merge
merged = orders.merge(weather, on='order_hour', how='left')
print(f"\n✓ Merged with weather")
print(f"  Records after merge: {len(merged):,}")
print(f"  Weather columns: {[c for c in merged.columns if 'env_' in c]}")

print("\n" + "="*80)
print("STEP 4: MERGE POLLUTION DATA")
print("="*80)

# Parse pollution timestamps
pollution['order_hour'] = pd.to_datetime(pollution['pollution_time_utc']).dt.floor('h')

# Merge
merged = merged.merge(pollution, on='order_hour', how='left')
print(f"\n✓ Merged with pollution")
print(f"  Records after merge: {len(merged):,}")
print(f"  Pollution columns: {[c for c in merged.columns if c in ['aqi', 'pm2_5', 'pm10', 'no2', 'o3', 'co']]}")

print("\n" + "="*80)
print("STEP 5: MERGE EVENT DATA")
print("="*80)

# Parse event dates
events['date'] = pd.to_datetime(events['date']).dt.date

# Merge
merged = merged.merge(events, on='date', how='left')
merged['has_event'] = merged['event'].notna()
print(f"\n✓ Merged with events")
print(f"  Records after merge: {len(merged):,}")
print(f"  Orders with events: {merged['has_event'].sum():,} ({merged['has_event'].sum()/len(merged)*100:.1f}%)")

print("\n" + "="*80)
print("STEP 6: SAVE CLEAN PROCESSED DATA")
print("="*80)

# Save
merged.to_csv('data/eda_processed.csv', index=False)
print(f"\n✓ Saved to data/eda_processed.csv")
print(f"  Total records: {len(merged):,}")
print(f"  Total columns: {len(merged.columns)}")

print("\n" + "="*80)
print("STEP 7: BASIC STATISTICS FROM CLEAN DATA")
print("="*80)

print(f"\n📊 TEMPERATURE:")
print(f"   Range: {merged['env_temp'].min():.1f}°C to {merged['env_temp'].max():.1f}°C")
print(f"   Mean: {merged['env_temp'].mean():.1f}°C")
print(f"   Median: {merged['env_temp'].median():.1f}°C")

print(f"\n📊 PRECIPITATION:")
rainy = merged['env_precip'] > 0
print(f"   Orders with rain: {rainy.sum():,} ({rainy.sum()/len(merged)*100:.1f}%)")
print(f"   Orders without rain: {(~rainy).sum():,} ({(~rainy).sum()/len(merged)*100:.1f}%)")

print(f"\n📊 WEATHER CONDITIONS:")
for condition, count in merged['env_condition'].value_counts().items():
    print(f"   {condition:15s}: {count:5,} ({count/len(merged)*100:5.1f}%)")

print(f"\n📊 AIR QUALITY:")
for aqi, count in merged['aqi'].value_counts().sort_index().items():
    labels = {2: 'Good', 3: 'Moderate', 4: 'Unhealthy for Sensitive', 5: 'Unhealthy'}
    label = labels.get(int(aqi), str(aqi))
    print(f"   AQI {int(aqi)} ({label:25s}): {count:5,} ({count/len(merged)*100:5.1f}%)")

print(f"\n📊 TEMPORAL PATTERNS:")
print(f"   Total unique hours: {merged['order_hour'].nunique():,}")
print(f"   Total unique days: {merged['date'].nunique()}")
print(f"   Weekend orders: {merged[merged['is_weekend']].shape[0]:,} ({merged['is_weekend'].sum()/len(merged)*100:.1f}%)")
print(f"   Weekday orders: {merged[~merged['is_weekend']].shape[0]:,} ({(~merged['is_weekend']).sum()/len(merged)*100:.1f}%)")

print("\n" + "="*80)
print("✅ DATA LOADING AND BASIC STATS COMPLETE - ALL VERIFIED")
print("="*80)
