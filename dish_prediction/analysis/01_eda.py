"""
STEP 01: EXPLORATORY DATA ANALYSIS (EDA)
Analyze patterns in the clean data - NO HALLUCINATIONS, ONLY REAL DATA
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the clean processed data
data = pd.read_csv('data/eda_processed.csv', parse_dates=['order_hour', 'Order Placed At'])

print("="*80)
print("EDA - EXPLORATORY DATA ANALYSIS")
print("="*80)

print(f"\n📊 Dataset: {len(data):,} orders across {data['order_hour'].nunique()} hours")

# ==============================================================================
# 1. TEMPORAL PATTERNS
# ==============================================================================
print("\n" + "="*80)
print("1. TEMPORAL PATTERNS")
print("="*80)

# Hourly distribution
hourly_orders = data.groupby('hour').size()
peak_hours = hourly_orders.nlargest(3)
low_hours = hourly_orders.nsmallest(3)

print(f"\n📈 HOURLY DISTRIBUTION:")
print(f"   Peak hours:")
for hour, count in peak_hours.items():
    print(f"      Hour {hour:2d}: {count:5,} orders ({count/len(data)*100:4.1f}%)")
print(f"   Lowest hours:")
for hour, count in low_hours.items():
    print(f"      Hour {hour:2d}: {count:5,} orders ({count/len(data)*100:4.1f}%)")

# Calculate peak vs off-peak
peak_mask = data['hour'].isin([19, 20, 21])
off_peak_mask = data['hour'].isin([11, 12, 13, 14, 15, 16])
peak_avg_dishes = data[peak_mask]['total_dishes'].mean()
off_peak_avg_dishes = data[off_peak_mask]['total_dishes'].mean()
peak_ratio = peak_avg_dishes / off_peak_avg_dishes if off_peak_avg_dishes > 0 else 0

print(f"\n🔥 PEAK HOURS (19-21) ANALYSIS:")
print(f"   Avg dishes per order in peak: {peak_avg_dishes:.2f}")
print(f"   Avg dishes per order off-peak: {off_peak_avg_dishes:.2f}")
print(f"   Peak ratio: {peak_ratio:.2f}x")

# Day of week
dow_orders = data.groupby('day_name').size()
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
print(f"\n📅 DAY OF WEEK:")
for day in dow_order:
    count = dow_orders.get(day, 0)
    print(f"   {day:10s}: {count:5,} orders")

# Weekend vs weekday
weekend_total = data[data['is_weekend']]['total_dishes'].sum()
weekend_days = data[data['is_weekend']]['date'].nunique()
weekday_total = data[~data['is_weekend']]['total_dishes'].sum()
weekday_days = data[~data['is_weekend']]['date'].nunique()

weekend_avg_per_day = weekend_total / weekend_days if weekend_days > 0 else 0
weekday_avg_per_day = weekday_total / weekday_days if weekday_days > 0 else 0
weekend_lift = (weekend_avg_per_day - weekday_avg_per_day) / weekday_avg_per_day * 100

print(f"\n🎉 WEEKEND vs WEEKDAY:")
print(f"   Weekend avg dishes/day: {weekend_avg_per_day:.1f}")
print(f"   Weekday avg dishes/day: {weekday_avg_per_day:.1f}")
print(f"   Weekend lift: {weekend_lift:+.1f}%")

# ==============================================================================
# 2. WEATHER PATTERNS
# ==============================================================================
print("\n" + "="*80)
print("2. WEATHER PATTERNS")
print("="*80)

print(f"\n🌡️ TEMPERATURE:")
print(f"   Range: {data['env_temp'].min():.1f}°C to {data['env_temp'].max():.1f}°C")
print(f"   Mean: {data['env_temp'].mean():.1f}°C")
print(f"   Std: {data['env_temp'].std():.1f}°C")

# Temperature bins
temp_bins = pd.cut(data['env_temp'], bins=[0, 10, 15, 20, 25, 30, 40], 
                   labels=['<10°C', '10-15°C', '15-20°C', '20-25°C', '25-30°C', '>30°C'])
temp_analysis = data.groupby(temp_bins).agg({
    'Order ID': 'count',
    'total_dishes': 'sum'
}).rename(columns={'Order ID': 'order_count'})

print(f"\n   Orders by temperature range:")
for temp_range, row in temp_analysis.iterrows():
    pct = row['order_count'] / len(data) * 100
    print(f"   {str(temp_range):10s}: {row['order_count']:5,} orders ({pct:4.1f}%) | {row['total_dishes']:6,} dishes")

# Rain analysis
rainy_orders = data[data['env_precip'] > 0]
non_rainy_orders = data[data['env_precip'] == 0]

rainy_avg_per_day = rainy_orders.groupby(rainy_orders['date']).size().mean() if len(rainy_orders) > 0 else 0
non_rainy_avg_per_day = non_rainy_orders.groupby(non_rainy_orders['date']).size().mean()

rain_effect = (rainy_avg_per_day - non_rainy_avg_per_day) / non_rainy_avg_per_day * 100

print(f"\n🌧️ RAIN IMPACT:")
print(f"   Rainy orders: {len(rainy_orders):,} ({len(rainy_orders)/len(data)*100:.1f}%)")
print(f"   Non-rainy orders: {len(non_rainy_orders):,} ({len(non_rainy_orders)/len(data)*100:.1f}%)")
print(f"   Rainy avg orders/day: {rainy_avg_per_day:.1f}")
print(f"   Non-rainy avg orders/day: {non_rainy_avg_per_day:.1f}")
print(f"   Rain effect: {rain_effect:+.1f}%")

# Weather conditions
print(f"\n☁️ WEATHER CONDITIONS:")
for condition, count in data['env_condition'].value_counts().items():
    pct = count / len(data) * 100
    dishes = data[data['env_condition'] == condition]['total_dishes'].sum()
    print(f"   {condition:15s}: {count:5,} orders ({pct:4.1f}%) | {dishes:6,} dishes")

# ==============================================================================
# 3. POLLUTION PATTERNS
# ==============================================================================
print("\n" + "="*80)
print("3. POLLUTION PATTERNS")
print("="*80)

print(f"\n🏭 AIR QUALITY INDEX (AQI):")
aqi_labels = {2: 'Good', 3: 'Moderate', 4: 'Unhealthy for Sensitive', 5: 'Unhealthy'}
for aqi_val, count in data['aqi'].value_counts().sort_index().items():
    pct = count / len(data) * 100
    label = aqi_labels.get(int(aqi_val), str(aqi_val))
    dishes = data[data['aqi'] == aqi_val]['total_dishes'].sum()
    print(f"   AQI {int(aqi_val)} ({label:25s}): {count:5,} ({pct:4.1f}%) | {dishes:6,} dishes")

# High vs low pollution
high_pollution = data[data['aqi'] >= 5]
low_pollution = data[data['aqi'] < 5]

high_avg = high_pollution.groupby('order_hour').size().mean() if len(high_pollution) > 0 else 0
low_avg = low_pollution.groupby('order_hour').size().mean() if len(low_pollution) > 0 else 0
pollution_effect = (high_avg - low_avg) / low_avg * 100 if low_avg > 0 else 0

print(f"\n   High pollution (AQI 5) avg orders/hour: {high_avg:.2f}")
print(f"   Low pollution (AQI <5) avg orders/hour: {low_avg:.2f}")
print(f"   Pollution effect: {pollution_effect:+.1f}%")

# ==============================================================================
# 4. DISH ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("4. DISH POPULARITY ANALYSIS")
print("="*80)

# Parse all dishes
from collections import Counter
import ast
all_dishes = []
for dishes_str in data['parsed_dishes']:
    try:
        # Convert string representation back to dict if needed
        if isinstance(dishes_str, str):
            dishes_dict = ast.literal_eval(dishes_str)
        else:
            dishes_dict = dishes_str
        for dish, qty in dishes_dict.items():
            all_dishes.extend([dish] * qty)
    except:
        continue

dish_counts = Counter(all_dishes)
total_dish_orders = len(all_dishes)

print(f"\n🍽️ TOP 20 DISHES:")
for i, (dish, count) in enumerate(dish_counts.most_common(20), 1):
    pct = count / total_dish_orders * 100
    print(f"   {i:2d}. {dish[:50]:50s}: {count:5,} ({pct:4.1f}%)")

# Chicken analysis
chicken_keywords = ['Chicken', 'chicken', 'Murgh', 'Tangdi', 'Tender']
chicken_dishes = [d for d in all_dishes if any(kw in d for kw in chicken_keywords)]
chicken_pct = len(chicken_dishes) / len(all_dishes) * 100

print(f"\n🍗 CHICKEN DOMINANCE:")
print(f"   Total dish orders: {len(all_dishes):,}")
print(f"   Chicken dish orders: {len(chicken_dishes):,} ({chicken_pct:.1f}%)")

# ==============================================================================
# 5. RESTAURANT ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("5. RESTAURANT DISTRIBUTION")
print("="*80)

if 'Restaurant ID' in data.columns:
    rest_orders = data.groupby('Restaurant ID').size().sort_values(ascending=False)
    print(f"\n🏪 TOP 10 RESTAURANTS:")
    for rest_id, count in rest_orders.head(10).items():
        pct = count / len(data) * 100
        rest_name = data[data['Restaurant ID'] == rest_id]['Restaurant name'].iloc[0] if 'Restaurant name' in data.columns else 'Unknown'
        print(f"   {rest_id}: {rest_name[:30]:30s} - {count:5,} orders ({pct:4.1f}%)")
else:
    print("\n⚠️ Restaurant ID column not found in data")

# ==============================================================================
# 6. EVENT ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("6. EVENT IMPACT")
print("="*80)

event_orders = data[data['has_event']].groupby('date').size()
non_event_orders = data[~data['has_event']].groupby('date').size()

event_avg = event_orders.mean() if len(event_orders) > 0 else 0
non_event_avg = non_event_orders.mean() if len(non_event_orders) > 0 else 0
event_lift = (event_avg - non_event_avg) / non_event_avg * 100 if non_event_avg > 0 else 0

print(f"\n🎊 EVENT DAYS:")
print(f"   Days with events: {data[data['has_event']]['date'].nunique()}")
print(f"   Days without events: {data[~data['has_event']]['date'].nunique()}")
print(f"   Event days avg orders/day: {event_avg:.1f}")
print(f"   Non-event days avg orders/day: {non_event_avg:.1f}")
print(f"   Event lift: {event_lift:+.1f}%")

# ==============================================================================
# SAVE EDA SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("SAVING EDA SUMMARY")
print("="*80)

summary = {
    'total_orders': len(data),
    'total_dishes': data['total_dishes'].sum(),
    'unique_hours': data['order_hour'].nunique(),
    'unique_days': data['date'].nunique(),
    'unique_dishes': len(set(all_dishes)),
    'temp_min': data['env_temp'].min(),
    'temp_max': data['env_temp'].max(),
    'temp_mean': data['env_temp'].mean(),
    'rainy_orders_pct': len(rainy_orders) / len(data) * 100,
    'weekend_lift_pct': weekend_lift,
    'peak_ratio': peak_ratio,
    'chicken_pct': chicken_pct,
    'rain_effect_pct': rain_effect,
    'pollution_effect_pct': pollution_effect,
    'event_lift_pct': event_lift
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('outputs/eda_summary_stats.csv', index=False)
print(f"\n✓ Saved summary to outputs/eda_summary_stats.csv")

print("\n" + "="*80)
print("✅ EDA COMPLETE - ALL STATISTICS VERIFIED FROM ACTUAL DATA")
print("="*80)
