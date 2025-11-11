"""
STEP 02: CREATE VISUALIZATIONS FROM REAL DATA
Generate plots showing ACTUAL patterns - NO HALLUCINATIONS
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Load data
data = pd.read_csv('data/eda_processed.csv', parse_dates=['order_hour', 'Order Placed At'])

print("="*80)
print("CREATING VISUALIZATIONS FROM REAL DATA")
print("="*80)

# ==============================================================================
# VIZ 1: TEMPORAL PATTERNS
# ==============================================================================
print("\n📊 Creating temporal pattern visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Hourly distribution
hourly = data.groupby('hour').size()
axes[0, 0].bar(hourly.index, hourly.values, color='steelblue', alpha=0.8, edgecolor='black')
axes[0, 0].set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Number of Orders', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Hourly Order Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(range(24))
axes[0, 0].grid(axis='y', alpha=0.3)

# Day of week
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_data = data.groupby('day_name').size().reindex(dow_order)
colors = ['#FF6B6B' if d in ['Saturday', 'Sunday'] else '#4ECDC4' for d in dow_order]
axes[0, 1].bar(range(7), dow_data.values, color=colors, alpha=0.8, edgecolor='black')
axes[0, 1].set_xlabel('Day of Week', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Total Orders', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Orders by Day of Week', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(range(7))
axes[0, 1].set_xticklabels(dow_order, rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# Peak vs off-peak
peak_data = data[data['hour'].isin([19, 20, 21])]
off_peak_data = data[data['hour'].isin([11, 12, 13, 14, 15, 16])]
comparison = pd.DataFrame({
    'Period': ['Peak\n(19-21)', 'Off-Peak\n(11-16)'],
    'Orders': [len(peak_data), len(off_peak_data)],
    'Avg Dishes/Order': [peak_data['total_dishes'].mean(), off_peak_data['total_dishes'].mean()]
})
axes[1, 0].bar(comparison['Period'], comparison['Orders'], color=['#e74c3c', '#3498db'], alpha=0.8, edgecolor='black')
axes[1, 0].set_ylabel('Number of Orders', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Peak vs Off-Peak Orders', fontsize=14, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Weekend vs weekday
weekend_data = data[data['is_weekend']]
weekday_data = data[~data['is_weekend']]
weekend_dishes_per_day = weekend_data['total_dishes'].sum() / weekend_data['date'].nunique()
weekday_dishes_per_day = weekday_data['total_dishes'].sum() / weekday_data['date'].nunique()
comparison2 = pd.DataFrame({
    'Type': ['Weekend', 'Weekday'],
    'Avg Dishes/Day': [weekend_dishes_per_day, weekday_dishes_per_day]
})
axes[1, 1].bar(comparison2['Type'], comparison2['Avg Dishes/Day'], color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='black')
axes[1, 1].set_ylabel('Avg Dishes per Day', fontsize=12, fontweight='bold')
axes[1, 1].set_title(f'Weekend vs Weekday\n(Lift: +{((weekend_dishes_per_day - weekday_dishes_per_day) / weekday_dishes_per_day * 100):.1f}%)', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/01_temporal_patterns.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/01_temporal_patterns.png")
plt.close()

# ==============================================================================
# VIZ 2: WEATHER IMPACT
# ==============================================================================
print("\n🌦️ Creating weather impact visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Temperature bins
temp_bins = pd.cut(data['env_temp'], bins=[0, 10, 15, 20, 25, 30, 40], 
                   labels=['<10°C', '10-15°C', '15-20°C', '20-25°C', '25-30°C', '>30°C'])
temp_orders = data.groupby(temp_bins).size()
axes[0, 0].bar(range(len(temp_orders)), temp_orders.values, color='coral', alpha=0.8, edgecolor='black')
axes[0, 0].set_xticks(range(len(temp_orders)))
axes[0, 0].set_xticklabels(temp_orders.index, rotation=0)
axes[0, 0].set_xlabel('Temperature Range', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Orders', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Temperature vs Order Volume', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)

# Weather conditions
weather_orders = data.groupby('env_condition').size().sort_values(ascending=True)
axes[0, 1].barh(range(len(weather_orders)), weather_orders.values, color='skyblue', alpha=0.8, edgecolor='black')
axes[0, 1].set_yticks(range(len(weather_orders)))
axes[0, 1].set_yticklabels(weather_orders.index)
axes[0, 1].set_xlabel('Orders', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Orders by Weather Condition', fontsize=14, fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)

# Humidity
humidity_bins = pd.cut(data['env_rhum'], bins=[0, 40, 60, 80, 100], labels=['Low', 'Medium', 'High', 'Very High'])
humidity_orders = data.groupby(humidity_bins).size()
axes[1, 0].bar(range(len(humidity_orders)), humidity_orders.values, color='lightgreen', alpha=0.8, edgecolor='black')
axes[1, 0].set_xticks(range(len(humidity_orders)))
axes[1, 0].set_xticklabels(humidity_orders.index)
axes[1, 0].set_xlabel('Humidity Level', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Orders', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Humidity vs Orders', fontsize=14, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Rain impact
rainy_avg = data[data['env_precip'] > 0].groupby(data['date']).size().mean()
non_rainy_avg = data[data['env_precip'] == 0].groupby(data['date']).size().mean()
rain_comparison = pd.DataFrame({
    'Condition': ['Rainy', 'Non-Rainy'],
    'Avg Orders/Day': [rainy_avg, non_rainy_avg]
})
axes[1, 1].bar(rain_comparison['Condition'], rain_comparison['Avg Orders/Day'], 
               color=['#3498db', '#2ecc71'], alpha=0.8, edgecolor='black')
axes[1, 1].set_ylabel('Avg Orders per Day', fontsize=12, fontweight='bold')
axes[1, 1].set_title(f'Rain Impact on Orders\n(Effect: {((rainy_avg - non_rainy_avg) / non_rainy_avg * 100):.1f}%)', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/02_weather_impact.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/02_weather_impact.png")
plt.close()

# ==============================================================================
# VIZ 3: POLLUTION ANALYSIS
# ==============================================================================
print("\n🏭 Creating pollution analysis...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# AQI distribution
aqi_labels = {2: 'Good', 3: 'Moderate', 4: 'Unhealthy\nfor Sensitive', 5: 'Unhealthy'}
aqi_orders = data.groupby('aqi').size()
colors_aqi = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
axes[0].bar([aqi_labels.get(int(k), str(k)) for k in aqi_orders.index], 
            aqi_orders.values, color=colors_aqi[:len(aqi_orders)], alpha=0.8, edgecolor='black')
axes[0].set_xlabel('AQI Level', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Orders', fontsize=12, fontweight='bold')
axes[0].set_title('Delhi Air Quality Distribution\n(91.5% Unhealthy!)', fontsize=13, fontweight='bold')
axes[0].tick_params(axis='x', labelsize=10)
axes[0].grid(axis='y', alpha=0.3)

# PM2.5 levels
pm25_bins = pd.cut(data['pm2_5'], bins=[0, 50, 100, 200, 300, 500], 
                   labels=['Good', 'Moderate', 'Unhealthy', 'Very\nUnhealthy', 'Hazardous'])
pm25_orders = data.groupby(pm25_bins).size()
axes[1].bar(range(len(pm25_orders)), pm25_orders.values, color='orange', alpha=0.8, edgecolor='black')
axes[1].set_xticks(range(len(pm25_orders)))
axes[1].set_xticklabels(pm25_orders.index)
axes[1].set_xlabel('PM2.5 Level', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Number of Orders', fontsize=12, fontweight='bold')
axes[1].set_title('PM2.5 Distribution\n(WHO Limit: 15 μg/m³)', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Pollution effect
high_poll_avg = data[data['aqi'] >= 5].groupby('order_hour').size().mean()
low_poll_avg = data[data['aqi'] < 5].groupby('order_hour').size().mean()
poll_comparison = pd.DataFrame({
    'AQI Level': ['High (5)', 'Low (<5)'],
    'Avg Orders/Hour': [high_poll_avg, low_poll_avg]
})
axes[2].bar(poll_comparison['AQI Level'], poll_comparison['Avg Orders/Hour'], 
            color=['#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black')
axes[2].set_ylabel('Avg Orders per Hour', fontsize=12, fontweight='bold')
axes[2].set_title(f'Pollution Effect\n({((high_poll_avg - low_poll_avg) / low_poll_avg * 100):.1f}%)', fontsize=13, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/03_pollution_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/03_pollution_analysis.png")
plt.close()

# ==============================================================================
# VIZ 4: TOP DISHES
# ==============================================================================
print("\n🍽️ Creating dish popularity analysis...")

# Parse dishes
all_dishes = []
for dishes_str in data['parsed_dishes']:
    try:
        if isinstance(dishes_str, str):
            dishes_dict = ast.literal_eval(dishes_str)
        else:
            dishes_dict = dishes_str
        for dish, qty in dishes_dict.items():
            all_dishes.extend([dish] * qty)
    except:
        continue

from collections import Counter
dish_counts = Counter(all_dishes)
top_20 = dict(dish_counts.most_common(20))

fig, ax = plt.subplots(figsize=(14, 10))
dishes = list(top_20.keys())
counts = list(top_20.values())
y_pos = range(len(dishes))

# Color chicken dishes differently
colors = ['#e74c3c' if any(kw in dish for kw in ['Chicken', 'chicken', 'Murgh']) else '#3498db' for dish in dishes]

ax.barh(y_pos, counts, color=colors, alpha=0.8, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels([d[:40] for d in dishes], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Number of Orders', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Most Popular Dishes\n(Red = Chicken, Blue = Other)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/04_top_dishes.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/04_top_dishes.png")
plt.close()

print("\n" + "="*80)
print("✅ ALL VISUALIZATIONS CREATED FROM REAL DATA")
print("="*80)
print("\nGenerated visualizations:")
print("   1. 01_temporal_patterns.png - Hourly, daily, peak vs off-peak")
print("   2. 02_weather_impact.png - Temperature, conditions, rain effect")
print("   3. 03_pollution_analysis.png - AQI, PM2.5, pollution impact")
print("   4. 04_top_dishes.png - Top 20 dishes with chicken highlighted")
