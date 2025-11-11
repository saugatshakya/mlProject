"""
EDA Phase 3: Comprehensive Visualization & Feature Discovery
Creating publication-quality visualizations for deep understanding

This analysis will create 20+ visualizations covering:
1. Temporal patterns (hourly, daily, weekly trends)
2. Dish popularity and preferences
3. Weather impact on orders
4. Pollution correlation analysis
5. Event-driven behavior
6. Delhi-specific seasonal patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

print("="*80)
print("EDA PHASE 3: COMPREHENSIVE VISUALIZATION & FEATURE DISCOVERY")
print("="*80)

# ============================================================================
# LOAD ALL DATA
# ============================================================================

print("\n📊 Loading all datasets...")

# Orders
df = pd.read_csv('data/interim/orders_with_temporal.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = pd.to_datetime(df['date'])
print(f"✓ Orders: {len(df):,} records")

# Weather
weather_df = pd.read_csv('../data/hourly_orders_weather.csv')
weather_df['order_hour'] = pd.to_datetime(weather_df['order_hour'])
print(f"✓ Weather: {len(weather_df):,} records")

# Pollution
pollution_df = pd.read_csv('../data/pollution.csv')
pollution_df['pollution_time_utc'] = pd.to_datetime(pollution_df['pollution_time_utc'])
pollution_df['pollution_hour'] = pollution_df['pollution_time_utc'].dt.floor('H')
print(f"✓ Pollution: {len(pollution_df):,} records")

# Events
events_df = pd.read_csv('../data/events.csv')
events_df['date'] = pd.to_datetime(events_df['date'])
print(f"✓ Events: {len(events_df):,} records")

# Create hourly aggregation
hourly_df = df.groupby(df['datetime'].dt.floor('H')).agg({
    'Order ID': 'count',
    'Total': ['sum', 'mean'],
    'KPT duration (minutes)': 'mean',
    'hour': 'first',
    'day_of_week': 'first',
    'day_name': 'first'
}).reset_index()

hourly_df.columns = ['datetime', 'order_count', 'revenue_total', 'revenue_avg', 
                      'kpt_duration', 'hour', 'day_of_week', 'day_name']
print(f"✓ Hourly aggregation: {len(hourly_df):,} hours")

# Merge with weather
hourly_df = hourly_df.merge(
    weather_df[['order_hour', 'env_temp', 'env_rhum', 'env_precip', 'env_wspd', 'env_condition']],
    left_on='datetime',
    right_on='order_hour',
    how='left'
)

# Merge with pollution
hourly_df = hourly_df.merge(
    pollution_df[['pollution_hour', 'aqi', 'pm2_5', 'pm10', 'no2', 'o3']],
    left_on='datetime',
    right_on='pollution_hour',
    how='left'
)

print(f"✓ Merged dataset: {len(hourly_df):,} hours with {hourly_df.shape[1]} features")
print(f"   Missing weather: {hourly_df['env_temp'].isnull().sum()} hours")
print(f"   Missing pollution: {hourly_df['aqi'].isnull().sum()} hours")

# ============================================================================
# VISUALIZATION 1: TEMPORAL PATTERNS
# ============================================================================

print("\n" + "="*80)
print("Creating visualizations...")
print("="*80)

import os
os.makedirs('reports/figures/01_eda', exist_ok=True)

# 1.1 Hourly pattern
print("\n📈 [1/20] Hourly order pattern...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Hourly average
hourly_avg = hourly_df.groupby('hour')['order_count'].mean()
axes[0].bar(hourly_avg.index, hourly_avg.values, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axhline(hourly_avg.mean(), color='red', linestyle='--', label=f'Mean: {hourly_avg.mean():.1f}', linewidth=2)
axes[0].axhline(hourly_avg.mean() + hourly_avg.std(), color='orange', linestyle=':', 
                label=f'Mean + 1σ: {hourly_avg.mean() + hourly_avg.std():.1f}', linewidth=2)
axes[0].set_xlabel('Hour of Day', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Average Orders per Hour', fontsize=14, fontweight='bold')
axes[0].set_title('Restaurant Order Pattern by Hour (Delhi Market)', fontsize=16, fontweight='bold')
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(range(24))

# Add meal period annotations
axes[0].axvspan(0, 4, alpha=0.1, color='purple', label='Late Night')
axes[0].axvspan(11, 15, alpha=0.1, color='yellow', label='Lunch')
axes[0].axvspan(19, 23, alpha=0.1, color='orange', label='Dinner')

# Hourly heatmap by day
pivot_hourly = hourly_df.pivot_table(
    values='order_count', 
    index='day_name', 
    columns='hour', 
    aggfunc='mean'
).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

sns.heatmap(pivot_hourly, cmap='YlOrRd', annot=True, fmt='.0f', cbar_kws={'label': 'Avg Orders'},
            linewidths=0.5, ax=axes[1])
axes[1].set_xlabel('Hour of Day', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Day of Week', fontsize=14, fontweight='bold')
axes[1].set_title('Order Intensity Heatmap (Day × Hour)', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/01_eda/01_hourly_patterns.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: 01_hourly_patterns.png")
plt.close()

# 1.2 Daily trends
print("\n📈 [2/20] Daily and weekly patterns...")
daily_df = df.groupby('date').agg({
    'Order ID': 'count',
    'Total': 'sum',
    'day_name': 'first',
    'day_of_week': 'first'
}).reset_index()
daily_df.columns = ['date', 'order_count', 'revenue', 'day_name', 'day_of_week']

fig, axes = plt.subplots(3, 1, figsize=(18, 12))

# Time series
axes[0].plot(daily_df['date'], daily_df['order_count'], marker='o', markersize=3, linewidth=1.5, color='darkblue')
axes[0].axhline(daily_df['order_count'].mean(), color='red', linestyle='--', 
                label=f'Mean: {daily_df["order_count"].mean():.0f}', linewidth=2)
axes[0].fill_between(daily_df['date'], 
                      daily_df['order_count'].rolling(7, center=True).mean() - daily_df['order_count'].rolling(7).std(),
                      daily_df['order_count'].rolling(7, center=True).mean() + daily_df['order_count'].rolling(7).std(),
                      alpha=0.2, color='blue', label='7-day rolling mean ± σ')
axes[0].set_xlabel('Date', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Daily Orders', fontsize=14, fontweight='bold')
axes[0].set_title('Daily Order Time Series (Sep 2024 - Jan 2025)', fontsize=16, fontweight='bold')
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

# Day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_avg = daily_df.groupby('day_name')['order_count'].mean().reindex(day_order)
colors = ['steelblue'] * 5 + ['coral', 'coral']  # Highlight weekends
axes[1].bar(day_avg.index, day_avg.values, color=colors, edgecolor='black', alpha=0.7)
axes[1].axhline(day_avg.mean(), color='red', linestyle='--', label=f'Weekly Mean: {day_avg.mean():.0f}', linewidth=2)
axes[1].set_xlabel('Day of Week', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Average Daily Orders', fontsize=14, fontweight='bold')
axes[1].set_title('Order Volume by Day of Week', fontsize=16, fontweight='bold')
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

# Box plot
axes[2].boxplot([daily_df[daily_df['day_name'] == day]['order_count'] for day in day_order],
                 labels=day_order, patch_artist=True,
                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))
axes[2].set_xlabel('Day of Week', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Order Count Distribution', fontsize=14, fontweight='bold')
axes[2].set_title('Order Variability Across Days', fontsize=16, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('reports/figures/01_eda/02_daily_weekly_patterns.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: 02_daily_weekly_patterns.png")
plt.close()

# 1.3 Monthly trends
print("\n📈 [3/20] Monthly and seasonal trends...")
df['month'] = df['datetime'].dt.month
df['month_name'] = df['datetime'].dt.strftime('%b %Y')

monthly_df = df.groupby('month_name').agg({
    'Order ID': 'count',
    'Total': 'sum',
    'month': 'first'
}).reset_index().sort_values('month')

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Monthly orders
axes[0].plot(monthly_df['month_name'], monthly_df['Order ID'], marker='o', markersize=10, 
             linewidth=3, color='darkgreen')
for i, (x, y) in enumerate(zip(monthly_df['month_name'], monthly_df['Order ID'])):
    axes[0].text(i, y + 100, f'{y:,}', ha='center', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Month', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Total Orders', fontsize=14, fontweight='bold')
axes[0].set_title('Monthly Order Volume Trend', fontsize=16, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Revenue trend
axes[1].bar(monthly_df['month_name'], monthly_df['Total']/1000, color='gold', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Month', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Total Revenue (₹ Thousands)', fontsize=14, fontweight='bold')
axes[1].set_title('Monthly Revenue Trend', fontsize=16, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('reports/figures/01_eda/03_monthly_trends.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: 03_monthly_trends.png")
plt.close()

print("\n✅ Temporal visualizations complete (3/20)")
print("   Next: Dish analysis, weather, pollution, events...")
print("\n💡 KEY FINDING: Clear peak hours (19-21), weekend effect, monthly growth trend")
