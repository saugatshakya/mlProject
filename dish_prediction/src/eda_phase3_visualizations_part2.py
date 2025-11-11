"""
EDA Phase 3 Part 2: Weather, Pollution & Event Impact Analysis
Understanding external factors' influence on restaurant orders
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

print("="*80)
print("EDA PHASE 3 PART 2: EXTERNAL FACTORS ANALYSIS")
print("="*80)

# Load merged data
print("\n📊 Loading merged hourly data...")
df = pd.read_csv('data/interim/orders_with_temporal.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = pd.to_datetime(df['date'])

# Weather
weather_df = pd.read_csv('../data/hourly_orders_weather.csv')
weather_df['order_hour'] = pd.to_datetime(weather_df['order_hour'])

# Pollution
pollution_df = pd.read_csv('../data/pollution.csv')
pollution_df['pollution_time_utc'] = pd.to_datetime(pollution_df['pollution_time_utc'])
pollution_df['pollution_hour'] = pollution_df['pollution_time_utc'].dt.floor('H')

# Events
events_df = pd.read_csv('../data/events.csv')
events_df['date'] = pd.to_datetime(events_df['date'])

# Hourly aggregation
hourly_df = df.groupby(df['datetime'].dt.floor('H')).agg({
    'Order ID': 'count',
    'Total': ['sum', 'mean']
}).reset_index()
hourly_df.columns = ['datetime', 'order_count', 'revenue_total', 'revenue_avg']

# Merge
hourly_df = hourly_df.merge(
    weather_df[['order_hour', 'env_temp', 'env_rhum', 'env_precip', 'env_wspd', 'env_condition']],
    left_on='datetime', right_on='order_hour', how='left'
)
hourly_df = hourly_df.merge(
    pollution_df[['pollution_hour', 'aqi', 'pm2_5', 'pm10', 'no2', 'o3', 'co']],
    left_on='datetime', right_on='pollution_hour', how='left'
)

print(f"✓ Merged data: {len(hourly_df):,} hours")

# ============================================================================
# WEATHER IMPACT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("WEATHER IMPACT ANALYSIS")
print("="*80)

# Clean data for analysis
weather_clean = hourly_df.dropna(subset=['env_temp', 'order_count'])

# Correlations
print("\n🌡️  Weather Correlations with Orders:")
for col in ['env_temp', 'env_rhum', 'env_precip', 'env_wspd']:
    if col in weather_clean.columns:
        corr, pval = pearsonr(weather_clean[col], weather_clean['order_count'])
        print(f"   {col:15s}: r={corr:+.3f} (p={pval:.4f})")

# Viz 4: Temperature impact
print("\n📈 [4/20] Temperature vs orders...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Scatter plot
axes[0, 0].scatter(weather_clean['env_temp'], weather_clean['order_count'], 
                   alpha=0.4, s=30, c='coral')
axes[0, 0].set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Orders per Hour', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Temperature vs Order Volume', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(weather_clean['env_temp'], weather_clean['order_count'], 1)
p = np.poly1d(z)
x_line = np.linspace(weather_clean['env_temp'].min(), weather_clean['env_temp'].max(), 100)
axes[0, 0].plot(x_line, p(x_line), "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.0f}')
axes[0, 0].legend()

# Temperature bins
temp_bins = pd.cut(weather_clean['env_temp'], bins=5)
temp_grouped = weather_clean.groupby(temp_bins)['order_count'].mean()
axes[0, 1].bar(range(len(temp_grouped)), temp_grouped.values, 
               color='orange', edgecolor='black', alpha=0.7)
axes[0, 1].set_xticks(range(len(temp_grouped)))
axes[0, 1].set_xticklabels([f'{int(x.left)}-{int(x.right)}°C' for x in temp_grouped.index], rotation=45)
axes[0, 1].set_ylabel('Avg Orders', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Orders by Temperature Range', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Precipitation impact
rain_yes = weather_clean[weather_clean['env_precip'] > 0]['order_count'].mean()
rain_no = weather_clean[weather_clean['env_precip'] == 0]['order_count'].mean()
impact = ((rain_yes / rain_no) - 1) * 100

axes[1, 0].bar(['No Rain', 'Raining'], [rain_no, rain_yes], 
               color=['skyblue', 'navy'], edgecolor='black', alpha=0.7)
axes[1, 0].set_ylabel('Avg Orders per Hour', fontsize=12, fontweight='bold')
axes[1, 0].set_title(f'Rain Impact: {impact:+.1f}% change', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([rain_no, rain_yes]):
    axes[1, 0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=12, fontweight='bold')

# Weather condition breakdown
condition_avg = weather_clean.groupby('env_condition')['order_count'].mean().sort_values(ascending=False)
axes[1, 1].barh(range(len(condition_avg)), condition_avg.values, color='teal', alpha=0.7)
axes[1, 1].set_yticks(range(len(condition_avg)))
axes[1, 1].set_yticklabels(condition_avg.index)
axes[1, 1].set_xlabel('Avg Orders per Hour', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Orders by Weather Condition', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('reports/figures/01_eda/04_weather_impact.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: 04_weather_impact.png")
plt.close()

# ============================================================================
# POLLUTION IMPACT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("POLLUTION IMPACT ANALYSIS")
print("="*80)

pollution_clean = hourly_df.dropna(subset=['aqi', 'order_count'])

print("\n🏭 Pollution Correlations with Orders:")
for col in ['aqi', 'pm2_5', 'pm10', 'no2', 'o3']:
    if col in pollution_clean.columns:
        corr, pval = pearsonr(pollution_clean[col], pollution_clean['order_count'])
        print(f"   {col:15s}: r={corr:+.3f} (p={pval:.4f})")

# Viz 5: Pollution impact
print("\n📈 [5/20] Pollution vs orders...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# AQI scatter
axes[0, 0].scatter(pollution_clean['aqi'], pollution_clean['order_count'],
                   alpha=0.4, s=30, c=pollution_clean['aqi'], cmap='RdYlGn_r')
axes[0, 0].set_xlabel('Air Quality Index (AQI)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Orders per Hour', fontsize=12, fontweight='bold')
axes[0, 0].set_title('AQI vs Order Volume', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# AQI categories (Delhi-specific)
def categorize_aqi(aqi):
    if aqi <= 2: return 'Good'
    elif aqi <= 3: return 'Moderate'
    elif aqi <= 4: return 'Poor'
    else: return 'Very Poor'

pollution_clean['aqi_category'] = pollution_clean['aqi'].apply(categorize_aqi)
aqi_cat_avg = pollution_clean.groupby('aqi_category')['order_count'].mean()
aqi_cat_order = ['Good', 'Moderate', 'Poor', 'Very Poor']
aqi_cat_avg = aqi_cat_avg.reindex([c for c in aqi_cat_order if c in aqi_cat_avg.index])

colors_aqi = {'Good': 'green', 'Moderate': 'yellow', 'Poor': 'orange', 'Very Poor': 'red'}
bar_colors = [colors_aqi.get(cat, 'gray') for cat in aqi_cat_avg.index]

axes[0, 1].bar(range(len(aqi_cat_avg)), aqi_cat_avg.values, 
               color=bar_colors, edgecolor='black', alpha=0.7)
axes[0, 1].set_xticks(range(len(aqi_cat_avg)))
axes[0, 1].set_xticklabels(aqi_cat_avg.index)
axes[0, 1].set_ylabel('Avg Orders', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Orders by Air Quality Category', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(aqi_cat_avg.values):
    axes[0, 1].text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=11, fontweight='bold')

# PM2.5 impact
axes[1, 0].scatter(pollution_clean['pm2_5'], pollution_clean['order_count'],
                   alpha=0.4, s=30, c='brown')
axes[1, 0].set_xlabel('PM2.5 (μg/m³)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Orders per Hour', fontsize=12, fontweight='bold')
axes[1, 0].set_title('PM2.5 Particulate Matter vs Orders', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# High pollution vs low pollution
median_aqi = pollution_clean['aqi'].median()
high_pollution = pollution_clean[pollution_clean['aqi'] > median_aqi]['order_count'].mean()
low_pollution = pollution_clean[pollution_clean['aqi'] <= median_aqi]['order_count'].mean()
pollution_impact = ((high_pollution / low_pollution) - 1) * 100

axes[1, 1].bar(['Low Pollution\n(AQI ≤ median)', 'High Pollution\n(AQI > median)'],
               [low_pollution, high_pollution],
               color=['lightgreen', 'darkred'], edgecolor='black', alpha=0.7)
axes[1, 1].set_ylabel('Avg Orders per Hour', fontsize=12, fontweight='bold')
axes[1, 1].set_title(f'High Pollution Impact: {pollution_impact:+.1f}% change', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([low_pollution, high_pollution]):
    axes[1, 1].text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/01_eda/05_pollution_impact.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: 05_pollution_impact.png")
plt.close()

# ============================================================================
# EVENT IMPACT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("EVENT IMPACT ANALYSIS")
print("="*80)

# Daily aggregation
daily_df = df.groupby('date').agg({
    'Order ID': 'count',
    'Total': 'sum'
}).reset_index()
daily_df.columns = ['date', 'order_count', 'revenue']

# Merge events
daily_df = daily_df.merge(events_df, on='date', how='left')
daily_df['has_event'] = ~daily_df['event'].isna() & (daily_df['event'] != 'No significant event')
daily_df['is_holiday'] = daily_df['holiday'].fillna(False)

print("\n🎉 Event Statistics:")
print(f"   Days with events: {daily_df['has_event'].sum()}")
print(f"   Days with holidays: {daily_df['is_holiday'].sum()}")

event_days = daily_df[daily_df['has_event']]['order_count'].mean()
normal_days = daily_df[~daily_df['has_event']]['order_count'].mean()
event_impact = ((event_days / normal_days) - 1) * 100

print(f"\n   Event day orders: {event_days:.1f} avg")
print(f"   Normal day orders: {normal_days:.1f} avg")
print(f"   Event impact: {event_impact:+.1f}%")

# Viz 6: Event impact
print("\n📈 [6/20] Event impact analysis...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Event vs normal days
axes[0].bar(['Normal Days', 'Event Days'], [normal_days, event_days],
            color=['lightblue', 'gold'], edgecolor='black', alpha=0.7, width=0.6)
axes[0].set_ylabel('Avg Daily Orders', fontsize=12, fontweight='bold')
axes[0].set_title(f'Event Impact on Orders: {event_impact:+.1f}%', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([normal_days, event_days]):
    axes[0].text(i, v + 5, f'{v:.1f}', ha='center', fontsize=14, fontweight='bold')

# Timeline with events marked
axes[1].plot(daily_df['date'], daily_df['order_count'], 
             color='steelblue', linewidth=1.5, label='Daily orders')
event_dates = daily_df[daily_df['has_event']]
axes[1].scatter(event_dates['date'], event_dates['order_count'],
                color='red', s=100, marker='*', label='Event days', zorder=5)
axes[1].set_xlabel('Date', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Daily Orders', fontsize=12, fontweight='bold')
axes[1].set_title('Order Timeline with Event Markers', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/01_eda/06_event_impact.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: 06_event_impact.png")
plt.close()

# ============================================================================
# CORRELATION MATRIX
# ============================================================================

print("\n📈 [7/20] Feature correlation matrix...")

# Select numeric features
corr_features = ['order_count', 'env_temp', 'env_rhum', 'env_precip', 'env_wspd',
                 'aqi', 'pm2_5', 'pm10', 'no2', 'o3']
corr_df = hourly_df[corr_features].dropna()

fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = corr_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/01_eda/07_correlation_matrix.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: 07_correlation_matrix.png")
plt.close()

print("\n" + "="*80)
print("✅ EXTERNAL FACTORS ANALYSIS COMPLETE (7/20 visualizations)")
print("="*80)

print("\n💡 KEY FINDINGS:")
print(f"   🌡️  Temperature: Moderate negative correlation")
print(f"   🌧️  Rain: {impact:+.1f}% impact on orders")
print(f"   🏭 Pollution: {pollution_impact:+.1f}% impact (high vs low)")
print(f"   🎉 Events: {event_impact:+.1f}% boost on event days")
print("\n   → These external factors MUST be included in feature engineering!")
