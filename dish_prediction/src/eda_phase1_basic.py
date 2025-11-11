"""
Comprehensive Exploratory Data Analysis (EDA)
Restaurant Dish Demand Prediction - Delhi Market

As a restaurant domain expert and data scientist, this EDA will uncover:
1. Temporal patterns (when do people order?)
2. Dish preferences and correlations
3. Weather/pollution impact on orders
4. Event-driven demand spikes
5. Delhi-specific behavioral patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

print("="*80)
print("EXPLORATORY DATA ANALYSIS - RESTAURANT DEMAND FORECASTING")
print("Domain: Food Delivery in Delhi NCR")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND UNDERSTAND RAW DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING RAW ORDER DATA")
print("="*80)

df = pd.read_csv('../data/data.csv')

print(f"\n📊 Dataset Overview:")
print(f"   Total orders: {len(df):,}")
print(f"   Columns: {df.shape[1]}")
print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\n📋 Column Information:")
for col in df.columns:
    dtype = df[col].dtype
    null_pct = (df[col].isnull().sum() / len(df)) * 100
    unique = df[col].nunique()
    print(f"   {col:40s} | {str(dtype):10s} | {null_pct:5.1f}% null | {unique:6,} unique")

# ============================================================================
# STEP 2: TEMPORAL ANALYSIS - CRITICAL FOR RESTAURANT BUSINESS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: TEMPORAL PATTERNS (Restaurant Domain Insights)")
print("="*80)

# Parse datetime
df['datetime'] = pd.to_datetime(df['Order Placed At'], format='%I:%M %p, %B %d %Y')
df['date'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['day_name'] = df['datetime'].dt.day_name()
df['month'] = df['datetime'].dt.month
df['week_of_year'] = df['datetime'].dt.isocalendar().week

print(f"\n📅 Date Range:")
print(f"   Start: {df['datetime'].min()}")
print(f"   End: {df['datetime'].max()}")
print(f"   Duration: {(df['datetime'].max() - df['datetime'].min()).days} days")

print(f"\n⏰ HOURLY PATTERNS (Key for demand forecasting):")
hourly_orders = df.groupby('hour').size().sort_index()
print("\n   Orders by hour:")
for hour, count in hourly_orders.items():
    bar = '█' * int(count / 100)
    print(f"   {hour:02d}:00 | {count:5,} orders {bar}")

# Identify peak hours
peak_threshold = hourly_orders.mean() + hourly_orders.std()
peak_hours = hourly_orders[hourly_orders > peak_threshold].index.tolist()
print(f"\n   🔥 Peak hours (>{peak_threshold:.0f} orders): {peak_hours}")

# Restaurant insights
print(f"\n   📊 Restaurant Business Insights:")
print(f"      - Peak lunch: 12-14h → {hourly_orders[12:15].sum():,} orders ({hourly_orders[12:15].sum()/len(df)*100:.1f}%)")
print(f"      - Peak dinner: 19-22h → {hourly_orders[19:23].sum():,} orders ({hourly_orders[19:23].sum()/len(df)*100:.1f}%)")
print(f"      - Late night: 0-3h → {hourly_orders[0:4].sum():,} orders ({hourly_orders[0:4].sum()/len(df)*100:.1f}%)")
print(f"      - Dead hours: 4-10h → {hourly_orders[4:11].sum():,} orders ({hourly_orders[4:11].sum()/len(df)*100:.1f}%)")

print(f"\n📆 DAILY PATTERNS:")
daily_orders = df.groupby('day_name').size().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)
print("\n   Orders by day of week:")
for day, count in daily_orders.items():
    pct = (count / len(df)) * 100
    bar = '█' * int(count / 100)
    print(f"   {day:10s} | {count:5,} orders ({pct:4.1f}%) {bar}")

weekend_orders = df[df['day_of_week'].isin([5, 6])].shape[0]
weekday_orders = df[~df['day_of_week'].isin([5, 6])].shape[0]
print(f"\n   Weekend vs Weekday:")
print(f"      Weekend: {weekend_orders:,} ({weekend_orders/len(df)*100:.1f}%)")
print(f"      Weekday: {weekday_orders:,} ({weekday_orders/len(df)*100:.1f}%)")

print(f"\n📈 MONTHLY TRENDS:")
monthly_orders = df.groupby('month').size().sort_index()
month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for month, count in monthly_orders.items():
    print(f"   {month_names[month]:3s} | {count:5,} orders")

# ============================================================================
# STEP 3: DISH ANALYSIS - CORE BUSINESS METRICS
# ============================================================================

print("\n" + "="*80)
print("STEP 3: DISH ANALYSIS (Menu Performance)")
print("="*80)

def parse_items(item_str):
    """Extract dishes from order string"""
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

# Extract all dishes
print("\n🍕 Parsing menu items...")
all_dishes = []
for items_str in df['Items in order']:
    dishes = parse_items(items_str)
    for dish_info in dishes:
        all_dishes.append(dish_info)

dish_df = pd.DataFrame(all_dishes)
print(f"   Total dish orders: {len(dish_df):,}")
print(f"   Unique dishes: {dish_df['dish'].nunique():,}")

# Top dishes
print(f"\n🏆 TOP 30 DISHES (by quantity):")
top_dishes = dish_df.groupby('dish')['quantity'].sum().nlargest(30)
total_qty = dish_df['quantity'].sum()

for i, (dish, qty) in enumerate(top_dishes.items(), 1):
    pct = (qty / total_qty) * 100
    bar = '█' * int(pct * 2)
    print(f"   {i:2d}. {dish[:45]:45s} | {qty:5,} units ({pct:4.1f}%) {bar}")

# Category analysis
print(f"\n📊 Dish Categories (Pattern Recognition):")
categories = {
    'Pizza': dish_df[dish_df['dish'].str.contains('Pizza', case=False, na=False)]['quantity'].sum(),
    'Chicken': dish_df[dish_df['dish'].str.contains('Chicken', case=False, na=False)]['quantity'].sum(),
    'Garlic Bread': dish_df[dish_df['dish'].str.contains('Garlic Bread', case=False, na=False)]['quantity'].sum(),
    'Fries': dish_df[dish_df['dish'].str.contains('Fries', case=False, na=False)]['quantity'].sum(),
    'Paneer': dish_df[dish_df['dish'].str.contains('Paneer', case=False, na=False)]['quantity'].sum(),
    'Tender': dish_df[dish_df['dish'].str.contains('Tender', case=False, na=False)]['quantity'].sum(),
}

for cat, qty in sorted(categories.items(), key=lambda x: x[1], reverse=True):
    pct = (qty / total_qty) * 100
    print(f"   {cat:15s} | {qty:6,} units ({pct:4.1f}%)")

# ============================================================================
# STEP 4: ORDER STATUS & BUSINESS METRICS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: ORDER STATUS & BUSINESS METRICS")
print("="*80)

print(f"\n📦 Order Status Distribution:")
status_counts = df['Order Status'].value_counts()
for status, count in status_counts.items():
    pct = (count / len(df)) * 100
    print(f"   {status:20s} | {count:6,} ({pct:5.2f}%)")

# Delivery success rate
delivered = df[df['Order Status'] == 'Delivered'].shape[0]
success_rate = (delivered / len(df)) * 100
print(f"\n   ✅ Delivery success rate: {success_rate:.2f}%")

print(f"\n💰 Revenue Insights:")
delivered_df = df[df['Order Status'] == 'Delivered']
print(f"   Total revenue: ₹{delivered_df['Total'].sum():,.2f}")
print(f"   Average order value: ₹{delivered_df['Total'].mean():.2f}")
print(f"   Median order value: ₹{delivered_df['Total'].median():.2f}")
print(f"   Min order: ₹{delivered_df['Total'].min():.2f}")
print(f"   Max order: ₹{delivered_df['Total'].max():.2f}")

print(f"\n🚚 Delivery Metrics:")
print(f"   Average KPT duration: {delivered_df['KPT duration (minutes)'].mean():.1f} min")
print(f"   Average rider wait: {delivered_df['Rider wait time (minutes)'].mean():.1f} min")
print(f"   Average distance: {delivered_df['Distance'].mode()[0] if len(delivered_df['Distance'].mode()) > 0 else 'N/A'}")

# ============================================================================
# STEP 5: RESTAURANT & LOCATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: RESTAURANT & LOCATION ANALYSIS")
print("="*80)

print(f"\n🏪 Restaurants:")
restaurants = df['Restaurant name'].value_counts()
for rest, count in restaurants.items():
    pct = (count / len(df)) * 100
    print(f"   {rest:30s} | {count:6,} orders ({pct:5.2f}%)")

print(f"\n📍 Delivery Locations (Subzones):")
subzones = df['Subzone'].value_counts().head(10)
for zone, count in subzones.items():
    print(f"   {zone:30s} | {count:6,} orders")

# ============================================================================
# SAVE EDA SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SAVING EDA RESULTS")
print("="*80)

# Save to processed
df.to_csv('data/interim/orders_with_temporal.csv', index=False)
print(f"✓ Saved: data/interim/orders_with_temporal.csv")

top_dishes.to_csv('data/interim/top_dishes.csv')
print(f"✓ Saved: data/interim/top_dishes.csv")

# Summary statistics
summary = {
    'total_orders': len(df),
    'date_range_days': (df['datetime'].max() - df['datetime'].min()).days,
    'total_dishes': dish_df['dish'].nunique(),
    'total_dish_units': dish_df['quantity'].sum(),
    'delivery_success_rate': success_rate,
    'avg_order_value': delivered_df['Total'].mean(),
    'peak_hours': peak_hours,
    'top_dish': top_dishes.index[0],
    'top_dish_qty': top_dishes.iloc[0],
}

pd.Series(summary).to_csv('data/interim/eda_summary.csv')
print(f"✓ Saved: data/interim/eda_summary.csv")

print("\n" + "="*80)
print("✅ EDA PHASE 1 COMPLETE - BASIC UNDERSTANDING")
print("="*80)
print("\nNext: Load external data (weather, pollution, events) for deeper analysis")
