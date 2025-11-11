"""
Step 04: Create Dish-Level Pivot Table
========================================
Transform order-level data into hour × dish pivot table for modeling.

Input: data/eda_processed.csv (21,321 orders)
Output: data/dish_pivot.csv (hourly × top dishes)
"""

import pandas as pd
import numpy as np
import ast
from collections import Counter

print("="*80)
print("STEP 04: CREATE DISH-LEVEL PIVOT TABLE")
print("="*80)

# ==============================================================================
# 1. Load processed data
# ==============================================================================
print("\n📂 Loading processed data...")
data = pd.read_csv('data/eda_processed.csv')
data['order_hour_dt'] = pd.to_datetime(data['order_hour'])

print(f"✓ Loaded {len(data):,} orders")
print(f"  Date range: {data['order_hour_dt'].min()} to {data['order_hour_dt'].max()}")

# ==============================================================================
# 2. Extract all dishes with quantities
# ==============================================================================
print("\n🍽️  Extracting dishes from all orders...")

# Parse dishes from each order
dish_records = []
for idx, row in data.iterrows():
    try:
        if isinstance(row['parsed_dishes'], str):
            dishes_dict = ast.literal_eval(row['parsed_dishes'])
        else:
            dishes_dict = row['parsed_dishes']
        
        # Create a record for each dish in the order
        for dish_name, quantity in dishes_dict.items():
            dish_records.append({
                'order_hour': row['order_hour'],
                'dish_name': dish_name,
                'quantity': quantity
            })
    except Exception as e:
        continue

dish_df = pd.DataFrame(dish_records)
print(f"✓ Extracted {len(dish_df):,} dish records from orders")

# ==============================================================================
# 3. Select top dishes for modeling
# ==============================================================================
print("\n🏆 Selecting top dishes...")

# Count total quantity per dish
dish_totals = dish_df.groupby('dish_name')['quantity'].sum().sort_values(ascending=False)

print(f"\nTop 30 dishes by total quantity:")
for i, (dish, qty) in enumerate(dish_totals.head(30).items(), 1):
    print(f"  {i:2}. {dish[:50]:<50} {qty:>6,} orders")

# Select top 30 dishes
top_30_dishes = dish_totals.head(30).index.tolist()
print(f"\n✓ Selected top {len(top_30_dishes)} dishes for modeling")

# Filter to top dishes only
dish_df_top = dish_df[dish_df['dish_name'].isin(top_30_dishes)].copy()
print(f"✓ Filtered to {len(dish_df_top):,} records ({len(dish_df_top)/len(dish_df)*100:.1f}% of all dishes)")

# ==============================================================================
# 4. Aggregate by hour and dish
# ==============================================================================
print("\n📊 Creating pivot table...")

# Aggregate: sum quantity per hour per dish
pivot_data = dish_df_top.groupby(['order_hour', 'dish_name'])['quantity'].sum().reset_index()

# Pivot to wide format: rows = hours, columns = dishes
pivot = pivot_data.pivot(index='order_hour', columns='dish_name', values='quantity')
pivot = pivot.fillna(0)  # Fill missing with 0 (no orders for that dish that hour)

print(f"✓ Created pivot table: {pivot.shape[0]} hours × {pivot.shape[1]} dishes")

# ==============================================================================
# 5. Merge with weather/pollution/events
# ==============================================================================
print("\n🌤️  Merging with context data...")

# Get unique hour records with context
context_cols = ['order_hour', 'env_temp', 'env_rhum', 'env_precip', 'env_wspd', 
                'env_condition', 'aqi', 'co', 'no', 'no2', 'o3', 'so2', 
                'pm2_5', 'pm10', 'nh3', 'event', 'holiday', 'has_event']

context_data = data[context_cols].drop_duplicates(subset=['order_hour']).copy()
context_data['order_hour_dt'] = pd.to_datetime(context_data['order_hour'])

# Add temporal features
context_data['hour'] = context_data['order_hour_dt'].dt.hour
context_data['day_of_week'] = context_data['order_hour_dt'].dt.dayofweek
context_data['day_of_month'] = context_data['order_hour_dt'].dt.day
context_data['month'] = context_data['order_hour_dt'].dt.month
context_data['is_weekend'] = context_data['day_of_week'].isin([5, 6]).astype(int)

print(f"✓ Context data: {len(context_data):,} unique hours")

# Merge pivot with context
pivot_full = pivot.reset_index().merge(context_data, on='order_hour', how='left')

print(f"✓ Merged data: {len(pivot_full):,} hours × {len(pivot_full.columns)} columns")

# ==============================================================================
# 6. Sort and save
# ==============================================================================
print("\n💾 Saving pivot table...")

# Sort by datetime
pivot_full = pivot_full.sort_values('order_hour_dt').reset_index(drop=True)

# Save to CSV
output_path = 'data/dish_pivot.csv'
pivot_full.to_csv(output_path, index=False)

print(f"✓ Saved to {output_path}")
print(f"  Shape: {pivot_full.shape[0]:,} hours × {pivot_full.shape[1]} columns")
print(f"  Size: {pivot_full.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

# ==============================================================================
# 7. Summary statistics
# ==============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nDish columns: {len(top_30_dishes)}")
print(f"Context columns: {len(context_cols)}")
print(f"Temporal columns: 5 (hour, day_of_week, day_of_month, month, is_weekend)")
print(f"Total columns: {len(pivot_full.columns)}")

print(f"\nTime period:")
print(f"  Start: {pivot_full['order_hour_dt'].min()}")
print(f"  End: {pivot_full['order_hour_dt'].max()}")
print(f"  Hours: {len(pivot_full):,}")
print(f"  Days: {pivot_full['order_hour_dt'].dt.date.nunique()}")

print(f"\nDish order statistics:")
dish_cols = top_30_dishes
for dish in dish_cols[:5]:  # Show stats for top 5
    orders = pivot_full[dish]
    print(f"  {dish[:40]:<40} Mean: {orders.mean():.2f}, Max: {orders.max():.0f}, Non-zero: {(orders>0).sum()}")

print(f"\nMissing values:")
missing = pivot_full.isnull().sum()
if missing.sum() == 0:
    print(f"  ✓ No missing values!")
else:
    print(f"  Columns with missing values:")
    for col in missing[missing > 0].index:
        print(f"    {col}: {missing[col]} ({missing[col]/len(pivot_full)*100:.1f}%)")

print("\n" + "="*80)
print("✅ PIVOT TABLE CREATED SUCCESSFULLY")
print("="*80)
print(f"\nNext step: Run 05_feature_engineering.py")
