"""
Script to process the original data.csv and create smaller CSV files for each model
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import csv

# Read the original data
print("Reading original data...")
df = pd.read_csv('../data/data.csv')
print(f"Original data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Create output directory
os.makedirs('uploads/original_data', exist_ok=True)

# ============================================================================
# 1. DISH PREDICTION DATA
# ============================================================================
print("\n" + "="*70)
print("CREATING DISH PREDICTION DATA")
print("="*70)

# Parse the 'Items in order' column to extract dish names and quantities
def parse_items(items_str):
    """Parse '1 x Dish1, 2 x Dish2' format"""
    if pd.isna(items_str):
        return []
    
    dishes = []
    items = items_str.split(',')
    for item in items:
        item = item.strip()
        if ' x ' in item:
            parts = item.split(' x ', 1)
            try:
                qty = int(parts[0].strip())
                dish = parts[1].strip()
                dishes.append({'dish': dish, 'quantity': qty})
            except:
                continue
    return dishes

# Parse timestamp
df['timestamp'] = pd.to_datetime(df['Order Placed At'], format='%I:%M %p, %B %d %Y')

# Extract hour
df['hour'] = df['timestamp'].dt.floor('H')

# Parse items
df['parsed_items'] = df['Items in order'].apply(parse_items)

# Aggregate by hour and dish
dish_hourly_data = []
for hour, group in df.groupby('hour'):
    # Count dishes for this hour
    dish_counts = {}
    for items_list in group['parsed_items']:
        for item in items_list:
            dish = item['dish']
            qty = item['quantity']
            dish_counts[dish] = dish_counts.get(dish, 0) + qty
    
    # Create row
    row = {'timestamp': hour}
    row.update(dish_counts)
    dish_hourly_data.append(row)

# Create DataFrame
dish_df = pd.DataFrame(dish_hourly_data).fillna(0)
dish_df = dish_df.sort_values('timestamp').reset_index(drop=True)

# Keep only top N dishes to reduce file size
dish_columns = [col for col in dish_df.columns if col != 'timestamp']
top_dishes = dish_df[dish_columns].sum().nlargest(15).index.tolist()
dish_df_final = dish_df[['timestamp'] + top_dishes]

print(f"Dish prediction data shape: {dish_df_final.shape}")
print(f"Top dishes: {top_dishes}")
print(f"Date range: {dish_df_final['timestamp'].min()} to {dish_df_final['timestamp'].max()}")

# Save
dish_output = 'uploads/original_data/dish_prediction.csv'
dish_df_final.to_csv(dish_output, index=False, quoting=csv.QUOTE_NONNUMERIC)
print(f"✓ Saved to {dish_output}")

# ============================================================================
# 2. DEMAND PREDICTION DATA
# ============================================================================
print("\n" + "="*70)
print("CREATING DEMAND PREDICTION DATA")
print("="*70)

# Count total orders per hour
demand_df = df.groupby('hour').size().reset_index(name='total_orders')
demand_df.rename(columns={'hour': 'timestamp'}, inplace=True)
demand_df = demand_df.sort_values('timestamp').reset_index(drop=True)

print(f"Demand prediction data shape: {demand_df.shape}")
print(f"Date range: {demand_df['timestamp'].min()} to {demand_df['timestamp'].max()}")
print(f"Total orders range: {demand_df['total_orders'].min()} to {demand_df['total_orders'].max()}")

# Save
demand_output = 'uploads/original_data/demand_prediction.csv'
demand_df.to_csv(demand_output, index=False, quoting=csv.QUOTE_NONNUMERIC)
print(f"✓ Saved to {demand_output}")

# ============================================================================
# 3. DISH RECOMMENDATION DATA
# ============================================================================
print("\n" + "="*70)
print("CREATING DISH RECOMMENDATION DATA")
print("="*70)

# Create order-dish pairs for association rules
recommend_data = []
for idx, row in df.iterrows():
    order_id = row['Order ID']
    items = row['parsed_items']
    if items:
        # Get unique dishes (not quantities)
        dishes = list(set([item['dish'] for item in items]))
        recommend_data.append({
            'order_id': order_id,
            'items': ', '.join(dishes)
        })

recommend_df = pd.DataFrame(recommend_data)

print(f"Recommendation data shape: {recommend_df.shape}")
print(f"Sample items: {recommend_df['items'].iloc[0]}")

# Save
recommend_output = 'uploads/original_data/dish_recommendation.csv'
recommend_df.to_csv(recommend_output, index=False, quoting=csv.QUOTE_NONNUMERIC)
print(f"✓ Saved to {recommend_output}")

# ============================================================================
# 4. PREP TIME PREDICTION DATA
# ============================================================================
print("\n" + "="*70)
print("CREATING PREP TIME PREDICTION DATA")
print("="*70)

# Filter for orders with prep time data
prep_df = df.copy()

# Keep only orders with KPT duration
prep_df = prep_df.dropna(subset=['KPT duration (minutes)'])
print(f"Orders with prep time data: {len(prep_df)}")

# Basic preprocessing (similar to notebook)
prep_df['timestamp'] = pd.to_datetime(prep_df['Order Placed At'], format='%I:%M %p, %B %d %Y')
prep_df['order_date'] = prep_df['timestamp'].dt.date

# Process distance
prep_df["Distance_km"] = prep_df["Distance"].replace({"<1km": "0.5km"})
prep_df["Distance_km"] = prep_df["Distance_km"].str.replace("km", "", regex=False).astype(float)

# Handle missing values
prep_df["Discount construct"] = prep_df["Discount construct"].fillna("No discount")
prep_df["Rider wait time (minutes)"] = prep_df["Rider wait time (minutes)"].fillna(prep_df["Rider wait time (minutes)"].median())

# Select relevant columns for prep time prediction (kitchen-focused only)
prep_columns = [
    'timestamp', 'order_date', 'KPT duration (minutes)',
    'Order Status', 'Items in order'  # Only kitchen-relevant features
]

prep_df_final = prep_df[prep_columns].copy()

print(f"Prep time prediction data shape: {prep_df_final.shape}")
print(f"Date range: {prep_df_final['timestamp'].min()} to {prep_df_final['timestamp'].max()}")
print(f"Prep time range: {prep_df_final['KPT duration (minutes)'].min():.1f} - {prep_df_final['KPT duration (minutes)'].max():.1f} minutes")
print(f"Features included: {prep_columns[3:]}")  # Show non-temporal features

# Save
prep_output = 'uploads/original_data/prep_time_prediction.csv'
prep_df_final.to_csv(prep_output, index=False, quoting=1)  # QUOTE_ALL to ensure consistent quoting
print(f"✓ Saved to {prep_output}")

# ============================================================================
# 5. PROMOTION EFFECTIVENESS DATA
# ============================================================================
print("\n" + "="*70)
print("CREATING PROMOTION EFFECTIVENESS DATA")
print("="*70)

# Load and process promotion effectiveness data
promo_source = '../data/data_4.csv'
promo_output = 'uploads/original_data/promotion_effectiveness.csv'

if os.path.exists(promo_source):
    promo_df = pd.read_csv(promo_source)

    # Create timestamp column from separate date/time columns
    # Assume year 2024 for all records
    promo_df['year'] = 2024
    # Rename day_of_month to day for pandas to_datetime
    promo_df['day'] = promo_df['day_of_month']
    promo_df['timestamp'] = pd.to_datetime(promo_df[['year', 'month', 'day', 'hour']])
    promo_df = promo_df.drop(['year', 'day'], axis=1)

    # Save with proper quoting
    promo_df.to_csv(promo_output, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"✓ Processed and saved promotion data to {promo_output}")

    # Read and show info
    promo_df = pd.read_csv(promo_output)
    print(f"Promotion effectiveness data shape: {promo_df.shape}")
    print(f"Columns: {promo_df.columns.tolist()}")
else:
    print(f"⚠️  Warning: {promo_source} not found, skipping promotion data")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Original data: {df.shape[0]} rows")
print(f"Dish prediction: {dish_df_final.shape[0]} hourly records, {len(top_dishes)} dishes")
print(f"Demand prediction: {demand_df.shape[0]} hourly records")
print(f"Dish recommendation: {recommend_df.shape[0]} orders")
print(f"Prep time prediction: {prep_df_final.shape[0]} orders")
if os.path.exists(promo_output):
    promo_df = pd.read_csv(promo_output)
    print(f"Promotion effectiveness: {promo_df.shape[0]} records")
print("\nAll files saved to uploads/original_data/")
