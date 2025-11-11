"""
Script to process the original data.csv and create smaller CSV files for each model
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

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
dish_df_final.to_csv(dish_output, index=False)
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
demand_df.to_csv(demand_output, index=False)
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
recommend_df.to_csv(recommend_output, index=False)
print(f"✓ Saved to {recommend_output}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Original data: {df.shape[0]} rows")
print(f"Dish prediction: {dish_df_final.shape[0]} hourly records, {len(top_dishes)} dishes")
print(f"Demand prediction: {demand_df.shape[0]} hourly records")
print(f"Dish recommendation: {recommend_df.shape[0]} orders")
print("\nAll files saved to uploads/original_data/")
