"""
Data Loading Module
Loads raw data from CSV files and performs initial parsing
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple


class DataLoader:
    """Load and parse restaurant order data"""
    
    def __init__(self, data_dir: str = '../data'):
        self.data_dir = Path(data_dir)
        
    def load_orders(self, filepath: str = None) -> pd.DataFrame:
        """
        Load raw order data
        
        Args:
            filepath: Path to orders CSV (default: ../data/data.csv)
            
        Returns:
            DataFrame with raw order data
        """
        if filepath is None:
            filepath = self.data_dir / 'data.csv'
        
        print(f"Loading orders from: {filepath}")
        df = pd.read_csv(filepath)
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(df['Order Placed At'], 
                                       format='%I:%M %p, %B %d %Y')
        
        print(f"✓ Loaded {len(df):,} orders")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df
    
    def load_weather(self, filepath: str = None) -> pd.DataFrame:
        """
        Load weather data
        
        Args:
            filepath: Path to weather CSV
            
        Returns:
            DataFrame with hourly weather data
        """
        if filepath is None:
            filepath = self.data_dir / 'hourly_orders_weather.csv'
        
        print(f"Loading weather from: {filepath}")
        df = pd.read_csv(filepath)
        
        # Parse datetime
        df['order_hour'] = pd.to_datetime(df['order_hour'])
        
        print(f"✓ Loaded {len(df):,} hourly weather records")
        
        return df
    
    def load_pollution(self, filepath: str = None) -> pd.DataFrame:
        """
        Load pollution data
        
        Args:
            filepath: Path to pollution CSV
            
        Returns:
            DataFrame with hourly pollution data
        """
        if filepath is None:
            filepath = self.data_dir / 'pollution.csv'
        
        print(f"Loading pollution from: {filepath}")
        df = pd.read_csv(filepath)
        
        # Parse datetime
        df['pollution_time_utc'] = pd.to_datetime(df['pollution_time_utc'])
        df['pollution_hour'] = df['pollution_time_utc'].dt.floor('H')
        
        print(f"✓ Loaded {len(df):,} hourly pollution records")
        
        return df
    
    def load_events(self, filepath: str = None) -> pd.DataFrame:
        """
        Load events/holidays data
        
        Args:
            filepath: Path to events CSV
            
        Returns:
            DataFrame with daily events
        """
        if filepath is None:
            filepath = self.data_dir / 'events.csv'
        
        print(f"Loading events from: {filepath}")
        df = pd.read_csv(filepath)
        
        # Parse date
        df['date'] = pd.to_datetime(df['date'])
        
        # Clean events (remove "No significant event")
        df['has_event'] = ~df['event'].str.contains('No significant event', 
                                                     case=False, na=False)
        
        print(f"✓ Loaded {len(df):,} daily event records")
        print(f"  Significant events: {df['has_event'].sum()}")
        
        return df
    
    def parse_dishes(self, items_str: str) -> List[Dict]:
        """
        Parse dishes from order items string
        
        Args:
            items_str: String like "2 x Pizza, 1 x Garlic Bread"
            
        Returns:
            List of dicts with dish name and quantity
        """
        if pd.isna(items_str):
            return []
        
        dishes = []
        parts = str(items_str).split(',')
        
        for part in parts:
            # Match pattern: "2 x Pizza" or "1 x Garlic Bread"
            match = re.search(r'(\d+)\s*x\s*(.+)', part.strip())
            if match:
                qty = int(match.group(1))
                dish = match.group(2).strip()
                dishes.append({'dish': dish, 'quantity': qty})
        
        return dishes
    
    def extract_dishes_from_orders(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract individual dish records from orders
        
        Args:
            orders_df: DataFrame with order data
            
        Returns:
            DataFrame with individual dish orders
        """
        print("\nExtracting dishes from orders...")
        
        all_dishes = []
        
        for idx, row in orders_df.iterrows():
            dishes = self.parse_dishes(row['Items in order'])
            
            for dish_info in dishes:
                all_dishes.append({
                    'datetime': row['datetime'],
                    'order_id': row['Order ID'],
                    'restaurant_id': row['Restaurant ID'],
                    'dish': dish_info['dish'],
                    'quantity': dish_info['quantity'],
                    'order_status': row['Order Status']
                })
        
        dish_df = pd.DataFrame(all_dishes)
        
        print(f"✓ Extracted {len(dish_df):,} dish records")
        print(f"  Unique dishes: {dish_df['dish'].nunique():,}")
        
        # Filter to delivered only
        delivered_df = dish_df[dish_df['order_status'] == 'Delivered'].copy()
        print(f"  Delivered dishes: {len(delivered_df):,}")
        
        return delivered_df
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all data sources
        
        Returns:
            Tuple of (orders, weather, pollution, events) DataFrames
        """
        print("="*80)
        print("LOADING ALL DATA SOURCES")
        print("="*80)
        
        orders = self.load_orders()
        weather = self.load_weather()
        pollution = self.load_pollution()
        events = self.load_events()
        
        print("\n✓ All data loaded successfully")
        
        return orders, weather, pollution, events


if __name__ == "__main__":
    # Test the loader
    loader = DataLoader()
    orders, weather, pollution, events = loader.load_all_data()
    
    # Extract dishes
    dishes = loader.extract_dishes_from_orders(orders)
    print(f"\nTop 5 dishes:")
    print(dishes['dish'].value_counts().head())
