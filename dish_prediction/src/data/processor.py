"""
Data Processing Module
Cleans, aggregates, and prepares data for feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional


class DataProcessor:
    """Process and aggregate restaurant data"""
    
    def __init__(self, top_n_dishes: int = 30):
        self.top_n_dishes = top_n_dishes
        self.top_dishes = None
        
    def get_top_dishes(self, dish_df: pd.DataFrame, n: int = None) -> List[str]:
        """
        Get top N dishes by quantity
        
        Args:
            dish_df: DataFrame with dish records
            n: Number of top dishes (default: self.top_n_dishes)
            
        Returns:
            List of top dish names
        """
        if n is None:
            n = self.top_n_dishes
        
        top = dish_df.groupby('dish')['quantity'].sum().nlargest(n)
        self.top_dishes = top.index.tolist()
        
        print(f"\nTop {n} dishes selected:")
        for i, (dish, qty) in enumerate(top.items(), 1):
            print(f"  {i:2d}. {dish[:50]:50s} | {qty:5,} units")
        
        return self.top_dishes
    
    def create_hourly_pivot(self, dish_df: pd.DataFrame, 
                           top_dishes: List[str] = None) -> pd.DataFrame:
        """
        Create hourly pivot table with dishes as columns
        
        Args:
            dish_df: DataFrame with dish records
            top_dishes: List of dishes to include (default: self.top_dishes)
            
        Returns:
            DataFrame with datetime index and dish quantities as columns
        """
        if top_dishes is None:
            top_dishes = self.top_dishes
        
        # Filter to top dishes
        df_filtered = dish_df[dish_df['dish'].isin(top_dishes)].copy()
        
        # Floor to hour
        df_filtered['hour'] = df_filtered['datetime'].dt.floor('H')
        
        # Aggregate by hour and dish
        hourly = df_filtered.groupby(['hour', 'dish'])['quantity'].sum().reset_index()
        
        # Pivot to wide format
        pivot = hourly.pivot(index='hour', columns='dish', values='quantity')
        pivot = pivot.fillna(0)  # No orders = 0
        
        # Reset index to make hour a column
        pivot = pivot.reset_index()
        pivot.columns.name = None
        
        print(f"\n✓ Created hourly pivot table:")
        print(f"  Shape: {pivot.shape[0]:,} hours × {pivot.shape[1]:,} columns")
        print(f"  Date range: {pivot['hour'].min()} to {pivot['hour'].max()}")
        
        return pivot
    
    def add_temporal_features(self, df: pd.DataFrame, 
                             datetime_col: str = 'hour') -> pd.DataFrame:
        """
        Add basic temporal features
        
        Args:
            df: DataFrame with datetime column
            datetime_col: Name of datetime column
            
        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # Basic temporal features
        df['hour_of_day'] = df[datetime_col].dt.hour
        df['day_of_week'] = df[datetime_col].dt.dayofweek
        df['day_of_month'] = df[datetime_col].dt.day
        df['week_of_year'] = df[datetime_col].dt.isocalendar().week
        df['month'] = df[datetime_col].dt.month
        df['date'] = df[datetime_col].dt.date
        
        # Binary flags
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # Peak hours from EDA
        df['is_peak_hour'] = df['hour_of_day'].isin([19, 20, 21]).astype(int)
        df['is_lunch_rush'] = df['hour_of_day'].isin([12, 13, 14]).astype(int)
        df['is_dinner_rush'] = df['hour_of_day'].isin([19, 20, 21, 22]).astype(int)
        df['is_late_night'] = df['hour_of_day'].isin([0, 1, 2, 3]).astype(int)
        
        # Meal period (categorical)
        def get_meal_period(hour):
            if hour in [0, 1, 2, 3]:
                return 'late_night'
            elif hour in [4, 5, 6, 7, 8, 9, 10]:
                return 'breakfast'
            elif hour in [11, 12, 13, 14, 15]:
                return 'lunch'
            elif hour in [16, 17, 18]:
                return 'evening'
            else:  # 19-23
                return 'dinner'
        
        df['meal_period'] = df['hour_of_day'].apply(get_meal_period)
        
        # Cyclical encoding (for hour and day of week)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        print(f"\n✓ Added temporal features:")
        print(f"  Basic: hour_of_day, day_of_week, month, week_of_year")
        print(f"  Flags: is_weekend, is_peak_hour, meal_period")
        print(f"  Cyclical: hour_sin/cos, day_sin/cos")
        
        return df
    
    def merge_external_data(self, df: pd.DataFrame,
                           weather_df: pd.DataFrame,
                           pollution_df: pd.DataFrame,
                           events_df: pd.DataFrame,
                           datetime_col: str = 'hour') -> pd.DataFrame:
        """
        Merge weather, pollution, and event data
        
        Args:
            df: Main DataFrame with hourly data
            weather_df: Weather DataFrame
            pollution_df: Pollution DataFrame
            events_df: Events DataFrame
            datetime_col: Name of datetime column in main df
            
        Returns:
            DataFrame with merged external data
        """
        df = df.copy()
        
        print("\nMerging external data...")
        
        # Merge weather (hourly)
        weather_cols = ['order_hour', 'env_temp', 'env_rhum', 'env_precip', 
                       'env_wspd', 'env_condition']
        df = df.merge(
            weather_df[weather_cols],
            left_on=datetime_col,
            right_on='order_hour',
            how='left'
        )
        df = df.drop('order_hour', axis=1)
        print(f"  ✓ Weather: {weather_df.shape[1]} columns merged")
        print(f"    Missing: {df['env_temp'].isnull().sum()} / {len(df)} hours")
        
        # Merge pollution (hourly)
        pollution_cols = ['pollution_hour', 'aqi', 'pm2_5', 'pm10', 'no2', 'o3', 'co']
        df = df.merge(
            pollution_df[pollution_cols],
            left_on=datetime_col,
            right_on='pollution_hour',
            how='left'
        )
        df = df.drop('pollution_hour', axis=1)
        print(f"  ✓ Pollution: {len(pollution_cols)-1} columns merged")
        print(f"    Missing: {df['aqi'].isnull().sum()} / {len(df)} hours")
        
        # Merge events (daily)
        # Add date column if not exists
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df[datetime_col]).dt.date
        
        df['date'] = pd.to_datetime(df['date'])
        events_df = events_df.copy()
        events_df['date'] = pd.to_datetime(events_df['date'])
        
        df = df.merge(
            events_df[['date', 'event', 'holiday', 'has_event']],
            on='date',
            how='left'
        )
        df['has_event'] = df['has_event'].fillna(False)
        df['holiday'] = df['holiday'].fillna(False)
        print(f"  ✓ Events: 3 columns merged")
        print(f"    Event days: {df['has_event'].sum()} / {df['date'].nunique()} unique days")
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, 
                           filepath: str = 'data/processed/hourly_data_with_features.csv'):
        """Save processed data to CSV"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"\n✓ Saved processed data to: {filepath}")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")


if __name__ == "__main__":
    # Test the processor
    from loader import DataLoader
    
    loader = DataLoader()
    orders, weather, pollution, events = loader.load_all_data()
    
    # Extract and process dishes
    dishes = loader.extract_dishes_from_orders(orders)
    
    processor = DataProcessor(top_n_dishes=30)
    top_dishes = processor.get_top_dishes(dishes)
    
    # Create hourly pivot
    hourly_pivot = processor.create_hourly_pivot(dishes, top_dishes)
    
    # Add temporal features
    hourly_pivot = processor.add_temporal_features(hourly_pivot, 'hour')
    
    # Merge external data
    full_data = processor.merge_external_data(
        hourly_pivot, weather, pollution, events, 'hour'
    )
    
    # Save
    processor.save_processed_data(full_data)
    
    print(f"\n✓ Processing complete!")
    print(f"Final shape: {full_data.shape}")
