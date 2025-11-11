"""
Feature Engineering for Hourly Order Volume Prediction
======================================================

This module creates time-series and aggregation features for predicting
hourly order volumes.

Key Features:
1. Temporal features (hour, day_of_week, weekend, cyclical encoding)
2. Lag features (1hr, 24hr lags)
3. Rolling window features (3hr mean/max)
4. Holiday features
5. Aggregation to hourly level

Author: Saugat Shakya
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging
import holidays

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for hourly order volume prediction.
    
    Handles temporal aggregation, lag features, rolling windows, and holiday features.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer with preprocessed data.
        
        Args:
            df: Preprocessed order data with order_date, order_hour, order_count
        """
        self.df = df.copy()
        self.hourly_df = None
        self.daily_df = None
        
    def aggregate_to_hourly(self) -> pd.DataFrame:
        """
        Aggregate order data to hourly level.
        
        Returns:
            DataFrame with one row per (date, hour) combination
        """
        logger.info("Aggregating to hourly level...")
        
        # Aggregate to hourly
        self.hourly_df = self.df.groupby(['order_date', 'order_hour']).agg({
            'order_count': 'sum'  # Total orders in that hour
        }).reset_index()
        
        # Rename target column
        self.hourly_df.rename(columns={'order_count': 'orders_per_hour'}, inplace=True)
        
        # Sort by date and hour
        self.hourly_df = self.hourly_df.sort_values(by=['order_date', 'order_hour'])
        
        logger.info(f"Hourly aggregation complete: {len(self.hourly_df)} hour-blocks")
        logger.info(f"Average orders per hour: {self.hourly_df['orders_per_hour'].mean():.2f}")
        
        return self.hourly_df
    
    def aggregate_to_daily(self) -> pd.DataFrame:
        """
        Aggregate order data to daily level.
        
        Returns:
            DataFrame with one row per date
        """
        logger.info("Aggregating to daily level...")
        
        self.daily_df = self.df.groupby('order_date').agg({
            'order_count': 'sum'
        }).reset_index()
        
        # Rename target column
        self.daily_df.rename(columns={'order_count': 'orders_per_day'}, inplace=True)
        
        logger.info(f"Daily aggregation complete: {len(self.daily_df)} days")
        logger.info(f"Average orders per day: {self.daily_df['orders_per_day'].mean():.2f}")
        
        return self.daily_df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from date/hour.
        
        Args:
            df: DataFrame with order_date and order_hour columns
            
        Returns:
            DataFrame with additional temporal features
        """
        logger.info("Creating temporal features...")
        
        df = df.copy()
        
        # Convert order_date to datetime if not already
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['order_date'].dt.dayofweek
        
        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for hour (0-23)
        df['sin_hour'] = np.sin(2 * np.pi * df['order_hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['order_hour'] / 24)
        
        # Cyclical encoding for day of week (0-6)
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month
        df['month'] = df['order_date'].dt.month
        
        # Day of month
        df['day_of_month'] = df['order_date'].dt.day
        
        logger.info(f"Created {8} temporal features")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'orders_per_hour') -> pd.DataFrame:
        """
        Create lag features for time series prediction.
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            
        Returns:
            DataFrame with lag features
        """
        logger.info("Creating lag features...")
        
        df = df.copy()
        
        # Ensure sorted by date and hour
        df = df.sort_values(by=['order_date', 'order_hour'])
        
        # 1-hour lag (previous hour)
        df['orders_lag_1hr'] = df[target_col].shift(1)
        
        # 24-hour lag (same hour yesterday)
        df['orders_lag_24hr'] = df[target_col].shift(24)
        
        # 48-hour lag (same hour 2 days ago)
        df['orders_lag_48hr'] = df[target_col].shift(48)
        
        # 168-hour lag (same hour last week)
        df['orders_lag_168hr'] = df[target_col].shift(168)
        
        logger.info(f"Created {4} lag features")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'orders_per_hour') -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            
        Returns:
            DataFrame with rolling features
        """
        logger.info("Creating rolling window features...")
        
        df = df.copy()
        
        # Ensure sorted
        df = df.sort_values(by=['order_date', 'order_hour'])
        
        # Shift by 1 to avoid data leakage (only look at past data)
        shifted_orders = df[target_col].shift(1)
        
        # 3-hour rolling statistics
        df['rolling_mean_3hr'] = shifted_orders.rolling(window=3, min_periods=1).mean()
        df['rolling_std_3hr'] = shifted_orders.rolling(window=3, min_periods=1).std()
        df['rolling_max_3hr'] = shifted_orders.rolling(window=3, min_periods=1).max()
        df['rolling_min_3hr'] = shifted_orders.rolling(window=3, min_periods=1).min()
        
        # 6-hour rolling statistics
        df['rolling_mean_6hr'] = shifted_orders.rolling(window=6, min_periods=1).mean()
        df['rolling_std_6hr'] = shifted_orders.rolling(window=6, min_periods=1).std()
        
        # 24-hour rolling statistics
        df['rolling_mean_24hr'] = shifted_orders.rolling(window=24, min_periods=1).mean()
        df['rolling_std_24hr'] = shifted_orders.rolling(window=24, min_periods=1).std()
        
        logger.info(f"Created {8} rolling window features")
        
        return df
    
    def create_holiday_features(self, df: pd.DataFrame, year: int = 2024) -> pd.DataFrame:
        """
        Create holiday-related features.
        
        Args:
            df: DataFrame with order_date column
            year: Year for holiday calendar
            
        Returns:
            DataFrame with holiday features
        """
        logger.info("Creating holiday features...")
        
        df = df.copy()
        
        # Get Indian holidays for the year
        ind_holidays = holidays.India(years=[year])
        
        # Is holiday
        df['is_holiday'] = df['order_date'].apply(lambda x: x in ind_holidays).astype(int)
        
        # Is post-holiday (day after holiday)
        df['is_post_holiday'] = df['order_date'].apply(
            lambda x: (x - pd.Timedelta(days=1)) in ind_holidays
        ).astype(int)
        
        # Is pre-holiday (day before holiday)
        df['is_pre_holiday'] = df['order_date'].apply(
            lambda x: (x + pd.Timedelta(days=1)) in ind_holidays
        ).astype(int)
        
        holiday_count = df['is_holiday'].sum()
        logger.info(f"Created holiday features. Found {holiday_count} holiday hours")
        
        return df
    
    def create_hourly_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features capturing typical hourly patterns.
        
        Args:
            df: DataFrame with order_hour and orders_per_hour
            
        Returns:
            DataFrame with hourly pattern features
        """
        logger.info("Creating hourly pattern features...")
        
        df = df.copy()
        
        # Average orders for each hour of day (across all days)
        hourly_avg = df.groupby('order_hour')['orders_per_hour'].mean().to_dict()
        df['hour_avg_orders'] = df['order_hour'].map(hourly_avg)
        
        # Average orders for each day of week
        dow_avg = df.groupby('day_of_week')['orders_per_hour'].mean().to_dict()
        df['dow_avg_orders'] = df['day_of_week'].map(dow_avg)
        
        # Average for each (hour, day_of_week) combination
        hour_dow_avg = df.groupby(['order_hour', 'day_of_week'])['orders_per_hour'].mean().to_dict()
        df['hour_dow_avg_orders'] = df.apply(
            lambda row: hour_dow_avg.get((row['order_hour'], row['day_of_week']), 0), 
            axis=1
        )
        
        logger.info(f"Created {3} hourly pattern features")
        
        return df
    
    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in features.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with filled values
        """
        logger.info("Filling missing values...")
        
        df = df.copy()
        
        # Fill NaN values from lag/rolling features with 0
        feature_cols = [col for col in df.columns if any(x in col for x in 
                       ['lag', 'rolling', '_avg_', 'std'])]
        
        for col in feature_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
        
        logger.info("Missing values filled")
        
        return df
    
    def run_feature_engineering(self) -> pd.DataFrame:
        """
        Run complete feature engineering pipeline.
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info("=" * 60)
        logger.info("Starting feature engineering pipeline")
        logger.info("=" * 60)
        
        # Aggregate to hourly level
        df = self.aggregate_to_hourly()
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Create lag features
        df = self.create_lag_features(df)
        
        # Create rolling features
        df = self.create_rolling_features(df)
        
        # Create holiday features
        df = self.create_holiday_features(df)
        
        # Create hourly pattern features
        df = self.create_hourly_patterns(df)
        
        # Fill missing values
        df = self.fill_missing_values(df)
        
        logger.info("=" * 60)
        logger.info("Feature engineering complete!")
        logger.info(f"Final shape: {df.shape}")
        logger.info(f"Total features: {len(df.columns)}")
        logger.info("=" * 60)
        
        return df


def get_feature_groups() -> dict:
    """
    Define feature groups for ablation study.
    
    Returns:
        Dictionary mapping group names to feature patterns
    """
    return {
        'temporal': ['day_of_week', 'is_weekend', 'sin_hour', 'cos_hour', 'sin_day', 
                    'cos_day', 'month', 'day_of_month', 'order_hour'],
        'lag': ['lag_1hr', 'lag_24hr', 'lag_48hr', 'lag_168hr'],
        'rolling': ['rolling_mean', 'rolling_std', 'rolling_max', 'rolling_min'],
        'holiday': ['is_holiday', 'is_post_holiday', 'is_pre_holiday'],
        'patterns': ['hour_avg_orders', 'dow_avg_orders', 'hour_dow_avg_orders'],
        'pollution': ['aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3'],
        'weather': ['temp', 'humidity', 'pressure', 'wind_speed', 'precipitation']
    }


if __name__ == "__main__":
    # Example usage
    from preprocessing import OrderDataPreprocessor
    
    # Paths
    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    PROCESSED_DATA = DATA_DIR / "processed" / "cleaned_orders.csv"
    FEATURES_OUTPUT = DATA_DIR / "processed" / "hourly_features.csv"
    
    # Load preprocessed data
    df = pd.read_csv(PROCESSED_DATA)
    
    # Run feature engineering
    engineer = FeatureEngineer(df)
    features_df = engineer.run_feature_engineering()
    
    # Save
    features_df.to_csv(FEATURES_OUTPUT, index=False)
    
    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING SUMMARY")
    print(f"{'='*60}")
    print(f"Total hour-blocks: {len(features_df):,}")
    print(f"Total features: {len(features_df.columns)}")
    print(f"Target (orders_per_hour) - Mean: {features_df['orders_per_hour'].mean():.2f}")
    print(f"Target (orders_per_hour) - Std: {features_df['orders_per_hour'].std():.2f}")
    print(f"Output: {FEATURES_OUTPUT}")
    print(f"{'='*60}\n")
