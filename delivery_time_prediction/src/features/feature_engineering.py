"""
Feature Engineering for Delivery Time Prediction
================================================

This module creates features for delivery time prediction including:
- Temporal features (hour, day, cyclical encoding)
- Lag features (historical delivery times)
- Rolling window features (moving averages)
- Restaurant-specific features
- Distance-based features
- Pollution/weather features

Author: Saugat Shakya
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Create engineered features for delivery time prediction.
    
    Features include:
        - Temporal (cyclical hour/day encoding)
        - Historical (lags, rolling windows)
        - Restaurant patterns
        - Distance bins
        - External (pollution if available)
    """
    
    def __init__(
        self,
        lag_periods: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None
    ):
        """
        Initialize FeatureEngineer.
        
        Args:
            lag_periods: List of lag periods (e.g., [1, 2, 3, 6, 12, 24])
            rolling_windows: List of rolling window sizes (e.g., [3, 6, 12, 24])
        """
        from ..config import LAG_PERIODS, ROLLING_WINDOWS
        
        self.lag_periods = lag_periods or LAG_PERIODS
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS
        
        logger.info(f"FeatureEngineer initialized")
        logger.info(f"  Lag periods: {self.lag_periods}")
        logger.info(f"  Rolling windows: {self.rolling_windows}")
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features with cyclical encoding.
        
        Args:
            df: DataFrame with temporal columns
            
        Returns:
            DataFrame with additional temporal features
        """
        logger.info("Creating temporal features...")
        
        df = df.copy()
        
        # Cyclical encoding for hour (24-hour cycle)
        if 'order_hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['order_hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['order_hour'] / 24)
        
        # Cyclical encoding for day of week (7-day cycle)
        if 'order_day' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['order_day'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['order_day'] / 7)
        
        # Cyclical encoding for month (12-month cycle)
        if 'order_month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['order_month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['order_month'] / 12)
        
        # Cyclical encoding for day of month (31-day cycle)
        if 'order_day_of_month' in df.columns:
            df['day_of_month_sin'] = np.sin(2 * np.pi * df['order_day_of_month'] / 31)
            df['day_of_month_cos'] = np.cos(2 * np.pi * df['order_day_of_month'] / 31)
        
        # Peak hour indicator
        if 'order_hour' in df.columns:
            df['is_peak_hour'] = df['order_hour'].isin([11, 12, 13, 19, 20, 21]).astype(int)
            df['is_lunch_hour'] = df['order_hour'].isin([11, 12, 13, 14]).astype(int)
            df['is_dinner_hour'] = df['order_hour'].isin([19, 20, 21, 22]).astype(int)
        
        logger.info(f"Created temporal features. New shape: {df.shape}")
        return df
    
    def create_lag_features(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'Total_time_taken',
        group_col: Optional[str] = 'Restaurant_ID_encoded'
    ) -> pd.DataFrame:
        """
        Create lag features for delivery time.
        
        Args:
            df: DataFrame sorted by time
            target_col: Target variable to create lags for
            group_col: Column to group by (e.g., restaurant)
            
        Returns:
            DataFrame with lag features
        """
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found, skipping lag features")
            return df
        
        logger.info(f"Creating lag features for '{target_col}'...")
        
        df = df.copy()
        
        # Sort by time
        if 'Order Placed At' in df.columns:
            df = df.sort_values('Order Placed At')
        
        # Create lags
        for lag in self.lag_periods:
            if group_col and group_col in df.columns:
                # Group-wise lags (e.g., per restaurant)
                df[f'{target_col}_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
            else:
                # Global lags
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        logger.info(f"Created {len(self.lag_periods)} lag features")
        return df
    
    def create_rolling_features(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'Total_time_taken',
        group_col: Optional[str] = 'Restaurant_ID_encoded'
    ) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: DataFrame sorted by time
            target_col: Target variable to create rolling features for
            group_col: Column to group by (e.g., restaurant)
            
        Returns:
            DataFrame with rolling window features
        """
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found, skipping rolling features")
            return df
        
        logger.info(f"Creating rolling window features for '{target_col}'...")
        
        df = df.copy()
        
        # Sort by time
        if 'Order Placed At' in df.columns:
            df = df.sort_values('Order Placed At')
        
        # Create rolling features
        for window in self.rolling_windows:
            if group_col and group_col in df.columns:
                # Group-wise rolling windows
                df[f'{target_col}_rolling_mean_{window}'] = (
                    df.groupby(group_col)[target_col]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                df[f'{target_col}_rolling_std_{window}'] = (
                    df.groupby(group_col)[target_col]
                    .rolling(window=window, min_periods=1)
                    .std()
                    .reset_index(level=0, drop=True)
                )
            else:
                # Global rolling windows
                df[f'{target_col}_rolling_mean_{window}'] = (
                    df[target_col].rolling(window=window, min_periods=1).mean()
                )
                df[f'{target_col}_rolling_std_{window}'] = (
                    df[target_col].rolling(window=window, min_periods=1).std()
                )
        
        logger.info(f"Created {len(self.rolling_windows) * 2} rolling features")
        return df
    
    def create_restaurant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create restaurant-specific features.
        
        Args:
            df: DataFrame with restaurant info
            
        Returns:
            DataFrame with restaurant features
        """
        logger.info("Creating restaurant features...")
        
        df = df.copy()
        
        if 'Restaurant_ID_encoded' in df.columns and 'Total_time_taken' in df.columns:
            # Average delivery time per restaurant
            restaurant_avg = df.groupby('Restaurant_ID_encoded')['Total_time_taken'].mean()
            df['restaurant_avg_delivery_time'] = df['Restaurant_ID_encoded'].map(restaurant_avg)
            
            # Number of orders per restaurant
            restaurant_counts = df.groupby('Restaurant_ID_encoded').size()
            df['restaurant_order_count'] = df['Restaurant_ID_encoded'].map(restaurant_counts)
            
            logger.info("Created restaurant-specific features")
        
        return df
    
    def create_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create distance-based features.
        
        Args:
            df: DataFrame with Distance column
            
        Returns:
            DataFrame with distance features
        """
        if 'Distance' not in df.columns:
            logger.warning("Distance column not found, skipping distance features")
            return df
        
        logger.info("Creating distance features...")
        
        df = df.copy()
        
        # Distance bins
        df['distance_bin'] = pd.cut(
            df['Distance'],
            bins=[0, 2, 4, 6, 8, 100],
            labels=['very_close', 'close', 'medium', 'far', 'very_far']
        ).astype(str)
        
        # One-hot encode distance bins
        distance_dummies = pd.get_dummies(df['distance_bin'], prefix='dist', dtype=int)
        df = pd.concat([df, distance_dummies], axis=1)
        df = df.drop(columns=['distance_bin'])
        
        # Distance squared (non-linear relationship)
        df['distance_squared'] = df['Distance'] ** 2
        
        logger.info("Created distance features")
        return df
    
    def create_pollution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create pollution-derived features if pollution data is present.
        
        Args:
            df: DataFrame with pollution columns
            
        Returns:
            DataFrame with pollution features
        """
        # Check if pollution features exist
        pollution_cols = ['aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        existing_pollution_cols = [col for col in pollution_cols if col in df.columns]
        
        if not existing_pollution_cols:
            logger.info("No pollution data found, skipping pollution features")
            return df
        
        logger.info(f"Creating pollution features from: {existing_pollution_cols}")
        
        df = df.copy()
        
        # Fill NaNs with median for pollution features
        for col in existing_pollution_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Create AQI bins if AQI exists
        if 'aqi' in df.columns:
            df['aqi_high'] = (df['aqi'] > 3).astype(int)  # AQI > 3 means poor air quality
        
        logger.info("Created pollution features")
        return df
    
    def engineer_features(self, df: pd.DataFrame, target_col: str = 'Total_time_taken') -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Preprocessed DataFrame
            target_col: Target variable name
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("="*80)
        logger.info("Starting feature engineering pipeline")
        logger.info("="*80)
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Create distance features
        df = self.create_distance_features(df)
        
        # Create pollution features
        df = self.create_pollution_features(df)
        
        # Create restaurant features
        df = self.create_restaurant_features(df)
        
        # Create lag and rolling features (only if target exists)
        if target_col in df.columns:
            df = self.create_lag_features(df, target_col=target_col)
            df = self.create_rolling_features(df, target_col=target_col)
        
        logger.info("="*80)
        logger.info("Feature engineering complete!")
        logger.info(f"Final shape: {df.shape}")
        logger.info(f"Total features: {len(df.columns)}")
        logger.info("="*80)
        
        return df
