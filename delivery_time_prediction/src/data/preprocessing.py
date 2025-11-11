"""
Data Preprocessing for Delivery Time Prediction
===============================================

This module handles data cleaning, transformation, and preparation.

Author: Saugat Shakya
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
import pytz
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Clean and preprocess delivery time data.
    
    Steps:
        1. Handle missing values
        2. Remove unnecessary columns
        3. Filter to delivered orders only
        4. Encode categorical variables
        5. Parse timestamps
        6. Merge with external data
    """
    
    def __init__(self):
        """Initialize DataPreprocessor."""
        logger.info("DataPreprocessor initialized")
        
        # Columns to drop
        self.drop_columns = [
            'Instructions',
            'Rating',
            'Review',
            'Cancellation / Rejection reason',
            'Restaurant compensation (Cancellation)',
            'Restaurant penalty (Rejection)',
            'Customer complaint tag',
            'Restaurant name',
            'Delivery',
            'Customer ID',
            'City'
        ]
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw delivery data.
        
        Args:
            df: Raw delivery DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting data cleaning... Initial shape: {df.shape}")
        
        # Make a copy
        df = df.copy()
        
        # Drop unnecessary columns
        cols_to_drop = [col for col in self.drop_columns if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped {len(cols_to_drop)} columns")
        
        # Filter to delivered orders only
        if 'Order Status' in df.columns:
            initial_count = len(df)
            df = df[df['Order Status'] == 'Delivered'].copy()
            df = df.drop(columns=['Order Status'])
            logger.info(f"Filtered to delivered orders: {len(df):,} (removed {initial_count - len(df):,})")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        if initial_count > len(df):
            logger.info(f"Removed {initial_count - len(df):,} duplicates")
        
        logger.info(f"Data cleaning complete. Final shape: {df.shape}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info("Handling missing values...")
        
        # Fill Discount construct with 'No Discount / NA'
        if 'Discount construct' in df.columns:
            df['Discount construct'] = df['Discount construct'].fillna('No Discount / NA')
        
        # Handle KPT duration
        if 'KPT duration (minutes)' in df.columns:
            kpt_median = df['KPT duration (minutes)'].median()
            df['KPT_missing'] = df['KPT duration (minutes)'].isna().astype(int)
            df['KPT duration (minutes)'] = df['KPT duration (minutes)'].fillna(kpt_median)
            logger.info(f"Filled KPT duration NaNs with median: {kpt_median:.2f}")
        
        # Handle Rider wait time
        if 'Rider wait time (minutes)' in df.columns:
            wait_median = df['Rider wait time (minutes)'].median()
            df['RiderWait_missing'] = df['Rider wait time (minutes)'].isna().astype(int)
            df['Rider wait time (minutes)'] = df['Rider wait time (minutes)'].fillna(wait_median)
            logger.info(f"Filled Rider wait time NaNs with median: {wait_median:.2f}")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: DataFrame with categorical columns
            
        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features...")
        
        # One-hot encode Subzone
        if 'Subzone' in df.columns:
            subzone_dummies = pd.get_dummies(df['Subzone'], prefix='Subzone', dtype=int)
            df = pd.concat([df.drop(columns=['Subzone']), subzone_dummies], axis=1)
            logger.info(f"One-hot encoded Subzone: {len(subzone_dummies.columns)} categories")
        
        # One-hot encode Order Ready Marked
        if 'Order Ready Marked' in df.columns:
            ordready_dummies = pd.get_dummies(
                df['Order Ready Marked'],
                prefix='OrderReady',
                dtype=int
            )
            df = pd.concat([df.drop(columns=['Order Ready Marked']), ordready_dummies], axis=1)
            logger.info(f"One-hot encoded Order Ready Marked: {len(ordready_dummies.columns)} categories")
        
        # Label encode Restaurant ID (if many unique values)
        if 'Restaurant ID' in df.columns:
            df['Restaurant_ID_encoded'] = df['Restaurant ID'].factorize()[0]
            df = df.drop(columns=['Restaurant ID'])
            logger.info("Label encoded Restaurant ID")
        
        return df
    
    def parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse timestamp columns.
        
        Args:
            df: DataFrame with timestamp columns
            
        Returns:
            DataFrame with parsed timestamps
        """
        logger.info("Parsing timestamps...")
        
        # Parse Order Placed At
        if 'Order Placed At' in df.columns:
            df['Order Placed At'] = pd.to_datetime(df['Order Placed At'], errors='coerce')
            
            # Extract temporal features
            df['order_hour'] = df['Order Placed At'].dt.hour
            df['order_day'] = df['Order Placed At'].dt.dayofweek
            df['order_month'] = df['Order Placed At'].dt.month
            df['order_day_of_month'] = df['Order Placed At'].dt.day
            df['is_weekend'] = df['order_day'].isin([5, 6]).astype(int)
            
            logger.info("Extracted temporal features from Order Placed At")
        
        return df
    
    def process_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process distance column.
        
        Args:
            df: DataFrame with Distance column
            
        Returns:
            DataFrame with processed Distance
        """
        if 'Distance' in df.columns:
            logger.info("Processing Distance column...")
            
            # Convert to string and clean
            df['Distance'] = df['Distance'].astype(str).str.strip()
            df['Distance'] = df['Distance'].str.replace('<', '', regex=False)
            df['Distance'] = df['Distance'].str.split('km').str[0]
            
            # Convert to float
            df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
            
            # Fill any remaining NaNs with median
            median_dist = df['Distance'].median()
            df['Distance'] = df['Distance'].fillna(median_dist)
            
            logger.info(f"Distance processed. Median: {median_dist:.2f} km")
        
        return df
    
    def merge_pollution_data(
        self, 
        df: pd.DataFrame, 
        pollution_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Merge pollution data with orders.
        
        Args:
            df: Main delivery DataFrame
            pollution_df: Pollution DataFrame (optional)
            
        Returns:
            DataFrame with pollution features merged
        """
        if pollution_df is None:
            logger.warning("No pollution data provided, skipping merge")
            return df
        
        logger.info("Merging pollution data...")
        
        # Ensure Order Placed At is datetime
        df['Order Placed At'] = pd.to_datetime(df['Order Placed At'], errors='coerce')
        
        # Convert to India time, then to UTC, then floor to hour
        india_tz = pytz.timezone('Asia/Kolkata')
        order_local = df['Order Placed At'].dt.tz_localize(india_tz, nonexistent='NaT', ambiguous='NaT')
        order_utc = order_local.dt.tz_convert('UTC')
        order_hour_utc = order_utc.dt.floor('H')
        df['order_hour_utc'] = order_hour_utc.dt.tz_localize(None)
        
        # Ensure pollution timestamp is datetime
        pollution_df['pollution_time_utc'] = pd.to_datetime(
            pollution_df['pollution_time_utc'],
            errors='coerce'
        )
        
        # Merge on matching hour
        df = df.merge(
            pollution_df,
            left_on='order_hour_utc',
            right_on='pollution_time_utc',
            how='left'
        )
        
        # Drop redundant columns
        df = df.drop(columns=['pollution_time_utc'], errors='ignore')
        
        logger.info(f"Pollution data merged. Shape: {df.shape}")
        return df
    
    def preprocess(
        self, 
        df: pd.DataFrame, 
        pollution_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw delivery DataFrame
            pollution_df: Optional pollution DataFrame
            
        Returns:
            Fully preprocessed DataFrame
        """
        logger.info("="*80)
        logger.info("Starting complete preprocessing pipeline")
        logger.info("="*80)
        
        # Clean data
        df = self.clean_data(df)
        
        # Parse timestamps
        df = self.parse_timestamps(df)
        
        # Process distance
        df = self.process_distance(df)
        
        # Merge pollution data
        if pollution_df is not None:
            df = self.merge_pollution_data(df, pollution_df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Drop Discount construct if still present
        df = df.drop(columns=['Discount construct'], errors='ignore')
        df = df.drop(columns=['Items in order'], errors='ignore')
        
        logger.info("="*80)
        logger.info("Preprocessing complete!")
        logger.info(f"Final shape: {df.shape}")
        logger.info(f"Final columns ({len(df.columns)}): {df.columns.tolist()}")
        logger.info("="*80)
        
        return df
