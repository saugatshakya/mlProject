"""
Data Preprocessing Pipeline for Demand Prediction
==================================================

This module handles:
1. Loading raw order data
2. Data cleaning and validation
3. Feature extraction from order data
4. Data type conversions
5. Filtering and quality checks

Author: Saugat Shakya
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderDataPreprocessor:
    """
    Preprocessor for food delivery order data.
    
    Handles cleaning, validation, and basic feature extraction from raw order data.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize preprocessor with data path.
        
        Args:
            data_path: Path to raw order data CSV file
        """
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load raw order data from CSV."""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        return self.df
    
    def parse_datetime(self) -> None:
        """Parse order datetime and extract date and time components."""
        logger.info("Parsing datetime fields...")
        
        # Parse datetime
        self.df["Order Placed At"] = pd.to_datetime(
            self.df["Order Placed At"], 
            format="%I:%M %p, %B %d %Y"
        )
        
        # Extract date and time
        self.df["order_date"] = self.df["Order Placed At"].dt.date
        self.df["order_time"] = self.df["Order Placed At"].dt.time
        
        # Drop original column
        self.df = self.df.drop(columns=["Order Placed At"])
        
        logger.info("Datetime parsing complete")
    
    def filter_delivered_orders(self) -> None:
        """Keep only successfully delivered orders."""
        original_count = len(self.df)
        
        if "Order Status" in self.df.columns:
            self.df = self.df[self.df["Order Status"] == "Delivered"]
            
        filtered_count = len(self.df)
        logger.info(f"Filtered to delivered orders: {filtered_count}/{original_count} "
                   f"({100*filtered_count/original_count:.1f}%)")
    
    def drop_irrelevant_columns(self) -> None:
        """Drop columns that don't contribute to order volume prediction."""
        columns_to_drop = [
            "Delivery",  # Only one value
            "City",  # Only one value
            "Order ID",  # Unique identifier
            "Order Status",  # Already filtered to "Delivered"
            "Instructions",  # Text data, not relevant
            "Gold discount",  # ~0.2% non-zero values
            "Brand pack discount",  # Very few non-zero values
            "Bill subtotal",  # Target-related, causes leakage
            "Packaging charges",  # Target-related
            "Total",  # Target-related
            "Rating",  # Not available at prediction time
            "Review",  # Not available at prediction time
            "Cancellation / Rejection reason",  # Not applicable
            "Restaurant compensation (Cancellation)",  # Not applicable
            "Restaurant penalty (Rejection)",  # Not applicable
            "KPT duration (minutes)",  # Not available at prediction time
            "Rider wait time (minutes)",  # Not available at prediction time
            "Order Ready Marked",  # Not available at prediction time
            "Customer complaint tag",  # Not available at prediction time
            "Customer ID"  # Privacy/not relevant
        ]
        
        # Drop columns that exist
        existing_to_drop = [col for col in columns_to_drop if col in self.df.columns]
        self.df = self.df.drop(columns=existing_to_drop, errors='ignore')
        
        logger.info(f"Dropped {len(existing_to_drop)} irrelevant columns")
        logger.info(f"Remaining columns: {list(self.df.columns)}")
    
    def extract_discount_percentage(self) -> None:
        """Extract discount percentage from discount construct column."""
        if "Discount construct" not in self.df.columns:
            logger.warning("'Discount construct' column not found, skipping...")
            return
        
        def extract_percent(x):
            if pd.isna(x):
                return 0
            match = re.search(r"(\d+)%", x)
            return float(match.group(1)) if match else 0
        
        self.df["discount_percent"] = self.df["Discount construct"].apply(extract_percent)
        self.df = self.df.drop(columns=["Discount construct"])
        
        logger.info("Extracted discount percentages")
    
    def convert_distance(self) -> None:
        """Convert distance from string format to numeric."""
        if "Distance" not in self.df.columns:
            logger.warning("'Distance' column not found, skipping...")
            return
        
        def parse_distance(x):
            if pd.isna(x):
                return None
            x = x.replace("km", "").strip()
            if x.startswith("<"):
                return 0.9  # For <1km
            try:
                return float(x)
            except:
                return None
        
        self.df["Distance"] = self.df["Distance"].apply(parse_distance)
        logger.info("Converted distance to numeric")
    
    def extract_temporal_features(self) -> None:
        """Extract hour from time and prepare temporal features."""
        # Convert order_date to datetime for further processing
        self.df["order_date"] = pd.to_datetime(self.df["order_date"], format="%Y-%m-%d")
        
        # Extract hour from order_time
        self.df["order_hour"] = pd.to_datetime(
            self.df["order_time"], 
            format="%H:%M:%S"
        ).dt.hour
        
        # Drop order_time after extraction
        self.df = self.df.drop(columns=["order_time"])
        
        logger.info("Extracted temporal features")
    
    def extract_order_count(self) -> None:
        """Extract order count from 'Items in order' column."""
        if "Items in order" not in self.df.columns:
            logger.warning("'Items in order' column not found, skipping...")
            return
        
        def extract_count(items):
            if pd.isna(items):
                return 1
            # Match the first number before 'x'
            match = re.match(r'(\d+)\s*x', items.strip())
            if match:
                return int(match.group(1))
            return 1  # Default to 1
        
        self.df['order_count'] = self.df['Items in order'].apply(extract_count)
        
        # Drop the original column
        self.df = self.df.drop(columns=['Items in order'])
        
        logger.info(f"Extracted order counts. Total items: {self.df['order_count'].sum()}")
    
    def run_preprocessing(self) -> pd.DataFrame:
        """
        Run complete preprocessing pipeline.
        
        Returns:
            Preprocessed DataFrame
        """
        logger.info("=" * 60)
        logger.info("Starting preprocessing pipeline")
        logger.info("=" * 60)
        
        # Load data
        self.load_data()
        
        # Preprocessing steps
        self.parse_datetime()
        self.filter_delivered_orders()
        self.drop_irrelevant_columns()
        self.extract_discount_percentage()
        self.convert_distance()
        self.extract_temporal_features()
        self.extract_order_count()
        
        logger.info("=" * 60)
        logger.info("Preprocessing complete!")
        logger.info(f"Final shape: {self.df.shape}")
        logger.info(f"Date range: {self.df['order_date'].min()} to {self.df['order_date'].max()}")
        logger.info("=" * 60)
        
        return self.df
    
    def save_processed_data(self, output_path: str) -> None:
        """
        Save preprocessed data to CSV.
        
        Args:
            output_path: Path to save processed data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")


def preprocess_pollution_data(pollution_path: str) -> pd.DataFrame:
    """
    Preprocess pollution data for merging with order data.
    
    Args:
        pollution_path: Path to pollution CSV file
        
    Returns:
        Preprocessed pollution DataFrame
    """
    logger.info(f"Loading pollution data from {pollution_path}")
    pollution = pd.read_csv(pollution_path)
    
    # Parse datetime
    pollution['pollution_time_utc'] = pd.to_datetime(pollution['pollution_time_utc'])
    pollution['pollution_date'] = pollution['pollution_time_utc'].dt.date
    pollution['pollution_hour'] = pollution['pollution_time_utc'].dt.hour
    
    # Select relevant features
    pollution_features = [
        'pollution_date', 'pollution_hour', 
        'aqi', 'co', 'no', 'no2', 'o3', 'so2', 
        'pm2_5', 'pm10', 'nh3'
    ]
    
    # Keep only columns that exist
    existing_features = [col for col in pollution_features if col in pollution.columns]
    pollution = pollution[existing_features]
    
    logger.info(f"Pollution data preprocessed: {pollution.shape}")
    
    return pollution


def merge_external_data(
    orders_df: pd.DataFrame,
    pollution_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Merge external data sources (pollution, weather) with order data.
    
    Args:
        orders_df: Preprocessed order data
        pollution_df: Preprocessed pollution data (optional)
        
    Returns:
        Merged DataFrame
    """
    merged_df = orders_df.copy()
    
    # Merge pollution data
    if pollution_df is not None:
        logger.info("Merging pollution data...")
        original_count = len(merged_df)
        
        merged_df = pd.merge(
            merged_df,
            pollution_df,
            left_on=['order_date', 'order_hour'],
            right_on=['pollution_date', 'pollution_hour'],
            how='inner'
        )
        
        # Drop redundant columns
        merged_df.drop(['pollution_date', 'pollution_hour'], axis=1, inplace=True, errors='ignore')
        
        logger.info(f"After pollution merge: {len(merged_df)}/{original_count} records retained")
    
    return merged_df


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Paths (adjust as needed)
    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    RAW_DATA_PATH = DATA_DIR / "raw" / "data.csv"
    POLLUTION_PATH = DATA_DIR / "raw" / "pollution.csv"
    PROCESSED_OUTPUT = DATA_DIR / "processed" / "cleaned_orders.csv"
    
    # Preprocess order data
    preprocessor = OrderDataPreprocessor(RAW_DATA_PATH)
    orders_df = preprocessor.run_preprocessing()
    
    # Preprocess pollution data (if exists)
    if POLLUTION_PATH.exists():
        pollution_df = preprocess_pollution_data(POLLUTION_PATH)
        
        # Merge data
        final_df = merge_external_data(orders_df, pollution_df)
    else:
        logger.warning("Pollution data not found, skipping merge")
        final_df = orders_df
    
    # Save
    preprocessor.df = final_df
    preprocessor.save_processed_data(PROCESSED_OUTPUT)
    
    print(f"\n{'='*60}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total orders: {len(final_df):,}")
    print(f"Date range: {final_df['order_date'].min()} to {final_df['order_date'].max()}")
    print(f"Features: {len(final_df.columns)}")
    print(f"Output: {PROCESSED_OUTPUT}")
    print(f"{'='*60}\n")
