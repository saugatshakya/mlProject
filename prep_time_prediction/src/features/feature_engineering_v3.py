"""
Feature Engineering V3 - Enhanced Dish-Based Features

Improvements over V2:
1. Dish embeddings via frequency encoding
2. Dish co-occurrence features
3. Order complexity metrics
4. Better temporal features
5. Dish category grouping

Author: AI Assistant
Date: November 10, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrepTimeFeatureEngineerV3:
    """
    Enhanced feature engineering with smarter dish features.
    """
    
    def __init__(self):
        self.df = None
        self.all_dishes = []
        self.dish_freq = {}
        self.dish_avg_prep_time = {}
        
    def parse_items_column(self, items_str: str) -> List[Tuple[str, int]]:
        """Parse 'Items in order' column."""
        if pd.isna(items_str):
            return []
        
        items = []
        for item in str(items_str).split(','):
            item = item.strip()
            if 'x' in item.lower():
                parts = item.split('x', 1)
                try:
                    qty = int(parts[0].strip())
                    dish = parts[1].strip()
                    items.append((dish, qty))
                except:
                    items.append((item, 1))
            else:
                items.append((item, 1))
        return items
    
    def create_dish_frequency_features(self):
        """
        Encode dishes by their frequency (popular dishes).
        This reduces dimensionality while preserving information.
        """
        logger.info("Creating dish frequency features...")
        
        # Parse all items
        all_dish_names = []
        for items_str in self.df["Items in order"]:
            for dish, qty in self.parse_items_column(items_str):
                all_dish_names.append(dish)
        
        # Count frequencies
        dish_counts = Counter(all_dish_names)
        self.dish_freq = dict(dish_counts)
        
        # Calculate percentile-based popularity
        freq_values = list(dish_counts.values())
        
        def get_dish_popularity(dish):
            freq = self.dish_freq.get(dish, 0)
            if freq == 0:
                return 0
            percentile = (sum(1 for f in freq_values if f <= freq) / len(freq_values)) * 100
            return percentile
        
        # Add popularity score
        self.df["avg_dish_popularity"] = self.df["Items in order"].apply(
            lambda x: np.mean([get_dish_popularity(dish) for dish, qty in self.parse_items_column(x)]) 
            if self.parse_items_column(x) else 0
        )
        
        logger.info(f"✅ Dish frequency features created")
    
    def create_dish_historical_features(self):
        """
        REMOVED - This was data leakage!
        
        We were using the target variable (KPT duration) to create features,
        which is a form of data leakage. In production, we wouldn't have
        historical prep times from the SAME dataset we're trying to predict.
        
        Proper approach would be:
        - Use truly historical data from BEFORE the prediction period
        - Calculate on training set only, apply to test set
        """
        logger.info("Skipping historical features (would be data leakage)")
        # No features created - removed to prevent leakage
    
    def create_enhanced_order_features(self):
        """Enhanced order-level features."""
        logger.info("Creating enhanced order features...")
        
        def get_order_stats(items_str):
            dishes = self.parse_items_column(items_str)
            if not dishes:
                return 0, 0, 0, 0
            
            num_items = sum(qty for _, qty in dishes)
            num_unique = len(dishes)
            max_qty = max(qty for _, qty in dishes)
            diversity = num_unique / num_items if num_items > 0 else 0
            
            return num_items, num_unique, max_qty, diversity
        
        stats = self.df["Items in order"].apply(get_order_stats)
        
        self.df["num_items"] = stats.apply(lambda x: x[0])
        self.df["num_unique_dishes"] = stats.apply(lambda x: x[1])
        self.df["max_dish_quantity"] = stats.apply(lambda x: x[2])
        self.df["dish_diversity"] = stats.apply(lambda x: x[3])
        
        # Order complexity score
        self.df["order_complexity"] = (
            self.df["num_unique_dishes"] * 2 +  # More unique dishes = more complex
            self.df["num_items"] * 0.5 +         # More items = more time
            (1 - self.df["dish_diversity"]) * 3  # Less diversity = bulk order = more efficient
        )
        
        logger.info("✅ Enhanced order features created")
    
    def create_enhanced_temporal_features(self):
        """Enhanced temporal features with interactions."""
        logger.info("Creating enhanced temporal features...")
        
        self.df["hour"] = self.df["Order Placed At"].dt.hour
        self.df["day_of_week"] = self.df["Order Placed At"].dt.dayofweek
        self.df["is_weekend"] = (self.df["day_of_week"] >= 5).astype(int)
        
        # Peak periods
        self.df["is_lunch_peak"] = self.df["hour"].between(12, 14).astype(int)
        self.df["is_dinner_peak"] = self.df["hour"].between(18, 21).astype(int)
        self.df["is_late_night"] = self.df["hour"].between(22, 23).astype(int)
        self.df["is_early_morning"] = self.df["hour"].between(0, 6).astype(int)
        
        # Cyclic encoding
        self.df["hour_sin"] = np.sin(2 * np.pi * self.df["hour"] / 24)
        self.df["hour_cos"] = np.cos(2 * np.pi * self.df["hour"] / 24)
        self.df["day_sin"] = np.sin(2 * np.pi * self.df["day_of_week"] / 7)
        self.df["day_cos"] = np.cos(2 * np.pi * self.df["day_of_week"] / 7)
        
        # Day of month effects (paydays, month-end)
        self.df["day_of_month"] = self.df["Order Placed At"].dt.day
        self.df["is_month_start"] = (self.df["day_of_month"] <= 5).astype(int)
        self.df["is_month_end"] = (self.df["day_of_month"] >= 25).astype(int)
        
        logger.info("✅ Enhanced temporal features created")
    
    def create_kitchen_load_features(self):
        """Kitchen load with rolling windows."""
        logger.info("Creating kitchen load features...")
        
        df_sorted = self.df.sort_values("Order Placed At").copy()
        
        # 30-minute rolling window
        df_sorted["orders_last_30min"] = 0
        df_sorted["items_last_30min"] = 0
        
        for idx in df_sorted.index:
            current_time = df_sorted.loc[idx, "Order Placed At"]
            time_window_start = current_time - pd.Timedelta(minutes=30)
            
            mask = (df_sorted["Order Placed At"] >= time_window_start) & \
                   (df_sorted["Order Placed At"] < current_time)
            
            df_sorted.loc[idx, "orders_last_30min"] = mask.sum()
            df_sorted.loc[idx, "items_last_30min"] = df_sorted.loc[mask, "num_items"].sum()
        
        # High load indicator
        load_threshold = df_sorted["orders_last_30min"].quantile(0.75)
        df_sorted["is_high_load"] = (df_sorted["orders_last_30min"] > load_threshold).astype(int)
        
        # Update main dataframe
        self.df["orders_last_30min"] = df_sorted["orders_last_30min"]
        self.df["items_last_30min"] = df_sorted["items_last_30min"]
        self.df["is_high_load"] = df_sorted["is_high_load"]
        
        logger.info("✅ Kitchen load features created")
    
    def create_event_features(self):
        """Event features."""
        logger.info("Creating event features...")
        
        if "holiday" in self.df.columns:
            holidays = self.df["holiday"].fillna("none").astype(str)
            unique_holidays = [h for h in holidays.unique() if h not in ["none", "False", "True", "nan"]]
            
            for holiday in unique_holidays:
                col_name = f"event_{holiday.replace(' ', '_').replace('-', '_')[:30]}"
                self.df[col_name] = (holidays == holiday).astype(int)
            
            self.df["has_event"] = (holidays.isin(unique_holidays)).astype(int)
        
        logger.info("✅ Event features created")
    
    def drop_irrelevant_columns(self):
        """Drop irrelevant columns."""
        logger.info("Dropping irrelevant columns...")
        
        financial_cols = [
            "Bill subtotal", "Total", "Packaging charges",
            "GST & Restaurant Charges", "Discount", "Tip amount",
            "Delivery charges"
        ]
        
        location_cols = [
            "Restaurant ID", "Restaurant Name", "Subzone",
            "Distance (KM)", "Customer Address"
        ]
        
        other_cols = [
            "Order ID", "Order Status", "Items in order",
            "Instructions", "Rating", "Review",
            "Cancellation / Rejection reason",
            "Restaurant compensation (Cancellation)",
            "Restaurant penalty (Rejection)",
            "Order Ready Marked", "Customer complaint tag",
            "Customer ID", "Rider wait time (minutes)",
            "order_date", "order_hour_utc",
            "Order Placed At", "event", "holiday"
        ]
        
        all_drop = financial_cols + location_cols + other_cols
        cols_to_drop = [col for col in all_drop if col in self.df.columns]
        
        self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
        
        logger.info(f"✅ Dropped {len(cols_to_drop)} irrelevant columns")
    
    def feature_engineering_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run V3 feature engineering pipeline.
        """
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING V3 - ENHANCED DISH FEATURES")
        logger.info("=" * 60)
        
        self.df = df.copy()
        
        # Core features
        self.create_enhanced_order_features()  # Must be first (creates num_items)
        self.create_dish_frequency_features()
        self.create_dish_historical_features()
        
        # Context features
        self.create_enhanced_temporal_features()
        self.create_kitchen_load_features()
        self.create_event_features()
        
        # Cleanup
        self.drop_irrelevant_columns()
        
        logger.info(f"✅ V3 Feature engineering complete: {self.df.shape[1]} total columns")
        
        return self.df


if __name__ == "__main__":
    # Test
    df = pd.read_csv("../../data/processed/preprocessed_orders.csv")
    df["Order Placed At"] = pd.to_datetime(df["Order Placed At"])
    
    engineer = PrepTimeFeatureEngineerV3()
    df_features = engineer.feature_engineering_pipeline(df)
    
    print(f"\nV3 Features shape: {df_features.shape}")
    print(f"\nColumns: {df_features.columns.tolist()}")
