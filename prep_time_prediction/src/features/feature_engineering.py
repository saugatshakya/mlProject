"""
Feature Engineering Module for Kitchen Prep Time Prediction
===========================================================

This module creates features for predicting kitchen preparation time:
1. Dish complexity features (complex dish identification)
2. Order features (item count, pricing)
3. Temporal features (hour, day, peak times)
4. Restaurant features (historical stats)
5. Location features (one-hot encoding)
6. Discount features
7. Load features (recent order volume)
8. Interaction features

Author: Saugat Shakya
Date: November 2025
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrepTimeFeatureEngineer:
    """Feature engineering for prep time prediction."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.df = None
        self.complex_dishes = []
        
    def parse_items_column(self, items_str: str) -> list:
        """
        Parse 'Items in order' column to extract dish names and quantities.
        
        Format: "1 x Dish Name, 2 x Another Dish"
        
        Args:
            items_str: String from Items in order column
            
        Returns:
            List of (dish_name, quantity) tuples
        """
        if pd.isna(items_str):
            return []
        
        parts = [p.strip() for p in str(items_str).split(",")]
        items = []
        
        for p in parts:
            m = re.match(r"(\d+)\s*x\s*(.*)", p)
            if m:
                qty = int(m.group(1))
                name = m.group(2).strip()
                items.append((name, qty))
        
        return items
    
    def count_total_items(self, items_str: str) -> int:
        """
        Count total number of items in an order.
        
        Args:
            items_str: String from Items in order column
            
        Returns:
            Total item count
        """
        parts = [p.strip() for p in str(items_str).split(",")]
        total = 0
        
        for p in parts:
            m = re.match(r"(\d+)\s*x", p)
            if m:
                total += int(m.group(1))
            else:
                total += 1
        
        return total
    
    def identify_complex_dishes(self, min_count: int = 20, extra_minutes: float = 3.0):
        """
        Identify dishes that take significantly longer to prepare.
        
        A dish is "complex" if:
        - Appears in at least min_count orders
        - Average KPT >= overall_mean + extra_minutes
        
        Args:
            min_count: Minimum number of orders for a dish to be considered
            extra_minutes: Minutes above overall mean to be considered complex
        """
        logger.info("Identifying complex dishes...")
        
        # Parse all dishes
        tmp = self.df[["Order ID", "Items in order", "KPT duration (minutes)"]].copy()
        tmp["parsed"] = tmp["Items in order"].apply(self.parse_items_column)
        
        # Explode to dish-level
        dish_df = tmp.explode("parsed")
        dish_df["dish_name"] = dish_df["parsed"].apply(lambda x: x[0] if x else None)
        dish_df["dish_qty"] = dish_df["parsed"].apply(lambda x: x[1] if x else 0)
        dish_df = dish_df.dropna(subset=["dish_name"])
        
        # Calculate dish statistics
        dish_stats = (
            dish_df.groupby("dish_name")["KPT duration (minutes)"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "dish_mean_kpt", "count": "dish_count"})
            .reset_index()
        )
        
        overall_mean = self.df["KPT duration (minutes)"].mean()
        
        # Identify complex dishes
        self.complex_dishes = dish_stats[
            (dish_stats["dish_count"] >= min_count) &
            (dish_stats["dish_mean_kpt"] >= overall_mean + extra_minutes)
        ]["dish_name"].tolist()
        
        logger.info(f"Found {len(self.complex_dishes)} complex dishes (min_count={min_count}, extra_minutes={extra_minutes})")
        if len(self.complex_dishes) > 0:
            logger.info(f"Examples: {self.complex_dishes[:5]}")
        
        return self.complex_dishes
    
    def create_complexity_features(self):
        """Create dish complexity features."""
        logger.info("Creating complexity features...")
        
        if not self.complex_dishes:
            self.identify_complex_dishes()
        
        # Parse items for each order
        tmp = self.df[["Order ID", "Items in order"]].copy()
        tmp["parsed"] = tmp["Items in order"].apply(self.parse_items_column)
        
        dish_df = tmp.explode("parsed")
        dish_df["dish_name"] = dish_df["parsed"].apply(lambda x: x[0] if x else None)
        dish_df["dish_qty"] = dish_df["parsed"].apply(lambda x: x[1] if x else 0)
        dish_df = dish_df.dropna(subset=["dish_name"])
        
        # Mark complex dishes
        dish_df["is_complex_dish"] = dish_df["dish_name"].isin(self.complex_dishes).astype(int)
        
        # Aggregate by order
        complex_per_order = (
            dish_df.assign(complex_qty=dish_df["dish_qty"] * dish_df["is_complex_dish"])
            .groupby("Order ID")["complex_qty"]
            .sum()
            .rename("num_complex_dishes")
            .reset_index()
        )
        
        has_complex_per_order = (
            dish_df.groupby("Order ID")["is_complex_dish"]
            .max()
            .rename("has_complex_dish")
            .reset_index()
        )
        
        order_complex = complex_per_order.merge(has_complex_per_order, on="Order ID", how="left")
        
        # Merge with main DataFrame
        self.df = self.df.merge(order_complex, on="Order ID", how="left")
        self.df["num_complex_dishes"] = self.df["num_complex_dishes"].fillna(0)
        self.df["has_complex_dish"] = self.df["has_complex_dish"].fillna(0).astype(int)
        
        logger.info("✅ Complexity features created")
        
        return self.df
    
    def create_order_features(self):
        """Create order-level features."""
        logger.info("Creating order features...")
        
        # Number of items
        self.df["num_items"] = self.df["Items in order"].apply(self.count_total_items)
        
        # Consolidate discounts
        discount_cols = [
            "Restaurant discount (Promo)",
            "Restaurant discount (Flat offs, Freebies & others)",
            "Gold discount",
            "Brand pack discount"
        ]
        
        # Total discount amount
        self.df["total_discount_amt"] = 0
        for col in discount_cols:
            if col in self.df.columns:
                self.df["total_discount_amt"] += self.df[col].fillna(0)
        
        # Drop individual discount columns
        self.df = self.df.drop(columns=discount_cols, errors='ignore')
        
        logger.info("✅ Order features created")
        
        return self.df
    
    def create_discount_features(self):
        """Create discount type features."""
        logger.info("Creating discount features...")
        
        # Has discount flag
        self.df["has_discount"] = (self.df["Discount construct"] != "No discount").astype(int)
        
        # Discount type
        dc = self.df["Discount construct"]
        self.df["discount_type"] = np.select(
            [
                dc.str.contains("% off", case=False, na=False),
                dc.str.contains("Flat Rs.", case=False, na=False),
                dc.str.contains("Buy 1 Get 1", case=False, na=False),
                dc.str.contains("Buy 7 Get 3", case=False, na=False),
            ],
            ["percent", "flat", "bogo", "bundle"],
            default="none"
        )
        
        # One-hot encode
        self.df = pd.get_dummies(self.df, columns=["discount_type"], prefix="disc", drop_first=True)
        
        # Drop original discount construct
        self.df = self.df.drop(columns=["Discount construct"], errors='ignore')
        
        logger.info("✅ Discount features created")
        
        return self.df
    
    def create_temporal_features(self):
        """Create time-based features."""
        logger.info("Creating temporal features...")
        
        # Basic temporal
        self.df["order_hour"] = self.df["Order Placed At"].dt.hour
        self.df["order_dayofweek"] = self.df["Order Placed At"].dt.dayofweek
        self.df["is_weekend"] = (self.df["order_dayofweek"] >= 5).astype(int)
        
        # Create local hour (needed for peak detection)
        import pytz
        india_tz = pytz.timezone("Asia/Kolkata")
        order_local = self.df["Order Placed At"].dt.tz_localize(india_tz, nonexistent="NaT", ambiguous="NaT")
        self.df["hour_local"] = order_local.dt.hour
        
        # Peak hours
        self.df["is_lunch_peak"] = self.df["hour_local"].between(12, 14).astype(int)
        self.df["is_dinner_peak"] = self.df["hour_local"].between(19, 22).astype(int)
        self.df["is_peak_hour"] = ((self.df["is_lunch_peak"] == 1) | (self.df["is_dinner_peak"] == 1)).astype(int)
        
        # Cyclic encoding for hour
        self.df["order_hour_sin"] = np.sin(2 * np.pi * self.df["hour_local"] / 24)
        self.df["order_hour_cos"] = np.cos(2 * np.pi * self.df["hour_local"] / 24)
        
        logger.info("✅ Temporal features created")
        
        return self.df
    
    def create_event_features(self):
        """Create event-related features."""
        logger.info("Creating event features...")
        
        # Has event flag
        if "event" in self.df.columns:
            self.df["has_event"] = (self.df["event"] != "No significant event").astype(int)
            
            # One-hot encode events
            self.df = pd.get_dummies(self.df, columns=["event"], prefix="event", drop_first=True)
            
            # Convert to int
            event_cols = [col for col in self.df.columns if col.startswith("event_")]
            self.df[event_cols] = self.df[event_cols].astype(int)
        
        logger.info("✅ Event features created")
        
        return self.df
    
    def create_location_features(self):
        """Create location features (subzone one-hot encoding)."""
        logger.info("Creating location features...")
        
        if "Subzone" in self.df.columns:
            # One-hot encode subzones
            self.df = pd.get_dummies(self.df, columns=["Subzone"], prefix="Subzone")
            
            logger.info(f"✅ Location features created ({self.df.filter(like='Subzone_').shape[1]} subzones)")
        
        return self.df
    
    def create_restaurant_features(self, target_col: str = "KPT duration (minutes)"):
        """
        Create restaurant-level features.
        
        These should be computed ONLY on training data to avoid leakage.
        This method just creates placeholder columns that will be filled during training.
        
        Args:
            target_col: Target column name
        """
        logger.info("Creating placeholder restaurant features...")
        
        # Create placeholder columns (will be filled during train/test split)
        self.df["rest_mean_KPT"] = 0.0
        self.df["rest_p75_KPT"] = 0.0
        self.df["rest_mean_wait"] = 0.0
        
        logger.info("✅ Restaurant feature placeholders created")
        
        return self.df
    
    def create_load_features(self):
        """
        Create kitchen load features (orders in last N minutes).
        
        This requires datetime sorting and rolling windows.
        """
        logger.info("Creating load features...")
        
        tmp = self.df[['Restaurant ID', 'Order Placed At']].copy()
        tmp = tmp.set_index('Order Placed At').sort_index()
        tmp['order_count'] = 1
        
        # Rolling sum over 30 minutes
        load = (
            tmp.groupby('Restaurant ID')
            .rolling('30min')['order_count']
            .sum()
            .rename('orders_last_30min')
            .reset_index()
        )
        
        # Merge back
        self.df = self.df.merge(load, on=['Restaurant ID', 'Order Placed At'], how='left')
        
        # Exclude current order from count
        self.df['orders_last_30min'] = (self.df['orders_last_30min'] - 1).clip(lower=0)
        
        logger.info("✅ Load features created")
        
        return self.df
    
    def create_engineered_features(self):
        """Create advanced engineered features."""
        logger.info("Creating engineered features...")
        
        # Average item value
        self.df["avg_item_value"] = self.df["Total"] / self.df["num_items"].clip(lower=1)
        
        # Complexity ratio
        self.df["complexity_ratio"] = self.df["num_complex_dishes"] / self.df["num_items"].clip(lower=1)
        
        # Big order flag
        self.df["is_big_order"] = (self.df["num_items"] >= 6).astype(int)
        
        # High value order (75th percentile)
        high_total_thresh = self.df["Total"].quantile(0.75)
        self.df["is_high_value_order"] = (self.df["Total"] >= high_total_thresh).astype(int)
        
        # High load flag
        if "orders_last_30min" in self.df.columns:
            self.df["is_high_load"] = (self.df["orders_last_30min"] >= 5).astype(int)
        
        # Peak * weekend interaction
        self.df["is_peak_weekend"] = ((self.df["is_peak_hour"] == 1) & (self.df["is_weekend"] == 1)).astype(int)
        
        # Cap num_complex_dishes at 3 (very few orders have >3)
        self.df["num_complex_capped"] = self.df["num_complex_dishes"].clip(upper=3)
        
        logger.info("✅ Engineered features created")
        
        return self.df
    
    def final_cleanup(self):
        """Final cleanup - drop columns not needed for modeling."""
        logger.info("Final cleanup...")
        
        cols_to_drop = [
            "Items in order",
            "Order Ready Marked",
            "order_hour_utc",  # Keep hour_local instead
        ]
        
        self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
        
        # Convert bool columns to int
        bool_cols = self.df.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            self.df[bool_cols] = self.df[bool_cols].astype(int)
        
        logger.info("✅ Final cleanup done")
        
        return self.df
    
    def feature_engineering_pipeline(self, df: pd.DataFrame):
        """
        Run full feature engineering pipeline.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with all features
        """
        logger.info("="*60)
        logger.info("FEATURE ENGINEERING PIPELINE")
        logger.info("="*60)
        
        self.df = df.copy()
        
        # Create all features
        self.create_complexity_features()
        self.create_order_features()
        self.create_discount_features()
        self.create_temporal_features()
        self.create_event_features()
        self.create_location_features()
        self.create_load_features()
        self.create_engineered_features()
        self.create_restaurant_features()
        self.final_cleanup()
        
        logger.info(f"✅ Feature engineering complete: {self.df.shape[1]} total columns")
        
        return self.df
    
    def save_features(self, output_path: str):
        """Save feature-engineered data."""
        logger.info(f"Saving features to: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        logger.info("✅ Features saved")


def main():
    """Example usage."""
    # Load preprocessed data
    df = pd.read_csv("../data/processed/orders_cleaned.csv")
    df["Order Placed At"] = pd.to_datetime(df["Order Placed At"])
    
    # Create features
    engineer = PrepTimeFeatureEngineer()
    df_features = engineer.feature_engineering_pipeline(df)
    
    # Save
    engineer.save_features("../data/processed/orders_with_features.csv")
    
    print(f"\n{'='*60}")
    print(f"Features created: {df_features.shape[1]} columns")
    print(f"Complex dishes found: {len(engineer.complex_dishes)}")
    print(f"\nSample features:")
    print(df_features[[
        "num_items", "num_complex_dishes", "order_hour", 
        "is_peak_hour", "is_weekend", "Total"
    ]].head())


if __name__ == "__main__":
    main()
