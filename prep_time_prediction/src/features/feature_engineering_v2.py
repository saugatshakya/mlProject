"""
Feature Engineering V2 - Kitchen Prep Time Prediction
DISH-FOCUSED APPROACH

This version focuses on WHAT IS BEING COOKED, not financial metrics.

Key Features:
1. One-hot encoded dishes with quantities (244 dishes)
2. Temporal features (hour, day, peaks)
3. Kitchen load (recent order volume)
4. Weather (may affect ingredient prep)
5. Events/holidays

REMOVED irrelevant features:
- Bill amounts, discounts, packaging charges
- Restaurant ID, location (treating as single restaurant)
- Distance, delivery info

Author: AI Assistant  
Date: November 10, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrepTimeFeatureEngineerV2:
    """
    Feature engineering focused on actual dish preparation.
    """
    
    def __init__(self):
        self.df = None
        self.all_dishes = []
        
    def parse_items_column(self, items_str: str) -> List[Tuple[str, int]]:
        """
        Parse 'Items in order' column.
        
        Input: "1 x Burger, 2 x Fries, 1 x Coke"
        Output: [("Burger", 1), ("Fries", 2), ("Coke", 1)]
        """
        if pd.isna(items_str):
            return []
        
        parsed = []
        for item in str(items_str).split(','):
            item = item.strip()
            if ' x ' in item:
                parts = item.split(' x ', 1)
                qty = int(parts[0].strip())
                dish = parts[1].strip()
                parsed.append((dish, qty))
        
        return parsed
    
    def create_dish_features(self):
        """
        Create one-hot encoded dish features with quantities.
        
        This is the CORE feature set:
        - Each dish gets a column: dish_GrilledChickenBurger
        - Value is the quantity (0 if not in order)
        - Directly represents what needs to be prepared
        """
        logger.info("Creating dish-based features...")
        
        # Collect all unique dishes
        all_dishes_set = set()
        order_dishes_list = []
        
        for idx, row in self.df.iterrows():
            items = row.get("Items in order", "")
            parsed = self.parse_items_column(items)
            
            dish_dict = {}
            for dish_name, qty in parsed:
                clean_name = dish_name.strip()
                all_dishes_set.add(clean_name)
                dish_dict[clean_name] = dish_dict.get(clean_name, 0) + qty
            
            order_dishes_list.append(dish_dict)
        
        self.all_dishes = sorted(all_dishes_set)
        logger.info(f"Found {len(self.all_dishes)} unique dishes")
        
        # Create one column per dish
        for dish in self.all_dishes:
            # Safe column name
            safe_name = dish.replace(' ', '_').replace('-', '_').replace('.', '').replace(',', '').replace('(', '').replace(')', '').replace('&', 'and').replace("'", '').replace('"', '')
            col_name = f"dish_{safe_name[:60]}"
            
            # Fill with quantities (0 if dish not in order)
            self.df[col_name] = [
                order_dish.get(dish, 0) 
                for order_dish in order_dishes_list
            ]
        
        logger.info(f"✅ Created {len(self.all_dishes)} dish feature columns")
    
    def create_basic_order_features(self):
        """
        Create basic order statistics.
        
        Features:
        - num_items: Total items in order
        - num_unique_dishes: Number of different dishes
        """
        logger.info("Creating basic order features...")
        
        num_items = []
        num_unique = []
        
        for idx, row in self.df.iterrows():
            items = row.get("Items in order", "")
            parsed = self.parse_items_column(items)
            
            # Total items
            total = sum(qty for _, qty in parsed)
            num_items.append(total)
            
            # Unique dishes
            unique = len(set(dish for dish, _ in parsed))
            num_unique.append(unique)
        
        self.df["num_items"] = num_items
        self.df["num_unique_dishes"] = num_unique
        
        logger.info("✅ Basic order features created")
    
    def create_temporal_features(self):
        """
        Create time-based features.
        
        Features:
        - order_hour (0-23)
        - order_dayofweek (0-6)
        - is_weekend
        - is_lunch_peak (11-14)
        - is_dinner_peak (18-21)
        - hour_sin, hour_cos (cyclic encoding)
        """
        logger.info("Creating temporal features...")
        
        # Ensure datetime
        if "Order Placed At" in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df["Order Placed At"]):
                self.df["Order Placed At"] = pd.to_datetime(self.df["Order Placed At"])
            
            # Extract time features
            self.df["order_hour"] = self.df["Order Placed At"].dt.hour
            self.df["order_dayofweek"] = self.df["Order Placed At"].dt.dayofweek
            self.df["is_weekend"] = (self.df["order_dayofweek"] >= 5).astype(int)
            
            # Peak hours
            self.df["is_lunch_peak"] = ((self.df["order_hour"] >= 11) & 
                                        (self.df["order_hour"] <= 14)).astype(int)
            self.df["is_dinner_peak"] = ((self.df["order_hour"] >= 18) & 
                                         (self.df["order_hour"] <= 21)).astype(int)
            self.df["is_peak_hour"] = (self.df["is_lunch_peak"] | 
                                       self.df["is_dinner_peak"]).astype(int)
            
            # Cyclic encoding for hour
            self.df["hour_sin"] = np.sin(2 * np.pi * self.df["order_hour"] / 24)
            self.df["hour_cos"] = np.cos(2 * np.pi * self.df["order_hour"] / 24)
            
            # Cyclic encoding for day
            self.df["day_sin"] = np.sin(2 * np.pi * self.df["order_dayofweek"] / 7)
            self.df["day_cos"] = np.cos(2 * np.pi * self.df["order_dayofweek"] / 7)
        
        logger.info("✅ Temporal features created")
    
    def create_load_features(self):
        """
        Create kitchen load features.
        
        Features:
        - orders_last_30min: Number of orders in last 30 minutes
        - items_last_30min: Total items in last 30 minutes
        """
        logger.info("Creating kitchen load features...")
        
        if "Order Placed At" in self.df.columns:
            # Sort by time
            self.df = self.df.sort_values("Order Placed At").reset_index(drop=True)
            
            orders_last_30 = []
            items_last_30 = []
            
            for idx, row in self.df.iterrows():
                current_time = row["Order Placed At"]
                
                # Get orders in last 30 minutes
                time_window = current_time - pd.Timedelta(minutes=30)
                recent_orders = self.df[
                    (self.df["Order Placed At"] >= time_window) & 
                    (self.df["Order Placed At"] < current_time)
                ]
                
                orders_last_30.append(len(recent_orders))
                items_last_30.append(recent_orders["num_items"].sum() if len(recent_orders) > 0 else 0)
            
            self.df["orders_last_30min"] = orders_last_30
            self.df["items_last_30min"] = items_last_30
            self.df["is_high_load"] = (self.df["orders_last_30min"] > self.df["orders_last_30min"].quantile(0.75)).astype(int)
        
        logger.info("✅ Kitchen load features created")
    
    def create_event_features(self):
        """
        Create event/holiday features (if available).
        """
        logger.info("Creating event features...")
        
        if "holiday" in self.df.columns:
            # One-hot encode holidays
            holidays = self.df["holiday"].fillna("none").astype(str)
            unique_holidays = [h for h in holidays.unique() if h not in ["none", "False", "True", "nan"]]
            
            for holiday in unique_holidays:
                col_name = f"event_{holiday.replace(' ', '_').replace('-', '_')[:30]}"
                self.df[col_name] = (holidays == holiday).astype(int)
            
            self.df["has_event"] = (holidays.isin(unique_holidays)).astype(int)
        
        logger.info("✅ Event features created")
    
    def create_weather_features(self):
        """
        Keep weather features (may affect ingredient prep speed).
        """
        logger.info("Checking weather features...")
        
        weather_cols = ["wx_temp_c", "wx_precip_mm", "wx_cloud_cover_pct"]
        available = [col for col in weather_cols if col in self.df.columns]
        
        if available:
            logger.info(f"✅ Weather features available: {available}")
        else:
            logger.info("⚠️  No weather features found")
    
    def drop_irrelevant_columns(self):
        """
        Drop columns that don't affect prep time.
        """
        logger.info("Dropping irrelevant columns...")
        
        # Financial features (irrelevant for prep time)
        financial_cols = [
            "Bill subtotal", "Packaging charges", "Total",
            "Restaurant discount (Promo)",
            "Restaurant discount (Flat offs, Freebies & others)",
            "Gold discount", "Brand pack discount",
            "Discount construct"
        ]
        
        # Location/delivery features (treating as single restaurant)
        location_cols = [
            "Restaurant ID", "Restaurant name", "Subzone", "City",
            "Distance", "Distance_km", "Delivery"
        ]
        
        # Other irrelevant
        other_cols = [
            "Order ID", "Order Status", "Items in order",
            "Instructions", "Rating", "Review",
            "Cancellation / Rejection reason",
            "Restaurant compensation (Cancellation)",
            "Restaurant penalty (Rejection)",
            "Order Ready Marked", "Customer complaint tag",
            "Customer ID", "Rider wait time (minutes)",
            "order_date", "order_hour_utc",
            "Order Placed At", "event", "holiday"  # Drop original datetime and event columns
        ]
        
        all_drop = financial_cols + location_cols + other_cols
        cols_to_drop = [col for col in all_drop if col in self.df.columns]
        
        self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
        
        logger.info(f"✅ Dropped {len(cols_to_drop)} irrelevant columns")
    
    def feature_engineering_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete feature engineering pipeline.
        
        Focus: What dishes are being prepared + when + kitchen load
        """
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING V2 - DISH-FOCUSED APPROACH")
        logger.info("=" * 60)
        
        self.df = df.copy()
        
        # Core features: THE DISHES THEMSELVES
        self.create_dish_features()  # 244 columns - one per dish
        
        # Supporting features
        self.create_basic_order_features()  # num_items, num_unique_dishes
        self.create_temporal_features()  # hour, day, peaks
        self.create_load_features()  # kitchen load in last 30 min
        self.create_event_features()  # holidays
        self.create_weather_features()  # weather (optional)
        
        # Cleanup
        self.drop_irrelevant_columns()
        
        logger.info(f"✅ Feature engineering complete: {self.df.shape[1]} total columns")
        logger.info(f"   - Dish features: {len(self.all_dishes)}")
        logger.info(f"   - Supporting features: ~{self.df.shape[1] - len(self.all_dishes)}")
        
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
    df = pd.read_csv("../data/processed/preprocessed_orders.csv")
    df["Order Placed At"] = pd.to_datetime(df["Order Placed At"])
    
    # Create features
    engineer = PrepTimeFeatureEngineerV2()
    df_features = engineer.feature_engineering_pipeline(df)
    
    # Save
    engineer.save_features("../data/processed/features_orders_v2.csv")
    
    print(f"\n{'='*60}")
    print(f"Features created: {df_features.shape[1]} columns")
    print(f"Dish features: {len(engineer.all_dishes)}")
    print(f"\nSample columns:")
    print(df_features.columns.tolist()[:20])
    print(f"\nSample data:")
    print(df_features.head())


if __name__ == "__main__":
    main()
