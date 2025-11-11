"""
Data Preprocessing Module for Kitchen Prep Time Prediction
==========================================================

This module handles:
1. Loading order data, events, and weather
2. Data cleaning and missing value handling
3. Basic preprocessing (datetime parsing, column dropping)
4. Merging external data sources

Author: Saugat Shakya
Date: November 2025
"""

import pandas as pd
import numpy as np
import pytz
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrepTimePreprocessor:
    """Preprocessor for kitchen preparation time prediction data."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.df = None
        self.events_df = None
        self.weather_df = None
        
    def load_order_data(self, filepath: str) -> pd.DataFrame:
        """
        Load order data from CSV.
        
        Args:
            filepath: Path to orders CSV
            
        Returns:
            DataFrame with order data
        """
        logger.info(f"Loading order data from: {filepath}")
        self.df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(self.df):,} orders")
        logger.info(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def drop_unwanted_columns(self):
        """Drop columns not needed for modeling."""
        cols_to_drop = [
            "Instructions",
            "Rating",
            "Review",
            "Cancellation / Rejection reason",
            "Restaurant compensation (Cancellation)",
            "Restaurant penalty (Rejection)",
            "Customer complaint tag",
            "Customer ID",
            "Restaurant name",
            "City",  # All same city
            "Delivery",  # Not relevant
            "Order Status",  # We only use delivered orders
        ]
        
        logger.info(f"Dropping {len(cols_to_drop)} unwanted columns")
        self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
        
        return self.df
    
    def handle_missing_values(self):
        """Handle missing values in dataset."""
        logger.info("Handling missing values...")
        
        # Discount construct - fill with "No discount"
        if "Discount construct" in self.df.columns:
            self.df["Discount construct"] = self.df["Discount construct"].fillna("No discount")
        
        # Rider wait time - fill with median
        if "Rider wait time (minutes)" in self.df.columns:
            median_wait = self.df["Rider wait time (minutes)"].median()
            self.df["Rider wait time (minutes)"] = self.df["Rider wait time (minutes)"].fillna(median_wait)
            logger.info(f"Filled rider wait time with median: {median_wait:.1f}")
        
        # Drop rows with missing target
        before_count = len(self.df)
        self.df = self.df.dropna(subset=["KPT duration (minutes)"])
        after_count = len(self.df)
        logger.info(f"Dropped {before_count - after_count} rows with missing KPT")
        
        return self.df
    
    def parse_datetime(self):
        """Parse Order Placed At column to datetime."""
        logger.info("Parsing datetime column...")
        
        self.df["Order Placed At"] = pd.to_datetime(
            self.df["Order Placed At"],
            format="%I:%M %p, %B %d %Y",
            errors="coerce"
        )
        
        # Drop rows with failed datetime parsing
        before_count = len(self.df)
        self.df = self.df.dropna(subset=["Order Placed At"])
        after_count = len(self.df)
        if before_count != after_count:
            logger.warning(f"Dropped {before_count - after_count} rows with invalid datetime")
        
        return self.df
    
    def process_distance(self):
        """Convert distance column to numeric km."""
        logger.info("Processing distance column...")
        
        # Handle "<1km" special case
        self.df["Distance_km"] = self.df["Distance"].replace({"<1km": "0.5km"})
        self.df["Distance_km"] = self.df["Distance_km"].str.replace("km", "", regex=False).astype(float)
        
        # Drop original Distance column
        self.df = self.df.drop(columns=["Distance"], errors='ignore')
        
        return self.df
    
    def load_events(self, filepath: str = None):
        """
        Load and merge event data.
        
        Args:
            filepath: Path to events CSV (default: delhi_major_events.csv in parent dir)
        """
        if filepath is None:
            filepath = Path(__file__).parent.parent.parent / "data" / "raw" / "delhi_major_events.csv"
        
        logger.info(f"Loading events from: {filepath}")
        self.events_df = pd.read_csv(filepath)
        self.events_df["date"] = pd.to_datetime(self.events_df["date"])
        
        # Create order_date for merging
        self.df["order_date"] = pd.to_datetime(self.df["Order Placed At"].dt.date)
        
        # Merge
        self.df = self.df.merge(
            self.events_df,
            left_on="order_date",
            right_on="date",
            how="left"
        )
        
        # Drop redundant date column
        self.df = self.df.drop(columns=["date"], errors='ignore')
        
        logger.info(f"Merged {len(self.events_df)} events")
        
        return self.df
    
    def fetch_weather_data(self):
        """
        Fetch historical weather data from Open-Meteo API.
        
        Returns:
            DataFrame with hourly weather data
        """
        logger.info("Fetching weather data from Open-Meteo...")
        
        # Delhi NCR coordinates
        LAT = 28.6139
        LON = 77.2090
        
        # Convert to UTC for API
        india_tz = pytz.timezone("Asia/Kolkata")
        order_local = self.df["Order Placed At"].dt.tz_localize(india_tz, nonexistent="NaT", ambiguous="NaT")
        order_utc = order_local.dt.tz_convert("UTC")
        order_hour_utc = order_utc.dt.floor("H")
        self.df["order_hour_utc"] = order_hour_utc.dt.tz_localize(None)
        
        # Date range
        start_date = self.df["order_hour_utc"].min().date()
        end_date = self.df["order_hour_utc"].max().date()
        
        # Variables to fetch
        hourly_vars = ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
            "pressure_msl",
            "cloud_cover"
        ])
        
        # API call
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={LAT}"
            f"&longitude={LON}"
            f"&start_date={start_date:%Y-%m-%d}"
            f"&end_date={end_date:%Y-%m-%d}"
            f"&hourly={hourly_vars}"
            f"&timezone=UTC"
        )
        
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            wx_json = resp.json()
            
            if "hourly" not in wx_json:
                raise RuntimeError(f"Open-Meteo did not return 'hourly' data")
            
            hourly = wx_json["hourly"]
            
            # Create weather DataFrame
            self.weather_df = pd.DataFrame({
                "weather_time_utc": pd.to_datetime(hourly["time"], errors="coerce"),
                "wx_temp_c": hourly.get("temperature_2m", [np.nan]*len(hourly["time"])),
                "wx_humidity_pct": hourly.get("relative_humidity_2m", [np.nan]*len(hourly["time"])),
                "wx_precip_mm": hourly.get("precipitation", [np.nan]*len(hourly["time"])),
                "wx_wind_speed_ms": hourly.get("wind_speed_10m", [np.nan]*len(hourly["time"])),
                "wx_wind_deg": hourly.get("wind_direction_10m", [np.nan]*len(hourly["time"])),
                "wx_weather_code": hourly.get("weather_code", [np.nan]*len(hourly["time"])),
                "wx_pressure_hpa": hourly.get("pressure_msl", [np.nan]*len(hourly["time"])),
                "wx_cloud_cover_pct": hourly.get("cloud_cover", [np.nan]*len(hourly["time"])),
            })
            
            self.weather_df["order_hour_utc"] = self.weather_df["weather_time_utc"].dt.floor("H")
            
            # Remove duplicates
            self.weather_df = self.weather_df.sort_values("order_hour_utc").drop_duplicates(
                subset=["order_hour_utc"],
                keep="first"
            )
            
            logger.info(f"Fetched {len(self.weather_df)} hourly weather records")
            
        except Exception as e:
            logger.error(f"Failed to fetch weather data: {e}")
            logger.warning("Continuing without weather data")
            self.weather_df = pd.DataFrame()
        
        return self.weather_df
    
    def merge_weather(self):
        """Merge weather data with orders."""
        if self.weather_df is None or self.weather_df.empty:
            logger.warning("No weather data to merge")
            return self.df
        
        logger.info("Merging weather data...")
        
        self.df = self.df.merge(
            self.weather_df.drop(columns=["weather_time_utc"], errors='ignore'),
            on="order_hour_utc",
            how="left"
        )
        
        # Drop some weather columns that showed low importance
        cols_to_drop = [
            "wx_humidity_pct",
            "wx_wind_speed_ms",
            "wx_wind_deg",
            "wx_weather_code",
            "wx_pressure_hpa",
        ]
        self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
        
        return self.df
    
    def preprocess_pipeline(self, order_filepath: str, events_filepath: str = None):
        """
        Run full preprocessing pipeline.
        
        Args:
            order_filepath: Path to orders CSV
            events_filepath: Path to events CSV (optional)
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("="*60)
        logger.info("PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        # Load data
        self.load_order_data(order_filepath)
        
        # Clean data
        self.drop_unwanted_columns()
        self.handle_missing_values()
        self.parse_datetime()
        self.process_distance()
        
        # External data
        if events_filepath or Path("../data/raw/delhi_major_events.csv").exists():
            self.load_events(events_filepath)
        
        self.fetch_weather_data()
        self.merge_weather()
        
        # Final cleanup
        self.df = self.df.drop(columns=["order_date"], errors='ignore')
        
        logger.info(f"✅ Preprocessing complete: {len(self.df):,} orders, {len(self.df.columns)} features")
        
        return self.df
    
    def save_processed_data(self, output_path: str):
        """Save processed data to CSV."""
        logger.info(f"Saving processed data to: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        logger.info("✅ Data saved")


def main():
    """Example usage."""
    preprocessor = PrepTimePreprocessor()
    
    # Preprocess data
    df = preprocessor.preprocess_pipeline(
        order_filepath="../data/raw/data.csv",
        events_filepath="../data/raw/delhi_major_events.csv"
    )
    
    # Save
    preprocessor.save_processed_data("../data/processed/orders_cleaned.csv")
    
    print(f"\n{'='*60}")
    print(f"Processed data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target (KPT) stats:")
    print(df["KPT duration (minutes)"].describe())


if __name__ == "__main__":
    main()
