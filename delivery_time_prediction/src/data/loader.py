"""
Data Loader for Delivery Time Prediction
========================================

This module handles loading raw data from CSV files.

Author: Saugat Shakya
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load raw delivery data from CSV files.
    
    Handles:
        - Main delivery/order data
        - Pollution data (if available)
        - Weather data (if available)
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to data directory. If None, uses config default.
        """
        from ..config import RAW_DATA_DIR
        self.data_path = data_path or RAW_DATA_DIR
        logger.info(f"DataLoader initialized with path: {self.data_path}")
    
    def load_order_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load main order/delivery data.
        
        Args:
            file_path: Path to data.csv. If None, uses default.
            
        Returns:
            DataFrame with raw order data
        """
        if file_path is None:
            file_path = self.data_path / "data.csv"
        
        logger.info(f"Loading order data from: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
            logger.info(f"Columns: {df.columns.tolist()}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def load_pollution_data(self, file_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """
        Load pollution data if available.
        
        Args:
            file_path: Path to pollution CSV
            
        Returns:
            DataFrame with pollution data or None if not found
        """
        if file_path is None:
            file_path = self.data_path / "delhi_pollution_orders.csv"
        
        try:
            df = pd.read_csv(file_path)
            df['pollution_time_utc'] = pd.to_datetime(df['pollution_time_utc'], errors='coerce')
            logger.info(f"Loaded pollution data: {len(df):,} rows")
            return df
        except FileNotFoundError:
            logger.warning(f"Pollution data not found at: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading pollution data: {e}")
            return None
    
    def load_all_data(self) -> dict:
        """
        Load all available data sources.
        
        Returns:
            Dictionary with keys: 'orders', 'pollution', 'weather'
        """
        data = {}
        
        # Load orders (required)
        data['orders'] = self.load_order_data()
        
        # Load pollution (optional)
        data['pollution'] = self.load_pollution_data()
        
        logger.info("All data loaded successfully")
        return data
