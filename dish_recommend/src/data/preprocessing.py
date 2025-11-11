"""
Data Preprocessing for Dish Recommendation System
==================================================

This module handles:
1. Loading order data
2. Parsing "Items in order" column to extract dish lists
3. Creating transaction baskets for association rules
4. Cleaning and normalizing dish names

Author: Saugat Shakya
Date: 2025-11-09
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DishOrderPreprocessor:
    """Preprocess order data for recommendation system."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.transactions = []
        self.dish_names = set()
        self.order_counts = {}
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load order data from CSV.
        
        Args:
            filepath: Path to data CSV file
            
        Returns:
            DataFrame with order data
        """
        logger.info(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df):,} orders")
        
        return df
    
    def parse_items_column(self, items_str: str) -> List[str]:
        """
        Parse the "Items in order" column to extract individual dishes.
        
        Format: "1 x Dish Name, 2 x Another Dish, ..."
        
        Args:
            items_str: String from Items in order column
            
        Returns:
            List of dish names (normalized)
        """
        if pd.isna(items_str) or items_str == '':
            return []
        
        dishes = []
        # Split by comma
        items = items_str.split(',')
        
        for item in items:
            item = item.strip()
            # Extract dish name after "x"
            # Pattern: "1 x Dish Name" or "2 x Dish Name"
            match = re.match(r'^\d+\s*x\s*(.+)$', item)
            if match:
                dish_name = match.group(1).strip()
                # Normalize dish name
                dish_name = self.normalize_dish_name(dish_name)
                dishes.append(dish_name)
        
        return dishes
    
    def normalize_dish_name(self, dish_name: str) -> str:
        """
        Normalize dish name (lowercase, remove extra spaces).
        
        Args:
            dish_name: Raw dish name
            
        Returns:
            Normalized dish name
        """
        # Convert to lowercase
        dish_name = dish_name.lower()
        # Remove extra whitespace
        dish_name = ' '.join(dish_name.split())
        # Remove special characters at start/end
        dish_name = dish_name.strip('.,;:-')
        
        return dish_name
    
    def create_transactions(self, df: pd.DataFrame, 
                           status_filter: str = 'Delivered') -> List[List[str]]:
        """
        Create transaction baskets from order data.
        
        Each transaction = list of dishes in one order.
        
        Args:
            df: DataFrame with order data
            status_filter: Only include orders with this status
            
        Returns:
            List of transactions (each transaction is a list of dishes)
        """
        logger.info(f"Creating transactions (filtering for: {status_filter})")
        
        # Filter by order status
        df_filtered = df[df['Order Status'] == status_filter].copy()
        logger.info(f"Filtered to {len(df_filtered):,} {status_filter} orders")
        
        transactions = []
        dish_counts = {}
        
        for idx, row in df_filtered.iterrows():
            items_str = row['Items in order']
            dishes = self.parse_items_column(items_str)
            
            # Only include orders with 2+ items (for co-occurrence)
            if len(dishes) >= 1:
                transactions.append(dishes)
                
                # Count individual dishes
                for dish in dishes:
                    self.dish_names.add(dish)
                    dish_counts[dish] = dish_counts.get(dish, 0) + 1
        
        logger.info(f"Created {len(transactions):,} transactions")
        logger.info(f"Found {len(self.dish_names):,} unique dishes")
        
        self.transactions = transactions
        self.order_counts = dish_counts
        
        return transactions
    
    def get_popular_dishes(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get most popular dishes by order count.
        
        Args:
            top_n: Number of top dishes to return
            
        Returns:
            DataFrame with dish names and order counts
        """
        df = pd.DataFrame([
            {'dish': dish, 'order_count': count}
            for dish, count in self.order_counts.items()
        ])
        df = df.sort_values('order_count', ascending=False).head(top_n)
        df = df.reset_index(drop=True)
        df.index = df.index + 1  # Start from 1
        
        return df
    
    def filter_rare_dishes(self, min_count: int = 5) -> List[List[str]]:
        """
        Remove rare dishes (ordered less than min_count times).
        
        This improves recommendation quality by focusing on popular items.
        
        Args:
            min_count: Minimum number of times a dish must appear
            
        Returns:
            Filtered transactions
        """
        # Get dishes that meet threshold
        valid_dishes = {
            dish for dish, count in self.order_counts.items()
            if count >= min_count
        }
        
        logger.info(f"Filtering dishes with <{min_count} orders")
        logger.info(f"Keeping {len(valid_dishes)} / {len(self.dish_names)} dishes")
        
        # Filter transactions
        filtered_transactions = []
        for transaction in self.transactions:
            filtered_items = [dish for dish in transaction if dish in valid_dishes]
            if len(filtered_items) >= 1:  # Keep if at least 1 item remains
                filtered_transactions.append(filtered_items)
        
        logger.info(f"Filtered transactions: {len(filtered_transactions):,}")
        
        return filtered_transactions
    
    def save_transactions(self, transactions: List[List[str]], 
                         output_path: str) -> None:
        """
        Save transactions to CSV.
        
        Args:
            transactions: List of transactions
            output_path: Path to save CSV
        """
        # Create DataFrame where each row is a transaction
        max_items = max(len(t) for t in transactions)
        
        data = []
        for transaction in transactions:
            row = transaction + [''] * (max_items - len(transaction))
            data.append(row)
        
        columns = [f'item_{i+1}' for i in range(max_items)]
        df = pd.DataFrame(data, columns=columns)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(transactions):,} transactions to: {output_path}")
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        transaction_sizes = [len(t) for t in self.transactions]
        
        stats = {
            'total_transactions': len(self.transactions),
            'total_unique_dishes': len(self.dish_names),
            'avg_items_per_order': np.mean(transaction_sizes),
            'median_items_per_order': np.median(transaction_sizes),
            'max_items_per_order': np.max(transaction_sizes),
            'min_items_per_order': np.min(transaction_sizes),
            'single_item_orders': sum(1 for t in self.transactions if len(t) == 1),
            'multi_item_orders': sum(1 for t in self.transactions if len(t) > 1),
        }
        
        return stats


def main():
    """Demo: Load and preprocess data."""
    # Initialize preprocessor
    preprocessor = DishOrderPreprocessor()
    
    # Load data (using absolute path to shared data folder)
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "data.csv"
    df = preprocessor.load_data(str(data_path))
    
    # Create transactions
    transactions = preprocessor.create_transactions(df)
    
    # Show statistics
    stats = preprocessor.get_statistics()
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.2f}")
        else:
            print(f"{key:30s}: {value:,}")
    
    # Show popular dishes
    print("\n" + "=" * 70)
    print("TOP 20 MOST POPULAR DISHES")
    print("=" * 70)
    popular = preprocessor.get_popular_dishes(top_n=20)
    print(popular.to_string())
    
    # Filter rare dishes
    filtered_transactions = preprocessor.filter_rare_dishes(min_count=10)
    
    # Save transactions
    output_path = Path(__file__).parent.parent.parent / "data" / "processed" / "transactions.csv"
    preprocessor.save_transactions(
        filtered_transactions,
        str(output_path)
    )
    
    print("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
