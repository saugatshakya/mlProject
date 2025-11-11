"""
Dish Recommendation Model Wrapper
==================================

Based on the actual workflow from dish_recommend project:
- Proper parsing of "Items in order" format: "1 x Dish Name, 2 x Another Dish, ..."
- Dish name normalization (lowercase, whitespace removal)
- Rare dish filtering (min_count threshold)
- Association rules (support, confidence, lift)
- Co-occurrence matrix

Author: Saugat Shakya
Date: 2025-11-09
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import pickle
import re


class DishRecommendationModel:
    """Dish recommendation using association rules and co-occurrence."""
    
    def __init__(self):
        self.transactions = []
        self.dish_support = {}  # How often each dish appears
        self.cooccurrence_matrix = {}  # How often dishes appear together
        self.association_rules = []  # Generated rules
        self.total_transactions = 0
        self.is_trained = False
        self.dish_names = set()
        self.metrics = {}
        
    def normalize_dish_name(self, dish_name: str) -> str:
        """
        Normalize dish name to match actual preprocessing workflow.
        - Convert to lowercase
        - Remove extra whitespace
        - Remove special characters at start/end
        """
        dish_name = dish_name.lower()
        dish_name = ' '.join(dish_name.split())
        dish_name = dish_name.strip('.,;:-')
        return dish_name
    
    def parse_items_column(self, items_str: str) -> List[str]:
        """
        Parse "Items in order" column.
        
        Expected format: "1 x Dish Name, 2 x Another Dish, ..."
        
        Args:
            items_str: String from Items column
            
        Returns:
            List of normalized dish names
        """
        if pd.isna(items_str) or items_str == '':
            return []
        
        dishes = []
        items = items_str.split(',')
        
        for item in items:
            item = item.strip()
            # Pattern: "1 x Dish Name" or "2 x Dish Name"
            match = re.match(r'^\d+\s*x\s*(.+)$', item)
            if match:
                dish_name = match.group(1).strip()
                dish_name = self.normalize_dish_name(dish_name)
                dishes.append(dish_name)
        
        return dishes
    
    def create_transactions(self, df: pd.DataFrame) -> List[List[str]]:
        """
        Create transaction baskets from order data.
        
        Supports two formats:
        1. Items format: order_id, items (comma-separated with "x")
        2. Single dish format: order_id, dish_name
        
        Args:
            df: DataFrame with order data
            
        Returns:
            List of transactions (each is a list of dishes)
        """
        print("\n" + "="*80)
        print("CREATING TRANSACTIONS")
        print("="*80)
        
        transactions = []
        dish_counts = {}
        
        # Check format
        if 'items' in df.columns:
            # Format 1: "Items in order" format
            print("✓ Detected 'items' column format")
            
            for idx, row in df.iterrows():
                items_str = row['items']
                dishes = self.parse_items_column(items_str)
                
                if len(dishes) >= 1:
                    transactions.append(dishes)
                    for dish in dishes:
                        self.dish_names.add(dish)
                        dish_counts[dish] = dish_counts.get(dish, 0) + 1
        
        elif 'dish_name' in df.columns and 'order_id' in df.columns:
            # Format 2: One row per dish
            print("✓ Detected 'order_id + dish_name' format")
            
            order_groups = df.groupby('order_id')['dish_name'].apply(list)
            
            for order_id, dishes in order_groups.items():
                # Normalize dish names
                dishes = [self.normalize_dish_name(d) for d in dishes]
                
                if len(dishes) >= 1:
                    transactions.append(dishes)
                    for dish in dishes:
                        self.dish_names.add(dish)
                        dish_counts[dish] = dish_counts.get(dish, 0) + 1
        
        else:
            raise ValueError(
                "CSV must have either:\n"
                "  1. 'items' column (format: '1 x Dish Name, 2 x Another, ...'), or\n"
                "  2. 'order_id' + 'dish_name' columns (one row per dish)"
            )
        
        print(f"\n✓ Created {len(transactions):,} transactions")
        print(f"✓ Found {len(self.dish_names):,} unique dishes")
        
        # Show top dishes
        sorted_dishes = sorted(dish_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 most ordered dishes:")
        for i, (dish, count) in enumerate(sorted_dishes[:10], 1):
            print(f"  {i:2d}. {dish[:50]:50s} | {count:4,} orders")
        
        self.transactions = transactions
        self.order_counts = dish_counts
        
        return transactions
    
    def filter_rare_dishes(self, transactions: List[List[str]], 
                          min_count: int = 5) -> List[List[str]]:
        """
        Remove rare dishes (ordered less than min_count times).
        Improves recommendation quality.
        
        Args:
            transactions: List of transactions
            min_count: Minimum times a dish must appear
            
        Returns:
            Filtered transactions
        """
        # Get dishes meeting threshold
        valid_dishes = {
            dish for dish, count in self.order_counts.items()
            if count >= min_count
        }
        
        print(f"\n✓ Filtering rare dishes (<{min_count} orders)")
        print(f"  Keeping {len(valid_dishes)} / {len(self.dish_names)} dishes")
        
        # Filter transactions
        filtered_transactions = []
        for transaction in transactions:
            filtered_items = [dish for dish in transaction if dish in valid_dishes]
            if len(filtered_items) >= 1:
                filtered_transactions.append(filtered_items)
        
        print(f"  Filtered transactions: {len(filtered_transactions):,}")
        
        return filtered_transactions
    
    def calculate_support(self, transactions: List[List[str]]):
        """Calculate support (frequency) for each dish."""
        print("\n✓ Calculating dish support...")
        
        dish_counts = Counter()
        for transaction in transactions:
            for dish in set(transaction):  # Count once per transaction
                dish_counts[dish] += 1
        
        # Convert to support (fraction)
        self.dish_support = {
            dish: count / len(transactions)
            for dish, count in dish_counts.items()
        }
        
        print(f"  Calculated support for {len(self.dish_support)} dishes")
    
    def calculate_cooccurrence(self, transactions: List[List[str]]):
        """
        Build co-occurrence matrix showing how often dishes appear together.
        
        Matrix[A][B] = number of transactions containing both A and B
        """
        print("\n✓ Building co-occurrence matrix...")
        
        # Use defaultdict for counting, then convert to regular dict
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for transaction in transactions:
            dishes = list(set(transaction))  # Remove duplicates within transaction
            for i, dish1 in enumerate(dishes):
                for dish2 in dishes[i+1:]:
                    # Count in both directions
                    cooccurrence[dish1][dish2] += 1
                    cooccurrence[dish2][dish1] += 1
        
        # Convert to regular dict for pickling
        cooccurrence_dict = {}
        for dish1, inner_dict in cooccurrence.items():
            cooccurrence_dict[dish1] = dict(inner_dict)
        
        self.cooccurrence_matrix = cooccurrence_dict
        
        total_pairs = sum(len(inner) for inner in cooccurrence_dict.values())
        print(f"  Found {total_pairs:,} dish pair co-occurrences")
    
    def generate_association_rules(self, min_support=0.01, min_confidence=0.3):
        """
        Generate association rules (if A then B).
        
        Metrics:
        - Support: P(A ∩ B) = count(A and B) / total_transactions
        - Confidence: P(B|A) = count(A and B) / count(A)
        - Lift: confidence / P(B)
        
        Args:
            min_support: Minimum support threshold (fraction)
            min_confidence: Minimum confidence threshold
        """
        print("\n✓ Generating association rules...")
        print(f"  Min support: {min_support:.2%}")
        print(f"  Min confidence: {min_confidence:.2%}")
        
        rules = []
        
        for dish_a in self.cooccurrence_matrix:
            for dish_b in self.cooccurrence_matrix[dish_a]:
                # Calculate metrics
                count_ab = self.cooccurrence_matrix[dish_a][dish_b]
                support_ab = count_ab / self.total_transactions
                
                # Support of A
                count_a = self.dish_support[dish_a] * self.total_transactions
                
                # Confidence: P(B|A)
                confidence = count_ab / count_a
                
                # Lift: confidence / P(B)
                lift = confidence / self.dish_support[dish_b]
                
                # Filter by thresholds
                if support_ab >= min_support and confidence >= min_confidence:
                    rules.append({
                        'antecedent': dish_a,
                        'consequent': dish_b,
                        'support': support_ab,
                        'confidence': confidence,
                        'lift': lift,
                        'count': count_ab
                    })
        
        # Sort by lift (descending) then confidence
        rules = sorted(rules, key=lambda x: (x['lift'], x['confidence']), 
                      reverse=True)
        
        self.association_rules = rules
        
        print(f"  Generated {len(rules):,} association rules")
    
    def train(self, csv_path, min_support=0.01, min_confidence=0.3, min_dish_count=5):
        """
        Train recommendation engine.
        
        Args:
            csv_path: Path to CSV with order data
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
            min_dish_count: Filter dishes with less than this many orders
            
        Returns:
            dict: Training metrics
        """
        print("\n" + "="*80)
        print("TRAINING DISH RECOMMENDATION ENGINE")
        print("="*80)
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"\n✓ Loaded {len(df):,} rows from {csv_path}")
        
        # Create transactions
        transactions = self.create_transactions(df)
        
        # Filter rare dishes
        transactions = self.filter_rare_dishes(transactions, min_count=min_dish_count)
        
        self.transactions = transactions
        self.total_transactions = len(transactions)
        
        # Calculate support
        self.calculate_support(transactions)
        
        # Build co-occurrence matrix
        self.calculate_cooccurrence(transactions)
        
        # Generate association rules
        self.generate_association_rules(min_support, min_confidence)
        
        self.is_trained = True
        
        # Collect metrics
        self.metrics = {
            'num_transactions': self.total_transactions,
            'num_dishes': len(self.dish_support),
            'num_rules': len(self.association_rules),
            'min_support': min_support,
            'min_confidence': min_confidence,
            'min_dish_count': min_dish_count
        }
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Transactions: {self.total_transactions:,}")
        print(f"Unique dishes: {len(self.dish_support):,}")
        print(f"Association rules: {len(self.association_rules):,}")
        
        return self.metrics
    
    def recommend(self, dish_name: str, top_n: int = 5):
        """
        Get recommendations for a given dish.
        
        Args:
            dish_name: Input dish name (will be normalized)
            top_n: Number of recommendations
            
        Returns:
            list: Recommended dishes with metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Normalize input
        dish_name = self.normalize_dish_name(dish_name)
        
        # Find rules where this dish is the antecedent
        recommendations = [
            rule for rule in self.association_rules
            if rule['antecedent'] == dish_name
        ]
        
        if not recommendations:
            return []
        
        # Format top N recommendations
        results = []
        for rule in recommendations[:top_n]:
            results.append({
                'dish': rule['consequent'],
                'confidence': float(rule['confidence']),
                'lift': float(rule['lift']),
                'support': float(rule['support']),
                'count': int(rule['count'])
            })
        
        return results
    
    def search_dishes(self, query: str, limit: int = 10):
        """
        Search for dishes matching query.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            list: Matching dish names
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        query = query.lower()
        matches = [
            dish for dish in self.dish_support.keys()
            if query in dish
        ]
        
        # Sort by order count
        matches = sorted(
            matches,
            key=lambda d: self.order_counts.get(d, 0),
            reverse=True
        )
        
        return matches[:limit]
    
    def save(self, filepath):
        """Save model to disk."""
        model_data = {
            'transactions': self.transactions,
            'dish_support': self.dish_support,
            'cooccurrence_matrix': self.cooccurrence_matrix,
            'association_rules': self.association_rules,
            'total_transactions': self.total_transactions,
            'dish_names': self.dish_names,
            'order_counts': self.order_counts,
            'metrics': self.metrics,
            'is_trained': self.is_trained
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.transactions = model_data['transactions']
        self.dish_support = model_data['dish_support']
        self.cooccurrence_matrix = model_data['cooccurrence_matrix']
        self.association_rules = model_data['association_rules']
        self.total_transactions = model_data['total_transactions']
        self.dish_names = model_data['dish_names']
        self.order_counts = model_data['order_counts']
        self.metrics = model_data['metrics']
        self.is_trained = model_data['is_trained']
        
        print(f"\n✓ Model loaded from {filepath}")
        print(f"  Dishes: {len(self.dish_support)}")
        print(f"  Rules: {len(self.association_rules)}")
