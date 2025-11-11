"""
Dish Recommendation Model Wrapper
==================================

Association rules-based recommendation system.
Recommends dishes that are frequently ordered together.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')


class DishRecommendationModel:
    """Wrapper for dish recommendation model."""
    
    def __init__(self):
        self.association_rules = None
        self.cooccurrence_matrix = None
        self.dish_support = None
        self.all_dishes = set()
        self.metrics = {}
        self.trained = False
        self.model_path = Path('models/dish_recommendation.pkl')
        
    def is_trained(self):
        """Check if model is trained."""
        return self.trained
    
    def get_status(self):
        """Get model status."""
        return {
            'trained': self.trained,
            'metrics': self.metrics,
            'num_dishes': len(self.all_dishes),
            'num_rules': len(self.association_rules) if self.association_rules is not None else 0
        }
    
    def parse_items(self, items_str):
        """Parse items string into list of dishes."""
        if pd.isna(items_str):
            return []
        
        # Handle different formats
        items_str = str(items_str).strip()
        
        # Split by common delimiters
        if ',' in items_str:
            items = [item.strip() for item in items_str.split(',')]
        elif ';' in items_str:
            items = [item.strip() for item in items_str.split(';')]
        elif '|' in items_str:
            items = [item.strip() for item in items_str.split('|')]
        else:
            items = [items_str]
        
        # Clean items
        items = [item for item in items if item and len(item) > 0]
        
        return items
    
    def create_transactions(self, df):
        """
        Create transaction baskets from order data.
        
        Args:
            df: DataFrame with columns ['order_id', 'items'] or ['order_id', 'dish_name']
            
        Returns:
            list: List of transactions (sets of dishes)
        """
        transactions = []
        
        if 'items' in df.columns:
            # Format 1: order_id, items (comma-separated)
            for items_str in df['items']:
                items = self.parse_items(items_str)
                if len(items) > 0:
                    transactions.append(set(items))
                    self.all_dishes.update(items)
        
        elif 'dish_name' in df.columns or 'dish' in df.columns:
            # Format 2: order_id, dish_name (one row per dish)
            dish_col = 'dish_name' if 'dish_name' in df.columns else 'dish'
            for order_id, group in df.groupby('order_id'):
                dishes = set(group[dish_col].dropna().unique())
                if len(dishes) > 0:
                    transactions.append(dishes)
                    self.all_dishes.update(dishes)
        
        else:
            raise ValueError("Data must have 'items' column or 'order_id' and 'dish_name' columns")
        
        return transactions
    
    def calculate_support(self, transactions):
        """Calculate support for each dish."""
        dish_counts = Counter()
        for transaction in transactions:
            dish_counts.update(transaction)
        
        total_transactions = len(transactions)
        dish_support = {dish: count / total_transactions for dish, count in dish_counts.items()}
        
        return dish_support
    
    def calculate_cooccurrence(self, transactions):
        """Calculate co-occurrence matrix."""
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for transaction in transactions:
            dishes = list(transaction)
            for i, dish1 in enumerate(dishes):
                for dish2 in dishes[i+1:]:
                    cooccurrence[dish1][dish2] += 1
                    cooccurrence[dish2][dish1] += 1
        
        # Convert to regular dict for serialization
        cooccurrence_dict = {}
        for dish1, inner_dict in cooccurrence.items():
            cooccurrence_dict[dish1] = dict(inner_dict)
        
        return cooccurrence_dict
    
    def generate_rules(self, transactions, min_support=0.001, min_confidence=0.1):
        """
        Generate association rules.
        
        Args:
            transactions: List of transaction sets
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
            
        Returns:
            DataFrame: Association rules with support, confidence, lift
        """
        total_transactions = len(transactions)
        
        # Calculate support
        self.dish_support = self.calculate_support(transactions)
        
        # Calculate co-occurrence
        cooccurrence = self.calculate_cooccurrence(transactions)
        
        # Generate rules
        rules = []
        
        for dish1 in self.all_dishes:
            if self.dish_support[dish1] < min_support:
                continue
            
            for dish2 in self.all_dishes:
                if dish1 == dish2:
                    continue
                
                if self.dish_support[dish2] < min_support:
                    continue
                
                # Count co-occurrence (use .get() to handle dishes with no co-occurrences)
                cooccur_count = cooccurrence.get(dish1, {}).get(dish2, 0)
                
                if cooccur_count == 0:
                    continue
                
                # Calculate metrics
                support = cooccur_count / total_transactions
                
                if support < min_support:
                    continue
                
                # Confidence: P(dish2 | dish1) = cooccur / count(dish1)
                dish1_count = self.dish_support[dish1] * total_transactions
                confidence = cooccur_count / dish1_count
                
                if confidence < min_confidence:
                    continue
                
                # Lift: confidence / P(dish2)
                lift = confidence / self.dish_support[dish2]
                
                rules.append({
                    'antecedent': dish1,
                    'consequent': dish2,
                    'support': support,
                    'confidence': confidence,
                    'lift': lift,
                    'count': cooccur_count
                })
        
        # Convert to DataFrame and sort
        rules_df = pd.DataFrame(rules)
        
        if len(rules_df) > 0:
            rules_df = rules_df.sort_values('lift', ascending=False)
        
        return rules_df, cooccurrence
    
    def train(self, filepath):
        """
        Train recommendation model.
        
        Args:
            filepath: Path to CSV file with order data
            
        Returns:
            dict: Training results
        """
        print(f"\n{'='*70}")
        print("TRAINING DISH RECOMMENDATION MODEL")
        print(f"{'='*70}")
        
        # Load data
        print(f"\nLoading data from: {filepath}")
        df = pd.read_csv(filepath)
        
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Create transactions
        print("\nCreating transaction baskets...")
        transactions = self.create_transactions(df)
        
        print(f"Total transactions: {len(transactions)}")
        print(f"Unique dishes: {len(self.all_dishes)}")
        
        # Generate rules
        print("\nGenerating association rules...")
        self.association_rules, self.cooccurrence_matrix = self.generate_rules(
            transactions, min_support=0.001, min_confidence=0.1
        )
        
        num_rules = len(self.association_rules)
        print(f"Generated {num_rules} rules")
        
        if num_rules > 0:
            print(f"\nTop 5 rules by lift:")
            for idx, row in self.association_rules.head(5).iterrows():
                print(f"  {row['antecedent']} → {row['consequent']}")
                print(f"    Lift: {row['lift']:.2f}, Confidence: {row['confidence']:.2%}, Support: {row['support']:.4f}")
        
        # Calculate metrics
        self.metrics = {
            'num_transactions': len(transactions),
            'num_dishes': len(self.all_dishes),
            'num_rules': num_rules,
            'avg_lift': float(self.association_rules['lift'].mean()) if num_rules > 0 else 0,
            'avg_confidence': float(self.association_rules['confidence'].mean()) if num_rules > 0 else 0
        }
        
        # Save model
        self.model_path.parent.mkdir(exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'association_rules': self.association_rules,
                'cooccurrence_matrix': self.cooccurrence_matrix,
                'dish_support': self.dish_support,
                'all_dishes': self.all_dishes,
                'metrics': self.metrics
            }, f)
        
        print(f"\nModel saved to: {self.model_path}")
        self.trained = True
        
        print(f"{'='*70}\n")
        
        return {
            'status': 'success',
            'metrics': self.metrics
        }
    
    def recommend(self, dish_name, top_n=5):
        """
        Get recommendations for a dish.
        
        Args:
            dish_name: Name of the dish
            top_n: Number of recommendations
            
        Returns:
            dict: Recommendations with confidence scores
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
        
        # Find matching rules
        rules = self.association_rules[self.association_rules['antecedent'] == dish_name]
        
        if len(rules) == 0:
            # Try co-occurrence
            if dish_name in self.cooccurrence_matrix:
                cooccur = self.cooccurrence_matrix[dish_name]
                recommendations = [
                    {
                        'dish': dish,
                        'confidence': count / (self.dish_support[dish_name] * self.metrics['num_transactions']),
                        'lift': (count / (self.dish_support[dish_name] * self.metrics['num_transactions'])) / self.dish_support[dish],
                        'method': 'cooccurrence'
                    }
                    for dish, count in sorted(cooccur.items(), key=lambda x: x[1], reverse=True)[:top_n]
                ]
            else:
                # Return popular dishes
                popular = sorted(self.dish_support.items(), key=lambda x: x[1], reverse=True)[:top_n]
                recommendations = [
                    {
                        'dish': dish,
                        'confidence': support,
                        'lift': 1.0,
                        'method': 'popular'
                    }
                    for dish, support in popular
                ]
        else:
            # Use association rules
            recommendations = []
            for idx, row in rules.head(top_n).iterrows():
                recommendations.append({
                    'dish': row['consequent'],
                    'confidence': float(row['confidence']),
                    'lift': float(row['lift']),
                    'method': 'association_rules'
                })
        
        return {
            'query_dish': dish_name,
            'recommendations': recommendations,
            'num_recommendations': len(recommendations)
        }
    
    def search(self, query):
        """
        Search for dishes.
        
        Args:
            query: Search query
            
        Returns:
            dict: Matching dishes
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
        
        query = query.lower()
        matches = [dish for dish in self.all_dishes if query in dish.lower()]
        matches = sorted(matches, key=lambda x: self.dish_support[x], reverse=True)
        
        return {
            'query': query,
            'matches': matches[:20],  # Top 20 matches
            'num_matches': len(matches)
        }
    
    def get_popular(self, top_n=20):
        """
        Get most popular dishes.
        
        Args:
            top_n: Number of dishes
            
        Returns:
            dict: Popular dishes with support
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
        
        popular = sorted(self.dish_support.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        dishes = []
        for dish, support in popular:
            dishes.append({
                'dish': dish,
                'support': float(support),
                'popularity': f"{support*100:.2f}%"
            })
        
        return {
            'dishes': dishes,
            'num_dishes': len(dishes)
        }
