"""
Dish Recommendation Engine
===========================

This module implements multiple recommendation approaches:
1. Association Rules (Apriori algorithm) - Find frequent itemsets
2. Co-occurrence Matrix - Count how often dishes appear together
3. Collaborative Filtering - Based on order patterns

Author: Saugat Shakya
Date: 2025-11-09
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DishRecommender:
    """Dish recommendation engine using association rules."""
    
    def __init__(self):
        """Initialize recommender."""
        self.transactions = []
        self.dish_support = {}  # How often each dish appears
        self.cooccurrence_matrix = {}  # How often dishes appear together
        self.association_rules = []  # Generated rules
        self.total_transactions = 0
        
    def fit(self, transactions: List[List[str]], 
            min_support: float = 0.01,
            min_confidence: float = 0.3):
        """
        Train recommendation engine on transaction data.
        
        Args:
            transactions: List of transactions (each is list of dishes)
            min_support: Minimum support threshold (fraction of transactions)
            min_confidence: Minimum confidence for association rules
        """
        logger.info("="*60)
        logger.info("TRAINING RECOMMENDATION ENGINE")
        logger.info("="*60)
        
        self.transactions = transactions
        self.total_transactions = len(transactions)
        
        logger.info(f"Total transactions: {self.total_transactions:,}")
        logger.info(f"Min support: {min_support:.2%}")
        logger.info(f"Min confidence: {min_confidence:.2%}")
        
        # Step 1: Calculate support for individual dishes
        self._calculate_dish_support()
        
        # Step 2: Build co-occurrence matrix
        self._build_cooccurrence_matrix()
        
        # Step 3: Generate association rules
        self._generate_association_rules(
            min_support=min_support,
            min_confidence=min_confidence
        )
        
        logger.info("✅ Training complete!")
    
    def _calculate_dish_support(self):
        """Calculate support (frequency) for each dish."""
        logger.info("Calculating dish support...")
        
        dish_counts = Counter()
        for transaction in self.transactions:
            # Count each dish once per transaction
            for dish in set(transaction):
                dish_counts[dish] += 1
        
        # Convert to support (fraction)
        self.dish_support = {
            dish: count / self.total_transactions
            for dish, count in dish_counts.items()
        }
        
        logger.info(f"Calculated support for {len(self.dish_support)} dishes")
    
    def _build_cooccurrence_matrix(self):
        """
        Build co-occurrence matrix showing how often dishes appear together.
        
        Matrix[A][B] = number of transactions containing both A and B
        """
        logger.info("Building co-occurrence matrix...")
        
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for transaction in self.transactions:
            # For each pair of dishes in the transaction
            dishes = list(set(transaction))  # Remove duplicates
            for i, dish1 in enumerate(dishes):
                for dish2 in dishes[i+1:]:
                    # Count co-occurrence in both directions
                    cooccurrence[dish1][dish2] += 1
                    cooccurrence[dish2][dish1] += 1
        
        self.cooccurrence_matrix = dict(cooccurrence)
        
        total_pairs = sum(len(dishes) for dishes in cooccurrence.values())
        logger.info(f"Found {total_pairs:,} dish co-occurrences")
    
    def _generate_association_rules(self, min_support: float, 
                                   min_confidence: float):
        """
        Generate association rules (if A then B).
        
        For each pair (A, B):
        - Support: P(A ∩ B) = count(A and B) / total_transactions
        - Confidence: P(B|A) = count(A and B) / count(A)
        - Lift: P(B|A) / P(B) = confidence / support(B)
        
        Args:
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
        """
        logger.info("Generating association rules...")
        
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
        
        logger.info(f"Generated {len(rules):,} association rules")
    
    def recommend(self, dish_name: str, top_n: int = 5) -> pd.DataFrame:
        """
        Get recommendations for a given dish.
        
        Args:
            dish_name: Input dish name (will be normalized)
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame with recommended dishes and metrics
        """
        # Normalize input
        dish_name = dish_name.lower().strip()
        
        # Find rules where this dish is the antecedent
        recommendations = [
            rule for rule in self.association_rules
            if rule['antecedent'] == dish_name
        ]
        
        if not recommendations:
            logger.warning(f"No recommendations found for '{dish_name}'")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(recommendations[:top_n])
        df = df[['consequent', 'confidence', 'lift', 'support', 'count']]
        df.columns = ['recommended_dish', 'confidence', 'lift', 'support', 
                     'times_ordered_together']
        
        # Add index starting from 1
        df.index = range(1, len(df) + 1)
        
        return df
    
    def recommend_by_cooccurrence(self, dish_name: str, 
                                 top_n: int = 5) -> pd.DataFrame:
        """
        Simple recommendation based on co-occurrence count.
        
        This is simpler than association rules - just counts
        how many times dishes appear together.
        
        Args:
            dish_name: Input dish name
            top_n: Number of recommendations
            
        Returns:
            DataFrame with recommendations
        """
        dish_name = dish_name.lower().strip()
        
        if dish_name not in self.cooccurrence_matrix:
            logger.warning(f"Dish '{dish_name}' not found in data")
            return pd.DataFrame()
        
        # Get co-occurrence counts
        cooccurrences = self.cooccurrence_matrix[dish_name]
        
        # Sort by count
        sorted_dishes = sorted(cooccurrences.items(), 
                              key=lambda x: x[1], reverse=True)
        
        # Get top N
        top_dishes = sorted_dishes[:top_n]
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                'recommended_dish': dish,
                'times_ordered_together': count,
                'cooccurrence_rate': count / (self.dish_support[dish_name] * 
                                             self.total_transactions)
            }
            for dish, count in top_dishes
        ])
        
        df.index = range(1, len(df) + 1)
        
        return df
    
    def get_top_rules(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top association rules by lift.
        
        Args:
            top_n: Number of rules to return
            
        Returns:
            DataFrame with top rules
        """
        if not self.association_rules:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.association_rules[:top_n])
        df = df[['antecedent', 'consequent', 'confidence', 'lift', 
                'support', 'count']]
        df.index = range(1, len(df) + 1)
        
        return df
    
    def save_model(self, output_dir: str):
        """
        Save trained model.
        
        Args:
            output_dir: Directory to save model files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save association rules
        if self.association_rules:
            rules_df = pd.DataFrame(self.association_rules)
            rules_path = output_dir / "association_rules.csv"
            rules_df.to_csv(rules_path, index=False)
            logger.info(f"Saved {len(rules_df)} rules to: {rules_path}")
        
        # Save dish support
        support_df = pd.DataFrame([
            {'dish': dish, 'support': support}
            for dish, support in self.dish_support.items()
        ])
        support_df = support_df.sort_values('support', ascending=False)
        support_path = output_dir / "dish_support.csv"
        support_df.to_csv(support_path, index=False)
        logger.info(f"Saved dish support to: {support_path}")
        
        # Save co-occurrence matrix (top pairs only to save space)
        cooccurrence_data = []
        for dish1 in self.cooccurrence_matrix:
            for dish2, count in self.cooccurrence_matrix[dish1].items():
                if dish1 < dish2:  # Avoid duplicates
                    cooccurrence_data.append({
                        'dish1': dish1,
                        'dish2': dish2,
                        'count': count
                    })
        
        if cooccurrence_data:
            cooccurrence_df = pd.DataFrame(cooccurrence_data)
            cooccurrence_df = cooccurrence_df.sort_values('count', 
                                                         ascending=False)
            cooccurrence_path = output_dir / "cooccurrence_matrix.csv"
            cooccurrence_df.to_csv(cooccurrence_path, index=False)
            logger.info(f"Saved co-occurrence matrix to: {cooccurrence_path}")


def main():
    """Demo: Train recommendation engine."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data.preprocessing import DishOrderPreprocessor
    
    # Load and preprocess data
    preprocessor = DishOrderPreprocessor()
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "data.csv"
    df = preprocessor.load_data(str(data_path))
    transactions = preprocessor.create_transactions(df)
    filtered_transactions = preprocessor.filter_rare_dishes(min_count=10)
    
    # Train recommender
    recommender = DishRecommender()
    recommender.fit(filtered_transactions, min_support=0.001, min_confidence=0.1)
    
    # Show top rules
    print("\n" + "="*70)
    print("TOP 20 ASSOCIATION RULES (by Lift)")
    print("="*70)
    top_rules = recommender.get_top_rules(top_n=20)
    if not top_rules.empty:
        print(top_rules.to_string())
    else:
        print("No rules generated with current thresholds")
    
    # Test recommendations using both methods
    test_dishes = [
        "bone in jamaican grilled chicken",
        "bageecha pizza",
        "chilli cheese garlic bread"
    ]
    
    for dish in test_dishes:
        print("\n" + "="*70)
        print(f"RECOMMENDATIONS FOR: {dish.upper()}")
        print("="*70)
        
        # Try association rules first
        recs = recommender.recommend(dish, top_n=5)
        if not recs.empty:
            print("\nUsing Association Rules:")
            print(recs.to_string())
        
        # Show co-occurrence recommendations
        print("\nUsing Co-occurrence:")
        recs_cooccur = recommender.recommend_by_cooccurrence(dish, top_n=5)
        if not recs_cooccur.empty:
            print(recs_cooccur.to_string())
        else:
            print("No recommendations found")
    
    # Save model
    models_dir = Path(__file__).parent.parent.parent / "models"
    recommender.save_model(str(models_dir))
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
