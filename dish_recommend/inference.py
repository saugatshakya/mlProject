"""
Simple Inference API for Dish Recommendations
==============================================

This module provides a simple API to get dish recommendations
for production use.

Usage:
    from inference import DishRecommendationAPI
    
    api = DishRecommendationAPI()
    api.load_model('models/')
    recommendations = api.recommend('bageecha pizza', top_n=5)
    print(recommendations)

Author: Saugat Shakya
Date: 2025-11-09
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DishRecommendationAPI:
    """Simple API for getting dish recommendations."""
    
    def __init__(self):
        """Initialize API."""
        self.association_rules = None
        self.cooccurrence_matrix = None
        self.dish_support = None
        self.loaded = False
        
    def load_model(self, model_dir: str):
        """
        Load trained recommendation model.
        
        Args:
            model_dir: Directory containing model files
        """
        model_dir = Path(model_dir)
        
        logger.info(f"Loading model from: {model_dir}")
        
        # Load association rules
        rules_path = model_dir / "association_rules.csv"
        if rules_path.exists():
            self.association_rules = pd.read_csv(rules_path)
            logger.info(f"Loaded {len(self.association_rules)} association rules")
        
        # Load co-occurrence matrix
        cooccur_path = model_dir / "cooccurrence_matrix.csv"
        if cooccur_path.exists():
            df = pd.read_csv(cooccur_path)
            # Convert to dict for faster lookup
            self.cooccurrence_matrix = {}
            for _, row in df.iterrows():
                dish1, dish2, count = row['dish1'], row['dish2'], row['count']
                if dish1 not in self.cooccurrence_matrix:
                    self.cooccurrence_matrix[dish1] = {}
                if dish2 not in self.cooccurrence_matrix:
                    self.cooccurrence_matrix[dish2] = {}
                self.cooccurrence_matrix[dish1][dish2] = count
                self.cooccurrence_matrix[dish2][dish1] = count
            logger.info(f"Loaded co-occurrence matrix")
        
        # Load dish support
        support_path = model_dir / "dish_support.csv"
        if support_path.exists():
            df = pd.read_csv(support_path)
            self.dish_support = dict(zip(df['dish'], df['support']))
            logger.info(f"Loaded support for {len(self.dish_support)} dishes")
        
        self.loaded = True
        logger.info("✅ Model loaded successfully!")
    
    def recommend(self, dish_name: str, top_n: int = 5, 
                 method: str = 'both') -> pd.DataFrame:
        """
        Get recommendations for a dish.
        
        Args:
            dish_name: Name of the dish
            top_n: Number of recommendations to return
            method: 'association', 'cooccurrence', or 'both'
            
        Returns:
            DataFrame with recommendations
        """
        if not self.loaded:
            raise ValueError("Model not loaded! Call load_model() first.")
        
        # Normalize dish name
        dish_name = dish_name.lower().strip()
        
        if method in ['association', 'both']:
            # Try association rules first
            if self.association_rules is not None:
                rules = self.association_rules[
                    self.association_rules['antecedent'] == dish_name
                ].copy()
                
                if not rules.empty:
                    rules = rules.nlargest(top_n, 'lift')
                    rules = rules[['consequent', 'confidence', 'lift', 'support']]
                    rules.columns = ['dish', 'confidence', 'lift', 'support']
                    rules['method'] = 'association_rules'
                    rules.index = range(1, len(rules) + 1)
                    
                    if method == 'association':
                        return rules
        
        if method in ['cooccurrence', 'both']:
            # Use co-occurrence
            if self.cooccurrence_matrix is not None and dish_name in self.cooccurrence_matrix:
                cooccur = self.cooccurrence_matrix[dish_name]
                sorted_dishes = sorted(cooccur.items(), 
                                      key=lambda x: x[1], reverse=True)
                top_dishes = sorted_dishes[:top_n]
                
                df = pd.DataFrame([
                    {
                        'dish': dish,
                        'times_ordered_together': count,
                        'method': 'cooccurrence'
                    }
                    for dish, count in top_dishes
                ])
                df.index = range(1, len(df) + 1)
                return df
        
        # No recommendations found
        return pd.DataFrame()
    
    def get_popular_dishes(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get most popular dishes.
        
        Args:
            top_n: Number of dishes to return
            
        Returns:
            DataFrame with popular dishes
        """
        if not self.loaded or self.dish_support is None:
            raise ValueError("Model not loaded!")
        
        df = pd.DataFrame([
            {'dish': dish, 'popularity': support}
            for dish, support in self.dish_support.items()
        ])
        df = df.nlargest(top_n, 'popularity')
        df.index = range(1, len(df) + 1)
        
        return df
    
    def search_dishes(self, query: str) -> List[str]:
        """
        Search for dishes matching a query.
        
        Args:
            query: Search term
            
        Returns:
            List of matching dish names
        """
        if not self.loaded or self.dish_support is None:
            raise ValueError("Model not loaded!")
        
        query = query.lower()
        matches = [
            dish for dish in self.dish_support.keys()
            if query in dish
        ]
        
        # Sort by popularity
        matches = sorted(matches, 
                        key=lambda d: self.dish_support[d], 
                        reverse=True)
        
        return matches
    
    def batch_recommend(self, dishes: List[str], top_n: int = 5) -> Dict:
        """
        Get recommendations for multiple dishes.
        
        Args:
            dishes: List of dish names
            top_n: Number of recommendations per dish
            
        Returns:
            Dictionary mapping dish names to recommendations
        """
        results = {}
        for dish in dishes:
            recs = self.recommend(dish, top_n=top_n)
            if not recs.empty:
                results[dish] = recs
        
        return results


def main():
    """Demo: Use the API to get recommendations."""
    # Initialize API
    api = DishRecommendationAPI()
    
    # Load model
    model_dir = Path(__file__).parent / "models"
    api.load_model(str(model_dir))
    
    print("\n" + "="*70)
    print("DISH RECOMMENDATION API - DEMO")
    print("="*70)
    
    # Get popular dishes
    print("\n📊 TOP 10 POPULAR DISHES:")
    print("-" * 70)
    popular = api.get_popular_dishes(top_n=10)
    for idx, row in popular.iterrows():
        print(f"{idx:2d}. {row['dish']:40s} ({row['popularity']:.1%})")
    
    # Search for dishes
    print("\n🔍 SEARCH: 'chicken'")
    print("-" * 70)
    matches = api.search_dishes('chicken')
    for i, dish in enumerate(matches[:10], 1):
        print(f"{i:2d}. {dish}")
    
    # Get recommendations for specific dishes
    test_dishes = [
        "bageecha pizza",
        "bone in jamaican grilled chicken",
        "chilli cheese garlic bread"
    ]
    
    for dish in test_dishes:
        print(f"\n💡 RECOMMENDATIONS FOR: '{dish.upper()}'")
        print("-" * 70)
        recs = api.recommend(dish, top_n=5, method='both')
        if not recs.empty:
            print(recs.to_string())
        else:
            print("No recommendations found")
    
    # Batch recommendations
    print("\n📦 BATCH RECOMMENDATIONS:")
    print("-" * 70)
    batch_results = api.batch_recommend(test_dishes[:2], top_n=3)
    for dish, recs in batch_results.items():
        print(f"\n{dish}:")
        print(recs[['dish', 'times_ordered_together']].to_string())
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
