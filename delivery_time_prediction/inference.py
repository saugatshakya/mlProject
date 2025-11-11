"""
Inference Script for Delivery Time Prediction
=============================================

This script demonstrates how to use the trained model for predictions.

Usage:
    python inference.py --model models/final/xgboost_model.pkl --data data/raw/new_orders.csv

Author: Saugat Shakya
Date: 2025-01-27
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import argparse
import logging
from src.models.inference import DeliveryTimePredictor
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run inference on new data."""
    
    parser = argparse.ArgumentParser(description='Predict delivery times for new orders')
    parser.add_argument(
        '--model',
        type=str,
        default='models/baseline/xgboost_model.pkl',
        help='Path to trained model file'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to CSV file with new orders (optional)'
    )
    parser.add_argument(
        '--single',
        action='store_true',
        help='Interactive mode for single prediction'
    )
    
    args = parser.parse_args()
    
    # Load predictor
    logger.info(f"Loading model from {args.model}...")
    predictor = DeliveryTimePredictor(args.model)
    
    if args.single:
        # Interactive single prediction
        logger.info("\n=== Single Order Prediction ===")
        logger.info("Enter order details (or press Enter for defaults):\n")
        
        features = {}
        features['Distance'] = float(input("Distance (km) [5.0]: ") or 5.0)
        features['order_hour'] = int(input("Order hour (0-23) [19]: ") or 19)
        features['order_day'] = int(input("Order day (0=Mon, 6=Sun) [5]: ") or 5)
        features['is_weekend'] = int(input("Is weekend? (0/1) [1]: ") or 1)
        
        # TODO: Add more features based on your model
        logger.info("\nNote: This is a simplified example. Real prediction requires all features.")
        logger.info("For full predictions, provide a CSV file with all required features.")
        
    elif args.data:
        # Batch prediction from CSV
        logger.info(f"Loading data from {args.data}...")
        df = pd.read_csv(args.data)
        
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.preprocess(df)
        
        logger.info("Engineering features...")
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(df_clean, target_col='Total_time_taken')
        
        logger.info("Making predictions...")
        predictions = predictor.predict(df_features, return_confidence=True)
        
        # Add predictions to DataFrame
        df['predicted_delivery_time'] = predictions['predictions']
        if 'lower_bound' in predictions:
            df['prediction_lower'] = predictions['lower_bound']
            df['prediction_upper'] = predictions['upper_bound']
        
        # Save results
        output_path = Path(args.data).parent / f"{Path(args.data).stem}_predictions.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"\nPredictions saved to {output_path}")
        
        # Show summary
        logger.info(f"\nSummary:")
        logger.info(f"  Total orders: {len(df):,}")
        logger.info(f"  Avg predicted time: {predictions['predictions'].mean():.1f} minutes")
        logger.info(f"  Min predicted time: {predictions['predictions'].min():.1f} minutes")
        logger.info(f"  Max predicted time: {predictions['predictions'].max():.1f} minutes")
        
    else:
        # Show feature importance
        logger.info("\nNo data provided. Showing feature importance...\n")
        importance = predictor.get_feature_importance(top_n=20)
        print(importance.to_string(index=False))
        
        logger.info("\nTo make predictions:")
        logger.info("  python inference.py --model MODEL_PATH --data DATA_CSV")
        logger.info("  python inference.py --model MODEL_PATH --single")


if __name__ == "__main__":
    main()
