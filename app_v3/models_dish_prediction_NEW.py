"""
Dish Prediction Model Wrapper
==============================

Based on the actual workflow from dish_prediction project:
- Top N dishes selection
- Extensive feature engineering (temporal, lag, smoothed)
- CatBoost/XGBoost multi-output regression
- Features: hour, day, weekend, cyclical, lags (1h,2h,3h), rolling mean (3h)

Author: Saugat Shakya
Date: 2025-11-09
"""

import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
from pathlib import Path


class DishPredictionModel:
    """Dish demand prediction with proper feature engineering."""
    
    def __init__(self, top_n_dishes=10, model_type='catboost'):
        self.model = None
        self.dish_columns = []
        self.feature_columns = []
        self.is_trained = False
        self.top_n_dishes = top_n_dishes
        self.model_type = model_type
        self.metrics = {}
        
    def create_features(self, df):
        """
        Create features matching the actual dish_prediction workflow.
        
        Input format: CSV with columns [timestamp, dish1, dish2, ...]
        
        Features created:
        1. Temporal: hour, day_of_week, is_weekend, cyclical encoding (sin/cos)
        2. Lag features: 1h, 2h, 3h for each dish
        3. Smoothed history: 3-hour rolling mean for each dish
        
        Returns:
            features_df: DataFrame with all features
            targets_df: DataFrame with dish demands (targets)
        """
        df = df.copy()
        
        # Handle both timestamp column names
        if 'order_placed_at' in df.columns:
            df['timestamp'] = pd.to_datetime(df['order_placed_at'])
            df = df.drop('order_placed_at', axis=1)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("CSV must have 'timestamp' or 'order_placed_at' column")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Get dish columns (everything except timestamp)
        all_dish_cols = [col for col in df.columns if col != 'timestamp']
        
        # Select top N dishes by total volume
        dish_volumes = df[all_dish_cols].sum().sort_values(ascending=False)
        self.dish_columns = dish_volumes.head(self.top_n_dishes).index.tolist()
        
        print(f"\n✓ Selected top {len(self.dish_columns)} dishes by volume:")
        for i, dish in enumerate(self.dish_columns, 1):
            print(f"  {i:2d}. {dish[:50]:50s} | {dish_volumes[dish]:6,.0f} orders")
        
        # Create features dataframe
        features_df = pd.DataFrame(index=df.index)
        
        # =========================================================================
        # 1. TEMPORAL FEATURES
        # =========================================================================
        features_df['hour'] = df['timestamp'].dt.hour
        features_df['day_of_week'] = df['timestamp'].dt.dayofweek
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for hour (captures circular nature of time)
        features_df['sin_hour'] = np.sin(2 * np.pi * features_df['hour'] / 24)
        features_df['cos_hour'] = np.cos(2 * np.pi * features_df['hour'] / 24)
        
        # =========================================================================
        # 2. LAG FEATURES (1, 2, 3 hours) - Historical demand patterns
        # =========================================================================
        for dish in self.dish_columns:
            for lag in [1, 2, 3]:
                features_df[f'{dish}_lag{lag}'] = df[dish].shift(lag)
        
        # =========================================================================
        # 3. SMOOTHED DISH HISTORY (3-hour rolling mean) - Trend information
        # =========================================================================
        for dish in self.dish_columns:
            features_df[f'{dish}_smooth'] = df[dish].rolling(window=3, min_periods=1).mean()
        
        # Fill NaN values from lags with 0
        features_df = features_df.fillna(0)
        
        self.feature_columns = features_df.columns.tolist()
        
        print(f"✓ Created {len(self.feature_columns)} features")
        print(f"  - Temporal: 5 features")
        print(f"  - Lag (1h,2h,3h): {len(self.dish_columns) * 3} features")
        print(f"  - Smoothed (3h rolling): {len(self.dish_columns)} features")
        
        return features_df, df[self.dish_columns]
    
    def train(self, csv_path):
        """
        Train the multi-output prediction model.
        
        Args:
            csv_path: Path to CSV file with columns [timestamp, dish1, dish2, ...]
            
        Returns:
            dict: Training metrics
        """
        print("\n" + "="*80)
        print("TRAINING DISH PREDICTION MODEL")
        print("="*80)
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"\n✓ Loaded {len(df):,} rows from {csv_path}")
        
        # Create features
        X, y = self.create_features(df)
        
        # Train/test split (80-20 temporal)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"\n✓ Split: Train={len(X_train):,} | Test={len(X_test):,}")
        
        # Initialize model
        if self.model_type == 'catboost':
            base_model = CatBoostRegressor(
                iterations=300,
                depth=5,
                learning_rate=0.1,
                verbose=0,
                random_state=42
            )
        else:  # xgboost
            base_model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        
        self.model = MultiOutputRegressor(base_model)
        
        # Train
        print(f"\n✓ Training {self.model_type.upper()} model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        self.metrics = {
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'num_dishes': len(self.dish_columns),
            'num_features': len(self.feature_columns),
            'num_samples': len(df)
        }
        
        self.is_trained = True
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Overall Train R²: {train_r2:.4f} | MAE: {train_mae:.2f}")
        print(f"Overall Test R²:  {test_r2:.4f} | MAE: {test_mae:.2f}")
        
        # Per-dish metrics
        print(f"\nPer-Dish Test Metrics:")
        for i, dish in enumerate(self.dish_columns):
            dish_r2 = r2_score(y_test.iloc[:, i], y_test_pred[:, i])
            dish_mae = mean_absolute_error(y_test.iloc[:, i], y_test_pred[:, i])
            print(f"  {dish[:50]:50s} | R²={dish_r2:.4f} | MAE={dish_mae:.2f}")
        
        return self.metrics
    
    def predict(self, timestamp_str):
        """
        Predict dish demand for a specific hour.
        
        Args:
            timestamp_str: Timestamp string (e.g., "2024-11-09 14:00:00")
            
        Returns:
            dict: Predictions for each dish
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        timestamp = pd.to_datetime(timestamp_str)
        
        # Create features for prediction
        # Note: For real prediction, we'd need recent historical data
        # For now, using zero lag features (should be improved with actual history)
        features = pd.DataFrame({
            'hour': [timestamp.hour],
            'day_of_week': [timestamp.dayofweek],
            'is_weekend': [1 if timestamp.dayofweek >= 5 else 0],
            'sin_hour': [np.sin(2 * np.pi * timestamp.hour / 24)],
            'cos_hour': [np.cos(2 * np.pi * timestamp.hour / 24)]
        })
        
        # Add zero lag and smooth features (placeholder)
        for dish in self.dish_columns:
            for lag in [1, 2, 3]:
                features[f'{dish}_lag{lag}'] = 0
            features[f'{dish}_smooth'] = 0
        
        # Ensure column order matches training
        features = features[self.feature_columns]
        
        # Predict
        predictions = self.model.predict(features)[0]
        
        # Format results
        results = {}
        for dish, pred in zip(self.dish_columns, predictions):
            results[dish] = max(0, float(pred))  # Ensure non-negative
        
        return results
    
    def save(self, filepath):
        """Save model to disk."""
        model_data = {
            'model': self.model,
            'dish_columns': self.dish_columns,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'is_trained': self.is_trained,
            'top_n_dishes': self.top_n_dishes,
            'model_type': self.model_type
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.dish_columns = model_data['dish_columns']
        self.feature_columns = model_data['feature_columns']
        self.metrics = model_data['metrics']
        self.is_trained = model_data['is_trained']
        self.top_n_dishes = model_data.get('top_n_dishes', 10)
        self.model_type = model_data.get('model_type', 'catboost')
        
        print(f"\n✓ Model loaded from {filepath}")
        print(f"  Dishes: {len(self.dish_columns)}")
        print(f"  Features: {len(self.feature_columns)}")
        print(f"  Test R²: {self.metrics.get('test_r2', 'N/A')}")
