"""
Demand Prediction Model Wrapper
================================

Hourly order volume prediction using temporal features only.
Based on ablation study - temporal-only model achieves R² = 0.8647.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


class DemandPredictionModel:
    """Wrapper for demand prediction model."""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.metrics = {}
        self.trained = False
        self.model_path = Path('models/demand_prediction.pkl')
        self.data_path = None
        
    def is_trained(self):
        """Check if model is trained."""
        return self.trained
    
    def get_status(self):
        """Get model status."""
        return {
            'trained': self.trained,
            'metrics': self.metrics,
            'num_features': len(self.feature_columns) if self.feature_columns else 0
        }
    
    def create_features(self, df):
        """
        Create temporal features for demand prediction.
        Based on ablation study - using ONLY temporal features (R² = 0.8647).
        """
        df = df.copy()
        
        # Handle different timestamp column names
        if 'order_placed_at' in df.columns:
            df['timestamp'] = pd.to_datetime(df['order_placed_at'])
            df = df.drop('order_placed_at', axis=1)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("CSV must have 'timestamp' or 'order_placed_at' column")
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Basic temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclic encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Time-based features
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 23)).astype(int)
        
        # Lag features for total_orders
        df['total_orders_lag_1h'] = df['total_orders'].shift(1)
        df['total_orders_lag_24h'] = df['total_orders'].shift(24)
        df['total_orders_rolling_6h'] = df['total_orders'].rolling(window=6, min_periods=1).mean()
        df['total_orders_rolling_24h'] = df['total_orders'].rolling(window=24, min_periods=1).mean()
        
        # Fill NaN
        df = df.fillna(0)
        
        return df
    
    def train(self, filepath):
        """
        Train demand prediction model.
        
        Args:
            filepath: Path to CSV file with timestamp and total_orders columns
            
        Returns:
            dict: Training results with metrics
        """
        print(f"\n{'='*70}")
        print("TRAINING DEMAND PREDICTION MODEL")
        print(f"{'='*70}")
        
        # Load data
        print(f"\nLoading data from: {filepath}")
        df = pd.read_csv(filepath)
        self.data_path = filepath
        
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Create features
        print("\nCreating temporal features...")
        df_features = self.create_features(df)
        
        # Prepare X and y
        feature_cols = [col for col in df_features.columns if col not in ['timestamp', 'total_orders']]
        self.feature_columns = feature_cols
        
        X = df_features[feature_cols]
        y = df_features['total_orders']
        
        print(f"Features: {len(feature_cols)}")
        print(f"Feature list: {feature_cols[:5]}... ({len(feature_cols)} total)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Time series - no shuffle
        )
        
        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Train model
        print("\nTraining XGBoost model...")
        
        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print("\nEvaluating model...")
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.metrics = {
            'r2_score': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"\nModel Performance:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
        # Save model
        self.model_path.parent.mkdir(exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_columns': self.feature_columns,
                'metrics': self.metrics
            }, f)
        
        print(f"\nModel saved to: {self.model_path}")
        self.trained = True
        
        print(f"{'='*70}\n")
        
        return {
            'status': 'success',
            'metrics': self.metrics,
            'num_features': len(self.feature_columns)
        }
    
    def predict(self, hours=1):
        """
        Predict demand for next N hours.
        
        Args:
            hours: Number of hours to predict
            
        Returns:
            dict: Predictions for each hour
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
        
        # Load latest data
        df = pd.read_csv(self.data_path)
        df_features = self.create_features(df)
        
        # Get last timestamp
        last_timestamp = pd.to_datetime(df_features['timestamp'].iloc[-1])
        
        # Prepare predictions
        predictions = []
        
        for i in range(1, hours + 1):
            # Create features for next hour
            next_timestamp = last_timestamp + timedelta(hours=i)
            
            # Extract temporal features
            hour = next_timestamp.hour
            day_of_week = next_timestamp.dayofweek
            day_of_month = next_timestamp.day
            month = next_timestamp.month
            is_weekend = 1 if day_of_week >= 5 else 0
            
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_cos = np.cos(2 * np.pi * day_of_week / 7)
            
            is_morning = 1 if 6 <= hour < 12 else 0
            is_afternoon = 1 if 12 <= hour < 18 else 0
            is_evening = 1 if 18 <= hour < 23 else 0
            
            # Get lag features from data
            if i == 1:
                lag_1h = df_features['total_orders'].iloc[-1]
                lag_24h = df_features['total_orders'].iloc[-24] if len(df_features) >= 24 else 0
                rolling_6h = df_features['total_orders'].iloc[-6:].mean()
                rolling_24h = df_features['total_orders'].iloc[-24:].mean() if len(df_features) >= 24 else 0
            else:
                # Use previous prediction
                lag_1h = predictions[-1]['predicted_orders']
                lag_24h = df_features['total_orders'].iloc[-(24-i+1)] if len(df_features) >= (24-i+1) else 0
                rolling_6h = np.mean([p['predicted_orders'] for p in predictions[max(0, i-6):]] + list(df_features['total_orders'].iloc[-(6-i):]))
                rolling_24h = np.mean([p['predicted_orders'] for p in predictions] + list(df_features['total_orders'].iloc[-(24-i):]))
            
            # Create feature vector
            X_pred = pd.DataFrame([{
                'hour': hour,
                'day_of_week': day_of_week,
                'day_of_month': day_of_month,
                'month': month,
                'is_weekend': is_weekend,
                'hour_sin': hour_sin,
                'hour_cos': hour_cos,
                'day_sin': day_sin,
                'day_cos': day_cos,
                'is_morning': is_morning,
                'is_afternoon': is_afternoon,
                'is_evening': is_evening,
                'total_orders_lag_1h': lag_1h,
                'total_orders_lag_24h': lag_24h,
                'total_orders_rolling_6h': rolling_6h,
                'total_orders_rolling_24h': rolling_24h
            }])
            
            # Ensure correct column order
            X_pred = X_pred[self.feature_columns]
            
            # Predict
            y_pred = self.model.predict(X_pred)[0]
            
            predictions.append({
                'hour': hour,
                'timestamp': next_timestamp.strftime('%Y-%m-%d %H:00'),
                'predicted_orders': max(0, float(y_pred))  # No negative orders
            })
        
        return {
            'predictions': predictions,
            'total_predicted_orders': sum(p['predicted_orders'] for p in predictions)
        }
