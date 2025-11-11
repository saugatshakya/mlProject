"""
Demand Prediction Model Wrapper
================================

Hourly order volume prediction using enhanced XGBoost model.
Based on comprehensive analysis - enhanced model achieves R² = 0.9558, MAE = 0.765.
Includes advanced feature engineering: lags, rolling statistics, and pattern features.
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
    """Wrapper for enhanced demand prediction model."""

    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.metrics = {}
        self.trained = False
        self.model_path = Path('models/demand_prediction.pkl')
        self.data_path = None
        self.pattern_stats = {}  # Store pattern statistics for prediction
        
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
        Create enhanced features for demand prediction.
        Based on comprehensive analysis - enhanced features achieve R² = 0.9558.
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

        # Extract temporal components
        df['order_date'] = df['timestamp'].dt.date
        df['order_hour'] = df['timestamp'].dt.hour

        # Rename total_orders to orders_per_hour for consistency
        df['orders_per_hour'] = df['total_orders']

        # Basic temporal features (app_v2)
        df['hour'] = df['order_hour']
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Cyclic encoding (app_v2)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Time-based features (app_v2)
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 23)).astype(int)

        # Lag features (app_v2 + enhanced)
        df['total_orders_lag_1h'] = df['orders_per_hour'].shift(1)
        df['total_orders_lag_24h'] = df['orders_per_hour'].shift(24)
        # Additional lags for enhanced performance
        df['total_orders_lag_2h'] = df['orders_per_hour'].shift(2)
        df['total_orders_lag_3h'] = df['orders_per_hour'].shift(3)
        df['total_orders_lag_48h'] = df['orders_per_hour'].shift(48)
        df['total_orders_lag_168h'] = df['orders_per_hour'].shift(168)  # 1 week

        # Rolling window features (safe - no leakage)
        s = df['orders_per_hour'].shift(1)  # Use shifted series for rolling stats

        # Rolling means (app_v2 + enhanced)
        df['total_orders_rolling_6h'] = s.rolling(window=6, min_periods=1).mean()
        df['total_orders_rolling_24h'] = s.rolling(window=24, min_periods=1).mean()
        df['total_orders_rolling_3h'] = s.rolling(window=3, min_periods=1).mean()
        df['total_orders_rolling_12h'] = s.rolling(window=12, min_periods=1).mean()

        # Rolling standard deviations (enhanced)
        df['total_orders_rolling_6h_std'] = s.rolling(window=6, min_periods=1).std()
        df['total_orders_rolling_24h_std'] = s.rolling(window=24, min_periods=1).std()

        # Fill NaN for std columns
        df['total_orders_rolling_6h_std'] = df['total_orders_rolling_6h_std'].fillna(0)
        df['total_orders_rolling_24h_std'] = df['total_orders_rolling_24h_std'].fillna(0)

        # Fill all remaining NaN values
        df = df.fillna(0)

        return df

    def create_train_pattern_features(self, df):
        """Create pattern features using ONLY training data to avoid leakage."""
        # Calculate patterns ONLY from training data
        hourly_avg = df.groupby('order_hour')['orders_per_hour'].mean().to_dict()
        dow_avg = df.groupby('day_of_week')['orders_per_hour'].mean().to_dict()

        # Hour-day combination patterns
        hour_dow_avg = df.groupby(['order_hour', 'day_of_week'])['orders_per_hour'].mean().to_dict()

        return {
            'hourly_avg': hourly_avg,
            'dow_avg': dow_avg,
            'hour_dow_avg': hour_dow_avg
        }

    def apply_pattern_features(self, df, pattern_stats):
        """Apply pre-calculated pattern features to any dataset."""
        df = df.copy()

        # Apply patterns calculated from training data only
        df['hour_avg_orders'] = df['order_hour'].map(pattern_stats['hourly_avg'])
        df['dow_avg_orders'] = df['day_of_week'].map(pattern_stats['dow_avg'])

        # Hour + day of week combination
        df['hour_dow_avg_orders'] = df.apply(
            lambda row: pattern_stats['hour_dow_avg'].get(
                (row['order_hour'], row['day_of_week']),
                pattern_stats['hourly_avg'].get(row['order_hour'], 0)
            ), axis=1
        )

        return df

    def train(self, filepath):
        """
        Train enhanced demand prediction model.

        Args:
            filepath: Path to CSV file with timestamp and total_orders columns

        Returns:
            dict: Training results with metrics
        """
        print(f"\n{'='*70}")
        print("TRAINING ENHANCED DEMAND PREDICTION MODEL")
        print(f"{'='*70}")

        # Load data
        print(f"\nLoading data from: {filepath}")
        df = pd.read_csv(filepath, quoting=1)
        self.data_path = filepath

        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Create enhanced features
        print("\nCreating enhanced features...")
        df_features = self.create_features(df)

        # Train/Test Split (time-aware - no future data in training)
        split_idx = int(len(df_features) * 0.8)
        train_df = df_features.iloc[:split_idx].copy()
        test_df = df_features.iloc[split_idx:].copy()

        print(f"✅ Train/Test split: Train {len(train_df)} hours, Test {len(test_df)} hours")
        print(f"   Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        print(f"   Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

        # Compute pattern features ONLY on training data (to avoid leakage)
        self.pattern_stats = self.create_train_pattern_features(train_df)
        print("✅ Pattern statistics computed from training data only")

        # Apply pattern features to both train and test
        train_df = self.apply_pattern_features(train_df, self.pattern_stats)
        test_df = self.apply_pattern_features(test_df, self.pattern_stats)
        print("✅ Pattern features applied to train and test sets")

        # Define enhanced feature set
        enhanced_features = [
            # App_v2 features
            'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'is_morning', 'is_afternoon', 'is_evening',
            'total_orders_lag_1h', 'total_orders_lag_24h',
            'total_orders_rolling_6h', 'total_orders_rolling_24h',
            # Additional lags
            'total_orders_lag_2h', 'total_orders_lag_3h', 'total_orders_lag_48h', 'total_orders_lag_168h',
            # Additional rolling means
            'total_orders_rolling_3h', 'total_orders_rolling_12h',
            # Rolling stds
            'total_orders_rolling_6h_std', 'total_orders_rolling_24h_std',
            # Pattern features
            'hour_avg_orders', 'dow_avg_orders', 'hour_dow_avg_orders'
        ]

        self.feature_columns = enhanced_features

        X_train = train_df[enhanced_features]
        X_test = test_df[enhanced_features]
        y_train = train_df['orders_per_hour']
        y_test = test_df['orders_per_hour']

        print(f"\nFeatures: {len(enhanced_features)}")
        print(f"Feature list: {enhanced_features[:5]}... ({len(enhanced_features)} total)")

        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        # Train enhanced XGBoost model (best configuration from analysis)
        print("\nTraining Enhanced XGBoost model...")

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
        
        print(f"\nTest Performance:")
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
        Predict demand for next N hours using enhanced features.

        Args:
            hours: Number of hours to predict

        Returns:
            dict: Predictions for each hour
        """
        if not self.trained:
            raise ValueError("Model not trained yet")

        # Load latest data
        df = pd.read_csv(self.data_path, quoting=1)
        df_features = self.create_features(df)

        # Apply pattern features using stored training patterns
        df_features = self.apply_pattern_features(df_features, self.pattern_stats)

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

            # Get lag features from data and previous predictions
            if i == 1:
                lag_1h = df_features['orders_per_hour'].iloc[-1]
                lag_2h = df_features['orders_per_hour'].iloc[-2] if len(df_features) >= 2 else 0
                lag_3h = df_features['orders_per_hour'].iloc[-3] if len(df_features) >= 3 else 0
                lag_24h = df_features['orders_per_hour'].iloc[-24] if len(df_features) >= 24 else 0
                lag_48h = df_features['orders_per_hour'].iloc[-48] if len(df_features) >= 48 else 0
                lag_168h = df_features['orders_per_hour'].iloc[-168] if len(df_features) >= 168 else 0

                rolling_3h = df_features['orders_per_hour'].iloc[-3:].mean() if len(df_features) >= 3 else df_features['orders_per_hour'].iloc[-1]
                rolling_6h = df_features['orders_per_hour'].iloc[-6:].mean() if len(df_features) >= 6 else df_features['orders_per_hour'].iloc[-1]
                rolling_12h = df_features['orders_per_hour'].iloc[-12:].mean() if len(df_features) >= 12 else df_features['orders_per_hour'].iloc[-1]
                rolling_24h = df_features['orders_per_hour'].iloc[-24:].mean() if len(df_features) >= 24 else df_features['orders_per_hour'].iloc[-1]

                rolling_6h_std = df_features['orders_per_hour'].iloc[-6:].std() if len(df_features) >= 6 else 0
                rolling_24h_std = df_features['orders_per_hour'].iloc[-24:].std() if len(df_features) >= 24 else 0
            else:
                # Use previous predictions
                lag_1h = predictions[-1]['predicted_orders']
                lag_2h = predictions[-2]['predicted_orders'] if len(predictions) >= 2 else lag_1h
                lag_3h = predictions[-3]['predicted_orders'] if len(predictions) >= 3 else lag_1h
                lag_24h = df_features['orders_per_hour'].iloc[-(24-i+1)] if len(df_features) >= (24-i+1) else 0
                lag_48h = df_features['orders_per_hour'].iloc[-(48-i+1)] if len(df_features) >= (48-i+1) else 0
                lag_168h = df_features['orders_per_hour'].iloc[-(168-i+1)] if len(df_features) >= (168-i+1) else 0

                # Rolling calculations using previous predictions
                recent_orders = [p['predicted_orders'] for p in predictions[max(0, i-24):]] + list(df_features['orders_per_hour'].iloc[-(24-i):])
                rolling_3h = np.mean(recent_orders[-3:]) if recent_orders else lag_1h
                rolling_6h = np.mean(recent_orders[-6:]) if recent_orders else lag_1h
                rolling_12h = np.mean(recent_orders[-12:]) if recent_orders else lag_1h
                rolling_24h = np.mean(recent_orders) if recent_orders else lag_1h

                rolling_6h_std = np.std(recent_orders[-6:]) if len(recent_orders) >= 6 else 0
                rolling_24h_std = np.std(recent_orders) if recent_orders else 0

            # Get pattern features using stored training patterns
            hour_avg_orders = self.pattern_stats['hourly_avg'].get(hour, 0)
            dow_avg_orders = self.pattern_stats['dow_avg'].get(day_of_week, 0)
            hour_dow_avg_orders = self.pattern_stats['hour_dow_avg'].get((hour, day_of_week), hour_avg_orders)

            # Create feature vector with all enhanced features
            X_pred = pd.DataFrame([{
                # App_v2 features
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
                'total_orders_rolling_24h': rolling_24h,
                # Additional lags
                'total_orders_lag_2h': lag_2h,
                'total_orders_lag_3h': lag_3h,
                'total_orders_lag_48h': lag_48h,
                'total_orders_lag_168h': lag_168h,
                # Additional rolling means
                'total_orders_rolling_3h': rolling_3h,
                'total_orders_rolling_12h': rolling_12h,
                # Rolling stds
                'total_orders_rolling_6h_std': rolling_6h_std,
                'total_orders_rolling_24h_std': rolling_24h_std,
                # Pattern features
                'hour_avg_orders': hour_avg_orders,
                'dow_avg_orders': dow_avg_orders,
                'hour_dow_avg_orders': hour_dow_avg_orders
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
