"""
Prep Time Prediction Model Wrapper
===================================

Kitchen preparation time prediction using gradient boosting.
Based on comprehensive analysis achieving high accuracy on real data.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

import warnings
warnings.filterwarnings('ignore')


class PrepTimePredictionModel:
    """Wrapper for prep time prediction model."""

    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.metrics = {}
        self.trained = False
        self.model_path = Path('models/prep_time_prediction.pkl')
        self.data_path = None
        self.label_encoders = {}
        self.scaler = None

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
        Create features for prep time prediction.
        Based on kitchen-focused feature engineering from the notebook.
        """
        df = df.copy()

        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Peak periods (when kitchen is busiest)
        df['is_lunch_peak'] = df['hour'].between(12, 14).astype(int)
        df['is_dinner_peak'] = df['hour'].between(18, 21).astype(int)
        df['is_late_night'] = df['hour'].between(22, 23).astype(int)
        df['is_early_morning'] = df['hour'].between(0, 6).astype(int)
        df['peak_intensity'] = df['is_lunch_peak'] + df['is_dinner_peak'] + df['is_late_night'] * 0.5

        # Parse items to get order complexity (kitchen-relevant feature)
        def parse_order_complexity(items_str):
            if pd.isna(items_str):
                return 0
            items = str(items_str).split(',')
            total_items = 0
            for item in items:
                if ' x ' in item:
                    try:
                        qty = int(item.split(' x ')[0].strip())
                        total_items += qty
                    except:
                        total_items += 1
                else:
                    total_items += 1
            return total_items

        df['order_complexity'] = df['Items in order'].apply(parse_order_complexity)
        df['order_complexity_log'] = np.log1p(df['order_complexity'])

        # Order size categories
        df['order_size_category'] = pd.cut(df['order_complexity'],
                                          bins=[0, 2, 4, 6, 10, 100],
                                          labels=["tiny", "small", "medium", "large", "xl"])

        # Create dummy variables for order size (drop first to avoid multicollinearity)
        size_dummies = pd.get_dummies(df['order_size_category'], prefix="size", drop_first=True)
        df = pd.concat([df, size_dummies], axis=1)

        # Interaction features (how kitchen factors combine)
        df['peak_x_complexity'] = df['peak_intensity'] * df['order_complexity']
        df['weekend_x_peak'] = df['is_weekend'] * df['peak_intensity']

        # Kitchen workload simulation (simplified - in real implementation would use rolling windows)
        # For now, use peak intensity as proxy for kitchen load
        df['kitchen_load_proxy'] = df['peak_intensity']

        # Encode categorical features that ARE kitchen-relevant
        categorical_cols = ['Order Status']  # Only keep order status as it might affect prep priority
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col + '_encoded'] = self.label_encoders[col].transform(df[col].astype(str))

        # Fill missing values (handle categorical columns properly)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # For categorical columns, fill with most frequent value or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                # Use most frequent value, or 'Unknown' if no mode exists
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                df[col] = df[col].fillna(fill_val)

        return df

    def train(self, filepath):
        """
        Train prep time prediction model.

        Args:
            filepath: Path to CSV file with prep time data

        Returns:
            dict: Training results with metrics
        """
        print(f"\n{'='*70}")
        print("TRAINING PREP TIME PREDICTION MODEL")
        print(f"{'='*70}")

        # Load data
        print(f"\nLoading data from: {filepath}")
        df = pd.read_csv(filepath, quoting=1)
        self.data_path = filepath

        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Create features
        print("\nCreating kitchen-focused features...")
        df_features = self.create_features(df)

        # Define features (exclude target and non-feature columns, only include numeric)
        exclude_cols = ['timestamp', 'order_date', 'KPT duration (minutes)', 'Items in order',
                       'Discount construct', 'Order Status']
        potential_features = [col for col in df_features.columns if col not in exclude_cols]

        # Only include numeric columns
        self.feature_columns = []
        for col in potential_features:
            if df_features[col].dtype in ['int64', 'float64', 'bool']:
                self.feature_columns.append(col)
            elif df_features[col].dtype == 'object':
                # Check if it's actually numeric strings
                try:
                    pd.to_numeric(df_features[col].dropna())
                    self.feature_columns.append(col)
                except:
                    continue

        X = df_features[self.feature_columns]
        y = df_features['KPT duration (minutes)']

        print(f"Features: {len(self.feature_columns)}")
        print(f"Feature list: {self.feature_columns[:5]}... ({len(self.feature_columns)} total)")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train XGBoost model
        print("\nTraining XGBoost model...")
        self.model =HistGradientBoostingRegressor(
        min_samples_leaf=20, max_iter=500, learning_rate=0.05, 
        max_depth=8, random_state=42
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        print("\nEvaluating model...")
        y_pred = self.model.predict(X_test_scaled)

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
                'metrics': self.metrics,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler
            }, f)

        print(f"\nModel saved to: {self.model_path}")
        self.trained = True

        print(f"{'='*70}\n")

        return {
            'status': 'success',
            'metrics': self.metrics,
            'num_features': len(self.feature_columns)
        }

    def predict(self, order_data):
        """
        Predict prep time for an order.

        Args:
            order_data: dict with order features

        Returns:
            dict: Prediction results
        """
        if not self.trained:
            raise ValueError("Model not trained yet")

        # Create DataFrame from order data
        df = pd.DataFrame([order_data])

        # Create features
        df_features = self.create_features(df)

        # Select features
        X = df_features[self.feature_columns]

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        y_pred = self.model.predict(X_scaled)[0]

        return {
            'predicted_prep_time': max(0, float(y_pred)),
            'features_used': len(self.feature_columns)
        }