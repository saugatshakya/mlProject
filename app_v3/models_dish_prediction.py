"""
Dish Prediction Model Wrapper (updated)
---------------------------------------

This wrapper now mirrors the feature engineering and training strategy used in
the original `dish_prediction/src/models/final_model.py` script:

- Top-N dish selection by volume (default top 10)
- Temporal features (hour, day_of_week, is_weekend, cyclical encodings)
- Lag features (1,2,3 hours)
- 3-hour rolling smoothing for dish history
- Optional external features (env/weather, pollution, events) if present
- Trains both CatBoost and XGBoost multi-output models and keeps CatBoost as
  the primary model (usually best performer)

The public class API is unchanged: `train(filepath)` and `predict(hour)` keep
the same signatures so the Flask app can call them without modification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

import xgboost as xgb


class DishPredictionModel:
    """Wrapper for dish prediction model (compatible with app endpoints)."""

    def __init__(self, top_n=10):
        self.model = None  # will hold the chosen best model (CatBoost if available)
        self.models = {}   # dict holding trained models (CatBoost, XGBoost)
        self.dish_columns = None
        self.feature_columns = None
        self.metrics = {}
        self.trained = False
        self.model_dir = Path('models/final')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = None
        self.top_n = top_n

    def is_trained(self):
        return self.trained

    def get_status(self):
        return {
            'trained': self.trained,
            'metrics': self.metrics,
            'num_dishes': len(self.dish_columns) if self.dish_columns else 0,
            'num_features': len(self.feature_columns) if self.feature_columns else 0
        }

    def _prepare_dataframe(self, df):
        """Normalize and ensure timestamp column exists and sorted."""
        df = df.copy()
        if 'order_placed_at' in df.columns:
            df['hour'] = pd.to_datetime(df['order_placed_at'])
        elif 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp'])
        elif 'hour' in df.columns:
            # assume pre-processed hourly index
            df['hour'] = pd.to_datetime(df['hour'])
        else:
            raise ValueError("CSV must have 'timestamp' or 'order_placed_at' or 'hour' column")

        df = df.sort_values('hour').reset_index(drop=True)
        return df

    def create_features(self, df):
        """Create features aligned with the original final_model pipeline."""
        df = self._prepare_dataframe(df)

        # temporal
        features = pd.DataFrame(index=df.index)
        features['hour'] = df['hour'].dt.hour
        features['day_of_week'] = df['hour'].dt.dayofweek
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['sin_hour'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['cos_hour'] = np.cos(2 * np.pi * features['hour'] / 24)

        # identify dish columns by excluding known meta/env columns
        exclude = set(['hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'week_of_year',
                       'month', 'date', 'is_weekend', 'is_friday', 'is_peak_hour',
                       'is_lunch_rush', 'is_dinner_rush', 'is_late_night', 'meal_period',
                       'sin_hour', 'cos_hour', 'hour_sin', 'hour_cos',
                       # environment/pollution/events columns commonly present
                       'env_temp', 'env_rhum', 'env_precip', 'env_wspd', 'env_condition',
                       'aqi', 'pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2', 'nh3',
                       'event', 'holiday', 'has_event', 'timestamp', 'order_placed_at'])

        dish_cols = [c for c in df.columns if c not in exclude and not c.startswith('Unnamed')]

        # If there are many dishes, pick top_n by total volume (as original)
        if len(dish_cols) > self.top_n:
            volumes = df[dish_cols].sum().sort_values(ascending=False)
            top_dishes = volumes.head(self.top_n).index.tolist()
            dish_cols = top_dishes

        # LAG features 1,2,3 hours for each dish
        for dish in dish_cols:
            for lag in [1, 2, 3]:
                features[f'{dish}_lag{lag}'] = df[dish].shift(lag).fillna(0)

        # 3-hour smoothed history
        for dish in dish_cols:
            features[f'{dish}_smooth3'] = df[dish].rolling(window=3, min_periods=1).mean()

        # optional external features
        ext_cols = ['env_temp', 'env_rhum', 'env_precip', 'env_wspd']
        for c in ext_cols:
            if c in df.columns:
                features[c] = df[c]

        if 'aqi' in df.columns:
            features['aqi'] = df['aqi']

        if 'has_event' in df.columns:
            features['has_event'] = df['has_event'].astype(int)
        if 'holiday' in df.columns:
            features['holiday'] = df['holiday'].astype(int)

        features = features.fillna(0)
        return features, dish_cols

    def train(self, filepath):
        """Train both CatBoost and XGBoost multi-output models and keep CatBoost."""
        print('\n' + '='*70)
        print('TRAINING DISH PREDICTION MODEL (final-style)')
        print('='*70)

        df = pd.read_csv(filepath, quoting=1)
        self.data_path = filepath

        features_df, dish_cols = self.create_features(df)
        self.dish_columns = dish_cols

        X = features_df
        y = df[dish_cols]

        # Train/test temporal split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | Dishes: {len(dish_cols)}")

        # Prepare models
        models = {}
        if CatBoostRegressor is not None:
            models['CatBoost'] = MultiOutputRegressor(
                CatBoostRegressor(
                    iterations=300,
                    depth=5,
                    learning_rate=0.1,
                    verbose=0,
                    random_state=42
                )
            )
        models['XGBoost'] = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        )

        results = []
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            print(f"  Test R²: {test_r2:.4f}, Test MAE: {test_mae:.4f}")

            # per-dish metrics
            dish_metrics = []
            for i, dish in enumerate(dish_cols):
                dish_r2 = r2_score(y_test[dish], y_test_pred[:, i])
                dish_mae = mean_absolute_error(y_test[dish], y_test_pred[:, i])
                dish_metrics.append({'model': name, 'dish': dish, 'test_r2': float(dish_r2), 'test_mae': float(dish_mae)})
                print(f"    {dish}: R² = {dish_r2:.4f}, MAE = {dish_mae:.4f}")

            results.extend(dish_metrics)

            # Save model artifact
            save_name = f"{name.lower()}_multi_output.pkl"
            joblib.dump(model, self.model_dir / save_name)
            print(f"  ✓ Saved {name} to {self.model_dir / save_name}")

            # keep in memory
            self.models[name] = model

        # choose best model (prefer CatBoost if available)
        if 'CatBoost' in self.models:
            self.model = self.models['CatBoost']
            chosen = 'CatBoost'
        else:
            self.model = self.models['XGBoost']
            chosen = 'XGBoost'

        # aggregate overall metrics for chosen model using test predictions
        y_test_pred = self.model.predict(X_test)
        overall_r2 = r2_score(y_test, y_test_pred)
        overall_mae = mean_absolute_error(y_test, y_test_pred)
        overall_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))

        self.metrics = {
            'model_chosen': chosen,
            'r2_score': float(overall_r2),
            'mae': float(overall_mae),
            'rmse': overall_rmse,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'per_dish': results
        }

        # Save feature names and dish names
        with open(self.model_dir / 'feature_names.txt', 'w') as f:
            for feat in X.columns:
                f.write(f"{feat}\n")
        with open(self.model_dir / 'dish_names.txt', 'w') as f:
            for d in dish_cols:
                f.write(f"{d}\n")

        self.feature_columns = list(X.columns)
        self.trained = True

        print('\n' + '='*70 + '\n')

        return {
            'status': 'success',
            'metrics': self.metrics,
            'num_dishes': len(self.dish_columns),
            'num_features': len(self.feature_columns)
        }

    def predict(self, hours_ahead=None):
        """
        Predict dish demand for N hours ahead.
        
        Args:
            hours_ahead: Number of hours into the future (1-24), None means 1 hour
            
        Returns:
            dict: Predictions for each hour with timestamps and dish forecasts
        """
        if not self.trained:
            raise ValueError('Model not trained yet')

        df = pd.read_csv(self.data_path, quoting=1)
        df_with_features = self._prepare_dataframe(df)
        X_all, _ = self.create_features(df)
        
        # Get the latest timestamp from data
        latest_timestamp = df_with_features['hour'].iloc[-1]
        
        # Default to 1 hour
        if hours_ahead is None:
            hours_ahead = 1
        
        from datetime import timedelta
        
        # Generate predictions for each hour from 1 to hours_ahead
        all_predictions = []
        
        for h in range(1, hours_ahead + 1):
            target_timestamp = latest_timestamp + timedelta(hours=h)
            target_hour = target_timestamp.hour
            
            # Find similar hour in historical data or synthesize features
            mask = X_all['hour'] == target_hour
            if mask.sum() == 0:
                # Synthesize features for this hour
                X_pred = X_all[self.feature_columns].iloc[-1:].copy()
                X_pred['hour'] = target_hour
                X_pred['sin_hour'] = np.sin(2 * np.pi * target_hour / 24)
                X_pred['cos_hour'] = np.cos(2 * np.pi * target_hour / 24)
                X_pred['day_of_week'] = target_timestamp.dayofweek
                X_pred['is_weekend'] = 1 if target_timestamp.dayofweek >= 5 else 0
            else:
                X_pred = X_all[self.feature_columns][mask].iloc[-1:]

            y_pred = self.model.predict(X_pred)[0]

            # Create predictions for this hour
            hour_dishes = []
            total_orders = 0
            for dish, val in zip(self.dish_columns, y_pred):
                orders = max(0, float(val))
                hour_dishes.append({'dish': dish, 'predicted_orders': orders})
                total_orders += orders

            hour_dishes = sorted(hour_dishes, key=lambda x: x['predicted_orders'], reverse=True)
            
            all_predictions.append({
                'timestamp': target_timestamp.strftime('%Y-%m-%d %H:%M'),
                'hour': int(target_hour),
                'hours_ahead': h,
                'dishes': hour_dishes,
                'total_predicted_orders': total_orders
            })

        return {
            'hours_ahead': int(hours_ahead),
            'predictions': all_predictions,
            'total_hours': len(all_predictions)
        }
