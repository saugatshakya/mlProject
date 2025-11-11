"""
Promotion Effectiveness Model Wrapper
====================================

Predicts promotion impact on orders and sales using Random Forest.
Based on comprehensive analysis with feature engineering and ablation study.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class PromotionEffectivenessModel:
    """Wrapper for promotion effectiveness prediction model."""

    def __init__(self):
        self.orders_model = None
        self.sales_model = None
        self.feature_columns = None
        self.metrics = {}
        self.trained = False
        self.model_path = Path('models/promotion_effectiveness.pkl')
        self.data_path = None

    def is_trained(self):
        """Check if model is trained."""
        return self.trained

    def load(self):
        """Load trained volume models from promotion_effectiveness directory."""
        try:
            # Try to load volume models from promotion_effectiveness directory
            volume_orders_path = Path('../promotion_effectiveness/models/orders_model.pkl')
            volume_sales_path = Path('../promotion_effectiveness/models/sales_model.pkl')
            
            if volume_orders_path.exists() and volume_sales_path.exists():
                import joblib
                self.orders_model = joblib.load(volume_orders_path)
                self.sales_model = joblib.load(volume_sales_path)
                
                # Define feature columns based on volume model training
                self.feature_columns = [
                    'hour', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos',
                    'temperature_mean', 'precipitation_mean', 'wind_speed_mean', 'is_event_max',
                    'flat_%_active_max', 'flat_rs_active_max', 'buy_1_get_1_active_max', 'buy_7_get_3_active_max'
                ]
                
                self.trained = True
                self.metrics = {'source': 'volume_models'}
                print(f"Loaded volume models from promotion_effectiveness directory")
                return True
            
            # Fallback to original model path
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.orders_model = model_data.get('orders_model')
                self.sales_model = model_data.get('sales_model')
                self.feature_columns = model_data.get('feature_columns', [])
                self.metrics = model_data.get('metrics', {})
                self.trained = model_data.get('trained', False)
                
                print(f"Loaded promotion effectiveness model from {self.model_path}")
                return True
            else:
                print(f"No saved model found at {self.model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_status(self):
        """Get model status."""
        return {
            'trained': self.trained,
            'metrics': self.metrics,
            'num_features': len(self.feature_columns) if self.feature_columns else 0
        }

    def train(self, filepath):
        """
        Train promotion effectiveness models for both orders and sales using HOURLY AGGREGATION.
        This matches the approach from final_promotion_effectiveness_clean.ipynb

        Args:
            filepath: Path to CSV file with promotion data (order-level data)

        Returns:
            dict: Training results with metrics
        """
        print(f"\n{'='*70}")
        print("TRAINING PROMOTION EFFECTIVENESS MODELS (HOURLY AGGREGATION)")
        print(f"{'='*70}")

        # Load order-level promotion data
        print(f"\nLoading order-level data from: {filepath}")
        df = pd.read_csv(filepath, quoting=1)
        self.data_path = filepath

        print(f"Order-level data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Parse timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
        else:
            print("ERROR: No timestamp column found!")
            return {'status': 'error', 'message': 'No timestamp column'}

        # Create promotion active flags
        promotion_cols = ['flat_%', 'flat_rs', 'buy_1_get_1', 'buy_7_get_3']
        for col in promotion_cols:
            if col in df.columns:
                df[f'{col}_active'] = (df[col] > 0).astype(int)
            else:
                df[f'{col}_active'] = 0

        # AGGREGATE BY HOUR (this is the key step from the notebook!)
        print("\n🔄 Aggregating data by hour...")
        agg_dict = {'subtotal': ['count', 'sum']}  # count = num_orders, sum = total_sales
        
        # Aggregate numeric weather columns
        numeric_cols = ['temperature', 'humidity', 'precipitation', 'wind_speed']
        for col in numeric_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'
        
        # Aggregate event flag
        if 'is_event' in df.columns:
            agg_dict['is_event'] = 'max'
        
        # Aggregate promotion active flags
        for promo in promotion_cols:
            active_col = f'{promo}_active'
            if active_col in df.columns:
                agg_dict[active_col] = 'max'

        df_agg = df.groupby(['date', 'hour']).agg(agg_dict).reset_index()
        
        # Flatten multi-level column names
        new_cols = ['date', 'hour']
        for col in df_agg.columns[2:]:
            if isinstance(col, tuple):
                if col[1] == 'count':
                    new_cols.append('num_orders')
                elif col[1] == 'sum':
                    new_cols.append('total_sales')
                else:
                    new_cols.append(f'{col[0]}_{col[1]}')
            else:
                new_cols.append(col)
        df_agg.columns = new_cols

        print(f"Aggregated data shape: {df_agg.shape}")
        print(f"Aggregated columns: {df_agg.columns.tolist()}")

        # Create temporal features
        df_agg['date'] = pd.to_datetime(df_agg['date'])
        df_agg['day_of_week'] = df_agg['date'].dt.dayofweek
        df_agg['is_weekend'] = df_agg['day_of_week'].isin([5, 6]).astype(int)
        df_agg['hour_sin'] = np.sin(2 * np.pi * df_agg['hour'] / 24)
        df_agg['hour_cos'] = np.cos(2 * np.pi * df_agg['hour'] / 24)

        # Build feature list
        self.feature_columns = ['hour', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos']
        
        for col in ['temperature_mean', 'precipitation_mean', 'wind_speed_mean', 'is_event_max'] + \
                   [f'{p}_active_max' for p in promotion_cols]:
            if col in df_agg.columns:
                self.feature_columns.append(col)

        print(f"\nFeatures ({len(self.feature_columns)}): {self.feature_columns}")

        X = df_agg[self.feature_columns]
        y_orders = df_agg['num_orders']
        y_sales = df_agg['total_sales']

        print(f"Training data: {len(X)} hourly records")
        print(f"Orders range: {y_orders.min():.0f} - {y_orders.max():.0f}")
        print(f"Sales range: ₹{y_sales.min():.0f} - ₹{y_sales.max():.0f}")

        # Train/test split
        X_train, X_test, y_train_ord, y_test_ord = train_test_split(
            X, y_orders, test_size=0.2, random_state=42
        )
        _, _, y_train_sal, y_test_sal = train_test_split(
            X, y_sales, test_size=0.2, random_state=42
        )

        # Train models (using same hyperparameters as notebook)
        print("\n🎯 Training Orders Model...")
        self.orders_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        self.orders_model.fit(X_train, y_train_ord)
        y_pred_ord = self.orders_model.predict(X_test)
        
        orders_r2 = r2_score(y_test_ord, y_pred_ord)
        orders_mae = mean_absolute_error(y_test_ord, y_pred_ord)
        orders_rmse = np.sqrt(mean_squared_error(y_test_ord, y_pred_ord))

        print(f"Orders Model - R²: {orders_r2:.4f}, MAE: {orders_mae:.2f}, RMSE: {orders_rmse:.2f}")

        print("\n🎯 Training Sales Model...")
        self.sales_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        self.sales_model.fit(X_train, y_train_sal)
        y_pred_sal = self.sales_model.predict(X_test)
        
        sales_r2 = r2_score(y_test_sal, y_pred_sal)
        sales_mae = mean_absolute_error(y_test_sal, y_pred_sal)
        sales_rmse = np.sqrt(mean_squared_error(y_test_sal, y_pred_sal))

        print(f"Sales Model - R²: {sales_r2:.4f}, MAE: {sales_mae:.2f}, RMSE: {sales_rmse:.2f}")

        # Store metrics
        self.metrics = {
            'orders': {'r2': orders_r2, 'mae': orders_mae, 'rmse': orders_rmse},
            'sales': {'r2': sales_r2, 'mae': sales_mae, 'rmse': sales_rmse}
        }

        # Save models
        print(f"\n💾 Saving models to: {self.model_path}")
        self.model_path.parent.mkdir(exist_ok=True)

        model_data = {
            'orders_model': self.orders_model,
            'sales_model': self.sales_model,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'trained': True
        }

        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)

        self.trained = True
        print(f"✅ Models saved successfully!")
        print(f"{'='*70}\n")

        return {
            'status': 'success',
            'metrics': self.metrics,
            'num_features': len(self.feature_columns)
        }

        # Train sales model
        if 'sales_per_hour' in df_features.columns:
            y_sales = df_features['sales_per_hour']
            print(f"\nTraining Sales Model...")
            print(f"Sales range: {y_sales.min():.0f} - {y_sales.max():.0f}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_sales, test_size=0.2, random_state=42
            )

            self.sales_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )

            self.sales_model.fit(X_train, y_train)
            y_pred = self.sales_model.predict(X_test)

            sales_r2 = r2_score(y_test, y_pred)
            sales_mae = mean_absolute_error(y_test, y_pred)
            sales_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            print(f"Sales Model - R²: {sales_r2:.3f}, MAE: {sales_mae:.2f}, RMSE: {sales_rmse:.2f}")

            self.metrics['sales'] = {
                'r2': sales_r2,
                'mae': sales_mae,
                'rmse': sales_rmse
            }

            print(f"Orders Model - R²: {orders_r2:.4f}, MAE: {orders_mae:.4f}, RMSE: {orders_rmse:.4f}")

        # Train sales model
        if 'sales_per_hour' in df_features.columns:
            y_sales = df_features['sales_per_hour']
            print(f"\nTraining Sales Model...")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_sales, test_size=0.2, random_state=42
            )

            self.sales_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )

            self.sales_model.fit(X_train, y_train)
            y_pred = self.sales_model.predict(X_test)

            sales_r2 = r2_score(y_test, y_pred)
            sales_mae = mean_absolute_error(y_test, y_pred)
            sales_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            print(f"Sales Model - R²: {sales_r2:.4f}, MAE: {sales_mae:.4f}, RMSE: {sales_rmse:.4f}")

        # Store metrics
        self.metrics = {
            'orders_model_r2': orders_r2 if 'orders_per_hour' in df_features.columns else None,
            'orders_model_mae': orders_mae if 'orders_per_hour' in df_features.columns else None,
            'orders_model_rmse': orders_rmse if 'orders_per_hour' in df_features.columns else None,
            'sales_model_r2': sales_r2 if 'sales_per_hour' in df_features.columns else None,
            'sales_model_mae': sales_mae if 'sales_per_hour' in df_features.columns else None,
            'sales_model_rmse': sales_rmse if 'sales_per_hour' in df_features.columns else None,
            'num_features': len(self.feature_columns)
        }

        # Save models
        self.model_path.parent.mkdir(exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'orders_model': self.orders_model,
                'sales_model': self.sales_model,
                'feature_columns': self.feature_columns,
                'metrics': self.metrics
            }, f)

        print(f"\nModels saved to: {self.model_path}")
        self.trained = True

        print(f"{'='*70}\n")

        return {
            'status': 'success',
            'metrics': self.metrics,
            'num_features': len(self.feature_columns)
        }

    def predict_promotion_impact(self, promotion_data):
        """
        Predict promotion impact using the volume prediction approach from the notebook.
        Predicts hourly orders and sales for a promotion period, comparing with baseline (no promo).
        
        Args:
            promotion_data: dict with promotion details including start_date, duration, hours, etc.
            
        Returns:
            dict: Predicted orders, sales, baseline, and impact metrics
        """
        if not self.trained:
            raise ValueError("Model not trained yet")

        # Extract promotion period details
        if 'start_date' in promotion_data:
            start_date = pd.to_datetime(promotion_data['start_date'])
            duration_days = promotion_data.get('duration_days', 1)
            start_hour = promotion_data.get('start_hour', 12)
            end_hour = promotion_data.get('end_hour', 14)
            
            # Generate all dates in the promotion period
            promotion_dates = [start_date + pd.Timedelta(days=i) for i in range(duration_days)]
        else:
            # Legacy single-day format
            start_hour = promotion_data.get('start_hour', 12)
            end_hour = promotion_data.get('end_hour', start_hour + 2)
            day_of_month = promotion_data.get('day_of_month', 15)
            month = promotion_data.get('month', 6)
            year = promotion_data.get('year', 2024)
            promotion_dates = [pd.Timestamp(year=year, month=month, day=day_of_month)]
        
        # Weather data (same for all hours)
        temperature = promotion_data.get('temperature', 25.0)
        precipitation = promotion_data.get('precipitation', 0.0)
        wind_speed = promotion_data.get('wind_speed', 5.0)
        is_event = promotion_data.get('is_event', 0)
        
        # Determine which promotion is active
        flat_percent_active = 0
        flat_rs_active = 0
        buy_1_get_1_active = 0
        buy_7_get_3_active = 0
        
        # Check direct flags first
        if 'flat_%' in promotion_data and promotion_data['flat_%'] > 0:
            flat_percent_active = 1
        if 'flat_rs' in promotion_data and promotion_data['flat_rs'] > 0:
            flat_rs_active = 1
        if 'buy_1_get_1' in promotion_data and promotion_data['buy_1_get_1'] > 0:
            buy_1_get_1_active = 1
        if 'buy_7_get_3' in promotion_data and promotion_data['buy_7_get_3'] > 0:
            buy_7_get_3_active = 1
        
        # Accumulate predictions across all hours in the promotion period
        total_predicted_orders = 0
        total_predicted_sales = 0
        total_baseline_orders = 0
        total_baseline_sales = 0
        all_promotion_hours = []
        
        for current_date in promotion_dates:
            day_of_week = current_date.dayofweek
            is_weekend = 1 if day_of_week in [5, 6] else 0
            
            # Handle hour wrap-around (e.g., 22-02 wraps around midnight)
            hours_in_day = []
            current_hour = start_hour
            while True:
                hours_in_day.append(current_hour)
                if current_hour == end_hour:
                    break
                current_hour = (current_hour + 1) % 24
                if len(hours_in_day) > 24:
                    break
            
            for hour in hours_in_day:
                all_promotion_hours.append(f"{current_date.strftime('%Y-%m-%d')} {hour:02d}:00")
                
                # Create feature dict matching the notebook approach
                feature_dict = {
                    'hour': hour,
                    'day_of_week': day_of_week,
                    'is_weekend': is_weekend,
                    'hour_sin': np.sin(2 * np.pi * hour / 24),
                    'hour_cos': np.cos(2 * np.pi * hour / 24),
                    'temperature_mean': temperature,
                    'precipitation_mean': precipitation,
                    'wind_speed_mean': wind_speed,
                    'is_event_max': is_event,
                    'flat_%_active_max': flat_percent_active,
                    'flat_rs_active_max': flat_rs_active,
                    'buy_1_get_1_active_max': buy_1_get_1_active,
                    'buy_7_get_3_active_max': buy_7_get_3_active
                }
                
                # Create DataFrame with only the features the model was trained on
                X_pred = pd.DataFrame([{k: v for k, v in feature_dict.items() if k in self.feature_columns}])
                
                # Predict WITH promotion
                if self.orders_model:
                    hourly_orders = self.orders_model.predict(X_pred)[0]
                    total_predicted_orders += max(0, float(hourly_orders))
                
                if self.sales_model:
                    hourly_sales = self.sales_model.predict(X_pred)[0]
                    total_predicted_sales += max(0, float(hourly_sales))
                
                # Predict baseline (WITHOUT promotion)
                baseline_dict = feature_dict.copy()
                baseline_dict.update({
                    'flat_%_active_max': 0,
                    'flat_rs_active_max': 0,
                    'buy_1_get_1_active_max': 0,
                    'buy_7_get_3_active_max': 0
                })
                X_baseline = pd.DataFrame([{k: v for k, v in baseline_dict.items() if k in self.feature_columns}])
                
                if self.orders_model:
                    baseline_hourly_orders = self.orders_model.predict(X_baseline)[0]
                    total_baseline_orders += max(0, float(baseline_hourly_orders))
                
                if self.sales_model:
                    baseline_hourly_sales = self.sales_model.predict(X_baseline)[0]
                    total_baseline_sales += max(0, float(baseline_hourly_sales))

        # Return results
        results = {
            'promotion_dates': [d.strftime('%Y-%m-%d') for d in promotion_dates],
            'duration_days': len(promotion_dates),
            'daily_hours': len(hours_in_day) if 'hours_in_day' in locals() else 0,
            'total_hours': len(all_promotion_hours),
            'promotion_hours': all_promotion_hours,
            'predicted_orders': total_predicted_orders,
            'predicted_sales': total_predicted_sales,
            'baseline_orders': total_baseline_orders,
            'baseline_sales': total_baseline_sales,
            'promotion_impact_orders': total_predicted_orders - total_baseline_orders,
            'promotion_impact_sales': total_predicted_sales - total_baseline_sales
        }
        
        return results