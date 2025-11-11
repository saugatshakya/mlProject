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

    def create_features(self, df):
        """
        Create features for promotion effectiveness prediction.
        Based on comprehensive feature engineering from the analysis.
        """
        df = df.copy()

        # Parse timestamp if it's a string
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['order_date'] = df['timestamp'].dt.date
            df['order_hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
        else:
            # Fallback to separate columns if timestamp doesn't exist
            df['order_hour'] = df['hour']
            df['day_of_week'] = df['day']
            df['day_of_month'] = df['day_of_month']
            df['month'] = df['month']

        # Basic temporal features
        df['hour'] = df['order_hour']
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

        # Promotion features - handle different data formats
        if 'promo_discount' in df.columns and 'restro_discount' in df.columns:
            # Original data format with individual discount columns
            df['has_promotion'] = ((df['promo_discount'] > 0) | (df['restro_discount'] > 0) |
                                  (df['flat_%'] > 0) | (df['flat_rs'] > 0) |
                                  (df['buy_1_get_1'] > 0) | (df['buy_7_get_3'] > 0)).astype(int)

            # Create promotion type encoding based on which promotion is active
            def get_promotion_type(row):
                if row['flat_%'] > 0:
                    return 'percentage_discount'
                elif row['flat_rs'] > 0:
                    return 'flat_discount'
                elif row['buy_1_get_1'] > 0:
                    return 'buy_1_get_1'
                elif row['buy_7_get_3'] > 0:
                    return 'buy_7_get_3'
                elif row['promo_discount'] > 0 or row['restro_discount'] > 0:
                    return 'other_discount'
                else:
                    return 'no_promo'

            df['promotion_type'] = df.apply(get_promotion_type, axis=1)
        elif 'promotion_type' in df.columns:
            # Generated data format with promotion_type column
            df['has_promotion'] = (df['promotion_type'] != 'no_promo').astype(int)
        else:
            # No promotion data available
            df['has_promotion'] = 0
            df['promotion_type'] = 'no_promo'

        df['promotion_type_encoded'] = pd.Categorical(df['promotion_type']).codes

        # Order value features
        df['subtotal_log'] = np.log1p(df['subtotal'])
        df['total_log'] = np.log1p(df['total'])

        # Weather features (if available)
        if 'temperature' in df.columns:
            df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
        else:
            df['temperature'] = 25.0

        if 'humidity' in df.columns:
            df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')
        else:
            df['humidity'] = 60.0

        if 'precipitation' in df.columns:
            df['precipitation'] = pd.to_numeric(df['precipitation'], errors='coerce')
        else:
            df['precipitation'] = 0.0

        if 'wind_speed' in df.columns:
            df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce')
        else:
            df['wind_speed'] = 5.0

        # Distance feature (if available)
        if 'Distance_km' in df.columns:
            df['distance_km'] = pd.to_numeric(df['Distance_km'], errors='coerce')
        elif 'distance_km' not in df.columns:
            df['distance_km'] = 3.0  # Default

        # Location features (if available)
        if 'rest_lat' in df.columns:
            df['rest_lat'] = pd.to_numeric(df['rest_lat'], errors='coerce')
        else:
            df['rest_lat'] = 28.6139  # Default Delhi coordinates

        if 'rest_lon' in df.columns:
            df['rest_lon'] = pd.to_numeric(df['rest_lon'], errors='coerce')
        else:
            df['rest_lon'] = 77.2090

        # Promotion-specific columns (if available)
        promo_cols = ['upto', 'flat_%', 'flat_rs', 'buy_1_get_1', 'buy_7_get_3']
        for col in promo_cols:
            if col not in df.columns:
                df[col] = 0

        return df

    def train(self, filepath):
        """
        Train promotion effectiveness models for both orders and sales.

        Args:
            filepath: Path to CSV file with promotion data

        Returns:
            dict: Training results with metrics
        """
        print(f"\n{'='*70}")
        print("TRAINING PROMOTION EFFECTIVENESS MODELS")
        print(f"{'='*70}")

        # Load promotion data
        print(f"\nLoading promotion data from: {filepath}")
        promo_df = pd.read_csv(filepath, quoting=1)
        self.data_path = filepath

        print(f"Promotion data shape: {promo_df.shape}")
        print(f"Promotion columns: {promo_df.columns.tolist()}")

        # Load demand data to get orders information
        demand_filepath = Path('uploads/original_data/demand_prediction.csv')
        if demand_filepath.exists():
            print(f"\nLoading demand data from: {demand_filepath}")
            demand_df = pd.read_csv(demand_filepath, quoting=1)
            print(f"Demand data shape: {demand_df.shape}")
            print(f"Demand columns: {demand_df.columns.tolist()}")

            # Merge datasets on timestamp to get orders data for promotion timestamps
            print(f"\nMerging datasets on timestamp...")
            promo_df['timestamp'] = pd.to_datetime(promo_df['timestamp'])
            demand_df['timestamp'] = pd.to_datetime(demand_df['timestamp'])

            # Merge to get orders data for promotion timestamps
            merged_df = promo_df.merge(demand_df, on='timestamp', how='left')
            print(f"Merged data shape: {merged_df.shape}")
            print(f"Orders data available for {merged_df['total_orders'].notna().sum()} records")
        else:
            print(f"\nWarning: Demand data not found at {demand_filepath}")
            print("Training sales model only - orders prediction will use estimation")
            merged_df = promo_df.copy()
            merged_df['total_orders'] = np.nan

        df = merged_df.copy()

        # Create features
        print("\nCreating promotion-focused features...")
        df_features = self.create_features(df)

        # Define features - exclude target variables and raw inputs, and non-numeric columns
        exclude_cols = ['timestamp', 'order_date', 'total', 'subtotal', 'total_orders',
                       'promo_discount', 'restro_discount', 'packaging_charges',
                       'weather_condition', 'promotion_type']  # Exclude string columns

        # Only include numeric columns
        potential_features = [col for col in df_features.columns if col not in exclude_cols]
        self.feature_columns = []

        for col in potential_features:
            # Check if column contains only numeric values
            try:
                sample_values = df_features[col].dropna().head(10)
                pd.to_numeric(sample_values, errors='coerce')
                # Only include if it's actually numeric
                if df_features[col].dtype in ['int64', 'float64', 'bool'] or df_features[col].astype(str).str.match(r'^\d+\.?\d*$').all():
                    self.feature_columns.append(col)
            except:
                continue

        print(f"Feature columns ({len(self.feature_columns)}): {self.feature_columns}")
        X = df_features[self.feature_columns]

        # Train orders model if orders data is available
        if 'total_orders' in df_features.columns and df_features['total_orders'].notna().sum() > 100:
            print(f"\nTraining Orders Model...")
            orders_data = df_features[df_features['total_orders'].notna()]
            X_orders = orders_data[self.feature_columns]
            y_orders = orders_data['total_orders']

            print(f"Orders training data: {len(orders_data)} samples")
            print(f"Orders range: {y_orders.min():.0f} - {y_orders.max():.0f}")

            X_train_ord, X_test_ord, y_train_ord, y_test_ord = train_test_split(
                X_orders, y_orders, test_size=0.2, random_state=42
            )

            self.orders_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )

            self.orders_model.fit(X_train_ord, y_train_ord)
            y_pred_ord = self.orders_model.predict(X_test_ord)

            orders_r2 = r2_score(y_test_ord, y_pred_ord)
            orders_mae = mean_absolute_error(y_test_ord, y_pred_ord)
            orders_rmse = np.sqrt(mean_squared_error(y_test_ord, y_pred_ord))

            print(f"Orders Model - R²: {orders_r2:.3f}, MAE: {orders_mae:.2f}, RMSE: {orders_rmse:.2f}")

            self.metrics['orders'] = {
                'r2': orders_r2,
                'mae': orders_mae,
                'rmse': orders_rmse
            }
        else:
            print("\nSkipping orders model - insufficient orders data")
            self.metrics['orders'] = None

        # Train sales model
        if 'total' in df_features.columns:
            print(f"\nTraining Sales Model...")
            y_sales = df_features['total']

            print(f"Sales training data: {len(df_features)} samples")
            print(f"Sales range: {y_sales.min():.0f} - {y_sales.max():.0f}")

            X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(
                X, y_sales, test_size=0.2, random_state=42
            )

            self.sales_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )

            self.sales_model.fit(X_train_sales, y_train_sales)
            y_pred_sales = self.sales_model.predict(X_test_sales)

            sales_r2 = r2_score(y_test_sales, y_pred_sales)
            sales_mae = mean_absolute_error(y_test_sales, y_pred_sales)
            sales_rmse = np.sqrt(mean_squared_error(y_test_sales, y_pred_sales))

            print(f"Sales Model - R²: {sales_r2:.3f}, MAE: {sales_mae:.2f}, RMSE: {sales_rmse:.2f}")

            self.metrics['sales'] = {
                'r2': sales_r2,
                'mae': sales_mae,
                'rmse': sales_rmse
            }
        else:
            print("\nSkipping sales model - no total column found")
            self.metrics['sales'] = None

        # Save model
        print(f"\nSaving models to: {self.model_path}")
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

        print(f"\nModels saved to: {self.model_path}")
        self.trained = True

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
        Predict promotion impact for a multi-day period using exact volume prediction approach from final_promotion_effectiveness_clean.ipynb
        Predicts total orders and sales during the entire promotion period, not just per hour.
        
        Args:
            promotion_data: dict with promotion details including period/duration
            
        Returns:
            dict: Total predicted orders and sales for the promotion period
        """
        if not self.trained:
            raise ValueError("Model not trained yet")

        # Extract promotion period details - support both old and new formats
        if 'start_date' in promotion_data:
            # New format: multi-day promotion
            start_date = pd.to_datetime(promotion_data['start_date'])
            duration_days = promotion_data.get('duration_days', 1)
            start_hour = promotion_data.get('start_hour', 12)
            end_hour = promotion_data.get('end_hour', 14)
            
            # Generate all dates in the promotion period
            promotion_dates = [start_date + pd.Timedelta(days=i) for i in range(duration_days)]
        else:
            # Legacy format: single day
            start_hour = promotion_data.get('start_hour', promotion_data.get('hour', 12))
            end_hour = promotion_data.get('end_hour', start_hour + 2)
            day_of_month = promotion_data.get('day_of_month', 15)
            month = promotion_data.get('month', 6)
            year = promotion_data.get('year', 2024)
            promotion_dates = [pd.Timestamp(year=year, month=month, day=day_of_month)]
        
        # Get weather data (same for all days/hours in the period)
        temperature = promotion_data.get('temperature', 25.0)
        precipitation = promotion_data.get('precipitation', 0.0)
        wind_speed = promotion_data.get('wind_speed', 5.0)
        is_event = promotion_data.get('is_event', 0)
        
        # Handle promotion inputs - support both promotion_type and direct flags
        flat_percent_active = 0
        flat_rs_active = 0
        buy_1_get_1_active = 0
        buy_7_get_3_active = 0
        
        # Check for direct promotion flag inputs first
        if 'flat_%' in promotion_data and promotion_data['flat_%'] > 0:
            flat_percent_active = 1
        if 'flat_rs' in promotion_data and promotion_data['flat_rs'] > 0:
            flat_rs_active = 1
        if 'buy_1_get_1' in promotion_data and promotion_data['buy_1_get_1'] > 0:
            buy_1_get_1_active = 1
        if 'buy_7_get_3' in promotion_data and promotion_data['buy_7_get_3'] > 0:
            buy_7_get_3_active = 1
            
        # Fallback to promotion_type mapping if no direct flags provided
        if all(flag == 0 for flag in [flat_percent_active, flat_rs_active, buy_1_get_1_active, buy_7_get_3_active]):
            promotion_type = promotion_data.get('promotion_type', 'no_promo')
            if promotion_type == 'discount_10':
                flat_percent_active = 1
            elif promotion_type == 'discount_20':
                flat_percent_active = 1
            elif promotion_type == 'free_delivery':
                flat_rs_active = 1
            elif promotion_type == 'combo_deal':
                buy_1_get_1_active = 1
            # no_promo leaves all as 0

        # Predict for each day in the promotion period
        total_predicted_orders = 0
        total_predicted_sales = 0
        total_baseline_orders = 0
        total_baseline_sales = 0
        all_promotion_hours = []
        
        for current_date in promotion_dates:
            day_of_month = current_date.day
            month = current_date.month
            day_of_week = current_date.dayofweek
            is_weekend = 1 if day_of_week in [5, 6] else 0
            
            # Handle hour wrap-around (e.g., 22-02 becomes 22,23,00,01,02)
            hours_in_day = []
            current_hour = start_hour
            while True:
                hours_in_day.append(current_hour)
                if current_hour == end_hour:
                    break
                current_hour = (current_hour + 1) % 24
                if len(hours_in_day) > 24:  # Prevent infinite loop
                    break
            
            for hour in hours_in_day:
                all_promotion_hours.append(f"{current_date.strftime('%Y-%m-%d')} {hour:02d}:00")
                
                # Create feature vector for this hour (exact same as notebook)
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
                
                # Create DataFrame for prediction
                X_pred = pd.DataFrame([feature_dict])
                
                # Ensure all required feature columns exist
                if self.feature_columns:
                    for col in self.feature_columns:
                        if col not in X_pred.columns:
                            X_pred[col] = 0  # Default value
                    X_pred = X_pred[self.feature_columns]

                # Predict for this hour
                if self.orders_model:
                    hourly_orders = self.orders_model.predict(X_pred)[0]
                    total_predicted_orders += max(0, float(hourly_orders))
                
                if self.sales_model:
                    hourly_sales = self.sales_model.predict(X_pred)[0]
                    total_predicted_sales += max(0, float(hourly_sales))
                    
                    # Calculate baseline (no promotion) for this hour
                    baseline_dict = feature_dict.copy()
                    baseline_dict.update({
                        'flat_%_active_max': 0,
                        'flat_rs_active_max': 0,
                        'buy_1_get_1_active_max': 0,
                        'buy_7_get_3_active_max': 0
                    })
                    X_baseline = pd.DataFrame([baseline_dict])
                    if self.feature_columns:
                        for col in self.feature_columns:
                            if col not in X_baseline.columns:
                                X_baseline[col] = 0
                        X_baseline = X_baseline[self.feature_columns]
                    
                    baseline_hourly_orders = self.orders_model.predict(X_baseline)[0]
                    baseline_hourly_sales = self.sales_model.predict(X_baseline)[0]
                    total_baseline_orders += max(0, float(baseline_hourly_orders))
                    total_baseline_sales += max(0, float(baseline_hourly_sales))

        # Predict for each hour in the promotion period
        total_predicted_orders = 0
        total_predicted_sales = 0
        total_baseline_orders = 0
        total_baseline_sales = 0
        
        # Handle hour wrap-around (e.g., 22-02 becomes 22,23,00,01,02)
        hours_in_period = []
        current_hour = start_hour
        while True:
            hours_in_period.append(current_hour)
            if current_hour == end_hour:
                break
            current_hour = (current_hour + 1) % 24
            if len(hours_in_period) > 24:  # Prevent infinite loop
                break
        
        for hour in hours_in_period:
            # Create feature vector for this hour (exact same as notebook)
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
            
            # Create DataFrame for prediction
            X_pred = pd.DataFrame([feature_dict])
            
            # Ensure all required feature columns exist
            if self.feature_columns:
                for col in self.feature_columns:
                    if col not in X_pred.columns:
                        X_pred[col] = 0  # Default value
                X_pred = X_pred[self.feature_columns]

            # Predict for this hour
            if self.orders_model:
                hourly_orders = self.orders_model.predict(X_pred)[0]
                total_predicted_orders += max(0, float(hourly_orders))
            
            if self.sales_model:
                hourly_sales = self.sales_model.predict(X_pred)[0]
                total_predicted_sales += max(0, float(hourly_sales))
                
                # Calculate baseline (no promotion) for this hour
                baseline_dict = feature_dict.copy()
                baseline_dict.update({
                    'flat_%_active_max': 0,
                    'flat_rs_active_max': 0,
                    'buy_1_get_1_active_max': 0,
                    'buy_7_get_3_active_max': 0
                })
                X_baseline = pd.DataFrame([baseline_dict])
                if self.feature_columns:
                    for col in self.feature_columns:
                        if col not in X_baseline.columns:
                            X_baseline[col] = 0
                    X_baseline = X_baseline[self.feature_columns]
                
                baseline_hourly_orders = self.orders_model.predict(X_baseline)[0] if self.orders_model else hourly_orders
                baseline_hourly_sales = self.sales_model.predict(X_baseline)[0]
                
                results = {
            'promotion_dates': [d.strftime('%Y-%m-%d') for d in promotion_dates],
            'duration_days': len(promotion_dates),
            'daily_hours': len(hours_in_day) if 'hours_in_day' in locals() else len(set([h.split(' ')[1] for h in all_promotion_hours])),
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

        results = {
            'promotion_dates': [d.strftime('%Y-%m-%d') for d in promotion_dates],
            'duration_days': len(promotion_dates),
            'daily_hours': len(hours_in_day) if 'hours_in_day' in locals() else len(hours_in_period),
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