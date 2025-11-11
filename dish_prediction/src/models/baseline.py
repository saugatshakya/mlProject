"""
Baseline Models
Simple benchmarks to establish performance floor

Models:
1. Historical Mean (overall average)
2. Hourly Mean (average per hour of day)
3. Last Value (naive forecast)
4. Moving Average (7-day rolling mean)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class BaselineModels:
    """Simple baseline models for demand forecasting"""
    
    def __init__(self):
        self.results = {}
        
    def prepare_data(self, df: pd.DataFrame, dish_cols: List[str],
                    test_size: int = 336) -> Tuple:  # 336 hours = 2 weeks
        """
        Prepare train/test split
        
        Args:
            df: DataFrame with features
            dish_cols: List of dish column names
            test_size: Number of hours for test set
            
        Returns:
            Tuple of (train_df, test_df, dish_cols)
        """
        # Sort by time
        df = df.sort_values('hour').reset_index(drop=True)
        
        # Split
        train = df.iloc[:-test_size].copy()
        test = df.iloc[-test_size:].copy()
        
        print(f"\nData split:")
        print(f"  Train: {len(train):,} hours ({train['hour'].min()} to {train['hour'].max()})")
        print(f"  Test:  {len(test):,} hours ({test['hour'].min()} to {test['hour'].max()})")
        
        return train, test, dish_cols
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                model_name: str) -> Dict:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of model
            
        Returns:
            Dict with metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (avoid division by zero)
        mask = y_true > 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
        
        results = {
            'model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        }
        
        return results
    
    def baseline_mean(self, train: pd.DataFrame, test: pd.DataFrame,
                     dish_cols: List[str]) -> Dict:
        """
        Baseline 1: Historical mean (overall average)
        Predicts the overall average for each dish
        """
        print("\n" + "="*80)
        print("BASELINE 1: Historical Mean")
        print("="*80)
        
        all_results = []
        
        for dish in dish_cols:
            # Calculate mean from training data
            mean_value = train[dish].mean()
            
            # Predict (same value for all test points)
            y_pred = np.full(len(test), mean_value)
            y_true = test[dish].values
            
            # Evaluate
            results = self.evaluate(y_true, y_pred, f'Mean_{dish}')
            all_results.append(results)
        
        # Average across dishes
        avg_results = {
            'model': 'Historical Mean (Average)',
            'MAE': np.mean([r['MAE'] for r in all_results]),
            'RMSE': np.mean([r['RMSE'] for r in all_results]),
            'R²': np.mean([r['R²'] for r in all_results]),
            'MAPE': np.mean([r['MAPE'] for r in all_results if not np.isnan(r['MAPE'])])
        }
        
        print(f"\n✓ Results (averaged across {len(dish_cols)} dishes):")
        print(f"  MAE:  {avg_results['MAE']:.3f}")
        print(f"  RMSE: {avg_results['RMSE']:.3f}")
        print(f"  R²:   {avg_results['R²']:.3f}")
        print(f"  MAPE: {avg_results['MAPE']:.1f}%")
        
        self.results['Historical Mean'] = avg_results
        return avg_results
    
    def baseline_hourly_mean(self, train: pd.DataFrame, test: pd.DataFrame,
                            dish_cols: List[str]) -> Dict:
        """
        Baseline 2: Hourly mean (average per hour of day)
        Predicts based on hour-of-day patterns
        """
        print("\n" + "="*80)
        print("BASELINE 2: Hourly Mean")
        print("="*80)
        
        all_results = []
        
        for dish in dish_cols:
            # Calculate mean per hour from training data
            hourly_means = train.groupby('hour_of_day')[dish].mean()
            
            # Predict based on hour of day
            y_pred = test['hour_of_day'].map(hourly_means).values
            y_true = test[dish].values
            
            # Evaluate
            results = self.evaluate(y_true, y_pred, f'HourlyMean_{dish}')
            all_results.append(results)
        
        # Average across dishes
        avg_results = {
            'model': 'Hourly Mean (Average)',
            'MAE': np.mean([r['MAE'] for r in all_results]),
            'RMSE': np.mean([r['RMSE'] for r in all_results]),
            'R²': np.mean([r['R²'] for r in all_results]),
            'MAPE': np.mean([r['MAPE'] for r in all_results if not np.isnan(r['MAPE'])])
        }
        
        print(f"\n✓ Results (averaged across {len(dish_cols)} dishes):")
        print(f"  MAE:  {avg_results['MAE']:.3f}")
        print(f"  RMSE: {avg_results['RMSE']:.3f}")
        print(f"  R²:   {avg_results['R²']:.3f}")
        print(f"  MAPE: {avg_results['MAPE']:.1f}%")
        
        self.results['Hourly Mean'] = avg_results
        return avg_results
    
    def baseline_last_value(self, train: pd.DataFrame, test: pd.DataFrame,
                           dish_cols: List[str]) -> Dict:
        """
        Baseline 3: Last value (naive forecast)
        Predicts the last observed value
        """
        print("\n" + "="*80)
        print("BASELINE 3: Last Value (Naive)")
        print("="*80)
        
        all_results = []
        
        for dish in dish_cols:
            # Use lag_1h feature if available, else last train value
            if f'{dish}_lag_1h' in test.columns:
                y_pred = test[f'{dish}_lag_1h'].values
            else:
                # Fall back to last training value (naive)
                last_value = train[dish].iloc[-1]
                y_pred = np.full(len(test), last_value)
            
            y_true = test[dish].values
            
            # Evaluate
            results = self.evaluate(y_true, y_pred, f'LastValue_{dish}')
            all_results.append(results)
        
        # Average across dishes
        avg_results = {
            'model': 'Last Value (Naive)',
            'MAE': np.mean([r['MAE'] for r in all_results]),
            'RMSE': np.mean([r['RMSE'] for r in all_results]),
            'R²': np.mean([r['R²'] for r in all_results]),
            'MAPE': np.mean([r['MAPE'] for r in all_results if not np.isnan(r['MAPE'])])
        }
        
        print(f"\n✓ Results (averaged across {len(dish_cols)} dishes):")
        print(f"  MAE:  {avg_results['MAE']:.3f}")
        print(f"  RMSE: {avg_results['RMSE']:.3f}")
        print(f"  R²:   {avg_results['R²']:.3f}")
        print(f"  MAPE: {avg_results['MAPE']:.1f}%")
        
        self.results['Last Value'] = avg_results
        return avg_results
    
    def baseline_moving_average(self, train: pd.DataFrame, test: pd.DataFrame,
                               dish_cols: List[str], window: int = 168) -> Dict:
        """
        Baseline 4: Moving average (7-day rolling mean)
        Uses rolling_mean_168h feature if available
        """
        print("\n" + "="*80)
        print(f"BASELINE 4: Moving Average ({window}h = 7 days)")
        print("="*80)
        
        all_results = []
        
        for dish in dish_cols:
            # Use rolling mean feature if available
            rolling_col = f'{dish}_rolling_mean_{window}h'
            
            if rolling_col in test.columns:
                y_pred = test[rolling_col].values
            else:
                # Calculate manually
                full_data = pd.concat([train[dish], test[dish]])
                rolling_mean = full_data.rolling(window=window, min_periods=1).mean()
                y_pred = rolling_mean.iloc[-len(test):].values
            
            y_true = test[dish].values
            
            # Evaluate
            results = self.evaluate(y_true, y_pred, f'MA{window}_{dish}')
            all_results.append(results)
        
        # Average across dishes
        avg_results = {
            'model': f'Moving Avg ({window}h)',
            'MAE': np.mean([r['MAE'] for r in all_results]),
            'RMSE': np.mean([r['RMSE'] for r in all_results]),
            'R²': np.mean([r['R²'] for r in all_results]),
            'MAPE': np.mean([r['MAPE'] for r in all_results if not np.isnan(r['MAPE'])])
        }
        
        print(f"\n✓ Results (averaged across {len(dish_cols)} dishes):")
        print(f"  MAE:  {avg_results['MAE']:.3f}")
        print(f"  RMSE: {avg_results['RMSE']:.3f}")
        print(f"  R²:   {avg_results['R²']:.3f}")
        print(f"  MAPE: {avg_results['MAPE']:.1f}%")
        
        self.results[f'MA_{window}h'] = avg_results
        return avg_results
    
    def run_all_baselines(self, df: pd.DataFrame, dish_cols: List[str],
                         test_size: int = 336) -> pd.DataFrame:
        """
        Run all baseline models and compare
        
        Args:
            df: DataFrame with features
            dish_cols: List of dish columns
            test_size: Test set size in hours
            
        Returns:
            DataFrame with comparison results
        """
        print("="*80)
        print("RUNNING ALL BASELINE MODELS")
        print("="*80)
        
        # Prepare data
        train, test, dish_cols = self.prepare_data(df, dish_cols, test_size)
        
        # Run baselines
        self.baseline_mean(train, test, dish_cols)
        self.baseline_hourly_mean(train, test, dish_cols)
        self.baseline_last_value(train, test, dish_cols)
        self.baseline_moving_average(train, test, dish_cols, window=168)
        
        # Create comparison table
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('MAE')
        
        print("\n" + "="*80)
        print("BASELINE COMPARISON (Sorted by MAE)")
        print("="*80)
        print(results_df.to_string())
        
        # Save results
        output_path = 'reports/baseline_results.csv'
        results_df.to_csv(output_path)
        print(f"\n✓ Saved results to: {output_path}")
        
        return results_df


if __name__ == "__main__":
    # Load TIER 1 features
    print("Loading TIER 1 features...")
    df = pd.read_csv('data/processed/tier1_features.csv')
    df['hour'] = pd.to_datetime(df['hour'])
    
    # Get dish columns
    metadata_cols = ['hour', 'hour_of_day', 'day_of_week', 'day_of_month',
                     'week_of_year', 'month', 'date', 'is_weekend', 'is_friday',
                     'is_peak_hour', 'is_lunch_rush', 'is_dinner_rush',
                     'is_late_night', 'meal_period', 'hour_sin', 'hour_cos',
                     'day_sin', 'day_cos', 'env_temp', 'env_rhum', 'env_precip',
                     'env_wspd', 'env_condition', 'aqi', 'pm2_5', 'pm10', 'no2',
                     'o3', 'co', 'event', 'holiday', 'has_event', 'is_raining']
    
    # Get base dish columns (not lag/rolling features)
    all_cols = df.columns.tolist()
    dish_cols = []
    for col in all_cols:
        if col not in metadata_cols and not any(x in col for x in ['_lag_', '_rolling_', '_cv_']):
            dish_cols.append(col)
    
    print(f"\nFound {len(dish_cols)} base dish columns")
    print(f"Sample: {dish_cols[:3]}")
    
    # Run baselines
    baseline = BaselineModels()
    results = baseline.run_all_baselines(df, dish_cols, test_size=336)
    
    print("\n✅ Baseline evaluation complete!")
