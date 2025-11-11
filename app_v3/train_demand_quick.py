"""
Quick Demand Prediction Hyperparameter Tuning
==============================================

Faster version with predefined parameter sets to test.
Good for quick experimentation and validation.

Usage:
    python train_demand_quick.py
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering functions from main training script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_demand_hyperparameter_tuning import (
    create_features, 
    create_train_pattern_features, 
    apply_pattern_features
)


def test_hyperparameter_configs(X_train, y_train, X_test, y_test):
    """Test multiple predefined hyperparameter configurations."""
    
    configs = {
        'current': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'random_state': 42,
            'n_jobs': -1
        },
        'shallow_fast': {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        },
        'deep_slow': {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'random_state': 42,
            'n_jobs': -1
        },
        'balanced': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'random_state': 42,
            'n_jobs': -1
        },
        'regularized': {
            'n_estimators': 250,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.2,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': -1
        },
        'aggressive': {
            'n_estimators': 400,
            'max_depth': 9,
            'learning_rate': 0.02,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 1,
            'random_state': 42,
            'n_jobs': -1
        }
    }
    
    results = []
    
    print("\n" + "="*70)
    print("TESTING HYPERPARAMETER CONFIGURATIONS")
    print("="*70)
    
    for config_name, params in configs.items():
        print(f"\n{'─'*70}")
        print(f"Testing: {config_name.upper()}")
        print(f"{'─'*70}")
        print("Parameters:")
        for param, value in params.items():
            if param not in ['random_state', 'n_jobs']:
                print(f"  {param}: {value}")
        
        # Train model
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            'config': config_name,
            'r2_score': float(r2_score(y_test, y_pred)),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mape': float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100)
        }
        
        print(f"\nResults:")
        print(f"  R² Score: {metrics['r2_score']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        
        results.append({
            'config': config_name,
            'params': params,
            'metrics': metrics,
            'model': model
        })
    
    return results


def main():
    print("\n" + "="*70)
    print("QUICK DEMAND PREDICTION HYPERPARAMETER TESTING")
    print("="*70)
    
    # Load data
    data_path = '../data/hourly_orders_weather.csv'
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path, quoting=1)
    print(f"Data shape: {df.shape}")
    
    # Create features
    print("\nCreating enhanced features...")
    df_features = create_features(df)
    
    # Train/Test Split
    split_idx = int(len(df_features) * 0.8)
    train_df = df_features.iloc[:split_idx].copy()
    test_df = df_features.iloc[split_idx:].copy()
    
    print(f"✅ Train/Test split: Train {len(train_df)} hours, Test {len(test_df)} hours")
    
    # Pattern features
    pattern_stats = create_train_pattern_features(train_df)
    train_df = apply_pattern_features(train_df, pattern_stats)
    test_df = apply_pattern_features(test_df, pattern_stats)
    print("✅ Pattern features applied")
    
    # Define features
    feature_columns = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'is_morning', 'is_afternoon', 'is_evening',
        'total_orders_lag_1h', 'total_orders_lag_24h',
        'total_orders_rolling_6h', 'total_orders_rolling_24h',
        'total_orders_lag_2h', 'total_orders_lag_3h', 'total_orders_lag_48h', 'total_orders_lag_168h',
        'total_orders_rolling_3h', 'total_orders_rolling_12h',
        'total_orders_rolling_6h_std', 'total_orders_rolling_24h_std',
        'hour_avg_orders', 'dow_avg_orders', 'hour_dow_avg_orders'
    ]
    
    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    y_train = train_df['orders_per_hour']
    y_test = test_df['orders_per_hour']
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Test configurations
    results = test_hyperparameter_configs(X_train, y_train, X_test, y_test)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF ALL CONFIGURATIONS")
    print("="*70)
    
    summary_df = pd.DataFrame([
        {
            'Config': r['config'],
            'R²': r['metrics']['r2_score'],
            'MAE': r['metrics']['mae'],
            'RMSE': r['metrics']['rmse'],
            'MAPE': r['metrics']['mape']
        }
        for r in results
    ])
    
    summary_df = summary_df.sort_values('MAE')
    
    print("\n" + summary_df.to_string(index=False))
    
    # Best configuration
    best_result = min(results, key=lambda x: x['metrics']['mae'])
    
    print("\n" + "="*70)
    print("BEST CONFIGURATION")
    print("="*70)
    print(f"\nConfig: {best_result['config'].upper()}")
    print(f"\nParameters:")
    for param, value in best_result['params'].items():
        if param not in ['random_state', 'n_jobs']:
            print(f"  {param}: {value}")
    
    print(f"\nMetrics:")
    print(f"  R² Score: {best_result['metrics']['r2_score']:.4f}")
    print(f"  MAE: {best_result['metrics']['mae']:.4f}")
    print(f"  RMSE: {best_result['metrics']['rmse']:.4f}")
    print(f"  MAPE: {best_result['metrics']['mape']:.2f}%")
    
    # Save results
    output_dir = Path('hyperparameter_tuning_results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save best model
    model_path = output_dir / f'best_model_quick_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_result['model'], f)
    
    # Save best params
    params_path = output_dir / f'best_params_quick_{timestamp}.json'
    with open(params_path, 'w') as f:
        # Remove non-serializable items
        params_to_save = {k: v for k, v in best_result['params'].items() 
                          if k not in ['random_state', 'n_jobs']}
        json.dump(params_to_save, f, indent=2)
    
    # Save summary
    summary_path = output_dir / f'summary_quick_{timestamp}.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n✅ Results saved to: {output_dir}")
    print(f"   - Model: {model_path.name}")
    print(f"   - Parameters: {params_path.name}")
    print(f"   - Summary: {summary_path.name}")
    
    print("\n" + "="*70)
    print("RECOMMENDED PARAMETERS FOR models_demand_prediction.py:")
    print("="*70)
    print("\nReplace the XGBRegressor initialization with:")
    print("\nself.model = XGBRegressor(")
    for param, value in best_result['params'].items():
        if param not in ['random_state', 'n_jobs']:
            print(f"    {param}={value},")
    print("    random_state=42,")
    print("    n_jobs=-1")
    print(")")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
