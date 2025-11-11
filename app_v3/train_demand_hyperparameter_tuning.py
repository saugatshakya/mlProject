"""
Demand Prediction Hyperparameter Tuning Script
===============================================

Standalone script to find optimal hyperparameters for demand prediction model.
Uses the same feature engineering as models_demand_prediction.py.
Performs extensive hyperparameter search using GridSearchCV and RandomizedSearchCV.

Usage:
    python train_demand_hyperparameter_tuning.py --data_path ../data/hourly_orders_weather.csv
"""

import pandas as pd
import numpy as np
import pickle
import json
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


def create_features(df):
    """
    Create enhanced features for demand prediction.
    Same feature engineering as in DemandPredictionModel.
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

    # Basic temporal features
    df['hour'] = df['order_hour']
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

    # Lag features
    df['total_orders_lag_1h'] = df['orders_per_hour'].shift(1)
    df['total_orders_lag_24h'] = df['orders_per_hour'].shift(24)
    df['total_orders_lag_2h'] = df['orders_per_hour'].shift(2)
    df['total_orders_lag_3h'] = df['orders_per_hour'].shift(3)
    df['total_orders_lag_48h'] = df['orders_per_hour'].shift(48)
    df['total_orders_lag_168h'] = df['orders_per_hour'].shift(168)  # 1 week

    # Rolling window features (safe - no leakage)
    s = df['orders_per_hour'].shift(1)  # Use shifted series for rolling stats

    # Rolling means
    df['total_orders_rolling_6h'] = s.rolling(window=6, min_periods=1).mean()
    df['total_orders_rolling_24h'] = s.rolling(window=24, min_periods=1).mean()
    df['total_orders_rolling_3h'] = s.rolling(window=3, min_periods=1).mean()
    df['total_orders_rolling_12h'] = s.rolling(window=12, min_periods=1).mean()

    # Rolling standard deviations
    df['total_orders_rolling_6h_std'] = s.rolling(window=6, min_periods=1).std()
    df['total_orders_rolling_24h_std'] = s.rolling(window=24, min_periods=1).std()

    # Fill NaN for std columns
    df['total_orders_rolling_6h_std'] = df['total_orders_rolling_6h_std'].fillna(0)
    df['total_orders_rolling_24h_std'] = df['total_orders_rolling_24h_std'].fillna(0)

    # Fill all remaining NaN values
    df = df.fillna(0)

    return df


def create_train_pattern_features(df):
    """Create pattern features using ONLY training data to avoid leakage."""
    hourly_avg = df.groupby('order_hour')['orders_per_hour'].mean().to_dict()
    dow_avg = df.groupby('day_of_week')['orders_per_hour'].mean().to_dict()
    hour_dow_avg = df.groupby(['order_hour', 'day_of_week'])['orders_per_hour'].mean().to_dict()

    return {
        'hourly_avg': hourly_avg,
        'dow_avg': dow_avg,
        'hour_dow_avg': hour_dow_avg
    }


def apply_pattern_features(df, pattern_stats):
    """Apply pre-calculated pattern features to any dataset."""
    df = df.copy()

    df['hour_avg_orders'] = df['order_hour'].map(pattern_stats['hourly_avg'])
    df['dow_avg_orders'] = df['day_of_week'].map(pattern_stats['dow_avg'])

    df['hour_dow_avg_orders'] = df.apply(
        lambda row: pattern_stats['hour_dow_avg'].get(
            (row['order_hour'], row['day_of_week']),
            pattern_stats['hourly_avg'].get(row['order_hour'], 0)
        ), axis=1
    )

    return df


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2_score': float(r2_score(y_test, y_pred)),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mape': float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100)
    }
    
    return metrics


def grid_search_tuning(X_train, y_train, cv_splits=3):
    """
    Perform Grid Search for hyperparameter tuning.
    More thorough but slower.
    """
    print("\n" + "="*70)
    print("GRID SEARCH HYPERPARAMETER TUNING")
    print("="*70)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    print(f"\nParameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\nTotal combinations: {total_combinations}")
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    # Create base model
    base_model = XGBRegressor(random_state=42, n_jobs=-1)
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    print("\nStarting Grid Search...")
    print(f"Using {cv_splits}-fold Time Series Cross-Validation")
    
    grid_search.fit(X_train, y_train)
    
    print("\n" + "="*70)
    print("GRID SEARCH RESULTS")
    print("="*70)
    print(f"\nBest Score (MAE): {-grid_search.best_score_:.4f}")
    print(f"\nBest Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


def random_search_tuning(X_train, y_train, cv_splits=3, n_iter=50):
    """
    Perform Randomized Search for hyperparameter tuning.
    Faster, good for initial exploration.
    """
    print("\n" + "="*70)
    print("RANDOMIZED SEARCH HYPERPARAMETER TUNING")
    print("="*70)
    
    # Define parameter distributions
    param_distributions = {
        'n_estimators': [50, 100, 150, 200, 250, 300, 400],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 2, 3, 4, 5, 6],
        'gamma': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [0, 0.01, 0.1, 1]
    }
    
    print(f"\nParameter distributions:")
    for param, values in param_distributions.items():
        print(f"  {param}: {len(values)} options")
    
    print(f"\nRandom iterations: {n_iter}")
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    # Create base model
    base_model = XGBRegressor(random_state=42, n_jobs=-1)
    
    # Randomized search
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=True
    )
    
    print("\nStarting Randomized Search...")
    print(f"Using {cv_splits}-fold Time Series Cross-Validation")
    
    random_search.fit(X_train, y_train)
    
    print("\n" + "="*70)
    print("RANDOMIZED SEARCH RESULTS")
    print("="*70)
    print(f"\nBest Score (MAE): {-random_search.best_score_:.4f}")
    print(f"\nBest Parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return random_search.best_estimator_, random_search.best_params_, random_search.cv_results_


def save_results(best_model, best_params, metrics, cv_results, output_dir, method_name):
    """Save tuning results and best model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save best model
    model_path = output_dir / f'best_demand_model_{method_name}_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n✅ Best model saved to: {model_path}")
    
    # Save best parameters
    params_path = output_dir / f'best_params_{method_name}_{timestamp}.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"✅ Best parameters saved to: {params_path}")
    
    # Save metrics
    metrics_path = output_dir / f'metrics_{method_name}_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to: {metrics_path}")
    
    # Save CV results
    cv_results_df = pd.DataFrame(cv_results)
    cv_results_path = output_dir / f'cv_results_{method_name}_{timestamp}.csv'
    cv_results_df.to_csv(cv_results_path, index=False)
    print(f"✅ CV results saved to: {cv_results_path}")
    
    # Create summary report
    report = f"""
Demand Prediction Hyperparameter Tuning Report
{'='*70}

Method: {method_name.upper()}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BEST HYPERPARAMETERS:
{'-'*70}
{json.dumps(best_params, indent=2)}

TEST SET PERFORMANCE:
{'-'*70}
R² Score: {metrics['r2_score']:.4f}
MAE: {metrics['mae']:.4f}
RMSE: {metrics['rmse']:.4f}
MAPE: {metrics['mape']:.2f}%

FILES SAVED:
{'-'*70}
Model: {model_path.name}
Parameters: {params_path.name}
Metrics: {metrics_path.name}
CV Results: {cv_results_path.name}

{'='*70}
"""
    
    report_path = output_dir / f'tuning_report_{method_name}_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ Report saved to: {report_path}")
    
    print(report)


def main():
    parser = argparse.ArgumentParser(description='Demand Prediction Hyperparameter Tuning')
    parser.add_argument('--data_path', type=str, 
                        default='uploads/original_data/demand_prediction.csv',
                        help='Path to the training data CSV file')
    parser.add_argument('--method', type=str, 
                        choices=['grid', 'random', 'both'],
                        default='random',
                        help='Tuning method: grid, random, or both')
    parser.add_argument('--cv_splits', type=int, default=3,
                        help='Number of cross-validation splits')
    parser.add_argument('--n_iter', type=int, default=50,
                        help='Number of iterations for random search')
    parser.add_argument('--output_dir', type=str, 
                        default='hyperparameter_tuning_results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DEMAND PREDICTION HYPERPARAMETER TUNING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Method: {args.method}")
    print(f"  CV splits: {args.cv_splits}")
    print(f"  Random iterations: {args.n_iter}")
    print(f"  Output directory: {args.output_dir}")
    
    # Load data
    print(f"\nLoading data from: {args.data_path}")
    df = pd.read_csv(args.data_path, quoting=1)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create features
    print("\nCreating enhanced features...")
    df_features = create_fecreate_features(df)
    
    # Train/Test Split (time-aware - no future data in training)
    split_idx = int(len(df_features) * 0.8)
    train_df = df_features.iloc[:split_idx].copy()
    test_df = df_features.iloc[split_idx:].copy()
    
    print(f"✅ Train/Test split: Train {len(train_df)} hours, Test {len(test_df)} hours")
    print(f"   Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"   Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    
    # Compute pattern features ONLY on training data
    pattern_stats = create_train_pattern_features(train_df)
    print("✅ Pattern statistics computed from training data only")
    
    # Apply pattern features
    train_df = apply_pattern_features(train_df, pattern_stats)
    test_df = apply_pattern_features(test_df, pattern_stats)
    print("✅ Pattern features applied to train and test sets")
    
    # Define feature set
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
    
    print(f"\nFeatures: {len(feature_columns)}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Perform hyperparameter tuning
    if args.method == 'grid' or args.method == 'both':
        best_model_grid, best_params_grid, cv_results_grid = grid_search_tuning(
            X_train, y_train, cv_splits=args.cv_splits
        )
        
        # Evaluate on test set
        metrics_grid = evaluate_model(best_model_grid, X_test, y_test)
        
        # Save results
        save_results(
            best_model_grid, best_params_grid, metrics_grid, 
            cv_results_grid, args.output_dir, 'grid_search'
        )
    
    if args.method == 'random' or args.method == 'both':
        best_model_random, best_params_random, cv_results_random = random_search_tuning(
            X_train, y_train, cv_splits=args.cv_splits, n_iter=args.n_iter
        )
        
        # Evaluate on test set
        metrics_random = evaluate_model(best_model_random, X_test, y_test)
        
        # Save results
        save_results(
            best_model_random, best_params_random, metrics_random,
            cv_results_random, args.output_dir, 'random_search'
        )
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nTo use the best hyperparameters in your app:")
    print("1. Check the best_params_*.json file")
    print("2. Update the XGBRegressor parameters in models_demand_prediction.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
