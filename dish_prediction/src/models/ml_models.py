"""
ML Model Comparison
Train and compare 15+ algorithms with selected features

Models:
- Linear: Ridge, Lasso, ElasticNet
- Trees: RandomForest, ExtraTrees, GradientBoosting
- Boosting: XGBoost, LightGBM, CatBoost
- Others: KNN, SVR

Evaluation: 5-fold time-series CV
Metrics: MAE, RMSE, R², MAPE
Target: Beat baseline MAE=0.443
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
import time
from typing import Dict, List, Tuple
warnings.filterwarnings('ignore')


from sklearn.multioutput import MultiOutputRegressor

class MLModelComparison:
    """Compare multiple ML algorithms for demand forecasting"""
    
    def __init__(self, n_splits: int = 5, multi_output: bool = False):
        self.n_splits = n_splits
        self.multi_output = multi_output
        self.results = []
        self.best_model = None
        self.best_score = float('inf')
        self.scaler = StandardScaler()
        
    def get_models(self) -> Dict:
        """Initialize all models with reasonable default parameters"""
        # Models that support multi-output natively
        models = {
            'DecisionTree': DecisionTreeRegressor(
                max_depth=10, min_samples_split=20, random_state=42
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=100, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ),
            'KNN': KNeighborsRegressor(n_neighbors=10, weights='distance'),
        }

        # Models that need to be wrapped for multi-output
        single_output_models = {
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=0.1, random_state=42, max_iter=5000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=5000),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                min_samples_split=10, random_state=42
            ),
            'AdaBoost': AdaBoostRegressor(
                n_estimators=100, learning_rate=0.1, random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                tree_method='hist', n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbose=-1
            ),
            'CatBoost': CatBoostRegressor(
                iterations=100, depth=6, learning_rate=0.1,
                random_state=42, verbose=0
            ),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }

        if self.multi_output:
            for name, model in single_output_models.items():
                models[name] = MultiOutputRegressor(model, n_jobs=-1)
        else:
            models.update(single_output_models)
        
        return models
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() == 0:
            return 100.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def evaluate_model(self, model, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       scale: bool = True) -> Dict:
        """
        Evaluate a single model
        
        Args:
            model: ML model instance
            X_train, y_train: Training data
            X_test, y_test: Test data
            scale: Whether to scale features
            
        Returns:
            Dictionary of evaluation metrics
        """
        start_time = time.time()
        
        # Scale features if needed
        if scale:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train model
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mape': self.calculate_mape(y_train, y_train_pred),
            'test_mape': self.calculate_mape(y_test, y_test_pred),
            'train_time': train_time
        }
        
        # Calculate overfitting gap
        metrics['overfit_gap_mae'] = metrics['test_mae'] - metrics['train_mae']
        metrics['overfit_gap_r2'] = metrics['train_r2'] - metrics['test_r2']
        
        return metrics
    
    def cross_validate_model(self, model_name: str, model, X: np.ndarray, 
                            y: np.ndarray, scale: bool = True) -> Dict:
        """
        Cross-validate a model using time-series CV
        
        Args:
            model_name: Name of the model
            model: ML model instance
            X, y: Full dataset
            scale: Whether to scale features
            
        Returns:
            Dictionary of CV results
        """
        print(f"\nTraining {model_name}...")
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            metrics = self.evaluate_model(model, X_train, y_train, 
                                         X_test, y_test, scale=scale)
            fold_results.append(metrics)
        
        # Average across folds
        avg_results = {
            'model': model_name,
            'cv_folds': self.n_splits,
            'train_mae_mean': np.mean([r['train_mae'] for r in fold_results]),
            'train_mae_std': np.std([r['train_mae'] for r in fold_results]),
            'test_mae_mean': np.mean([r['test_mae'] for r in fold_results]),
            'test_mae_std': np.std([r['test_mae'] for r in fold_results]),
            'train_rmse_mean': np.mean([r['train_rmse'] for r in fold_results]),
            'test_rmse_mean': np.mean([r['test_rmse'] for r in fold_results]),
            'train_r2_mean': np.mean([r['train_r2'] for r in fold_results]),
            'test_r2_mean': np.mean([r['test_r2'] for r in fold_results]),
            'train_mape_mean': np.mean([r['train_mape'] for r in fold_results]),
            'test_mape_mean': np.mean([r['test_mape'] for r in fold_results]),
            'overfit_gap_mae': np.mean([r['overfit_gap_mae'] for r in fold_results]),
            'overfit_gap_r2': np.mean([r['overfit_gap_r2'] for r in fold_results]),
            'train_time_mean': np.mean([r['train_time'] for r in fold_results])
        }
        
        print(f"  ✓ Test MAE: {avg_results['test_mae_mean']:.4f} ± {avg_results['test_mae_std']:.4f}")
        print(f"    Test R²: {avg_results['test_r2_mean']:.4f}")
        print(f"    Overfit Gap (MAE): {avg_results['overfit_gap_mae']:.4f}")
        print(f"    Training Time: {avg_results['train_time_mean']:.2f}s")
        
        return avg_results
    
    def run_comparison(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Run full model comparison
        
        Args:
            X: Feature DataFrame
            y: Target variable
            
        Returns:
            DataFrame with comparison results
        """
        print("="*80)
        print("ML MODEL COMPARISON")
        print("="*80)
        print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
        if self.multi_output:
            print(f"Targets: {y.columns.tolist()}")
        else:
            print(f"Target: {y.name}")
        print(f"Cross-validation: {self.n_splits}-fold Time-Series CV")
        
        # Align data and convert to numpy arrays
        X_aligned = X[X.columns].fillna(X.mean())
        y_aligned = y[y.columns] if self.multi_output else y
        
        # Get all models
        models = self.get_models()
        
        # Models that need scaling
        scale_models = ['Ridge', 'Lasso', 'ElasticNet', 'KNN', 'SVR']
        
        # Train and evaluate each model
        for model_name, model in models.items():
            try:
                scale = model_name in scale_models
                
                # Pass DataFrame to preserve feature names
                results = self.cross_validate_model(
                    model_name, model, X_aligned, y_aligned, scale=scale
                )
                self.results.append(results)
                
                # Track best model
                if results['test_mae_mean'] < self.best_score:
                    self.best_score = results['test_mae_mean']
                    self.best_model = model_name
                    
            except Exception as e:
                print(f"  ✗ Error training {model_name}: {str(e)}")
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('test_mae_mean').reset_index(drop=True)
        
        return results_df
    
    def print_summary(self, results_df: pd.DataFrame):
        """Print comparison summary"""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        print(f"\n{'Rank':<6} {'Model':<20} {'Test MAE':<12} {'Test R²':<10} {'Overfit Gap':<12} {'Time (s)':<10}")
        print("-" * 80)
        
        for i, row in results_df.head(15).iterrows():
            mae_str = f"{row['test_mae_mean']:.4f} ± {row['test_mae_std']:.4f}"
            r2_str = f"{row['test_r2_mean']:.4f}"
            gap_str = f"{row['overfit_gap_mae']:.4f}"
            time_str = f"{row['train_time_mean']:.2f}"
            
            # Highlight if beats baseline
            marker = "✓" if row['test_mae_mean'] < 0.443 else " "
            
            print(f"{marker} {i+1:<5} {row['model']:<20} {mae_str:<12} {r2_str:<10} {gap_str:<12} {time_str:<10}")
        
        print("\n" + "="*80)
        print(f"Best Model: {self.best_model}")
        print(f"Best MAE: {self.best_score:.4f}")
        
        baseline_mae = 0.443
        if self.best_score < baseline_mae:
            improvement = (baseline_mae - self.best_score) / baseline_mae * 100
            print(f"✓ Improved over baseline by {improvement:.1f}%")
        else:
            print(f"⚠ Did not beat baseline (MAE = {baseline_mae:.4f})")
        
        # Check for overfitting
        best_model_row = results_df[results_df['model'] == self.best_model].iloc[0]
        if abs(best_model_row['overfit_gap_mae']) > 0.05:
            print(f"⚠ Warning: Significant overfitting detected (gap = {best_model_row['overfit_gap_mae']:.4f})")
        
        print("="*80)
    
    def save_results(self, results_df: pd.DataFrame, 
                    output_path: str = 'reports/model_comparison.csv'):
        """Save results to CSV"""
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    MULTI_OUTPUT_MODE = True
    TOP_N_DISHES = 5
    
    # Load data with TIER 2 selected features
    print("Loading data with TIER 2 selected features...")
    try:
        # Load the full TIER 2 feature set first
        df_full = pd.read_csv('data/processed/tier2_features.csv')
        df_full['hour'] = pd.to_datetime(df_full['hour'])
        
        # Load the selected feature names
        selected_features_df = pd.read_csv('reports/selected_features_tier2.txt', sep='|', skiprows=4, header=None)
        # Column 0 contains the feature name with numbering, e.g., "  1. feature_name  "
        raw_feature_col = selected_features_df[0].tolist()
        selected_features = [s.split('.', 1)[1].strip() for s in raw_feature_col]

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure `tier2_features.py` and `feature_selector.py` have been run.")
        exit()
        
    # Get top N dishes by total quantity
    metadata_cols = ['hour', 'date']
    all_dishes = [col for col in df_full.columns if '_lag_' not in col and '_rolling_' not in col and col not in metadata_cols and df_full[col].dtype in ['int64', 'float64']]
    top_dishes = df_full[all_dishes].sum().sort_values(ascending=False).head(TOP_N_DISHES).index.tolist()
    
    print(f"\nRunning in {'Multi-Output' if MULTI_OUTPUT_MODE else 'Single-Output'} mode.")
    
    if MULTI_OUTPUT_MODE:
        print(f"Predicting Top {TOP_N_DISHES} dishes: {top_dishes}")
        y = df_full[top_dishes]
    else:
        target_dish = 'Bageecha Pizza'
        print(f"Predicting single dish: {target_dish}")
        y = df_full[target_dish]

    # Use only the selected features
    X = df_full[selected_features]
    
    # Drop NaN rows from features and targets
    combined = pd.concat([X, y], axis=1).dropna()
    X = combined[selected_features]
    y = combined[y.columns] if MULTI_OUTPUT_MODE else combined[y.name]

    print(f"\nDataset prepared:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    
    # Run comparison
    comparison = MLModelComparison(n_splits=5, multi_output=MULTI_OUTPUT_MODE)
    results = comparison.run_comparison(X, y)
    
    # Print summary
    comparison.print_summary(results)
    
    # Save results
    output_filename = 'model_comparison_multi_output.csv' if MULTI_OUTPUT_MODE else 'model_comparison_tier2.csv'
    comparison.save_results(results, output_path=f'reports/{output_filename}')
    
    print(f"\n✅ {'Multi-output' if MULTI_OUTPUT_MODE else 'TIER 2'} model comparison complete!")
