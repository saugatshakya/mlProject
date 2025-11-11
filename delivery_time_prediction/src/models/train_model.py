"""
Model Training for Delivery Time Prediction
===========================================

This module trains and compares multiple regression models:
- XGBoost
- LightGBM
- CatBoost
- Random Forest

Includes hyperparameter tuning, cross-validation, and model evaluation.

Author: Saugat Shakya
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
import json
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


class ModelTrainer:
    """
    Train and evaluate delivery time prediction models.
    
    Workflow:
        1. Train baseline models with default parameters
        2. Compare model performance
        3. Tune best model(s)
        4. Final evaluation
    """
    
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5
    ):
        """
        Initialize ModelTrainer.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed
            cv_folds: Number of cross-validation folds
        """
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
        logger.info(f"ModelTrainer initialized")
        logger.info(f"  Test size: {test_size}")
        logger.info(f"  Random state: {random_state}")
        logger.info(f"  CV folds: {cv_folds}")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'Total_time_taken',
        drop_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for modeling.
        
        Args:
            df: Full dataset with features and target
            target_col: Name of target variable
            drop_cols: Additional columns to drop
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Preparing data for modeling...")
        
        # Drop columns
        cols_to_drop = [target_col, 'Order Placed At', 'order_hour_utc']
        if drop_cols:
            cols_to_drop.extend(drop_cols)
        
        # Remove duplicates and NaNs from cols_to_drop
        cols_to_drop = list(set([col for col in cols_to_drop if col in df.columns]))
        
        # Prepare X and y
        X = df.drop(columns=cols_to_drop)
        y = df[target_col]
        
        # Drop rows with NaN in target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Fill remaining NaNs in features with median
        for col in X.columns:
            if X[col].isna().any():
                if X[col].dtype in ['float64', 'int64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Feature columns ({len(X.columns)}): {X.columns.tolist()}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        logger.info(f"Train set: {X_train.shape[0]:,} samples")
        logger.info(f"Test set: {X_test.shape[0]:,} samples")
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_name: str
    ) -> Dict:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_train, X_test: Feature matrices
            y_train, y_test: Target vectors
            model_name: Name for logging
            
        Returns:
            Dictionary of metrics
        """
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mape': mean_absolute_percentage_error(y_train, y_train_pred) * 100,
            'test_mape': mean_absolute_percentage_error(y_test, y_test_pred) * 100,
        }
        
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Train R²: {metrics['train_r2']:.4f} | Test R²: {metrics['test_r2']:.4f}")
        logger.info(f"  Train MAE: {metrics['train_mae']:.2f} | Test MAE: {metrics['test_mae']:.2f}")
        logger.info(f"  Train RMSE: {metrics['train_rmse']:.2f} | Test RMSE: {metrics['test_rmse']:.2f}")
        logger.info(f"  Train MAPE: {metrics['train_mape']:.2f}% | Test MAPE: {metrics['test_mape']:.2f}%")
        
        return metrics
    
    def train_baseline_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict:
        """
        Train baseline models with default/reasonable parameters.
        
        Args:
            X_train, X_test: Feature matrices
            y_train, y_test: Target vectors
            
        Returns:
            Dictionary of results
        """
        logger.info("="*80)
        logger.info("TRAINING BASELINE MODELS")
        logger.info("="*80)
        
        results = {}
        
        # 1. XGBoost
        logger.info("\n1. Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        results['xgboost'] = self.evaluate_model(
            xgb_model, X_train, X_test, y_train, y_test, "XGBoost"
        )
        
        # 2. LightGBM
        logger.info("\n2. Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        self.models['lightgbm'] = lgb_model
        results['lightgbm'] = self.evaluate_model(
            lgb_model, X_train, X_test, y_train, y_test, "LightGBM"
        )
        
        # 3. CatBoost
        logger.info("\n3. Training CatBoost...")
        cat_model = CatBoostRegressor(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            random_state=self.random_state,
            verbose=False
        )
        cat_model.fit(X_train, y_train)
        self.models['catboost'] = cat_model
        results['catboost'] = self.evaluate_model(
            cat_model, X_train, X_test, y_train, y_test, "CatBoost"
        )
        
        # 4. Random Forest
        logger.info("\n4. Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        results['random_forest'] = self.evaluate_model(
            rf_model, X_train, X_test, y_train, y_test, "Random Forest"
        )
        
        self.results = results
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models.
        
        Returns:
            DataFrame with comparison metrics
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)
        
        # Create comparison table
        comparison = []
        for model_name, metrics in self.results.items():
            comparison.append({
                'Model': model_name,
                'Test R²': metrics['test_r2'],
                'Test MAE': metrics['test_mae'],
                'Test RMSE': metrics['test_rmse'],
                'Test MAPE': metrics['test_mape'],
                'Train R²': metrics['train_r2'],
            })
        
        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.sort_values('Test R²', ascending=False)
        
        print("\n" + df_comparison.to_string(index=False))
        
        # Best model
        best_model_name = df_comparison.iloc[0]['Model']
        logger.info(f"\n🏆 Best Model: {best_model_name}")
        
        return df_comparison
    
    def tune_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None
    ):
        """
        Tune hyperparameters for a specific model.
        
        Args:
            model_name: Name of model to tune
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid for GridSearchCV
            
        Returns:
            Tuned model
        """
        logger.info(f"\nTuning {model_name}...")
        
        # Default parameter grids
        default_param_grids = {
            'xgboost': {
                'n_estimators': [500, 1000],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            },
            'lightgbm': {
                'n_estimators': [500, 1000],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            },
            'catboost': {
                'iterations': [500, 1000],
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1]
            },
            'random_forest': {
                'n_estimators': [200, 300],
                'max_depth': [12, 15, 20],
                'min_samples_split': [10, 20],
                'min_samples_leaf': [5, 10]
            }
        }
        
        if param_grid is None:
            param_grid = default_param_grids.get(model_name, {})
        
        # Get base model
        base_model = self.models[model_name]
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=self.cv_folds,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV R²: {grid_search.best_score_:.4f}")
        
        # Save tuned model
        self.models[f'{model_name}_tuned'] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def extract_feature_importance(
        self,
        model_name: str,
        feature_names: List[str],
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Extract and visualize feature importance.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            top_n: Number of top features to show
            
        Returns:
            DataFrame with feature importances
        """
        model = self.models[model_name]
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            logger.warning(f"Model {model_name} does not have feature_importances_")
            return pd.DataFrame()
        
        # Create DataFrame
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        df_importance = df_importance.sort_values('importance', ascending=False)
        
        # Save
        self.feature_importance[model_name] = df_importance
        
        # Log top features
        logger.info(f"\nTop {top_n} features for {model_name}:")
        for idx, row in df_importance.head(top_n).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return df_importance
    
    def save_models(self, output_dir: Path):
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving models to {output_dir}...")
        
        for model_name, model in self.models.items():
            model_path = output_dir / f"{model_name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"  Saved {model_name} to {model_path}")
        
        # Save results
        results_path = output_dir / "model_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"  Saved results to {results_path}")
    
    def plot_results(self, save_dir: Optional[Path] = None):
        """
        Plot model comparison and feature importance.
        
        Args:
            save_dir: Directory to save plots
        """
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Model comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(self.results.keys())
        metrics_to_plot = ['test_r2', 'test_mae', 'test_rmse', 'test_mape']
        titles = ['R² Score (Higher is Better)', 'MAE (Lower is Better)', 
                  'RMSE (Lower is Better)', 'MAPE (Lower is Better)']
        
        for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
            ax = axes[idx // 2, idx % 2]
            values = [self.results[m][metric] for m in models]
            ax.bar(models, values, color='skyblue', edgecolor='navy')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel(metric.upper())
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / "model_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
