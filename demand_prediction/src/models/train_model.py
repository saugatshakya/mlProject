"""
Model Training Pipeline for Hourly Order Volume Prediction
===========================================================

This module handles:
1. Train/test split (time-series aware)
2. Multiple model training (Linear Regression, Random Forest, XGBoost)
3. Hyperparameter tuning
4. Model evaluation and comparison
5. Model persistence

Author: Saugat Shakya
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from typing import Dict, Tuple, Any, List
import logging
from datetime import datetime

# Scikit-learn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# XGBoost
from xgboost import XGBRegressor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemandModelTrainer:
    """
    Trainer for hourly order volume prediction models.
    
    Handles model training, evaluation, hyperparameter tuning, and model selection.
    """
    
    def __init__(self, features_df: pd.DataFrame, target_col: str = 'orders_per_hour'):
        """
        Initialize trainer with feature data.
        
        Args:
            features_df: DataFrame with engineered features
            target_col: Name of target column
        """
        self.df = features_df.copy()
        self.target_col = target_col
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_df = None
        
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def prepare_train_test_split(
        self, 
        test_size: float = 0.2,
        features_to_drop: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare time-series aware train/test split.
        
        Args:
            test_size: Proportion of data for testing
            features_to_drop: List of columns to drop (e.g., date, identifiers)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing train/test split...")
        
        # Sort by date and hour to ensure chronological order
        self.df = self.df.sort_values(by=['order_date', 'order_hour'])
        
        # Time-based split (use earlier data for training)
        split_index = int(len(self.df) * (1 - test_size))
        train_df = self.df.iloc[:split_index]
        test_df = self.df.iloc[split_index:]
        
        # Store test_df for later analysis
        self.test_df = test_df.copy()
        
        # Define features to drop
        if features_to_drop is None:
            features_to_drop = [
                'order_date', 
                self.target_col,
                'orders_per_day'  # if exists
            ]
        
        # Prepare features and target
        self.X_train = train_df.drop(columns=[col for col in features_to_drop if col in train_df.columns])
        self.y_train = train_df[self.target_col]
        
        self.X_test = test_df.drop(columns=[col for col in features_to_drop if col in test_df.columns])
        self.y_test = test_df[self.target_col]
        
        logger.info(f"Train set: {len(self.X_train)} samples, {len(self.X_train.columns)} features")
        logger.info(f"Test set: {len(self.X_test)} samples")
        logger.info(f"Train period: {train_df['order_date'].min()} to {train_df['order_date'].max()}")
        logger.info(f"Test period: {test_df['order_date'].min()} to {test_df['order_date'].max()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def evaluate_model(
        self, 
        model: Any, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model performance on train and test sets.
        
        Args:
            model: Trained model
            X_train, y_train: Training data
            X_test, y_test: Test data
            model_name: Name for logging
            
        Returns:
            Dictionary of metrics
        """
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        logger.info(f"\n{model_name} Performance:")
        logger.info(f"  Train - MAE: {metrics['train_mae']:.4f}, RMSE: {metrics['train_rmse']:.4f}, R²: {metrics['train_r2']:.4f}")
        logger.info(f"  Test  - MAE: {metrics['test_mae']:.4f}, RMSE: {metrics['test_rmse']:.4f}, R²: {metrics['test_r2']:.4f}")
        
        return metrics
    
    def train_linear_regression(self) -> Tuple[Any, Dict]:
        """Train Linear Regression model."""
        logger.info("\n" + "="*60)
        logger.info("Training Linear Regression")
        logger.info("="*60)
        
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        metrics = self.evaluate_model(
            model, self.X_train, self.y_train, 
            self.X_test, self.y_test, "Linear Regression"
        )
        
        self.models['Linear Regression'] = model
        self.results['Linear Regression'] = metrics
        
        return model, metrics
    
    def train_random_forest(self, tune_hyperparameters: bool = False) -> Tuple[Any, Dict]:
        """
        Train Random Forest model.
        
        Args:
            tune_hyperparameters: Whether to run hyperparameter tuning
            
        Returns:
            Tuple of (model, metrics)
        """
        logger.info("\n" + "="*60)
        logger.info("Training Random Forest")
        logger.info("="*60)
        
        if tune_hyperparameters:
            logger.info("Running hyperparameter tuning...")
            
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            
            # Use TimeSeriesSplit for cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            grid_search = GridSearchCV(
                rf, param_grid, cv=tscv, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            # Use default parameters
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(self.X_train, self.y_train)
        
        metrics = self.evaluate_model(
            model, self.X_train, self.y_train,
            self.X_test, self.y_test, "Random Forest"
        )
        
        self.models['Random Forest'] = model
        self.results['Random Forest'] = metrics
        
        return model, metrics
    
    def train_xgboost(self, tune_hyperparameters: bool = False) -> Tuple[Any, Dict]:
        """
        Train XGBoost model.
        
        Args:
            tune_hyperparameters: Whether to run hyperparameter tuning
            
        Returns:
            Tuple of (model, metrics)
        """
        logger.info("\n" + "="*60)
        logger.info("Training XGBoost")
        logger.info("="*60)
        
        if tune_hyperparameters:
            logger.info("Running hyperparameter tuning...")
            
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            
            xgb = XGBRegressor(
                random_state=42,
                objective='reg:squarederror',
                n_jobs=-1
            )
            
            # Use TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=3)
            
            grid_search = GridSearchCV(
                xgb, param_grid, cv=tscv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            # Use tuned parameters from notebook
            model = XGBRegressor(
                learning_rate=0.05,
                max_depth=3,
                n_estimators=100,
                random_state=42,
                objective='reg:squarederror',
                n_jobs=-1
            )
            model.fit(self.X_train, self.y_train)
        
        metrics = self.evaluate_model(
            model, self.X_train, self.y_train,
            self.X_test, self.y_test, "XGBoost"
        )
        
        self.models['XGBoost'] = model
        self.results['XGBoost'] = metrics
        
        return model, metrics
    
    def train_all_models(self, tune_hyperparameters: bool = False) -> Dict:
        """
        Train all models and compare performance.
        
        Args:
            tune_hyperparameters: Whether to tune hyperparameters
            
        Returns:
            Dictionary of all results
        """
        logger.info("\n" + "="*60)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*60)
        
        # Train models
        self.train_linear_regression()
        self.train_random_forest(tune_hyperparameters=tune_hyperparameters)
        self.train_xgboost(tune_hyperparameters=tune_hyperparameters)
        
        # Find best model based on test R²
        best_r2 = -np.inf
        for name, metrics in self.results.items():
            if metrics['test_r2'] > best_r2:
                best_r2 = metrics['test_r2']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*60)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.results).T
        print(comparison_df.round(4))
        
        logger.info(f"\n✅ Best Model: {self.best_model_name} (Test R² = {best_r2:.4f})")
        
        return self.results
    
    def save_model(self, model: Any, model_name: str, output_dir: str) -> str:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            model_name: Name for the model file
            output_dir: Directory to save model
            
        Returns:
            Path to saved model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name.replace(' ', '_').lower()}_{timestamp}.pkl"
        filepath = output_dir / filename
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved to: {filepath}")
        
        return str(filepath)
    
    def save_results(self, output_dir: str) -> None:
        """
        Save training results and metrics.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results as JSON
        results_file = output_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Save results as CSV
        results_df = pd.DataFrame(self.results).T
        csv_file = output_dir / "training_results.csv"
        results_df.to_csv(csv_file)
        
        logger.info(f"Results CSV saved to: {csv_file}")
    
    def save_all_models(self, output_dir: str) -> None:
        """
        Save all trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            if model is not None:
                filename = f"{name.replace(' ', '_').lower()}_model.pkl"
                filepath = output_dir / filename
                
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
                
                logger.info(f"✅ Saved: {filename}")
        
        logger.info(f"\nAll models saved to: {output_dir}")
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            model_name: Name of model (if None, uses best model)
            
        Returns:
            DataFrame with feature importances
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            logger.warning(f"Model {model_name} does not have feature importance")
            return None
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # Paths
    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    FEATURES_DATA = DATA_DIR / "processed" / "hourly_features.csv"
    MODELS_DIR = Path(__file__).parent.parent.parent / "models"
    
    # Load features
    logger.info(f"Loading features from {FEATURES_DATA}")
    df = pd.read_csv(FEATURES_DATA)
    
    # Initialize trainer
    trainer = DemandModelTrainer(df)
    
    # Prepare data
    trainer.prepare_train_test_split(test_size=0.2)
    
    # Train all models (set tune_hyperparameters=True for full tuning)
    results = trainer.train_all_models(tune_hyperparameters=False)
    
    # Save best model
    trainer.save_model(trainer.best_model, trainer.best_model_name, MODELS_DIR)
    
    # Save results
    trainer.save_results(MODELS_DIR)
    
    # Get feature importance
    if trainer.best_model_name in ['Random Forest', 'XGBoost']:
        feature_importance = trainer.get_feature_importance()
        print(f"\n{'='*60}")
        print("TOP 20 MOST IMPORTANT FEATURES")
        print(f"{'='*60}")
        print(feature_importance.head(20))
