"""
Train Model Module - Kitchen Prep Time Prediction

This module handles:
- Train/test splitting
- Restaurant feature calculation (on train set only to avoid leakage)
- Target transformation (log1p)
- Model training with hyperparameter tuning
- Model evaluation and comparison
- Model persistence

Author: AI Assistant
Date: November 2025
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Any, List
from collections import OrderedDict

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class PrepTimeModelTrainer:
    """
    Train and evaluate models for kitchen prep time prediction.
    """
    
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        n_cv_folds: int = 5,
        n_random_search_iter: int = 20
    ):
        """
        Initialize trainer.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
        n_cv_folds : int
            Number of cross-validation folds
        n_random_search_iter : int
            Number of iterations for RandomizedSearchCV
        """
        self.test_size = test_size
        self.random_state = random_state
        self.n_cv_folds = n_cv_folds
        self.n_random_search_iter = n_random_search_iter
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.selected_features = None
        self.best_model = None
        self.best_model_name = None
        self.results_df = None
        
    @staticmethod
    def _custom_mae_scorer(y_log_true, y_log_pred, **kwargs):
        """
        Custom scorer: returns NEGATIVE MAE in minutes.
        
        The model predicts log(KPT+1), so we need to:
        1. Transform predictions back to minutes
        2. Transform true values back to minutes
        3. Calculate MAE
        4. Return negative (sklearn maximizes scores)
        
        Parameters:
        -----------
        y_log_true : array-like
            True target values in log space
        y_log_pred : array-like
            Predicted values in log space
            
        Returns:
        --------
        float : Negative MAE in minutes
        """
        y_true = np.expm1(y_log_true)
        y_pred = np.expm1(y_log_pred)
        return -mean_absolute_error(y_true, y_pred)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "KPT duration (minutes)"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training:
        - Handle missing values in target
        - Split into features and target
        - Apply log transform to target
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with all features
        target_col : str
            Name of target column
            
        Returns:
        --------
        X : pd.DataFrame
            Features
        y_log : pd.Series
            Log-transformed target
        """
        print("\n=== Preparing Data ===")
        
        # Drop rows with missing target
        df = df.dropna(subset=[target_col]).copy()
        print(f"Data shape after dropping missing target: {df.shape}")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Log transform target (to handle skewness)
        y_log = np.log1p(y)
        print(f"Target range: {y.min():.2f} to {y.max():.2f} minutes")
        print(f"Target log range: {y_log.min():.3f} to {y_log.max():.3f}")
        
        return X, y_log
    
    def calculate_restaurant_features(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        restaurant_col: str = "Restaurant ID",
        kpt_col: str = "KPT duration (minutes)",
        wait_col: str = "Rider wait time (minutes)"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate restaurant statistics from TRAINING SET ONLY.
        
        This prevents data leakage - we don't want the model to "see"
        test set restaurant statistics during training.
        
        Parameters:
        -----------
        df_train : pd.DataFrame
            Training dataframe
        df_test : pd.DataFrame
            Test dataframe
        restaurant_col : str
            Name of restaurant identifier column
        kpt_col : str
            Name of KPT column
        wait_col : str
            Name of wait time column
            
        Returns:
        --------
        df_train : pd.DataFrame
            Training data with restaurant features
        df_test : pd.DataFrame
            Test data with restaurant features
        """
        print("\n=== Calculating Restaurant Features (from train set only) ===")
        
        # Drop placeholder restaurant features if they exist
        rest_feature_cols = ['rest_mean_KPT', 'rest_p75_KPT', 'rest_mean_wait']
        df_train = df_train.drop(columns=rest_feature_cols, errors='ignore')
        df_test = df_test.drop(columns=rest_feature_cols, errors='ignore')
        
        # Calculate stats from training set
        rest_stats = df_train.groupby(restaurant_col).agg({
            kpt_col: ['mean', lambda x: x.quantile(0.75)],
            wait_col: 'mean'
        })
        
        rest_stats.columns = ['rest_mean_KPT', 'rest_p75_KPT', 'rest_mean_wait']
        rest_stats = rest_stats.reset_index()
        
        print(f"Calculated stats for {len(rest_stats)} restaurants")
        
        # Global averages (fallback for unseen restaurants)
        global_mean_kpt = df_train[kpt_col].mean()
        global_p75_kpt = df_train[kpt_col].quantile(0.75)
        global_mean_wait = df_train[wait_col].mean()
        
        # Merge with train
        df_train = df_train.merge(rest_stats, on=restaurant_col, how='left')
        
        # Merge with test (some restaurants may not be in train)
        df_test = df_test.merge(rest_stats, on=restaurant_col, how='left')
        
        # Fill missing values with global averages
        df_train['rest_mean_KPT'].fillna(global_mean_kpt, inplace=True)
        df_train['rest_p75_KPT'].fillna(global_p75_kpt, inplace=True)
        df_train['rest_mean_wait'].fillna(global_mean_wait, inplace=True)
        
        df_test['rest_mean_KPT'].fillna(global_mean_kpt, inplace=True)
        df_test['rest_p75_KPT'].fillna(global_p75_kpt, inplace=True)
        df_test['rest_mean_wait'].fillna(global_mean_wait, inplace=True)
        
        print("Restaurant features added successfully")
        
        return df_train, df_test
    
    def split_data(
        self,
        X: pd.DataFrame,
        y_log: pd.Series
    ) -> None:
        """
        Split data into train/test sets and select features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y_log : pd.Series
            Log-transformed target
        """
        print(f"\n=== Splitting Data ({int((1-self.test_size)*100)}% train, {int(self.test_size*100)}% test) ===")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_log,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        print(f"Train shape: {self.X_train.shape}")
        print(f"Test shape: {self.X_test.shape}")
        
        # Select features (exclude non-predictive columns)
        exclude_cols = [
            'Restaurant ID', 'Order ID', 'Order Placed At', 'order_date', 
            'Rider wait time (minutes)',
            'order_hour_utc'  # Keep order_hour but drop UTC version
        ]
        
        self.selected_features = [
            col for col in self.X_train.columns 
            if col not in exclude_cols
        ]
        
        print(f"Selected {len(self.selected_features)} features for modeling")
        
    def define_models(self) -> Dict[str, Dict]:
        """
        Define models and hyperparameter grids for tuning.
        
        Returns:
        --------
        Dict : Dictionary with model configurations
        """
        models_and_params = {
            # Linear Models
            "Ridge": {
                "pipeline": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", Ridge(max_iter=5000, random_state=self.random_state))
                ]),
                "params": {
                    "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                }
            },
            "Lasso": {
                "pipeline": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", Lasso(max_iter=5000, random_state=self.random_state))
                ]),
                "params": {
                    "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
                }
            },
            "ElasticNet": {
                "pipeline": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(max_iter=5000, random_state=self.random_state))
                ]),
                "params": {
                    "model__alpha": np.logspace(-3, 1, 5),   # 0.001 to 10
                    "model__l1_ratio": [0.1, 0.5, 0.9]
                }
            },
            
            # Tree-based Models
            "DecisionTree": {
                "pipeline": Pipeline([
                    ("model", DecisionTreeRegressor(random_state=self.random_state))
                ]),
                "params": {
                    "model__max_depth": [10, 20, 30, None],
                    "model__min_samples_split": [10, 20, 50],
                    "model__min_samples_leaf": [5, 10, 20]
                }
            },
            "RandomForest": {
                "pipeline": Pipeline([
                    ("model", RandomForestRegressor(random_state=self.random_state, n_jobs=-1))
                ]),
                "params": {
                    "model__n_estimators": [100, 200, 300],
                    "model__max_depth": [10, 20, None],
                    "model__min_samples_split": [10, 20],
                    "model__min_samples_leaf": [5, 10]
                }
            },
            "ExtraTrees": {
                "pipeline": Pipeline([
                    ("model", ExtraTreesRegressor(random_state=self.random_state, n_jobs=-1))
                ]),
                "params": {
                    "model__n_estimators": [100, 200, 300],
                    "model__max_depth": [10, 20, None],
                    "model__min_samples_split": [10, 20],
                    "model__min_samples_leaf": [5, 10]
                }
            },
            
            # Gradient Boosting Models
            "GradientBoosting": {
                "pipeline": Pipeline([
                    ("model", GradientBoostingRegressor(random_state=self.random_state))
                ]),
                "params": {
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__n_estimators": [100, 200, 300],
                    "model__max_depth": [3, 5, 7],
                    "model__min_samples_split": [10, 20]
                }
            },
            "HistGB": {
                "pipeline": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", HistGradientBoostingRegressor(
                        random_state=self.random_state
                    ))
                ]),
                "params": {
                    "model__learning_rate": [0.05, 0.1, 0.2],
                    "model__max_depth": [None, 6, 10, 15],
                    "model__max_iter": [200, 400],
                    "model__min_samples_leaf": [10, 20, 50]
                }
            },
            "XGBoost": {
                "pipeline": Pipeline([
                    ("model", XGBRegressor(random_state=self.random_state, n_jobs=-1, verbosity=0))
                ]),
                "params": {
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__max_depth": [3, 5, 7, 10],
                    "model__n_estimators": [100, 200, 300],
                    "model__min_child_weight": [1, 3, 5],
                    "model__subsample": [0.8, 0.9, 1.0]
                }
            },
            "LightGBM": {
                "pipeline": Pipeline([
                    ("model", LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbose=-1))
                ]),
                "params": {
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__max_depth": [5, 10, 15, -1],
                    "model__n_estimators": [100, 200, 300],
                    "model__num_leaves": [31, 50, 70],
                    "model__min_child_samples": [10, 20, 30]
                }
            }
        }
        
        return models_and_params
    
    def train_models(self) -> OrderedDict:
        """
        Train all models with hyperparameter tuning.
        
        Returns:
        --------
        OrderedDict : Dictionary of trained models
        """
        print("\n=== Training Models with Hyperparameter Tuning ===")
        
        # Define cross-validation
        cv = KFold(
            n_splits=self.n_cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Custom scorer
        mae_scorer = make_scorer(
            PrepTimeModelTrainer._custom_mae_scorer,
            greater_is_better=True  # We return negative MAE
        )
        
        # Get model configurations
        models_and_params = self.define_models()
        
        # Train each model
        best_searches = OrderedDict()
        
        for name, cfg in models_and_params.items():
            print(f"\n=== Tuning {name} ===")
            
            search = RandomizedSearchCV(
                estimator=cfg["pipeline"],
                param_distributions=cfg["params"],
                n_iter=self.n_random_search_iter,
                scoring=mae_scorer,
                cv=cv,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
            
            # Fit on selected features only
            search.fit(
                self.X_train[self.selected_features],
                self.y_train
            )
            
            print(f"Best CV MAE (minutes) for {name}: {-search.best_score_:.3f}")
            print(f"Best params: {search.best_params_}")
            
            best_searches[name] = search
        
        return best_searches
    
    def evaluate_models(
        self,
        best_searches: OrderedDict
    ) -> pd.DataFrame:
        """
        Evaluate all trained models on train and test sets.
        
        Parameters:
        -----------
        best_searches : OrderedDict
            Dictionary of RandomizedSearchCV results
            
        Returns:
        --------
        pd.DataFrame : Results dataframe
        """
        print("\n=== Evaluating Models ===")
        
        results = []
        
        for name, search in best_searches.items():
            best_model = search.best_estimator_
            
            # Train predictions
            y_train_log_pred = best_model.predict(self.X_train[self.selected_features])
            y_train_true = np.expm1(self.y_train)
            y_train_pred = np.expm1(y_train_log_pred)
            
            train_mae = mean_absolute_error(y_train_true, y_train_pred)
            train_r2 = r2_score(y_train_true, y_train_pred)
            
            # Test predictions
            y_test_log_pred = best_model.predict(self.X_test[self.selected_features])
            y_test_true = np.expm1(self.y_test)
            y_test_pred = np.expm1(y_test_log_pred)
            
            test_mae = mean_absolute_error(y_test_true, y_test_pred)
            test_r2 = r2_score(y_test_true, y_test_pred)
            
            results.append({
                "model": name,
                "best_params": search.best_params_,
                "cv_mae_minutes": -search.best_score_,
                "train_mae_minutes": train_mae,
                "train_r2": train_r2,
                "test_mae_minutes": test_mae,
                "test_r2": test_r2
            })
            
            print(f"\n{name}:")
            print(f"  Train MAE: {train_mae:.3f} min, R²: {train_r2:.4f}")
            print(f"  Test MAE:  {test_mae:.3f} min, R²: {test_r2:.4f}")
        
        # Sort by test MAE (best first)
        self.results_df = pd.DataFrame(results).sort_values("test_mae_minutes")
        
        return self.results_df
    
    def select_best_model(
        self,
        best_searches: OrderedDict,
        metric: str = "test_mae_minutes"
    ) -> Any:
        """
        Select the best model based on evaluation metric.
        
        Parameters:
        -----------
        best_searches : OrderedDict
            Dictionary of trained models
        metric : str
            Metric to use for selection
            
        Returns:
        --------
        Any : Best model (sklearn pipeline)
        """
        print(f"\n=== Selecting Best Model (by {metric}) ===")
        
        best_row = self.results_df.iloc[0]
        self.best_model_name = best_row["model"]
        
        print(f"Best model: {self.best_model_name}")
        print(f"Test MAE: {best_row['test_mae_minutes']:.3f} minutes")
        print(f"Test R²: {best_row['test_r2']:.4f}")
        
        self.best_model = best_searches[self.best_model_name].best_estimator_
        
        return self.best_model
    
    def retrain_on_full_data(
        self,
        X: pd.DataFrame,
        y_log: pd.Series
    ) -> None:
        """
        Retrain best model on full dataset (train + test).
        
        This is the final production model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            All features
        y_log : pd.Series
            All log-transformed targets
        """
        print("\n=== Retraining Best Model on Full Dataset ===")
        
        self.best_model.fit(X[self.selected_features], y_log)
        
        print("Model retrained successfully on all available data")
    
    def save_model(
        self,
        output_dir: str = "models/final",
        model_name: str = "best_model.pkl"
    ) -> None:
        """
        Save trained model and metadata to disk.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save model
        model_name : str
            Name of model file
        """
        print(f"\n=== Saving Model to {output_dir} ===")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_path / model_name
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"Model saved: {model_path}")
        
        # Save feature names
        features_path = output_path / "feature_names.txt"
        with open(features_path, 'w') as f:
            for feat in self.selected_features:
                f.write(f"{feat}\n")
        print(f"Feature names saved: {features_path}")
        
        # Save model configuration
        config = {
            "model_name": self.best_model_name,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "n_features": len(self.selected_features),
            "best_params": self.results_df.iloc[0]["best_params"],
            "test_mae_minutes": float(self.results_df.iloc[0]["test_mae_minutes"]),
            "test_r2": float(self.results_df.iloc[0]["test_r2"]),
            "train_mae_minutes": float(self.results_df.iloc[0]["train_mae_minutes"]),
            "train_r2": float(self.results_df.iloc[0]["train_r2"])
        }
        
        config_path = output_path / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Model config saved: {config_path}")
        
        # Save full results
        results_path = output_path / "model_comparison.csv"
        self.results_df.to_csv(results_path, index=False)
        print(f"Model comparison saved: {results_path}")
    
    def get_inference_predictions(self) -> pd.DataFrame:
        """
        Generate predictions on test set for error analysis.
        
        Returns:
        --------
        pd.DataFrame : Dataframe with actual, predicted, and error columns
        """
        print("\n=== Generating Inference Predictions ===")
        
        # Predict on test set
        y_test_log_pred = self.best_model.predict(self.X_test[self.selected_features])
        
        # Convert to minutes
        y_test_minutes = np.expm1(self.y_test)
        y_pred_minutes = np.expm1(y_test_log_pred)
        
        # Create inference dataframe
        inference_df = pd.DataFrame({
            "KPT_actual_min": y_test_minutes,
            "KPT_pred_min": y_pred_minutes
        }, index=self.X_test.index)
        
        inference_df["error_min"] = inference_df["KPT_pred_min"] - inference_df["KPT_actual_min"]
        inference_df["abs_error_min"] = inference_df["error_min"].abs()
        
        print(f"Generated predictions for {len(inference_df)} test samples")
        print(f"Mean absolute error: {inference_df['abs_error_min'].mean():.3f} minutes")
        
        return inference_df


def main():
    """
    Example usage of the PrepTimeModelTrainer.
    """
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from data.preprocessing import PrepTimePreprocessor
    from features.feature_engineering import PrepTimeFeatureEngineer
    
    print("=" * 80)
    print("KITCHEN PREP TIME PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    # 1. Load and preprocess data
    preprocessor = PrepTimePreprocessor()
    df = preprocessor.preprocess_pipeline(
        order_filepath="data/raw/orders.csv",
        events_filepath="data/raw/delhi_major_events.csv"
    )
    
    # Save preprocessed data
    preprocessor.save_processed_data("data/processed/preprocessed_orders.csv")
    
    # 2. Engineer features
    engineer = PrepTimeFeatureEngineer()
    df_features = engineer.feature_engineering_pipeline(df)
    
    # Save features
    engineer.save_features("data/processed/features_orders.csv")
    
    # 3. Prepare data for training
    trainer = PrepTimeModelTrainer(
        test_size=0.2,
        random_state=42,
        n_cv_folds=5,
        n_random_search_iter=20
    )
    
    X, y_log = trainer.prepare_data(df_features)
    
    # 4. Split data
    trainer.split_data(X, y_log)
    
    # 5. Calculate restaurant features (from train set only)
    # Merge back with original dataframe indices
    df_train = df_features.loc[trainer.X_train.index].copy()
    df_test = df_features.loc[trainer.X_test.index].copy()
    
    df_train, df_test = trainer.calculate_restaurant_features(df_train, df_test)
    
    # Update X_train and X_test with restaurant features
    trainer.X_train = df_train.drop(columns=["KPT duration (minutes)"])
    trainer.X_test = df_test.drop(columns=["KPT duration (minutes)"])
    
    # 6. Train models
    best_searches = trainer.train_models()
    
    # 7. Evaluate models
    results_df = trainer.evaluate_models(best_searches)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(results_df[["model", "cv_mae_minutes", "train_mae_minutes", "train_r2", 
                      "test_mae_minutes", "test_r2"]].to_string(index=False))
    
    # 8. Select best model
    best_model = trainer.select_best_model(best_searches)
    
    # 9. Retrain on full data
    trainer.retrain_on_full_data(X, y_log)
    
    # 10. Save model
    trainer.save_model()
    
    # 11. Generate inference predictions for error analysis
    inference_df = trainer.get_inference_predictions()
    inference_df.to_csv("data/processed/test_predictions.csv")
    print("\nTest predictions saved to: data/processed/test_predictions.csv")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best model: {trainer.best_model_name}")
    print(f"Test MAE: {trainer.results_df.iloc[0]['test_mae_minutes']:.3f} minutes")
    print(f"Test R²: {trainer.results_df.iloc[0]['test_r2']:.4f}")
    print("\nModel artifacts saved to: models/final/")


if __name__ == "__main__":
    main()
