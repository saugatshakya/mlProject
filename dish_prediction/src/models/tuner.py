"""
Hyperparameter Tuning with Optuna
Optimize top 3 models: ExtraTrees, CatBoost, RandomForest

Objectives:
- Beat baseline: MAE < 0.90
- Reduce overfitting: gap < 0.15
- Improve R²: > 0.40

Trials: 100 per model with Bayesian optimization
"""

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
import time
import pickle
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.multioutput import MultiOutputRegressor
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner:
    """Optimize hyperparameters for demand forecasting models

    Notes:
    - For models that support eval-based early stopping (CatBoost, LightGBM, XGBoost)
      we use a small validation split inside each CV fold and pass
      `early_stopping_rounds` to speed up training.
    - For multi-output targets (2D y) early stopping inside scikit-learn's
      MultiOutputRegressor is not attempted (complex to wire); tuning will
      still work but without eval-based early stopping for wrapped models.
    """
    
    def __init__(self, n_trials: int = 100, n_splits: int = 5, early_stopping_rounds: int = 50):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.best_params = {}
        self.best_models = {}
        self.tuning_results = []
        
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        mask = y_true != 0
        if mask.sum() == 0:
            return 100.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def objective_extratrees(self, trial: optuna.Trial, X: np.ndarray, 
                            y: np.ndarray) -> float:
        """Objective function for ExtraTrees"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = ExtraTreesRegressor(**params)
        return self._cross_validate(model, trial, X, y)
    
    def objective_catboost(self, trial: optuna.Trial, X: np.ndarray, 
                          y: np.ndarray) -> float:
        """Objective function for CatBoost"""
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'random_state': 42,
            'verbose': 0
        }
        
        model = CatBoostRegressor(**params)
        return self._cross_validate(model, trial, X, y)
    
    def objective_lightgbm(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Objective function for LightGBM"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        model = lgb.LGBMRegressor(**params)
        return self._cross_validate(model, trial, X, y)

    def objective_xgboost(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Objective function for XGBoost"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1
        }
        model = xgb.XGBRegressor(**params)
        return self._cross_validate(model, trial, X, y)

    def objective_elasticnet(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Objective function for ElasticNet"""
        params = {
            'alpha': trial.suggest_float('alpha', 0.0001, 10.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
            'max_iter': trial.suggest_int('max_iter', 1000, 5000),
            'random_state': 42
        }
        model = ElasticNet(**params)
        return self._cross_validate(model, trial, X, y)

    def objective_mlp(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Objective function for MLP Neural Network"""
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_layer_sizes = tuple([trial.suggest_int(f'n_units_l{i}', 32, 256) for i in range(n_layers)])
        
        params = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'solver': 'adam',
            'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.01, log=True),
            'max_iter': trial.suggest_int('max_iter', 200, 1000),
            'early_stopping': True,
            'validation_fraction': 0.15,
            'n_iter_no_change': 20,
            'random_state': 42
        }
        model = MLPRegressor(**params)
        return self._cross_validate(model, trial, X, y)

    def objective_randomforest(self, trial: optuna.Trial, X: np.ndarray, 
                              y: np.ndarray) -> float:
        """Objective function for RandomForest"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**params)
        return self._cross_validate(model, trial, X, y)
    
    def _cross_validate(self, model, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """
        Cross-validate a model and return mean MAE
        
        Args:
            model: ML model instance
            X, y: Dataset
            
        Returns:
            Mean test MAE across folds
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        mae_scores = []

        # Detect multi-output once
        is_multi_output = y.ndim > 1 and y.shape[1] > 1

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fill NaN values (use train mean)
            X_train_df = pd.DataFrame(X_train)
            X_train_df = X_train_df.fillna(X_train_df.mean())
            X_test_df = pd.DataFrame(X_test)
            X_test_df = X_test_df.fillna(X_train_df.mean())

            X_train_filled = X_train_df.values
            X_test_filled = X_test_df.values

            # If multi-output and model doesn't natively support it, wrap here
            if is_multi_output and isinstance(model, (CatBoostRegressor, lgb.LGBMRegressor, xgb.XGBRegressor, ElasticNet, MLPRegressor)):
                current_model = MultiOutputRegressor(model, n_jobs=-1)
            else:
                current_model = model

            # Early stopping will be used only for single-output and for boosters
            supports_eval = isinstance(model, (CatBoostRegressor, lgb.LGBMRegressor, xgb.XGBRegressor))
            use_early_stop = supports_eval and (not is_multi_output) and (self.early_stopping_rounds > 0)

            # MLPRegressor has built-in early stopping, ElasticNet doesn't need scaling (we'll scale data for MLP)
            needs_scaling = isinstance(model, MLPRegressor)
            
            if needs_scaling and not is_multi_output:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_filled)
                X_test_scaled = scaler.transform(X_test_filled)
            else:
                X_train_scaled = X_train_filled
                X_test_scaled = X_test_filled

            try:
                if use_early_stop:
                    # Reserve the last 10-15% of training fold as validation (time-aware)
                    val_size = max(1, int(0.12 * len(X_train_scaled)))
                    X_tr_inner = X_train_scaled[:-val_size]
                    y_tr_inner = y_train[:-val_size]
                    X_val = X_train_scaled[-val_size:]
                    y_val = y_train[-val_size:]

                    # Call fit with the appropriate API per library
                    if isinstance(model, CatBoostRegressor):
                        current_model.fit(X_tr_inner, y_tr_inner, eval_set=(X_val, y_val), early_stopping_rounds=self.early_stopping_rounds, verbose=False)
                    elif isinstance(model, lgb.LGBMRegressor):
                        current_model.fit(X_tr_inner, y_tr_inner, eval_set=[(X_val, y_val)], early_stopping_rounds=self.early_stopping_rounds, verbose=False)
                    elif isinstance(model, xgb.XGBRegressor):
                        current_model.fit(X_tr_inner, y_tr_inner, eval_set=[(X_val, y_val)], early_stopping_rounds=self.early_stopping_rounds, verbose=False)
                    else:
                        current_model.fit(X_train_scaled, y_train)
                else:
                    # MLPRegressor and others use standard fit
                    current_model.fit(X_train_scaled, y_train)

                y_pred = current_model.predict(X_test_scaled)
            except Exception as e:
                # Fallback: try simple fit/predict once more and propagate informative message
                try:
                    current_model.fit(X_train_filled, y_train)
                    y_pred = current_model.predict(X_test_filled)
                except Exception as final_e:
                    print(f"  [Error] Model fitting failed twice for trial: {final_e}")
                    raise final_e

            # Calculate MAE
            mae = mean_absolute_error(y_test, y_pred)
            mae_scores.append(mae)

        return np.mean(mae_scores)
    
    def tune_model(self, model_name: str, X: pd.DataFrame, y: pd.DataFrame) -> dict:
        """
        Tune hyperparameters for a specific model
        
        Args:
            model_name: Name of the model ('ExtraTrees', 'CatBoost', 'RandomForest')
            X, y: Dataset
            
        Returns:
            Dictionary with best parameters and performance
        """
        print(f"\n{'='*80}")
        print(f"TUNING: {model_name}")
        print(f"{'='*80}")
        print(f"Trials: {self.n_trials}")
        print(f"CV Folds: {self.n_splits}")
        
        # Select objective function
        if model_name == 'ExtraTrees':
            objective = lambda trial: self.objective_extratrees(trial, X.values, y.values)
        elif model_name == 'CatBoost':
            objective = lambda trial: self.objective_catboost(trial, X.values, y.values)
        elif model_name == 'RandomForest':
            objective = lambda trial: self.objective_randomforest(trial, X.values, y.values)
        elif model_name == 'LightGBM':
            objective = lambda trial: self.objective_lightgbm(trial, X.values, y.values)
        elif model_name == 'XGBoost':
            objective = lambda trial: self.objective_xgboost(trial, X.values, y.values)
        elif model_name == 'ElasticNet':
            objective = lambda trial: self.objective_elasticnet(trial, X.values, y.values)
        elif model_name == 'MLP':
            objective = lambda trial: self.objective_mlp(trial, X.values, y.values)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        start_time = time.time()
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        tuning_time = time.time() - start_time
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"\n✓ Tuning complete!")
        print(f"  Best MAE: {best_score:.4f}")
        print(f"  Tuning time: {tuning_time:.1f}s")
        print(f"  Best parameters:")
        for param, value in best_params.items():
            print(f"    {param}: {value}")
        
        # Train final model with best parameters
        print(f"\nTraining final model with best parameters...")
        if model_name == 'ExtraTrees':
            base_model = ExtraTreesRegressor(**best_params, random_state=42, n_jobs=-1)
        elif model_name == 'CatBoost':
            base_model = CatBoostRegressor(**best_params, random_state=42, verbose=0)
        elif model_name == 'RandomForest':
            base_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        elif model_name == 'LightGBM':
            base_model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
        elif model_name == 'XGBoost':
            base_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        elif model_name == 'ElasticNet':
            base_model = ElasticNet(**best_params, random_state=42)
        elif model_name == 'MLP':
            base_model = MLPRegressor(**best_params, random_state=42)

        # Wrap model for multi-output if needed
        if y.ndim > 1 and y.shape[1] > 1:
            final_model = MultiOutputRegressor(base_model, n_jobs=-1)
        else:
            final_model = base_model

        # Evaluate on full dataset with time-series split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        X_train_filled = X_train.fillna(X_train.mean()).values
        X_test_filled = X_test.fillna(X_train.mean()).values
        
        final_model.fit(X_train_filled, y_train)
        
        # Predictions
        y_train_pred = final_model.predict(X_train_filled)
        y_test_pred = final_model.predict(X_test_filled)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mape = self.calculate_mape(y_train, y_train_pred)
        test_mape = self.calculate_mape(y_test, y_test_pred)
        overfit_gap = test_mae - train_mae
        
        results = {
            'model': model_name,
            'best_params': best_params,
            'cv_mae': best_score,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'overfit_gap': overfit_gap,
            'tuning_time': tuning_time
        }
        
        print(f"\nFinal Performance:")
        print(f"  Train MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Overfit Gap: {overfit_gap:.4f}")
        
        # Store best model and parameters
        self.best_models[model_name] = final_model
        self.best_params[model_name] = best_params
        self.tuning_results.append(results)

        # --- PLOTTING DIAGNOSTICS ---
        try:
            output_dir = Path('reports/plots')
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Saving diagnostic plots to {output_dir}...")

            # Plot for each target variable in multi-output
            num_targets = y_test.shape[1] if y_test.ndim > 1 else 1
            target_names = y_test.columns if num_targets > 1 else [y_test.name]

            for i in range(num_targets):
                y_test_i = y_test.iloc[:, i] if num_targets > 1 else y_test
                y_test_pred_i = y_test_pred[:, i] if num_targets > 1 else y_test_pred
                target_name = target_names[i]

                # Predicted vs Actual
                plt.figure(figsize=(8, 8))
                sns.scatterplot(x=y_test_i, y=y_test_pred_i, alpha=0.6)
                plt.plot([y_test_i.min(), y_test_i.max()], [y_test_i.min(), y_test_i.max()], 'r--', lw=2)
                plt.xlabel(f"Actual {target_name}")
                plt.ylabel(f"Predicted {target_name}")
                plt.title(f"{model_name}: Predicted vs Actual for {target_name}", fontsize=14)
                plt.tight_layout()
                plt.savefig(output_dir / f'{model_name}_{target_name}_pred_vs_actual.png')
                plt.close()

                # Residuals
                residuals = y_test_i - y_test_pred_i
                plt.figure(figsize=(8, 6))
                sns.histplot(residuals, bins=50, kde=True)
                plt.title(f"{model_name}: Residuals for {target_name}", fontsize=14)
                plt.xlabel("Prediction Error")
                plt.tight_layout()
                plt.savefig(output_dir / f'{model_name}_{target_name}_residuals.png')
                plt.close()

            # Feature Importance (if available)
            # For multi-output, we inspect the first estimator
            estimator = final_model.estimator if hasattr(final_model, 'estimator') else final_model
            if hasattr(estimator, 'feature_importances_'):
                fi = pd.DataFrame({
                    'feature': X.columns,
                    'importance': estimator.feature_importances_
                }).sort_values('importance', ascending=False).head(30)

                plt.figure(figsize=(10, 12))
                sns.barplot(x='importance', y='feature', data=fi)
                plt.title(f"{model_name}: Top 30 Feature Importances", fontsize=14)
                plt.tight_layout()
                plt.savefig(output_dir / f'{model_name}_feature_importance.png')
                plt.close()
            print("  ✓ Plots saved.")

        except Exception as e:
            print(f"  ⚠ Warning: Could not generate plots. Reason: {e}")
        
        return results
    
    def run_tuning(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """
        Run hyperparameter tuning for all models
        
        Args:
            X: Feature DataFrame
            y: Target variable(s)
            
        Returns:
            DataFrame with tuning results
        """
        print("="*80)
        print("HYPERPARAMETER TUNING")
        print("="*80)
        print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
        if y.ndim > 1:
            print(f"Targets: {y.columns.tolist()}")
        else:
            print(f"Target: {y.name}")
        
        # Models to tune - Expanded list
        models = ['CatBoost', 'ExtraTrees', 'LightGBM', 'XGBoost', 'RandomForest']
        
        # Tune each model
        for model_name in models:
            self.tune_model(model_name, X, y)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.tuning_results)
        results_df = results_df.sort_values('test_mae').reset_index(drop=True)
        
        return results_df
    
    def print_summary(self, results_df: pd.DataFrame):
        """Print tuning summary"""
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING SUMMARY")
        print("="*80)
        
        baseline_mae = 0.443 # From multi-output baseline
        
        print(f"\n{'Rank':<6} {'Model':<20} {'Test MAE':<12} {'Test R²':<10} {'Overfit Gap':<12} {'Time (s)':<12}")
        print("-" * 80)
        
        for i, row in results_df.iterrows():
            mae_str = f"{row['test_mae']:.4f}"
            r2_str = f"{row['test_r2']:.4f}"
            gap_str = f"{row['overfit_gap']:.4f}"
            time_str = f"{row['tuning_time']/60:.1f} min"
            
            print(f"  {i+1:<5} {row['model']:<20} {mae_str:<12} {r2_str:<10} {gap_str:<12} {time_str:<12}")
        
        print("\n" + "="*80)
        best_model = results_df.iloc[0]
        print(f"Best Model: {best_model['model']}")
        print(f"Best MAE: {best_model['test_mae']:.4f}")
        print(f"Best R²: {best_model['test_r2']:.4f}")
        
        if best_model['test_mae'] < baseline_mae:
            improvement = (baseline_mae - best_model['test_mae']) / baseline_mae * 100
            print(f"✓ Improved over baseline by {improvement:.1f}%")
        
        if abs(best_model['overfit_gap']) < 0.15:
            print(f"✓ Low overfitting (gap = {best_model['overfit_gap']:.4f})")
        else:
            print(f"⚠ Moderate overfitting (gap = {best_model['overfit_gap']:.4f})")
        
        print("="*80)
    
    def save_results(self, results_df: pd.DataFrame,
                    output_dir: str = 'models/tuned'):
        """Save tuned models and results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results CSV
        results_df.to_csv(output_path / 'tuning_results.csv', index=False)
        print(f"\n✓ Results saved to: {output_path / 'tuning_results.csv'}")
        
        # Save best parameters
        params_df = pd.DataFrame([
            {'model': name, **params} 
            for name, params in self.best_params.items()
        ])
        params_df.to_csv(output_path / 'best_parameters.csv', index=False)
        print(f"✓ Parameters saved to: {output_path / 'best_parameters.csv'}")
        
        # Save models
        for model_name, model in self.best_models.items():
            model_path = output_path / f'{model_name.lower()}_tuned.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Model saved to: {model_path}")


if __name__ == "__main__":
    # Load data with selected features
    print("Loading data...")
    
    try:
        # Load all features and targets from the tier2 dataset
        full_df = pd.read_csv('data/processed/tier2_features.csv')
        full_df['hour'] = pd.to_datetime(full_df['hour'])
        
        # Load the list of selected feature names
        with open('reports/selected_features_tier2.txt', 'r') as f:
            lines = f.readlines()
            feature_cols = []
            for line in lines:
                if '|' not in line or not line.strip():
                    continue
                # Extract feature name between the first '.' and the first '|'
                feature_part = line.split('|')[0]
                dot_index = feature_part.find('.')
                if dot_index != -1:
                    feature_name = feature_part[dot_index + 1:].strip()
                    feature_cols.append(feature_name)

        # Define top 5 dishes as targets
        target_cols = [
            'Bageecha Pizza', 'Makhani Paneer Pizza', 'Margherita Pizza',
            'Peri Peri Paneer Pizza', 'Chilli Cheese Garlic Bread'
        ]
        
        # Ensure all target columns exist
        missing_targets = [t for t in target_cols if t not in full_df.columns]
        if missing_targets:
            raise ValueError(f"Target columns not found in dataset: {missing_targets}")

        X = full_df[feature_cols]
        y = full_df[target_cols]

    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")
        print("Please ensure 'tier2_features.csv' and 'selected_features_tier2.txt' exist.")
        exit()

    print(f"\nDataset prepared:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Targets: {y.columns.tolist()}")
    
    # Run tuning (15 trials for speed, 50 for better results)
    tuner = HyperparameterTuner(n_trials=15, n_splits=5, early_stopping_rounds=20)
    results = tuner.run_tuning(X, y)
    
    # Print summary
    tuner.print_summary(results)
    
    # Save results
    tuner.save_results(results, output_dir='models/tuned_multi_output')
    
    print("\n✅ Hyperparameter tuning complete!")
