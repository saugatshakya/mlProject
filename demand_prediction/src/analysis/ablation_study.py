"""
Ablation Study for Hourly Order Volume Prediction
==================================================

This module systematically removes feature groups to measure their impact
on model performance. Answers the question: "Do these features actually help?"

Feature Groups Tested:
1. Time-series features (lags + rolling windows)
2. Temporal features (hour, day, cyclical encoding)
3. Holiday features
4. Pattern features (hourly/daily averages)
5. External data (pollution, weather if available)

Author: Saugat Shakya
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class AblationStudy:
    """
    Ablation study for feature importance analysis.
    
    Systematically trains models with different feature subsets to measure
    the actual contribution of each feature group.
    """
    
    def __init__(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ):
        """
        Initialize ablation study.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.results = {}
        self.models = {}
        
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Define feature groups for ablation.
        
        Returns:
            Dictionary mapping group names to column patterns
        """
        all_columns = list(self.X_train.columns)
        
        # Define patterns for each group
        groups = {
            'timeseries': [],  # Lags + rolling windows
            'temporal': [],    # Hour, day, cyclical
            'holiday': [],     # Holiday features
            'patterns': [],    # Hourly/daily averages
            'pollution': [],   # Pollution data
            'weather': []      # Weather data
        }
        
        # Classify columns
        for col in all_columns:
            if any(x in col for x in ['lag', 'rolling']):
                groups['timeseries'].append(col)
            elif any(x in col for x in ['hour', 'day', 'sin_', 'cos_', 'weekend', 'month']):
                groups['temporal'].append(col)
            elif any(x in col for x in ['holiday']):
                groups['holiday'].append(col)
            elif any(x in col for x in ['_avg_', 'pattern']):
                groups['patterns'].append(col)
            elif any(x in col for x in ['aqi', 'pm', 'co', 'no', 'o3', 'so2', 'nh3']):
                groups['pollution'].append(col)
            elif any(x in col for x in ['temp', 'humidity', 'pressure', 'wind', 'precipitation']):
                groups['weather'].append(col)
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if len(v) > 0}
        
        logger.info("Feature groups identified:")
        for group, features in groups.items():
            logger.info(f"  {group}: {len(features)} features")
        
        return groups
    
    def train_model_with_features(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        experiment_name: str
    ) -> Dict[str, float]:
        """
        Train model with specific feature subset and evaluate.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            experiment_name: Name of experiment
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"\nTraining: {experiment_name}")
        logger.info(f"  Features: {X_train.shape[1]}")
        
        # Train XGBoost model with best parameters
        model = XGBRegressor(
            learning_rate=0.05,
            max_depth=3,
            n_estimators=100,
            random_state=42,
            objective='reg:squarederror',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        metrics = {
            'experiment': experiment_name,
            'n_features': X_train.shape[1],
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        logger.info(f"  Test R²: {metrics['test_r2']:.4f}, MAE: {metrics['test_mae']:.4f}")
        
        return metrics, model
    
    def run_ablation_study(self) -> pd.DataFrame:
        """
        Run complete ablation study.
        
        Returns:
            DataFrame with all results
        """
        logger.info("=" * 60)
        logger.info("STARTING ABLATION STUDY")
        logger.info("=" * 60)
        
        # Get feature groups
        groups = self.get_feature_groups()
        
        results_list = []
        
        # 1. FULL MODEL (baseline)
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 1: FULL MODEL (Baseline)")
        logger.info("="*60)
        
        metrics, model = self.train_model_with_features(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            "FULL MODEL"
        )
        results_list.append(metrics)
        self.models['FULL MODEL'] = model
        baseline_r2 = metrics['test_r2']
        
        # 2. Remove each group individually
        for group_name, group_features in groups.items():
            logger.info("\n" + "="*60)
            logger.info(f"EXPERIMENT: WITHOUT {group_name.upper()}")
            logger.info("="*60)
            
            # Create feature set without this group
            remaining_features = [col for col in self.X_train.columns 
                                 if col not in group_features]
            
            X_train_subset = self.X_train[remaining_features]
            X_test_subset = self.X_test[remaining_features]
            
            metrics, model = self.train_model_with_features(
                X_train_subset, self.y_train,
                X_test_subset, self.y_test,
                f"NO {group_name.upper()}"
            )
            
            # Calculate impact
            metrics['r2_drop'] = baseline_r2 - metrics['test_r2']
            metrics['r2_drop_pct'] = (metrics['r2_drop'] / baseline_r2) * 100
            
            results_list.append(metrics)
            self.models[f"NO {group_name.upper()}"] = model
        
        # 3. ONLY TIME-SERIES FEATURES (if available)
        if 'timeseries' in groups and len(groups['timeseries']) > 0:
            logger.info("\n" + "="*60)
            logger.info("EXPERIMENT: ONLY TIME-SERIES FEATURES")
            logger.info("="*60)
            
            X_train_subset = self.X_train[groups['timeseries']]
            X_test_subset = self.X_test[groups['timeseries']]
            
            metrics, model = self.train_model_with_features(
                X_train_subset, self.y_train,
                X_test_subset, self.y_test,
                "ONLY TIME-SERIES"
            )
            
            metrics['r2_drop'] = baseline_r2 - metrics['test_r2']
            metrics['r2_drop_pct'] = (metrics['r2_drop'] / baseline_r2) * 100
            
            results_list.append(metrics)
            self.models["ONLY TIME-SERIES"] = model
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_list)
        self.results = results_df
        
        logger.info("\n" + "="*60)
        logger.info("ABLATION STUDY COMPLETE")
        logger.info("="*60)
        
        return results_df
    
    def plot_ablation_results(self, output_dir: str) -> None:
        """
        Create comprehensive ablation study visualizations.
        
        Args:
            output_dir: Directory to save figures
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = self.results.copy()
        
        # === FIGURE 1: Ablation Study Overview (4 panels) ===
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Panel 1: R² Comparison
        ax = axes[0, 0]
        x = range(len(df))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], df['train_r2'], width, 
               label='Train R²', alpha=0.7, color='skyblue')
        ax.bar([i + width/2 for i in x], df['test_r2'], width,
               label='Test R²', alpha=0.7, color='coral')
        
        # Baseline line
        baseline_r2 = df[df['experiment'] == 'FULL MODEL']['test_r2'].values[0]
        ax.axhline(y=baseline_r2, color='green', linestyle='--', 
                  label=f'Baseline (R²={baseline_r2:.4f})', linewidth=2)
        
        ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax.set_title('R² Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['experiment'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Panel 2: Performance Drop from Baseline
        ax = axes[0, 1]
        df_with_drop = df[df['experiment'] != 'FULL MODEL'].copy()
        
        colors = ['red' if x > 0 else 'green' for x in df_with_drop['r2_drop']]
        
        ax.barh(df_with_drop['experiment'], df_with_drop['r2_drop'], color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('R² Drop from Baseline', fontsize=12, fontweight='bold')
        ax.set_title('Performance Drop (Negative = Improvement!)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Panel 3: Feature Count vs Performance
        ax = axes[1, 0]
        ax.scatter(df['n_features'], df['test_r2'], s=200, alpha=0.6, c='purple')
        
        for i, row in df.iterrows():
            ax.annotate(row['experiment'], 
                       (row['n_features'], row['test_r2']),
                       fontsize=8, ha='right', va='bottom')
        
        ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test R² Score', fontsize=12, fontweight='bold')
        ax.set_title('Feature Count vs Performance', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Panel 4: MAE Comparison
        ax = axes[1, 1]
        ax.barh(df['experiment'], df['test_mae'], color='teal', alpha=0.7)
        ax.set_xlabel('Mean Absolute Error (orders)', fontsize=12, fontweight='bold')
        ax.set_title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / '01_ablation_study_overview.png', bbox_inches='tight')
        logger.info(f"Saved: {output_dir / '01_ablation_study_overview.png'}")
        plt.close()
        
        # === FIGURE 2: Feature Group Importance ===
        df_groups = df[df['experiment'].str.startswith('NO ')].copy()
        
        if len(df_groups) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Sort by R² drop (most harmful first)
            df_groups = df_groups.sort_values('r2_drop', ascending=False)
            
            # Panel 1: R² drop
            ax = axes[0]
            colors = ['red' if x > 0 else 'green' for x in df_groups['r2_drop']]
            ax.barh(df_groups['experiment'], df_groups['r2_drop'], color=colors, alpha=0.7)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel('R² Drop When Removed', fontsize=12, fontweight='bold')
            ax.set_title('Feature Group Importance\n(Positive = Hurts Performance)', 
                        fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (idx, row) in enumerate(df_groups.iterrows()):
                ax.text(row['r2_drop'], i, f"  {row['r2_drop']:.4f} ({row['r2_drop_pct']:.2f}%)",
                       va='center', fontsize=9)
            
            # Panel 2: Impact per feature
            ax = axes[1]
            df_groups['impact_per_feature'] = df_groups['r2_drop'] / (
                df[df['experiment'] == 'FULL MODEL']['n_features'].values[0] - df_groups['n_features']
            )
            
            df_groups_sorted = df_groups.sort_values('impact_per_feature', ascending=False)
            colors = ['red' if x > 0 else 'green' for x in df_groups_sorted['impact_per_feature']]
            
            ax.barh(df_groups_sorted['experiment'], df_groups_sorted['impact_per_feature'], 
                   color=colors, alpha=0.7)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel('R² Drop per Feature', fontsize=12, fontweight='bold')
            ax.set_title('Impact per Feature\n(Efficiency of Harm)', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(output_dir / '02_feature_group_importance.png', bbox_inches='tight')
            logger.info(f"Saved: {output_dir / '02_feature_group_importance.png'}")
            plt.close()
    
    def save_results(self, output_dir: str) -> None:
        """
        Save ablation study results.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_file = output_dir / 'ablation_study_results.csv'
        self.results.to_csv(csv_file, index=False)
        logger.info(f"Results saved to: {csv_file}")
        
        # Save as JSON
        json_file = output_dir / 'ablation_study_results.json'
        with open(json_file, 'w') as f:
            json.dump(self.results.to_dict('records'), f, indent=2)
        logger.info(f"Results saved to: {json_file}")


if __name__ == "__main__":
    # Example usage
    from train_model import DemandModelTrainer
    
    # Paths
    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    FEATURES_DATA = DATA_DIR / "processed" / "hourly_features.csv"
    OUTPUT_DIR = Path(__file__).parent.parent.parent / "docs" / "figures" / "ablation_study"
    
    # Load data
    df = pd.read_csv(FEATURES_DATA)
    
    # Prepare train/test split
    trainer = DemandModelTrainer(df)
    trainer.prepare_train_test_split(test_size=0.2)
    
    # Run ablation study
    study = AblationStudy(
        trainer.X_train, trainer.y_train,
        trainer.X_test, trainer.y_test
    )
    
    results = study.run_ablation_study()
    
    # Display results
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    print(results[['experiment', 'n_features', 'test_r2', 'test_mae', 'r2_drop']].round(4))
    
    # Save results
    study.save_results(OUTPUT_DIR)
    
    # Generate visualizations
    study.plot_ablation_results(OUTPUT_DIR)
    
    print(f"\n{'='*60}")
    print("✅ Ablation study complete!")
    print(f"{'='*60}")
