"""
Ablation Study for Delivery Time Prediction
===========================================

This module systematically removes feature groups to measure their impact
on model performance. Answers: "Do these features actually help?"

Feature Groups Tested:
1. Temporal features (hour, day, cyclical encoding)
2. Lag features (historical delivery times)
3. Rolling window features (moving averages)
4. Restaurant features (per-restaurant patterns)
5. Distance features (distance bins, squared)
6. Pollution features (AQI, pollutants)

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

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


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
        
        logger.info("AblationStudy initialized")
        logger.info(f"  Train samples: {len(X_train):,}")
        logger.info(f"  Test samples: {len(X_test):,}")
        logger.info(f"  Total features: {len(X_train.columns)}")
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Define feature groups for ablation.
        
        Returns:
            Dictionary mapping group names to column lists
        """
        all_columns = list(self.X_train.columns)
        
        # Define patterns for each group
        groups = {
            'temporal': [],
            'lag': [],
            'rolling': [],
            'restaurant': [],
            'distance': [],
            'pollution': [],
            'other': []
        }
        
        # Classify columns
        for col in all_columns:
            if any(x in col for x in ['hour', 'day', 'month', 'weekend', 'peak', 'lunch', 'dinner', '_sin', '_cos']):
                groups['temporal'].append(col)
            elif 'lag_' in col:
                groups['lag'].append(col)
            elif 'rolling_' in col:
                groups['rolling'].append(col)
            elif 'restaurant' in col.lower():
                groups['restaurant'].append(col)
            elif any(x in col.lower() for x in ['distance', 'dist_']):
                groups['distance'].append(col)
            elif any(x in col for x in ['aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']):
                groups['pollution'].append(col)
            else:
                groups['other'].append(col)
        
        # Log group sizes
        logger.info("\nFeature groups identified:")
        for group_name, features in groups.items():
            if features:
                logger.info(f"  {group_name}: {len(features)} features")
        
        return groups
    
    def train_baseline(self) -> Dict:
        """
        Train baseline model with all features.
        
        Returns:
            Metrics dictionary
        """
        logger.info("\n" + "="*80)
        logger.info("BASELINE: Training with ALL features")
        logger.info("="*80)
        
        # Train XGBoost with all features
        model = XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = model.predict(self.X_test)
        
        metrics = {
            'r2': r2_score(self.y_test, y_pred),
            'mae': mean_absolute_error(self.y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred))
        }
        
        self.results['baseline'] = metrics
        self.models['baseline'] = model
        
        logger.info(f"  R² Score: {metrics['r2']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.2f} minutes")
        logger.info(f"  RMSE: {metrics['rmse']:.2f} minutes")
        
        return metrics
    
    def ablate_group(self, group_name: str, features_to_remove: List[str]) -> Dict:
        """
        Train model without a specific feature group.
        
        Args:
            group_name: Name of the feature group
            features_to_remove: List of features to exclude
            
        Returns:
            Metrics dictionary
        """
        if not features_to_remove:
            logger.warning(f"No features to remove for group '{group_name}'")
            return {}
        
        logger.info(f"\nABLATING: Removing {len(features_to_remove)} '{group_name}' features")
        
        # Create subset without these features
        X_train_ablated = self.X_train.drop(columns=features_to_remove, errors='ignore')
        X_test_ablated = self.X_test.drop(columns=features_to_remove, errors='ignore')
        
        logger.info(f"  Remaining features: {X_train_ablated.shape[1]}")
        
        # Train model
        model = XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_ablated, self.y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_ablated)
        
        metrics = {
            'r2': r2_score(self.y_test, y_pred),
            'mae': mean_absolute_error(self.y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred))
        }
        
        # Calculate impact (baseline - ablated)
        baseline_r2 = self.results['baseline']['r2']
        metrics['r2_drop'] = baseline_r2 - metrics['r2']
        metrics['feature_count'] = len(features_to_remove)
        
        logger.info(f"  R² Score: {metrics['r2']:.4f} (drop: {metrics['r2_drop']:.4f})")
        logger.info(f"  MAE: {metrics['mae']:.2f} minutes")
        logger.info(f"  RMSE: {metrics['rmse']:.2f} minutes")
        
        self.results[f'without_{group_name}'] = metrics
        self.models[f'without_{group_name}'] = model
        
        return metrics
    
    def run_full_ablation(self) -> pd.DataFrame:
        """
        Run complete ablation study on all feature groups.
        
        Returns:
            DataFrame with ablation results
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING FULL ABLATION STUDY")
        logger.info("="*80)
        
        # Get feature groups
        groups = self.get_feature_groups()
        
        # Train baseline
        baseline_metrics = self.train_baseline()
        
        # Ablate each group
        for group_name, features in groups.items():
            if features:  # Only ablate if group has features
                self.ablate_group(group_name, features)
        
        # Create summary DataFrame
        summary = self._create_summary()
        
        logger.info("\n" + "="*80)
        logger.info("ABLATION STUDY COMPLETE")
        logger.info("="*80)
        
        return summary
    
    def _create_summary(self) -> pd.DataFrame:
        """Create summary DataFrame from results."""
        summary_data = []
        
        baseline_r2 = self.results['baseline']['r2']
        baseline_mae = self.results['baseline']['mae']
        
        for name, metrics in self.results.items():
            if name == 'baseline':
                summary_data.append({
                    'Configuration': 'Baseline (All Features)',
                    'R² Score': metrics['r2'],
                    'R² Drop': 0.0,
                    'MAE': metrics['mae'],
                    'MAE Increase': 0.0,
                    'Features Removed': 0
                })
            else:
                group_name = name.replace('without_', '')
                summary_data.append({
                    'Configuration': f'Without {group_name}',
                    'R² Score': metrics['r2'],
                    'R² Drop': metrics.get('r2_drop', 0),
                    'MAE': metrics['mae'],
                    'MAE Increase': metrics['mae'] - baseline_mae,
                    'Features Removed': metrics.get('feature_count', 0)
                })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('R² Drop', ascending=False)
        
        # Display summary
        print("\n" + "="*80)
        print("ABLATION STUDY SUMMARY")
        print("="*80)
        print(df_summary.to_string(index=False))
        print("="*80)
        
        return df_summary
    
    def plot_results(self, save_path: Path = None):
        """
        Visualize ablation study results.
        
        Args:
            save_path: Path to save the plot
        """
        summary = self._create_summary()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: R² Drop
        ax1 = axes[0]
        configs = summary[summary['Configuration'] != 'Baseline (All Features)']['Configuration']
        r2_drops = summary[summary['Configuration'] != 'Baseline (All Features)']['R² Drop']
        
        colors = ['red' if x > 0.02 else 'orange' if x > 0.01 else 'green' for x in r2_drops]
        ax1.barh(configs, r2_drops, color=colors, edgecolor='black')
        ax1.set_xlabel('R² Score Drop', fontsize=12, fontweight='bold')
        ax1.set_title('Feature Group Importance (R² Drop When Removed)', fontsize=14, fontweight='bold')
        ax1.axvline(x=0.01, color='gray', linestyle='--', alpha=0.5, label='0.01 threshold')
        ax1.axvline(x=0.02, color='red', linestyle='--', alpha=0.5, label='0.02 threshold')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: MAE Increase
        ax2 = axes[1]
        mae_increases = summary[summary['Configuration'] != 'Baseline (All Features)']['MAE Increase']
        
        colors2 = ['red' if x > 1.0 else 'orange' if x > 0.5 else 'green' for x in mae_increases]
        ax2.barh(configs, mae_increases, color=colors2, edgecolor='black')
        ax2.set_xlabel('MAE Increase (minutes)', fontsize=12, fontweight='bold')
        ax2.set_title('Prediction Error Increase When Group Removed', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ablation plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_results(self, output_path: Path):
        """
        Save ablation results to JSON.
        
        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Saved ablation results to {output_path}")
