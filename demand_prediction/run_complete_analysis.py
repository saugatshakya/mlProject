"""
Complete Analysis Pipeline for Demand Prediction
================================================

This script runs the complete analysis pipeline:
1. Generate synthetic data (for demonstration)
2. Train multiple algorithms
3. Compare model performance
4. Generate comprehensive visualizations
5. Run ablation study
6. Feature importance analysis

Run this to see the complete project in action!

Author: Saugat Shakya
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent))  # demand_prediction folder

from src.features.feature_engineering import FeatureEngineer
from src.models.train_model import DemandModelTrainer
from src.analysis.ablation_study import AblationStudy

print("=" * 80)
print("DEMAND PREDICTION - COMPLETE ANALYSIS PIPELINE")
print("=" * 80)


def generate_synthetic_data(n_days=90):
    """
    Generate synthetic hourly order data for demonstration.
    
    Args:
        n_days: Number of days to generate
        
    Returns:
        DataFrame with synthetic order data
    """
    print("\n📊 Generating synthetic data...")
    
    # Generate date range
    start_date = datetime(2024, 1, 1)
    dates = []
    hours = []
    order_counts = []
    
    np.random.seed(42)
    
    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        day_of_week = current_date.weekday()
        
        for hour in range(24):
            # Base pattern: higher orders during lunch (12-14) and dinner (19-21)
            base_orders = 15
            
            # Hour effect
            if 12 <= hour <= 14:  # Lunch peak
                hour_effect = 8
            elif 19 <= hour <= 21:  # Dinner peak
                hour_effect = 12
            elif 7 <= hour <= 10:  # Breakfast
                hour_effect = 5
            elif 15 <= hour <= 18:  # Afternoon
                hour_effect = 3
            else:  # Late night / early morning
                hour_effect = -8
            
            # Day of week effect (weekends busier)
            if day_of_week >= 5:  # Weekend
                dow_effect = 5
            else:
                dow_effect = 0
            
            # Add some trend over time
            trend = day * 0.1
            
            # Random noise
            noise = np.random.normal(0, 3)
            
            # Calculate total orders
            total_orders = max(0, base_orders + hour_effect + dow_effect + trend + noise)
            
            dates.append(current_date.date())
            hours.append(hour)
            order_counts.append(int(total_orders))
    
    # Create DataFrame
    df = pd.DataFrame({
        'order_date': dates,
        'order_hour': hours,
        'order_count': order_counts
    })
    
    print(f"✅ Generated {len(df)} hourly records ({n_days} days)")
    print(f"   Date range: {df['order_date'].min()} to {df['order_date'].max()}")
    print(f"   Average orders per hour: {df['order_count'].mean():.2f}")
    
    return df


def run_feature_engineering(df):
    """Run feature engineering pipeline."""
    print("\n🔧 Running feature engineering...")
    
    engineer = FeatureEngineer(df)
    features_df = engineer.run_feature_engineering()
    
    print(f"✅ Created {len(features_df.columns)} features")
    
    return features_df


def train_all_models(features_df):
    """Train all models and return trainer."""
    print("\n🤖 Training all models...")
    
    trainer = DemandModelTrainer(features_df)
    trainer.prepare_train_test_split(test_size=0.2)
    
    print(f"\nTraining set: {len(trainer.X_train)} samples")
    print(f"Test set: {len(trainer.X_test)} samples")
    print(f"Features: {len(trainer.X_train.columns)}")
    
    # Train all models
    results = trainer.train_all_models(tune_hyperparameters=False)
    
    return trainer


def create_model_comparison_plots(trainer, output_dir):
    """Create comprehensive model comparison visualizations."""
    print("\n📊 Creating model comparison visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(trainer.results).T
    
    # === FIGURE 1: Model Comparison Overview (4 panels) ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: R² Scores
    ax = axes[0, 0]
    x = range(len(results_df))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], results_df['train_r2'], width, 
           label='Train R²', alpha=0.7, color='skyblue')
    ax.bar([i + width/2 for i in x], results_df['test_r2'], width,
           label='Test R²', alpha=0.7, color='coral')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('R² Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index, rotation=0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(results_df.iterrows()):
        ax.text(i - width/2, row['train_r2'], f"{row['train_r2']:.3f}", 
               ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, row['test_r2'], f"{row['test_r2']:.3f}", 
               ha='center', va='bottom', fontsize=9)
    
    # Panel 2: MAE Comparison
    ax = axes[0, 1]
    ax.barh(results_df.index, results_df['test_mae'], color='teal', alpha=0.7)
    ax.set_xlabel('Mean Absolute Error (orders)', fontsize=12, fontweight='bold')
    ax.set_title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(results_df.iterrows()):
        ax.text(row['test_mae'], i, f"  {row['test_mae']:.2f}", 
               va='center', fontsize=10)
    
    # Panel 3: RMSE Comparison
    ax = axes[1, 0]
    ax.barh(results_df.index, results_df['test_rmse'], color='purple', alpha=0.7)
    ax.set_xlabel('Root Mean Squared Error', fontsize=12, fontweight='bold')
    ax.set_title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(results_df.iterrows()):
        ax.text(row['test_rmse'], i, f"  {row['test_rmse']:.2f}", 
               va='center', fontsize=10)
    
    # Panel 4: Summary Table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for idx, row in results_df.iterrows():
        table_data.append([
            idx,
            f"{row['test_r2']:.4f}",
            f"{row['test_mae']:.2f}",
            f"{row['test_rmse']:.2f}"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'Test R²', 'Test MAE', 'Test RMSE'],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.2, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code best scores
    best_r2_idx = results_df['test_r2'].idxmax()
    best_mae_idx = results_df['test_mae'].idxmin()
    best_rmse_idx = results_df['test_rmse'].idxmin()
    
    for i, idx in enumerate(results_df.index):
        if idx == best_r2_idx:
            table[(i+1, 1)].set_facecolor('#90EE90')
        if idx == best_mae_idx:
            table[(i+1, 2)].set_facecolor('#90EE90')
        if idx == best_rmse_idx:
            table[(i+1, 3)].set_facecolor('#90EE90')
    
    ax.set_title('Performance Summary Table', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    fig.savefig(output_dir / '01_model_comparison.png', bbox_inches='tight')
    print(f"✅ Saved: {output_dir / '01_model_comparison.png'}")
    plt.close()


def create_residual_analysis(trainer, output_dir):
    """Create residual analysis plots."""
    print("\n📊 Creating residual analysis...")
    
    output_dir = Path(output_dir)
    
    # Get best model predictions
    y_pred = trainer.best_model.predict(trainer.X_test)
    y_true = trainer.y_test
    residuals = y_true - y_pred
    
    # === FIGURE 2: Residual Analysis (4 panels) ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(y_pred, y_true, alpha=0.5, s=20, color='blue')
    
    # Perfect prediction line
    min_val = min(y_pred.min(), y_true.min())
    max_val = max(y_pred.max(), y_true.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Predicted Orders', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Orders', fontsize=12, fontweight='bold')
    ax.set_title(f'{trainer.best_model_name} - Predicted vs Actual', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add R² to plot
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
           fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 2: Residual Plot
    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, color='green')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Orders', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Panel 3: Residual Distribution
    ax = axes[1, 0]
    ax.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Residuals', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics
    ax.text(0.02, 0.98, f'Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 4: Error by Magnitude
    ax = axes[1, 1]
    error_bins = pd.cut(y_true, bins=5)
    error_by_bin = pd.DataFrame({'true': y_true, 'abs_error': np.abs(residuals), 'bin': error_bins})
    error_summary = error_by_bin.groupby('bin')['abs_error'].agg(['mean', 'std'])
    
    x_pos = range(len(error_summary))
    ax.bar(x_pos, error_summary['mean'], yerr=error_summary['std'], 
           alpha=0.7, color='orange', capsize=5)
    ax.set_xlabel('Actual Order Range', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax.set_title('Error by Order Volume', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{int(interval.left)}-{int(interval.right)}" 
                        for interval in error_summary.index], rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / '02_residual_analysis.png', bbox_inches='tight')
    print(f"✅ Saved: {output_dir / '02_residual_analysis.png'}")
    plt.close()


def create_feature_importance_plots(trainer, output_dir):
    """Create feature importance visualizations."""
    print("\n📊 Creating feature importance plots...")
    
    output_dir = Path(output_dir)
    
    # Get feature importance
    feature_importance = trainer.get_feature_importance()
    
    if feature_importance is None:
        print("⚠️ Best model doesn't have feature importance")
        return
    
    # === FIGURE 3: Feature Importance (2 panels) ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Panel 1: Top 20 Features
    ax = axes[0]
    top_20 = feature_importance.head(20)
    
    ax.barh(range(len(top_20)), top_20['importance'], color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 20 Most Important Features - {trainer.best_model_name}', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(top_20.iterrows()):
        ax.text(row['importance'], i, f"  {row['importance']:.4f}", 
               va='center', fontsize=9)
    
    # Panel 2: Feature Groups
    ax = axes[1]
    
    # Categorize features
    def categorize_feature(name):
        if any(x in name for x in ['lag', 'rolling']):
            return 'Time-Series'
        elif any(x in name for x in ['hour', 'day', 'sin_', 'cos_', 'weekend', 'month']):
            return 'Temporal'
        elif any(x in name for x in ['holiday']):
            return 'Holiday'
        elif any(x in name for x in ['_avg_']):
            return 'Patterns'
        else:
            return 'Other'
    
    feature_importance['category'] = feature_importance['feature'].apply(categorize_feature)
    category_importance = feature_importance.groupby('category')['importance'].sum().sort_values(ascending=False)
    
    colors = plt.cm.Set3(range(len(category_importance)))
    wedges, texts, autotexts = ax.pie(
        category_importance.values, 
        labels=category_importance.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax.set_title('Feature Importance by Category', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(output_dir / '03_feature_importance.png', bbox_inches='tight')
    print(f"✅ Saved: {output_dir / '03_feature_importance.png'}")
    plt.close()


def run_ablation_study_analysis(trainer, output_dir):
    """Run ablation study."""
    print("\n🔬 Running ablation study...")
    
    output_dir = Path(output_dir)
    
    study = AblationStudy(
        trainer.X_train, trainer.y_train,
        trainer.X_test, trainer.y_test
    )
    
    results = study.run_ablation_study()
    
    # Save results
    study.save_results(output_dir)
    
    # Generate plots
    study.plot_ablation_results(output_dir)
    
    return results


def create_summary_report(trainer, ablation_results, output_dir):
    """Create summary report."""
    print("\n📝 Creating summary report...")
    
    output_dir = Path(output_dir)
    
    report = []
    report.append("=" * 80)
    report.append("DEMAND PREDICTION - ANALYSIS SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    # Model Comparison
    report.append("1. MODEL COMPARISON")
    report.append("-" * 80)
    results_df = pd.DataFrame(trainer.results).T
    report.append(results_df.to_string())
    report.append("")
    report.append(f"✅ Best Model: {trainer.best_model_name}")
    report.append(f"   Test R²: {results_df.loc[trainer.best_model_name, 'test_r2']:.4f}")
    report.append(f"   Test MAE: {results_df.loc[trainer.best_model_name, 'test_mae']:.2f} orders")
    report.append("")
    
    # Feature Importance
    report.append("2. TOP 10 MOST IMPORTANT FEATURES")
    report.append("-" * 80)
    feature_importance = trainer.get_feature_importance()
    if feature_importance is not None:
        for i, row in feature_importance.head(10).iterrows():
            report.append(f"   {i+1:2d}. {row['feature']:30s} - {row['importance']:.6f}")
    report.append("")
    
    # Ablation Study
    report.append("3. ABLATION STUDY RESULTS")
    report.append("-" * 80)
    ablation_summary = ablation_results[['experiment', 'n_features', 'test_r2', 'test_mae']].copy()
    report.append(ablation_summary.to_string(index=False))
    report.append("")
    
    # Key findings
    baseline_r2 = ablation_results[ablation_results['experiment'] == 'FULL MODEL']['test_r2'].values[0]
    ablation_with_drop = ablation_results[ablation_results['experiment'] != 'FULL MODEL'].copy()
    
    if len(ablation_with_drop) > 0:
        worst_config = ablation_with_drop.loc[ablation_with_drop['r2_drop'].idxmax()]
        
        report.append("4. KEY FINDINGS")
        report.append("-" * 80)
        report.append(f"   • Baseline (FULL MODEL) R²: {baseline_r2:.4f}")
        report.append(f"   • Most critical feature group: {worst_config['experiment']}")
        report.append(f"     - Removing causes R² drop of {worst_config['r2_drop']:.4f} ({worst_config['r2_drop_pct']:.2f}%)")
        report.append("")
    
    # Save report
    report_text = "\n".join(report)
    report_file = output_dir / 'ANALYSIS_SUMMARY.txt'
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"✅ Saved: {report_file}")
    
    # Also print to console
    print("\n" + report_text)


def main():
    """Run complete analysis pipeline."""
    
    # Setup paths
    BASE_DIR = Path(__file__).parent  # demand_prediction folder
    OUTPUT_DIR = BASE_DIR / "docs" / "figures" / "comprehensive"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Generate synthetic data
        df = generate_synthetic_data(n_days=90)
        
        # Step 2: Feature engineering
        features_df = run_feature_engineering(df)
        
        # Step 3: Train all models
        trainer = train_all_models(features_df)
        
        # Step 3a: Save all models
        print("\n💾 Saving models...")
        trainer.save_all_models(MODELS_DIR)
        
        # Step 4: Create visualizations
        create_model_comparison_plots(trainer, OUTPUT_DIR)
        create_residual_analysis(trainer, OUTPUT_DIR)
        create_feature_importance_plots(trainer, OUTPUT_DIR)
        
        # Step 5: Ablation study
        ablation_dir = BASE_DIR / "docs" / "figures" / "ablation_study"
        ablation_results = run_ablation_study_analysis(trainer, ablation_dir)
        
        # Step 6: Summary report
        create_summary_report(trainer, ablation_results, BASE_DIR / "docs")
        
        print("\n" + "=" * 80)
        print("✅ COMPLETE ANALYSIS FINISHED!")
        print("=" * 80)
        print(f"\n📁 All outputs saved to:")
        print(f"   Comprehensive figures: {OUTPUT_DIR}")
        print(f"   Ablation study: {ablation_dir}")
        print(f"   Summary report: {BASE_DIR / 'docs' / 'ANALYSIS_SUMMARY.txt'}")
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
