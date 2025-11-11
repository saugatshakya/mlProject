"""
Complete Training Pipeline for Delivery Time Prediction
=======================================================

This script runs the full training pipeline:
1. Load and preprocess data
2. Engineer features
3. Train multiple models
4. Run ablation study
5. Save models and results

Usage:
    python run_training.py

Author: Saugat Shakya
Date: 2025-01-27
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

import logging
import pandas as pd
from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    BASELINE_MODELS_DIR,
    FINAL_MODELS_DIR,
    FIGURES_DIR,
    TARGET
)
from src.data.loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.train_model import ModelTrainer
from src.analysis.ablation_study import AblationStudy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run complete training pipeline."""
    
    logger.info("="*80)
    logger.info("DELIVERY TIME PREDICTION - TRAINING PIPELINE")
    logger.info("="*80)
    
    # -------------------------------------------------------------------------
    # STEP 1: Load Data
    # -------------------------------------------------------------------------
    logger.info("\nSTEP 1: Loading data...")
    loader = DataLoader()
    data = loader.load_all_data()
    
    df_orders = data['orders']
    df_pollution = data['pollution']
    
    # -------------------------------------------------------------------------
    # STEP 2: Preprocess Data
    # -------------------------------------------------------------------------
    logger.info("\nSTEP 2: Preprocessing data...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.preprocess(df_orders, df_pollution)
    
    # Save processed data
    processed_file = PROCESSED_DATA_DIR / "delivery_data_processed.csv"
    df_clean.to_csv(processed_file, index=False)
    logger.info(f"Saved processed data to {processed_file}")
    
    # -------------------------------------------------------------------------
    # STEP 3: Feature Engineering
    # -------------------------------------------------------------------------
    logger.info("\nSTEP 3: Engineering features...")
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df_clean, target_col=TARGET)
    
    # Save features
    features_file = PROCESSED_DATA_DIR / "delivery_features.csv"
    df_features.to_csv(features_file, index=False)
    logger.info(f"Saved features to {features_file}")
    
    # -------------------------------------------------------------------------
    # STEP 4: Train Models
    # -------------------------------------------------------------------------
    logger.info("\nSTEP 4: Training models...")
    trainer = ModelTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df_features, target_col=TARGET)
    
    # Train baseline models
    results = trainer.train_baseline_models(X_train, X_test, y_train, y_test)
    
    # Compare models
    comparison_df = trainer.compare_models()
    
    # Save comparison
    comparison_file = BASELINE_MODELS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"Saved model comparison to {comparison_file}")
    
    # Extract feature importance for best model
    best_model_name = comparison_df.iloc[0]['Model']
    importance_df = trainer.extract_feature_importance(
        best_model_name,
        X_train.columns.tolist(),
        top_n=30
    )
    
    # Save feature importance
    importance_file = BASELINE_MODELS_DIR / f"{best_model_name}_feature_importance.csv"
    importance_df.to_csv(importance_file, index=False)
    logger.info(f"Saved feature importance to {importance_file}")
    
    # Plot results
    trainer.plot_results(save_dir=FIGURES_DIR)
    
    # Save models
    trainer.save_models(BASELINE_MODELS_DIR)
    
    # -------------------------------------------------------------------------
    # STEP 5: Ablation Study
    # -------------------------------------------------------------------------
    logger.info("\nSTEP 5: Running ablation study...")
    ablation = AblationStudy(X_train, y_train, X_test, y_test)
    ablation_results = ablation.run_full_ablation()
    
    # Save ablation results
    ablation_file = BASELINE_MODELS_DIR / "ablation_study.csv"
    ablation_results.to_csv(ablation_file, index=False)
    logger.info(f"Saved ablation results to {ablation_file}")
    
    # Plot ablation results
    ablation_plot = FIGURES_DIR / "ablation_study.png"
    ablation.plot_results(save_path=ablation_plot)
    
    # Save ablation JSON
    ablation_json = BASELINE_MODELS_DIR / "ablation_results.json"
    ablation.save_results(ablation_json)
    
    # -------------------------------------------------------------------------
    # STEP 6: Tune Best Model (Optional)
    # -------------------------------------------------------------------------
    logger.info("\nSTEP 6: Tuning best model...")
    logger.info(f"Best model: {best_model_name}")
    
    # Simplified param grid for faster tuning
    param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [6, 8],
        'learning_rate': [0.05, 0.1]
    }
    
    try:
        tuned_model = trainer.tune_model(
            best_model_name,
            X_train,
            y_train,
            param_grid=param_grid
        )
        
        # Evaluate tuned model
        tuned_metrics = trainer.evaluate_model(
            tuned_model,
            X_train,
            X_test,
            y_train,
            y_test,
            f"{best_model_name}_tuned"
        )
        
        # Save tuned model
        import joblib
        tuned_model_path = FINAL_MODELS_DIR / f"{best_model_name}_tuned_model.pkl"
        joblib.dump(tuned_model, tuned_model_path)
        logger.info(f"Saved tuned model to {tuned_model_path}")
        
    except Exception as e:
        logger.warning(f"Model tuning failed: {e}")
        logger.info("Using baseline model as final model")
    
    # -------------------------------------------------------------------------
    # DONE
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("TRAINING PIPELINE COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nResults saved to:")
    logger.info(f"  - Models: {BASELINE_MODELS_DIR}")
    logger.info(f"  - Figures: {FIGURES_DIR}")
    logger.info(f"  - Processed data: {PROCESSED_DATA_DIR}")
    logger.info("\nNext steps:")
    logger.info("  1. Review ablation study to understand feature importance")
    logger.info("  2. Check model comparison to see performance")
    logger.info("  3. Use inference.py for predictions")
    logger.info("="*80)


if __name__ == "__main__":
    main()
