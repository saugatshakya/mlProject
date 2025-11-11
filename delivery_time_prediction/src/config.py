"""
Configuration file for Delivery Time Prediction
================================================

Centralized configuration for paths, parameters, and model settings.
"""

from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = ROOT_DIR / "models"
BASELINE_MODELS_DIR = MODELS_DIR / "baseline"
FINAL_MODELS_DIR = MODELS_DIR / "final"

# Output paths
REPORTS_DIR = ROOT_DIR / "reports"
DOCS_DIR = ROOT_DIR / "docs"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data files
RAW_DATA_FILE = RAW_DATA_DIR / "data.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "delivery_data_processed.csv"
FEATURES_FILE = PROCESSED_DATA_DIR / "delivery_features.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering parameters
LAG_PERIODS = [1, 2, 3, 6, 12, 24]  # Hours to look back
ROLLING_WINDOWS = [3, 6, 12, 24]     # Rolling window sizes

# Model configurations
MODELS_CONFIG = {
    'xgboost': {
        'n_estimators': 1000,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'early_stopping_rounds': 50
    },
    'lightgbm': {
        'n_estimators': 1000,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE
    },
    'catboost': {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.05,
        'random_state': RANDOM_STATE,
        'verbose': False
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
}

# Target variable
TARGET = 'Total_time_taken'  # Delivery time in minutes

# Columns to drop during preprocessing
DROP_COLUMNS = [
    'Instructions',
    'Rating',
    'Review',
    'Cancellation / Rejection reason',
    'Restaurant compensation (Cancellation)',
    'Restaurant penalty (Rejection)',
    'Customer complaint tag',
    'Restaurant name',
    'Delivery',
    'Customer ID',
    'City'
]

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                 MODELS_DIR, BASELINE_MODELS_DIR, FINAL_MODELS_DIR,
                 REPORTS_DIR, DOCS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
