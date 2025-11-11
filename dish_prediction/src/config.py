"""
Configuration file for Dish Demand Prediction Project
Contains all constants, paths, and hyperparameters
"""

from pathlib import Path
import os

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Base directory
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

# Data paths
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
INTERIM_DATA_DIR = DATA_DIR / 'interim'

# External data (in project/data/)
EXTERNAL_DATA_DIR = PROJECT_ROOT / 'data'
ORDERS_DATA = EXTERNAL_DATA_DIR / 'data.csv'
WEATHER_DATA = EXTERNAL_DATA_DIR / 'hourly_orders_weather.csv'
POLLUTION_DATA = EXTERNAL_DATA_DIR / 'pollution.csv'
EVENTS_DATA = EXTERNAL_DATA_DIR / 'events.csv'

# Model paths
MODELS_DIR = BASE_DIR / 'models'
BASELINE_MODELS_DIR = MODELS_DIR / 'baseline'
TUNED_MODELS_DIR = MODELS_DIR / 'tuned'
FINAL_MODELS_DIR = MODELS_DIR / 'final'

# Reports paths
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
EDA_FIGURES_DIR = FIGURES_DIR / '01_eda'
FEATURE_FIGURES_DIR = FIGURES_DIR / '02_features'
MODEL_FIGURES_DIR = FIGURES_DIR / '03_models'
RESULT_FIGURES_DIR = FIGURES_DIR / '04_results'

# ============================================================================
# DATA PROCESSING CONFIGURATION
# ============================================================================

# Top N dishes to analyze
TOP_N_DISHES = 30

# Date format in raw data
DATE_FORMAT = '%I:%M %p, %B %d %Y'

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

# Lag features (in hours)
LAG_FEATURES = [1, 2, 3, 6, 12, 24, 48, 72, 168]  # Up to 1 week

# Rolling window sizes (in hours)
ROLLING_WINDOWS = [3, 6, 12, 24, 48, 168]

# Rolling statistics to calculate
ROLLING_STATS = ['mean', 'std', 'min', 'max']

# Temporal features
TEMPORAL_FEATURES = [
    'hour', 'day_of_week', 'day', 'month', 'week_of_year',
    'is_weekend', 'is_peak_lunch', 'is_peak_dinner',
    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos'
]

# Peak hours definition (Delhi restaurant context)
PEAK_LUNCH_HOURS = [12, 13, 14]
PEAK_DINNER_HOURS = [19, 20, 21, 22]

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Models to compare
BASELINE_MODELS = ['mean', 'median', 'last_value', 'moving_average']

LINEAR_MODELS = ['linear_regression', 'ridge', 'lasso', 'elasticnet']

TREE_MODELS = ['decision_tree', 'random_forest', 'extra_trees', 'gradient_boosting']

BOOSTING_MODELS = ['xgboost', 'lightgbm', 'catboost']

OTHER_MODELS = ['knn', 'svr']  # Optional

# All models
ALL_MODELS = BASELINE_MODELS + LINEAR_MODELS + TREE_MODELS + BOOSTING_MODELS

# ============================================================================
# HYPERPARAMETER TUNING CONFIGURATION
# ============================================================================

# Number of trials for Optuna
N_TRIALS = 100

# Cross-validation folds
CV_FOLDS = 5

# Hyperparameter search spaces
XGBOOST_PARAM_SPACE = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
}

LIGHTGBM_PARAM_SPACE = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [15, 31, 63, 127],
    'min_child_samples': [5, 10, 20, 30],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
}

RANDOM_FOREST_PARAM_SPACE = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================

# Metrics to calculate
METRICS = ['r2', 'mae', 'rmse', 'mape']

# Scoring metric for model selection
SCORING_METRIC = 'r2'

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Figure size
FIGURE_SIZE = (12, 6)
LARGE_FIGURE_SIZE = (16, 10)

# DPI for saving figures
DPI = 300

# Style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Color palettes
COLOR_PALETTE = 'Set2'
DIVERGING_PALETTE = 'RdYlGn'

# ============================================================================
# DELHI-SPECIFIC CONFIGURATION
# ============================================================================

# Delhi seasons (for feature engineering)
DELHI_SEASONS = {
    'winter': [12, 1, 2],      # Winter (cold, smog)
    'spring': [3, 4],          # Pleasant
    'summer': [5, 6, 7, 8],    # Hot
    'monsoon': [7, 8, 9],      # Rainy (overlaps with summer)
    'autumn': [10, 11],        # Pleasant
}

# AQI categories (Delhi air quality)
AQI_CATEGORIES = {
    'good': (0, 50),
    'satisfactory': (51, 100),
    'moderate': (101, 200),
    'poor': (201, 300),
    'very_poor': (301, 400),
    'severe': (401, 500),
}

# Major festivals/events in Delhi
DELHI_MAJOR_EVENTS = [
    'Diwali', 'Holi', 'Navratri', 'Eid', 'Christmas',
    'Republic Day', 'Independence Day', 'Gandhi Jayanti',
    'Durga Puja', 'Dussehra', 'New Year'
]

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Logging level
LOG_LEVEL = 'INFO'

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR,
        MODELS_DIR, BASELINE_MODELS_DIR, TUNED_MODELS_DIR, FINAL_MODELS_DIR,
        REPORTS_DIR, FIGURES_DIR, EDA_FIGURES_DIR, FEATURE_FIGURES_DIR,
        MODEL_FIGURES_DIR, RESULT_FIGURES_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("✓ All directories created")
    print(f"✓ Base directory: {BASE_DIR}")
    print(f"✓ Orders data: {ORDERS_DATA}")
    print(f"✓ Configuration loaded successfully")
