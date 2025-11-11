# 🚀 Delivery Time Prediction

**Predict food delivery times using machine learning with weather, pollution, and temporal features.**

## 📊 Project Overview

This project predicts delivery times for food orders using multiple machine learning models including XGBoost, LightGBM, CatBoost, and Random Forest. The models incorporate temporal patterns, restaurant behavior, distance metrics, and external factors like pollution.

### Key Features

- **Multiple Model Comparison**: XGBoost, LightGBM, CatBoost, Random Forest
- **Feature Engineering**:
  - Temporal features (hour, day, cyclical encoding)
  - Lag features (historical delivery times)
  - Rolling windows (moving averages)
  - Restaurant-specific patterns
  - Distance-based features
  - Pollution/weather data
- **Ablation Study**: Systematic analysis of feature group importance
- **Production-Ready**: Modular code, inference module, comprehensive docs

### Performance

- **Best Model**: XGBoost / LightGBM (determined by training)
- **Expected Metrics**:
  - R² Score: ~0.75-0.85
  - MAE: ~3-5 minutes
  - RMSE: ~4-7 minutes

---

## 📁 Project Structure

```
delivery_time_prediction/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_training.py              # Main training pipeline
├── inference.py                 # Inference script
│
├── src/
│   ├── config.py                # Configuration and paths
│   ├── data/
│   │   ├── loader.py            # Data loading
│   │   └── preprocessing.py     # Data cleaning & preprocessing
│   ├── features/
│   │   └── feature_engineering.py  # Feature engineering
│   ├── models/
│   │   ├── train_model.py       # Model training & comparison
│   │   └── inference.py         # Prediction module
│   └── analysis/
│       └── ablation_study.py    # Feature importance analysis
│
├── data/
│   ├── raw/                     # Place data.csv here
│   └── processed/               # Generated processed data
│
├── models/
│   ├── baseline/                # Baseline models & results
│   └── final/                   # Final tuned models
│
├── reports/
│   └── figures/                 # Generated plots
│
└── docs/                        # Documentation
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd delivery_time_prediction

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your data files in `data/raw/`:

- `data.csv` - Main delivery/order data
- `delhi_pollution_orders.csv` - Pollution data (optional)

### 3. Train Models

```bash
# Run complete training pipeline
python run_training.py
```

This will:

1. Load and preprocess data
2. Engineer features
3. Train multiple models (XGBoost, LightGBM, CatBoost, Random Forest)
4. Run ablation study
5. Save models and results

### 4. Make Predictions

```bash
# Predict on new data
python inference.py --model models/baseline/xgboost_model.pkl --data data/raw/new_orders.csv

# Interactive single prediction
python inference.py --model models/baseline/xgboost_model.pkl --single

# Show feature importance
python inference.py --model models/baseline/xgboost_model.pkl
```

---

## 📊 Pipeline Details

### Data Preprocessing

The preprocessing pipeline (`src/data/preprocessing.py`):

1. **Data Cleaning**

   - Remove unnecessary columns (instructions, reviews, complaints)
   - Filter to delivered orders only
   - Handle missing values
   - Remove duplicates

2. **Timestamp Processing**

   - Parse order timestamps
   - Convert to UTC for pollution data merge
   - Extract temporal features

3. **Feature Encoding**

   - One-hot encode categorical variables (Subzone, Order Ready Marked)
   - Label encode Restaurant ID
   - Process distance (remove '<', 'km' text)

4. **External Data Merge**
   - Merge pollution data on hourly basis
   - Handle timezone conversions

### Feature Engineering

The feature engineering pipeline (`src/features/feature_engineering.py`):

1. **Temporal Features**

   - Cyclical encoding (hour, day, month using sin/cos)
   - Peak hour indicators (lunch, dinner)
   - Weekend flags

2. **Historical Features**

   - Lag features (1, 2, 3, 6, 12, 24 periods)
   - Rolling windows (3, 6, 12, 24 periods)
   - Per-restaurant historical patterns

3. **Restaurant Features**

   - Average delivery time per restaurant
   - Order volume per restaurant

4. **Distance Features**

   - Distance bins (very_close, close, medium, far, very_far)
   - Distance squared (non-linear relationship)

5. **Pollution Features** (if available)
   - AQI levels
   - Individual pollutants (PM2.5, PM10, NO2, etc.)

### Model Training

The training pipeline (`src/models/train_model.py`):

1. **Baseline Models**

   - XGBoost
   - LightGBM
   - CatBoost
   - Random Forest

2. **Evaluation Metrics**

   - R² Score (coefficient of determination)
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - MAPE (Mean Absolute Percentage Error)

3. **Model Comparison**

   - Cross-validation
   - Test set evaluation
   - Feature importance extraction

4. **Hyperparameter Tuning** (optional)
   - GridSearchCV for best model
   - Save tuned model to `models/final/`

### Ablation Study

The ablation study (`src/analysis/ablation_study.py`) systematically removes feature groups to measure their impact:

**Feature Groups Tested**:

- Temporal features
- Lag features
- Rolling window features
- Restaurant features
- Distance features
- Pollution features

**Output**:

- R² drop when each group is removed
- MAE increase when each group is removed
- Identifies most important feature groups

---

## 📈 Results & Outputs

### Saved Files

After training, you'll find:

**Models** (`models/baseline/`):

- `xgboost_model.pkl`
- `lightgbm_model.pkl`
- `catboost_model.pkl`
- `random_forest_model.pkl`
- `model_results.json` - Performance metrics
- `ablation_study.csv` - Feature importance analysis

**Data** (`data/processed/`):

- `delivery_data_processed.csv` - Cleaned data
- `delivery_features.csv` - Engineered features

**Visualizations** (`reports/figures/`):

- `model_comparison.png` - Model performance comparison
- `ablation_study.png` - Feature group importance

### Interpreting Results

**Model Comparison**:

- Look for highest R² and lowest MAE
- Check for overfitting (train vs test gap)

**Ablation Study**:

- Large R² drop → feature group is important
- Green bars → low impact
- Red bars → high impact

---

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Paths
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering
LAG_PERIODS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12, 24]

# Target variable
TARGET = 'Total_time_taken'
```

---

## 📚 API Usage

### Training

```python
from src.data.loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.train_model import ModelTrainer

# Load data
loader = DataLoader()
data = loader.load_all_data()

# Preprocess
preprocessor = DataPreprocessor()
df_clean = preprocessor.preprocess(data['orders'], data['pollution'])

# Engineer features
engineer = FeatureEngineer()
df_features = engineer.engineer_features(df_clean)

# Train models
trainer = ModelTrainer()
X_train, X_test, y_train, y_test = trainer.prepare_data(df_features)
results = trainer.train_baseline_models(X_train, X_test, y_train, y_test)
```

### Inference

```python
from src.models.inference import DeliveryTimePredictor
import pandas as pd

# Load model
predictor = DeliveryTimePredictor('models/baseline/xgboost_model.pkl')

# Predict on DataFrame
predictions = predictor.predict(new_data_df)

# Predict single order
prediction = predictor.predict_single({
    'Distance': 5.0,
    'order_hour': 19,
    'is_weekend': 1,
    # ... other features
})
```

---

## 🎯 Next Steps

1. **Review Results**

   - Check `models/baseline/model_comparison.csv`
   - Analyze `reports/figures/ablation_study.png`

2. **Improve Model**

   - Add more features (traffic data, restaurant ratings, etc.)
   - Try different hyperparameters
   - Ensemble multiple models

3. **Deploy**
   - Use `inference.py` for production predictions
   - Integrate with API/web app
   - Set up monitoring

---

## 📝 Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- lightgbm >= 4.0.0
- catboost >= 1.2.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- shap >= 0.42.0 (optional, for explainability)

---

## 👤 Author

**Saugat Shakya**  
Date: 2025-01-27

---

## 📄 License

This project is part of ML2025 coursework.
