# Dish Demand Prediction

**Machine Learning-based demand forecasting for restaurant dishes using XGBoost**

## 📊 Project Overview

This project predicts hourly demand for restaurant dishes using historical order data, weather conditions, pollution levels, and special events. The model achieves **R² = 0.89** for top-performing dishes.

### Key Features

- **272 engineered features** including lag variables, rolling statistics, temporal patterns
- **XGBoost models** trained individually for each dish
- **Weather & pollution integration** (temperature, humidity, AQI, PM2.5, etc.)
- **Event detection** (holidays, major city events)
- **Comprehensive validation** with cross-validation and overfitting analysis

## 📁 Project Structure

```
dish_prediction/
├── data/
│   ├── raw/                    # Original data
│   │   └── dummy_orders.csv
│   └── processed/              # Processed data and results
│       ├── full_data.csv
│       ├── features_full.csv
│       ├── model_results.csv
│       └── ...
├── models/                     # Trained XGBoost models (.pkl)
│   ├── chilli_cheese_garlic_bread.pkl
│   ├── bageecha_pizza.pkl
│   └── ...
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── final.ipynb
│   └── basemodel.ipynb
├── reports/
│   ├── RESULTS.md             # Comprehensive analysis documentation
│   └── figures/               # All visualizations
│       ├── fig01_algorithm_comparison.png
│       ├── fig02_feature_ablation.png
│       └── ...
├── scripts/
│   └── run_all.sh            # Master script to run everything
└── src/                       # Source code (modular)
    ├── __init__.py
    ├── pipeline.py            # Main pipeline orchestrator
    ├── data_processor.py      # Data loading and merging
    ├── feature_engineer.py    # Feature creation
    ├── model_trainer.py       # Model training and evaluation
    └── visualization/
        └── generate_all_figures.py  # Comprehensive visualization suite
```

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.9
pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn
```

### Installation

```bash
# Clone repository
cd dish_prediction

# Install dependencies (if using pip)
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```

### Run Complete Analysis

```bash
# Option 1: Run master script
./scripts/run_all.sh

# Option 2: Run Python pipeline directly
python src/pipeline.py --top-n 10 --visualize

# Option 3: Run steps individually
python src/data_processor.py
python src/feature_engineer.py
python src/model_trainer.py
python src/visualization/generate_all_figures.py
```

## 📈 Results Summary

### Top 3 Dishes (by R²)

1. **Chilli Cheese Garlic Bread**: R² = 0.8918, MAE = 0.59
2. **Bageecha Pizza**: R² = 0.8502, MAE = 0.68
3. **Bone in Jamaican Grilled Chicken**: R² = 0.8234, MAE = 0.72

### Overall Performance (10 dishes)

- **Mean R²**: 0.792 (±0.089)
- **Mean MAE**: 0.687 (±0.098)
- **Mean Overfitting Gap**: 0.034 (excellent generalization)

## 📊 Methodology

### 1. Data Processing

- Hourly aggregation of dish orders
- Integration with weather data (temperature, humidity, precipitation, wind speed)
- Integration with pollution data (AQI, PM2.5, PM10, NO2, O3, CO, SO2, NH3)
- Event detection (holidays, major city events)

### 2. Feature Engineering (272 features)

- **Lag features**: 1h, 2h, 3h, 24h, 168h (weekly)
- **Rolling statistics**: 3h, 6h, 12h, 24h windows (mean, std)
- **Temporal features**: Hour, day of week, month, cyclical encoding
- **Environmental features**: Weather + pollution (13 features)
- **Event features**: Holiday flags, event indicators

### 3. Model Selection

Compared 8 algorithms:

- Linear: Ridge, Lasso, ElasticNet
- Tree-based: Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM

**Winner**: XGBoost (R² = 0.34 baseline → 0.89 with full features)

### 4. Feature Ablation Study

- Base only (lags + rolling + temporal): R² = -0.23
- - Weather: R² = -0.14 (+39% improvement)
- - Pollution: R² = -0.16 (+30% improvement)
- - Events: R² = -0.14 (+39% improvement)
- **All features**: R² = 0.19 baseline → **0.89 final** (+182% improvement)

### 5. Validation

- **Train-test split**: 80-20 temporal split
- **Cross-validation**: 5-fold CV (mean R² = 0.78)
- **Overfitting analysis**: Mean gap = 0.034 (excellent)
- **Residual analysis**: Normally distributed, minimal patterns

## 📊 Visualizations

The project generates 10+ comprehensive figures:

1. **Algorithm Comparison** - Performance across all models
2. **Feature Ablation** - Impact of weather/pollution/events
3. **Model Performance** - R² and MAE for all dishes
4. **Feature Importance** - Top features for best dishes
5. **Predictions vs Actuals** - Model accuracy visualization
6. **Residual Analysis** - 4-panel diagnostic plots
7. **Learning Curves** - Training convergence
8. **Cross-Validation** - CV score distributions
9. **Overfitting Analysis** - Train-test gap analysis
10. **Temporal Patterns** - Performance by hour/day
11. **Volume Correlations** - Dish volume vs accuracy

## 🔬 Key Findings

1. **Weather matters**: Temperature and precipitation significantly impact demand
2. **Pollution effects**: AQI and PM2.5 show moderate correlation with orders
3. **Event boosts**: Major events increase demand by ~25-30%
4. **Temporal patterns**: Strong hourly patterns (lunch 12-2pm, dinner 7-10pm)
5. **XGBoost superior**: Outperforms linear models by 60% (R² 0.34 → 0.89)

## 📝 Documentation

Comprehensive results and analysis in **[reports/RESULTS.md](reports/RESULTS.md)**

Includes:

- Detailed methodology
- Hypothesis testing results
- Statistical validation
- Model justifications
- Feature importance analysis
- Overfitting assessment
- Future recommendations

## 👤 Author

**Saugat Shakya**  
Machine Learning Project - 2025

## 📄 License

Educational project for ML2025 course.
