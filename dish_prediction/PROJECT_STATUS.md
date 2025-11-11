# Dish Demand Prediction Project - Complete Rebuild

## 🎯 Project Goal

Predict hourly demand for restaurant dishes in Delhi using machine learning, incorporating weather, pollution, and event data.

## 📊 Current Status

### ✅ Completed

1. **Project Structure**

   - Created clean, organized folder hierarchy
   - Separated data (raw/processed/interim)
   - Organized models (baseline/tuned/final)
   - Structured reports with figure categories
   - Modular src code (data/features/models/visualization/utils)

2. **Configuration**

   - `requirements.txt` - All Python dependencies
   - `src/config.py` - Centralized configuration
   - `__init__.py` files in all modules

3. **Data Preparation (Step 1 completed)**
   - Parsed order data from `../data/data.csv`
   - Extracted 38,232 dish-level records
   - Aggregated to hourly level
   - Selected top 30 dishes by volume
   - Created pivot table (2,505 hours × 30 dishes)
   - Saved to `data/processed/hourly_dish_pivot.csv`

### 🚧 In Progress

Building modular Python code in `src/`:

- `src/data/` - Data loading and processing
- `src/features/` - Feature engineering
- `src/models/` - All ML models
- `src/visualization/` - Plotting functions

### 📋 To Do

#### Phase 1: Core Functionality (Python modules)

- [ ] `src/data/loader.py` - Load and parse raw data
- [ ] `src/data/processor.py` - Clean and aggregate
- [ ] `src/data/external.py` - Merge weather/pollution/events
- [ ] `src/features/temporal.py` - Time-based features
- [ ] `src/features/lag.py` - Lag features
- [ ] `src/features/rolling.py` - Rolling statistics
- [ ] `src/features/domain.py` - Delhi-specific features
- [ ] `src/models/baseline.py` - Simple baselines
- [ ] `src/models/linear.py` - Linear regression models
- [ ] `src/models/tree.py` - Tree-based models
- [ ] `src/models/tuner.py` - Hyperparameter tuning
- [ ] `src/models/evaluator.py` - Evaluation metrics
- [ ] `src/visualization/*.py` - All plotting functions

#### Phase 2: Analysis Notebooks

- [ ] `00_eda.ipynb` - Exploratory Data Analysis
- [ ] `01_preprocessing.ipynb` - Data cleaning
- [ ] `02_feature_engineering.ipynb` - Feature creation
- [ ] `03_baseline_models.ipynb` - Simple baselines
- [ ] `04_algorithm_comparison.ipynb` - Compare all algorithms
- [ ] `05_hyperparameter_tuning.ipynb` - Optimize best models
- [ ] `06_final_models.ipynb` - Train final models
- [ ] `07_complete_analysis.ipynb` - **FINAL SUBMISSION**

## 🔬 Analysis Plan

### Data Overview

- **Source**: Restaurant order data from Delhi
- **Period**: Sep 2024 - Jan 2025 (153 days, 2,505 hours)
- **Orders**: 21,321 total, 21,131 delivered
- **Dishes**: 244 unique, focusing on top 30
- **Top Dish**: Bageecha Pizza (3,304 units)

### External Data

1. **Weather** (hourly)

   - Temperature, humidity, precipitation, wind speed, condition

2. **Pollution** (hourly)

   - AQI, PM2.5, PM10, NO2, O3, CO, SO2, NH3

3. **Events** (daily)
   - Holidays, festivals, major Delhi events

### Features to Engineer (~200-300 features)

1. **Temporal** (~15 features)

   - Hour, day, week, month
   - Cyclical encoding (sin/cos)
   - Weekend, peak hour flags

2. **Lag Features** (~270 features = 30 dishes × 9 lags)

   - 1h, 2h, 3h, 6h, 12h, 24h, 48h, 72h, 168h

3. **Rolling Statistics** (~720 features = 30 dishes × 6 windows × 4 stats)

   - Windows: 3h, 6h, 12h, 24h, 48h, 168h
   - Stats: mean, std, min, max

4. **Weather** (~5 features)

   - All weather variables

5. **Pollution** (~9 features)

   - All pollution metrics

6. **Events** (~3 features)

   - Holiday flag, event type, days to/from event

7. **Delhi-Specific** (~10 features)
   - Season, AQI severity, festival periods

### Algorithms to Compare

#### Baseline Models

1. Mean (seasonal)
2. Median (seasonal)
3. Last value
4. Moving average

#### Linear Models

5. Linear Regression
6. Ridge Regression
7. Lasso Regression
8. ElasticNet

#### Tree-Based Models

9. Decision Tree
10. Random Forest
11. Extra Trees
12. Gradient Boosting

#### Boosting Models

13. XGBoost
14. LightGBM
15. CatBoost

#### Optional

16. KNN Regression
17. SVR
18. Neural Network (MLP)

### Evaluation Strategy

**Metrics**:

- R² Score (primary)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

**Validation**:

- 80-20 train-test split (temporal)
- 5-fold time series cross-validation
- Overfitting checks (train-test gap)

**Selection Criteria**:

- Best R² on test set
- Low overfitting
- Computational efficiency
- Interpretability

### Hyperparameter Tuning

**Method**: Optuna (Bayesian optimization)

**Top 3 Algorithms** (based on initial comparison):

- 100 trials per algorithm
- 5-fold CV
- Save best parameters

**Search Spaces**:

- XGBoost: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma
- LightGBM: n_estimators, max_depth, learning_rate, num_leaves, min_child_samples, subsample, colsample_bytree
- Random Forest: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features

## 📊 Expected Outputs

### Data Files

- `data/processed/01_hourly_pivot.csv` ✓
- `data/processed/02_with_external.csv`
- `data/processed/03_with_features.csv`
- `data/processed/04_train_test.csv`
- `data/processed/05_final_results.csv`
- `data/processed/algorithm_comparison.csv`
- `data/processed/tuning_results.csv`

### Models

- `models/baseline/*.pkl` - Simple baseline models
- `models/tuned/*.pkl` - Hyperparameter-tuned models
- `models/final/*.pkl` - Production-ready models for top dishes

### Visualizations (~30-40 figures)

**01_eda/** (10 figures)

- Temporal patterns (hourly, daily, weekly)
- Dish volume distributions
- Correlation heatmaps
- Weather/pollution distributions
- Missing data patterns

**02_features/** (8 figures)

- Feature importance rankings
- Feature correlations
- Distribution plots
- Category-wise importance

**03_models/** (12 figures)

- Algorithm comparison (bar charts)
- Learning curves
- Residual plots
- Prediction vs actual scatter
- Overfitting analysis
- Cross-validation results

**04_results/** (8 figures)

- Final model performance
- Top dish predictions
- Error analysis
- Feature importance (final)
- Time series predictions
- Confidence intervals

### Documentation

- `reports/ANALYSIS.md` - Comprehensive technical analysis
- `reports/RESULTS.md` - Executive summary
- `README.md` - Project overview and instructions

### Final Deliverable

- **`notebooks/07_complete_analysis.ipynb`** - Complete end-to-end analysis for professor

## 🎓 For Professor Submission

The final notebook will include:

1. **Introduction** - Problem statement, objectives
2. **Data Exploration** - EDA with visualizations
3. **Data Preprocessing** - Cleaning, aggregation
4. **Feature Engineering** - All features with explanations
5. **Baseline Models** - Simple benchmarks
6. **Algorithm Comparison** - 15+ models compared
7. **Hyperparameter Tuning** - Optimization process
8. **Final Models** - Best models trained
9. **Evaluation** - Comprehensive metrics and validation
10. **Results** - Key findings and insights
11. **Conclusions** - Summary and recommendations
12. **Appendix** - Code, additional plots

## 📝 Next Steps

1. ✅ Project structure created
2. ✅ Configuration files set up
3. ✅ Data preparation (Step 1) completed
4. 🔄 Build modular Python code in `src/`
5. 🔄 Create analysis notebooks
6. 🔄 Generate all visualizations
7. 🔄 Write documentation
8. 🔄 Create final submission notebook

---

**Last Updated**: November 9, 2025  
**Status**: Phase 1 - Building modular code
