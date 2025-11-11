# Dish Demand Prediction - Quick Start Guide

## 📋 Project Overview

**Goal**: Predict hourly demand for restaurant dishes in Delhi using ML  
**Data**: Restaurant orders from Sep 2024 - Jan 2025  
**Models**: 15+ algorithms with hyperparameter tuning  
**Output**: Comprehensive analysis notebook for professor

## 🗂️ Project Structure

```
dish_prediction/
├── data/              # All datasets
├── models/            # Trained models (baseline/tuned/final)
├── notebooks/         # Jupyter notebooks (analysis steps)
├── reports/           # Documentation + visualizations
├── src/               # Modular Python code
├── requirements.txt   # Dependencies
└── PROJECT_STATUS.md  # Current progress
```

## 🚀 Getting Started

### 1. Check Current Status

```bash
cat PROJECT_STATUS.md
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Configuration

```bash
cd src
python config.py
```

### 4. Review What's Done

- ✅ Project structure created
- ✅ Configuration set up
- ✅ Step 1: hourly_dish_pivot.csv created (2,505 hours × 30 dishes)

## 📝 Workflow

### Phase 1: Build Modular Code (src/)

Create Python modules for:

1. **data/** - Load, process, merge external data
2. **features/** - Engineer temporal, lag, rolling, Delhi-specific features
3. **models/** - Implement all algorithms + tuning
4. **visualization/** - Create all plots
5. **utils/** - Helper functions

### Phase 2: Analysis Notebooks

Create step-by-step notebooks:

1. `00_eda.ipynb` - Data exploration
2. `01_preprocessing.ipynb` - Data cleaning
3. `02_feature_engineering.ipynb` - Feature creation
4. `03_baseline_models.ipynb` - Simple models
5. `04_algorithm_comparison.ipynb` - Compare 15+ algorithms
6. `05_hyperparameter_tuning.ipynb` - Optimize top 3
7. `06_final_models.ipynb` - Train final models
8. **`07_complete_analysis.ipynb`** - **FINAL FOR PROFESSOR**

### Phase 3: Generate Outputs

- 30-40 visualizations in `reports/figures/`
- Trained models in `models/`
- Documentation in `reports/`

## 📊 Key Files

### Configuration

- `src/config.py` - All settings, paths, hyperparameters

### Data Flow

1. Source: `../data/data.csv` (21,321 orders)
2. Step 1: `data/processed/hourly_dish_pivot.csv` ✓
3. Step 2: `data/processed/02_with_external.csv` (+ weather/pollution/events)
4. Step 3: `data/processed/03_with_features.csv` (+ engineered features)
5. Step 4: `data/processed/04_train_test.csv` (split for training)
6. Step 5: `data/processed/05_final_results.csv` (model predictions)

### Algorithms

**Baseline** (4): Mean, Median, Last Value, Moving Average  
**Linear** (4): Linear, Ridge, Lasso, ElasticNet  
**Tree** (4): Decision Tree, Random Forest, Extra Trees, Gradient Boosting  
**Boosting** (3): XGBoost, LightGBM, CatBoost  
**Total**: 15+ models

### Features (~200-300)

- **Temporal** (~15): Hour, day, week, cyclical encoding
- **Lag** (~270): 30 dishes × 9 lags
- **Rolling** (~720): 30 dishes × 6 windows × 4 stats
- **Weather** (~5): Temp, humidity, precip, wind, condition
- **Pollution** (~9): AQI, PM2.5, PM10, NO2, O3, CO, SO2, NH3, NO
- **Events** (~3): Holiday, event type, days to/from event
- **Delhi-specific** (~10): Season, AQI severity, festivals

## 📈 Evaluation

**Metrics**: R², MAE, RMSE, MAPE  
**Validation**: 80-20 split, 5-fold CV, overfitting checks  
**Selection**: Best R² with low overfitting

## 🎯 Deliverables

1. **Code**: Clean, modular, well-documented in `src/`
2. **Notebooks**: Step-by-step analysis in `notebooks/`
3. **Visualizations**: 30-40 figures in `reports/figures/`
4. **Models**: Trained models in `models/`
5. **Documentation**: `reports/ANALYSIS.md` and `reports/RESULTS.md`
6. **Final**: `notebooks/07_complete_analysis.ipynb` for professor

## 📞 Quick Commands

```bash
# Check structure
tree -L 2 -I '__pycache__|*.pyc|notebooks_backup'

# Run config
python src/config.py

# View status
cat PROJECT_STATUS.md

# Check data
ls -lh data/processed/

# Check models
ls -lh models/

# View figures
ls reports/figures/*/
```

## 📝 Notes

- Old work backed up in `notebooks_backup/`
- Using actual `../data/data.csv` (not dummy_orders.csv)
- Focus on top 30 dishes by volume
- Delhi-specific features for accuracy
- Comprehensive comparison of 15+ algorithms
- Hyperparameter tuning for top 3 models
- Professional structure ready for submission

---

**Last Updated**: November 9, 2025  
**Status**: Structure complete, ready to build modular code
