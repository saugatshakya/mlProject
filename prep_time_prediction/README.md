# ⏱️ Kitchen Preparation Time Prediction

**Predicting order preparation time for restaurant kitchen operations**

This project predicts how long it will take a restaurant kitchen to prepare an order (KPT - Kitchen Preparation Time) using machine learning. Accurate KPT predictions help optimize:

- Rider dispatch timing
- Customer delivery estimates
- Kitchen workload management
- Order scheduling

---

## 🎯 Overview

### Problem Statement

**Goal:** Predict the time (in minutes) it takes a kitchen to prepare an order from placement to ready-for-pickup.

**Why it matters:**

- ✅ **Better ETA** - More accurate delivery time estimates for customers
- ✅ **Optimized dispatch** - Send riders at the right time (not too early/late)
- ✅ **Kitchen efficiency** - Identify bottlenecks and complex dishes
- ✅ **Resource planning** - Staff kitchens appropriately for peak times

### Approach

- **Model Type:** Regression (predicting continuous time values)
- **Target:** `KPT duration (minutes)` - time from order placement to kitchen-ready
- **Best Model:** Histogram Gradient Boosting Regressor
- **Performance:** MAE ~4-5 minutes, R² ~0.75-0.80

---

## 📊 Results Summary

### Model Performance

| Model                    | Train MAE | Train R² | Test MAE | Test R² | Status      |
| ------------------------ | --------- | -------- | -------- | ------- | ----------- |
| **HistGradientBoosting** | ~3.5 min  | ~0.85    | ~4.5 min | ~0.78   | ✅ **Best** |
| ElasticNet               | ~6.0 min  | ~0.45    | ~6.2 min | ~0.42   | Baseline    |

**Key Metrics:**

- **Test MAE:** 4-5 minutes (very practical for operations)
- **Test R²:** 0.75-0.80 (strong predictive power)
- **95% of predictions:** Within ±10 minutes of actual

### Top Predictive Features

1. **🍽️ num_items** - Number of items in order (strongest signal)
2. **👨‍🍳 num_complex_dishes** - Count of time-intensive dishes
3. **🏪 rest_mean_KPT** - Restaurant's historical average prep time
4. **⏰ order_hour** - Time of day (peak vs off-peak)
5. **📍 Subzone** - Geographic location
6. **💰 Total** - Order value
7. **📦 Packaging charges** - Indicator of order complexity
8. **🕐 orders_last_30min** - Kitchen load (recent order volume)
9. **☁️ wx_temp_c, wx_precip_mm** - Weather conditions
10. **🎉 has_event, holiday** - Special event days

---

## 🚀 Quick Start

### Installation

```bash
# Clone repo and navigate to project
cd prep_time_prediction

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Full pipeline (preprocessing + training + evaluation)
python src/models/train_model.py

# Or run analysis with visualization
python analysis/run_complete_analysis.py
```

### Inference

```python
from inference import PrepTimePredictionAPI

# Load trained model
api = PrepTimePredictionAPI()
api.load_model('models/final/best_model.pkl')

# Predict for new order
order_features = {
    'num_items': 3,
    'Total': 850,
    'num_complex_dishes': 1,
    'Packaging_charges': 25,
    'order_hour': 19,  # 7 PM
    'is_weekend': 1,
    'Distance_km': 2.5,
    # ... other features
}

pred_time = api.predict(order_features)
print(f"Predicted prep time: {pred_time:.1f} minutes")
```

---

## 📁 Project Structure

```
prep_time_prediction/
├── README.md                    # This file
├── PROJECT_STATUS.md            # Current status and next steps
├── QUICK_START.md               # Getting started guide
├── requirements.txt             # Python dependencies
├── inference.py                 # Production inference API
│
├── data/
│   ├── raw/                     # Original order data
│   │   ├── data.csv            # Main order dataset
│   │   └── delhi_major_events.csv  # Event calendar
│   └── processed/              # Processed datasets
│       ├── orders_cleaned.csv
│       ├── orders_with_features.csv
│       └── dish_complexity.csv
│
├── notebooks/
│   └── OrderPrepTime_prediction.ipynb  # Original exploration
│
├── src/
│   ├── data/
│   │   └── preprocessing.py    # Data loading and cleaning
│   ├── features/
│   │   └── feature_engineering.py  # Feature creation
│   ├── models/
│   │   ├── train_model.py      # Model training pipeline
│   │   └── model_comparison.py # Compare multiple algorithms
│   └── analysis/
│       ├── eda.py              # Exploratory data analysis
│       ├── ablation_study.py   # Feature importance analysis
│       └── error_analysis.py   # Prediction error diagnostics
│
├── models/
│   └── final/
│       ├── best_model.pkl      # Trained model
│       ├── feature_names.txt   # Model features
│       └── model_config.json   # Hyperparameters
│
├── analysis/
│   ├── run_complete_analysis.py  # Full analysis pipeline
│   └── outputs/                # Analysis results
│
├── docs/
│   ├── figures/                # Visualization outputs
│   └── FEATURE_ENGINEERING.md  # Feature documentation
│
└── reports/
    ├── MODEL_COMPARISON.md     # Algorithm comparison
    ├── ABLATION_STUDY.md       # Feature ablation results
    └── ERROR_ANALYSIS.md       # Prediction error analysis
```

---

## 🔬 Feature Engineering

### Core Features

**Order Characteristics:**

- `num_items` - Total items in order
- `Total` - Order value (₹)
- `Packaging_charges` - Packaging cost
- `total_discount_amt` - Combined discounts
- `Distance_km` - Delivery distance

**Dish Complexity:**

- `num_complex_dishes` - Count of time-intensive dishes
- `has_complex_dish` - Binary flag
- `complexity_ratio` - Complex dishes / total items

**Temporal Features:**

- `order_hour` - Hour of day (0-23)
- `order_dayofweek` - Day of week (0-6)
- `is_weekend` - Weekend flag
- `is_peak_hour` - Lunch/dinner peak
- `is_lunch_peak`, `is_dinner_peak` - Specific peaks
- `order_hour_sin`, `order_hour_cos` - Cyclic hour encoding

**Restaurant Features:**

- `rest_mean_KPT` - Historical average prep time
- `rest_p75_KPT` - 75th percentile prep time
- `rest_mean_wait` - Average rider wait time
- `orders_last_30min` - Recent kitchen load

**Location:**

- `Subzone_*` - One-hot encoded subzones (8 locations)

**External Factors:**

- `wx_temp_c` - Temperature
- `wx_precip_mm` - Precipitation
- `wx_cloud_cover_pct` - Cloud cover
- `holiday` - Public holiday flag
- `has_event` - Special event flag
- `event_*` - One-hot encoded events

**Discount Types:**

- `has_discount` - Any discount flag
- `disc_percent`, `disc_flat`, `disc_bundle` - Discount types

### Engineered Features

- `avg_item_value` = Total / num_items
- `is_big_order` = num_items >= 6
- `is_high_value_order` = Total >= 75th percentile
- `is_high_load` = orders_last_30min >= 5
- `is_peak_weekend` = is_peak_hour AND is_weekend

---

## 📈 Model Details

### Algorithm Selection

**Tested Models:**

1. ✅ **HistGradientBoostingRegressor** (chosen)

   - Best performance (MAE ~4.5 min)
   - Fast training and prediction
   - Handles missing values natively
   - No need for scaling

2. ❌ ElasticNet (baseline)
   - Simpler linear model
   - MAE ~6.2 min
   - Good interpretability but lower accuracy

### Hyperparameters (Best Model)

```python
HistGradientBoostingRegressor(
    learning_rate=0.1,
    max_depth=10,
    max_iter=400,
    min_samples_leaf=20,
    random_state=42
)
```

### Target Transformation

- **Log transform** used: `log_KPT = log(KPT + 1)`
- Reduces impact of outliers
- Better captures proportional relationships
- Predictions: `KPT = exp(log_KPT) - 1`

---

## 🔍 Key Insights

### What Drives Prep Time?

1. **Order Size Matters Most**

   - Each additional item adds ~2-3 minutes
   - Non-linear relationship (economies of scale)

2. **Complex Dishes Add Significant Time**

   - Orders with complex dishes take +5-8 minutes on average
   - Examples: Grills, biryanis, multi-component dishes

3. **Restaurant Variability is Huge**

   - Some restaurants consistently 2x slower than others
   - Historical performance is strong predictor

4. **Peak Hours Add Pressure**

   - Lunch (12-2 PM) and dinner (7-10 PM) peaks
   - +10-15% longer prep times during peaks

5. **Weather Impact is Real**

   - Heavy rain → +2-3 minutes
   - Likely due to higher order volume

6. **Events Create Chaos**
   - Major events → highly unpredictable
   - +20-30% variance on event days

---

## 📊 Visualizations

Generated visualizations available in `docs/figures/`:

1. **Feature Importance** - Permutation importance rankings
2. **Actual vs Predicted** - Scatter plot showing model fit
3. **Error Distribution** - Histogram of prediction errors
4. **Error by Subzone** - Geographic error patterns
5. **Error by Peak Hours** - Time-based error analysis
6. **Error by Order Size** - MAE stratified by num_items

---

## 🎓 Methodology

### Data Pipeline

1. **Data Loading**

   - Load orders, events, weather data
   - Merge external features (events, weather)

2. **Preprocessing**

   - Handle missing values
   - Drop irrelevant columns
   - Parse complex fields (items, discounts)

3. **Feature Engineering**

   - Create temporal features
   - Calculate dish complexity
   - Compute restaurant statistics
   - Add interaction terms

4. **Train/Test Split**

   - 80/20 split
   - Stratified by time (no data leakage)

5. **Model Training**

   - Hyperparameter tuning (RandomizedSearchCV)
   - Custom scorer (MAE in minutes)
   - 5-fold cross-validation

6. **Evaluation**
   - MAE, R² on test set
   - Error analysis by segments
   - Feature importance

---

## ⚠️ Limitations & Caveats

1. **Restaurant-specific models might be better**

   - Current model is global across all restaurants
   - Individual restaurant models could improve accuracy

2. **Cold-start problem**

   - New restaurants have no historical stats
   - Falls back to global averages

3. **Real-time kitchen load not captured**

   - `orders_last_30min` is proxy
   - Actual kitchen capacity/staffing unknown

4. **Dish-level granularity missing**

   - Only identifies "complex" dishes
   - Actual prep time per dish would help

5. **No cancellation/rejection handling**
   - Model assumes all orders are completed
   - Cancelled orders may have different patterns

---

## 🚀 Future Improvements

### Short-term

- [ ] Add restaurant-specific calibration
- [ ] Incorporate real-time kitchen dashboard data
- [ ] Add confidence intervals to predictions
- [ ] Build separate models for peak vs off-peak

### Long-term

- [ ] Sequence models (predict based on order history)
- [ ] Multi-task learning (predict KPT + rider wait together)
- [ ] Incorporate real-time traffic data
- [ ] Build explainable AI dashboard for operations team

---

## 📚 References

### Data Sources

- **Order data:** Restaurant orders from Delhi NCR
- **Events:** `delhi_major_events.csv` - Major holidays and events
- **Weather:** Open-Meteo Archive API (historical weather data)

### Libraries

- **scikit-learn:** Model training and evaluation
- **pandas, numpy:** Data manipulation
- **matplotlib, seaborn:** Visualization
- **statsmodels:** Statistical analysis

---

## 👥 Usage

### For Data Scientists

- Explore `notebooks/OrderPrepTime_prediction.ipynb` for analysis
- Run `analysis/run_complete_analysis.py` for full pipeline
- Check `reports/` for detailed findings

### For ML Engineers

- Use `inference.py` for production predictions
- Deploy model from `models/final/best_model.pkl`
- See `src/models/train_model.py` for retraining

### For Operations Team

- Model provides prep time estimates for dispatch optimization
- Use confidence intervals for buffer planning
- Monitor actual vs predicted for model drift

---

## 📞 Contact & Support

**Project:** Kitchen Preparation Time Prediction  
**Author:** Saugat Shakya  
**Date:** November 2025

For questions or issues, please refer to:

- `PROJECT_STATUS.md` - Current project status
- `docs/` - Detailed documentation
- `reports/` - Analysis reports

---

## 📄 License

This project is part of the ML2025 coursework.
