# Quick Start Guide - Kitchen Prep Time Prediction

This guide will help you get started with the Kitchen Prep Time Prediction project in just a few minutes.

---

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Basic familiarity with pandas and scikit-learn

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Data

Place your order data CSV in the `data/raw/` directory:

- `data/raw/orders.csv` - Main order dataset
- `data/raw/delhi_major_events.csv` - Events/holidays data (optional)

**Required columns in orders.csv:**

- `Order Placed At` - Timestamp
- `Restaurant` - Restaurant identifier
- `Items` - Comma-separated dish list (e.g., "1 x Pizza, 2 x Burger")
- `Total`, `Packaging_charges` - Order value
- `Discount` - Discount type
- `Subzone` - Delivery location
- `Distance (km)` - Delivery distance
- `Rider wait time at restaurant (minutes)` - Wait time
- `KPT duration (minutes)` - **TARGET** - Kitchen prep time

### Step 3: Run the Complete Pipeline

```bash
cd prep_time_prediction
python src/models/train_model.py
```

This will:

1. Load and preprocess data
2. Merge external data (events, weather)
3. Engineer 50+ features
4. Train multiple models with hyperparameter tuning
5. Evaluate and select best model
6. Save trained model to `models/final/`

**Expected Runtime:** 5-15 minutes (depending on data size)

### Step 4: Make Predictions

```python
from inference import PrepTimePredictionAPI

# Load trained model
api = PrepTimePredictionAPI(model_dir="models/final")

# Prepare order features (dict or DataFrame)
order = {
    'num_items': 3,
    'num_complex_dishes': 1,
    'Total': 450.0,
    'order_hour': 19,
    'is_weekend': 0,
    # ... all other required features
}

# Get prediction with confidence interval
result = api.predict(order, return_confidence=True)
print(f"Predicted KPT: {result['prediction_minutes']:.1f} minutes")
print(f"95% CI: [{result['lower_bound']:.1f}, {result['upper_bound']:.1f}]")
```

---

## 📊 Expected Results

After training, you should see results similar to:

| Model      | Test MAE (min) | Test R² | Train MAE (min) | Train R² |
| ---------- | -------------- | ------- | --------------- | -------- |
| HistGB     | 4.5            | 0.78    | 3.5             | 0.85     |
| ElasticNet | 5.2            | 0.72    | 4.8             | 0.75     |

**Best Model:** HistGradientBoostingRegressor

**Performance:**

- Predicts prep time within ±4.5 minutes on average
- Explains 78% of variance in prep times
- Top features: num_items, num_complex_dishes, rest_mean_KPT

---

## 📁 Project Structure

After running the pipeline, your directory will look like:

```
prep_time_prediction/
├── data/
│   ├── raw/
│   │   ├── orders.csv                     # Your input data
│   │   └── delhi_major_events.csv         # Events data
│   └── processed/
│       ├── preprocessed_orders.csv        # After cleaning
│       ├── features_orders.csv            # After feature engineering
│       └── test_predictions.csv           # Test set predictions
│
├── models/
│   └── final/
│       ├── best_model.pkl                 # Trained model
│       ├── feature_names.txt              # Required features
│       ├── model_config.json              # Model metadata
│       └── model_comparison.csv           # All model results
│
├── src/
│   ├── data/
│   │   └── preprocessing.py               # Data cleaning
│   ├── features/
│   │   └── feature_engineering.py         # Feature creation
│   └── models/
│       └── train_model.py                 # Model training
│
├── inference.py                           # Prediction API
├── requirements.txt                       # Dependencies
└── README.md                              # Full documentation
```

---

## 🔧 Common Tasks

### Task 1: Retrain Model with New Data

```bash
# 1. Add new data to data/raw/orders.csv
# 2. Run training pipeline
python src/models/train_model.py
```

### Task 2: Make Predictions on New Orders

```python
from inference import predict_from_csv

# Predict for CSV file
df = predict_from_csv(
    input_csv='data/new_orders.csv',
    output_csv='data/predictions.csv'
)
```

### Task 3: Batch Predictions

```python
import pandas as pd
from inference import PrepTimePredictionAPI

# Load data
df = pd.read_csv('data/processed/features_orders.csv')

# Make predictions
api = PrepTimePredictionAPI()
df_with_preds = api.predict_batch(df, add_to_dataframe=True)

# Save results
df_with_preds.to_csv('data/all_predictions.csv', index=False)
```

### Task 4: Analyze Predictions

```python
import pandas as pd

# Load test predictions
df = pd.read_csv('data/processed/test_predictions.csv')

# Summary statistics
print(df[['KPT_actual_min', 'KPT_pred_min', 'abs_error_min']].describe())

# Find worst predictions
worst = df.nlargest(10, 'abs_error_min')
print(worst[['KPT_actual_min', 'KPT_pred_min', 'error_min']])
```

---

## 🎯 Understanding the Features

The model uses 50+ features across several categories:

### Order Features

- `num_items` - Number of dishes in order
- `num_complex_dishes` - Count of complex/slow dishes
- `Total`, `Packaging_charges` - Order value
- `total_discount_amt` - Discount amount

### Temporal Features

- `order_hour` - Hour of day (0-23)
- `order_day` - Day of week (0-6)
- `is_weekend`, `is_lunch_peak`, `is_dinner_peak` - Flags

### Restaurant Features

- `rest_mean_KPT` - Restaurant's average prep time
- `rest_p75_KPT` - Restaurant's 75th percentile prep time
- `rest_mean_wait` - Average rider wait time

### External Features

- `has_event` - Major event/holiday flag
- `wx_temp_c`, `wx_precip_mm` - Weather conditions
- `orders_last_30min` - Kitchen load indicator

### Location Features

- `Subzone_*` - One-hot encoded delivery zones (8 zones)

---

## 🐛 Troubleshooting

### Issue 1: Missing Features Error

**Error:** `ValueError: Missing required features: ['rest_mean_KPT', ...]`

**Solution:** You must engineer all features before prediction. Use the full pipeline:

```python
from src.data.preprocessing import PrepTimePreprocessor
from src.features.feature_engineering import PrepTimeFeatureEngineer

# Preprocess
preprocessor = PrepTimePreprocessor()
df = preprocessor.preprocess_pipeline('data/raw/orders.csv')

# Engineer features
engineer = PrepTimeFeatureEngineer()
df_features = engineer.feature_engineering_pipeline(df)

# Now you can predict
api.predict(df_features)
```

### Issue 2: Weather API Fails

**Error:** `Weather API request failed`

**Solution:** The Open-Meteo API is free but has rate limits. If it fails:

1. Wait a few minutes and retry
2. Or skip weather features (set wx\_\* columns to 0)

### Issue 3: Model Performance Lower Than Expected

**Possible causes:**

- Different data distribution (restaurant mix, time periods)
- Missing or incorrect features
- Need more training data

**Solutions:**

1. Check feature distributions: `df.describe()`
2. Verify all features are present: `api.get_feature_names()`
3. Retrain with more data if available

### Issue 4: ImportError

**Error:** `ModuleNotFoundError: No module named 'sklearn'`

**Solution:** Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📈 Next Steps

Once you're comfortable with the basics:

1. **Analyze Results**

   - Review `PROJECT_STATUS.md` for current status
   - Check `models/final/model_comparison.csv` for all model results

2. **Improve Model**

   - Try different hyperparameters in `src/models/train_model.py`
   - Add more features in `src/features/feature_engineering.py`
   - Collect more training data

3. **Deploy to Production**

   - Use `inference.py` API in your application
   - Set up monitoring for prediction drift
   - Retrain periodically with new data

4. **Advanced Analysis**
   - Run ablation study to understand feature importance
   - Analyze errors by segment (restaurant, time, order size)
   - Generate visualizations

---

## 📚 Additional Resources

- **README.md** - Comprehensive project documentation
- **PROJECT_STATUS.md** - Current project status
- **notebooks/OrderPrepTime_prediction.ipynb** - Original exploration

---

## 💡 Tips

1. **Start Small:** Test with a sample of data first (~1000 orders)
2. **Validate Features:** Always check feature distributions before training
3. **Monitor Performance:** Track MAE and R² on holdout set over time
4. **Retrain Regularly:** Retrain monthly with new data to avoid model drift

---

## 🤝 Need Help?

Common questions:

**Q: How accurate is the model?**
A: Test MAE is ~4.5 minutes. Expect predictions within ±4-5 minutes on average.

**Q: What's the minimum data size?**
A: Recommended: At least 5,000 orders with variety of restaurants, times, and order types.

**Q: Can I use this for a different restaurant/city?**
A: Yes, but retrain with data from your specific context. Feature importance may differ.

**Q: How often should I retrain?**
A: Retrain monthly or when performance degrades. Monitor MAE on recent orders.

---

**Ready to start?** Run `python src/models/train_model.py` and you'll have a working model in minutes! 🚀
