# 🍕 Delhi Food Delivery - Hourly Order Volume Prediction

**Predict hourly food delivery order volumes using time-series features, temporal patterns, and external data.**

---

## 📊 Project Overview

This project predicts **hourly order volumes** for food delivery restaurants in Delhi using:

- **Time-series features**: Lag features, rolling windows
- **Temporal patterns**: Hour of day, day of week, seasonal effects
- **Holiday effects**: Indian holidays and their impact
- **External data** (optional): Weather, pollution

### Key Results

| Model             | Test R²   | Test MAE        | Status   |
| ----------------- | --------- | --------------- | -------- |
| **XGBoost**       | **~0.73** | **~4.0 orders** | ✅ Best  |
| Random Forest     | ~0.68     | ~4.5 orders     | Good     |
| Linear Regression | ~0.45     | ~6.0 orders     | Baseline |

**Target**: `orders_per_hour` - Total number of orders in each hour block

---

## 📁 Project Structure

```
demand_prediction/
├── data/
│   ├── raw/                    # Raw data files (data.csv, pollution.csv)
│   └── processed/              # Processed data (cleaned_orders.csv, hourly_features.csv)
│
├── src/
│   ├── data/
│   │   └── preprocessing.py    # Data cleaning and preparation
│   ├── features/
│   │   └── feature_engineering.py  # Feature engineering pipeline
│   ├── models/
│   │   ├── train_model.py      # Model training and comparison
│   │   └── inference.py        # Prediction on new data
│   └── analysis/
│       ├── ablation_study.py   # Feature importance analysis
│       └── model_impact_analysis.py  # Model behavior analysis
│
├── models/                     # Saved trained models
│
├── docs/                       # Complete documentation
│   ├── 00_START_HERE.md       # Quick start guide
│   ├── 01_PROJECT_OVERVIEW.md # Executive summary
│   ├── 02_MODEL_IMPACT_ANALYSIS.md
│   ├── 03_ABLATION_STUDY.md   # What features matter?
│   ├── 04_INFERENCE_GUIDE.md  # How to use in production
│   └── figures/               # All visualizations
│
├── notebooks/                  # Original Jupyter notebooks
│   ├── order-volume-pred copy.ipynb
│   └── api-integrated-pred.ipynb
│
├── cache/                      # Cached API responses
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd demand_prediction
pip install -r requirements.txt
```

### 2. Prepare Data

Place your data files in `data/raw/`:

- `data.csv` - Order data
- `pollution.csv` - Pollution data (optional)

### 3. Run Complete Pipeline

```bash
# Step 1: Preprocess data
python src/data/preprocessing.py

# Step 2: Engineer features
python src/features/feature_engineering.py

# Step 3: Train models
python src/models/train_model.py

# Step 4: Run ablation study
python src/analysis/ablation_study.py
```

### 4. View Results

Open `docs/00_START_HERE.md` to explore all results and analysis!

---

## 📈 Features Engineered

### 1. Temporal Features (9 features)

- `order_hour` (0-23)
- `day_of_week` (0-6, Monday=0)
- `is_weekend` (0/1)
- `sin_hour`, `cos_hour` (cyclical encoding)
- `sin_day`, `cos_day` (cyclical day of week)
- `month`, `day_of_month`

### 2. Time-Series Features (12 features)

**Lag Features**:

- `orders_lag_1hr` - Orders 1 hour ago
- `orders_lag_24hr` - Orders same hour yesterday
- `orders_lag_48hr` - Orders same hour 2 days ago
- `orders_lag_168hr` - Orders same hour last week

**Rolling Windows** (using past data only):

- `rolling_mean_3hr`, `rolling_std_3hr`, `rolling_max_3hr`, `rolling_min_3hr`
- `rolling_mean_6hr`, `rolling_std_6hr`
- `rolling_mean_24hr`, `rolling_std_24hr`

### 3. Holiday Features (3 features)

- `is_holiday` - Indian national holidays
- `is_pre_holiday` - Day before holiday
- `is_post_holiday` - Day after holiday

### 4. Pattern Features (3 features)

- `hour_avg_orders` - Average orders for this hour across all days
- `dow_avg_orders` - Average orders for this day of week
- `hour_dow_avg_orders` - Average for (hour, day_of_week) combination

### 5. External Data (Optional)

- **Pollution**: AQI, PM2.5, PM10, NO2, O3, CO, etc.
- **Weather**: Temperature, humidity, precipitation, wind

**Total**: ~30-40 features depending on external data availability

---

## 🔬 Key Findings

### From Ablation Study

**Question**: Do time-series features (lags + rolling windows) actually help?

**Answer**: YES! Time-series features are critical for accurate predictions.

| Configuration        | Test R² | Impact                    |
| -------------------- | ------- | ------------------------- |
| **FULL MODEL**       | 0.726   | Baseline                  |
| **NO TIME-SERIES**   | 0.512   | -29% R² drop!             |
| **ONLY TIME-SERIES** | 0.698   | Nearly matches full model |

**Key Insights**:

1. ✅ Time-series features (lags + rolling) are ESSENTIAL
2. ⚠️ External data (weather, pollution) may ADD noise (needs testing)
3. ✅ Temporal patterns (hour, day) help slightly
4. ✅ Holiday features provide minor lift

See `docs/03_ABLATION_STUDY.md` for complete analysis.

---

## 💡 Model Performance

### By Hour of Day

**Best Performance** (lowest error):

- 🌙 **Late night** (1 AM - 5 AM): Low volume, easy to predict
- 🌆 **Mid-afternoon** (2 PM - 4 PM): Stable patterns

**Worst Performance** (highest error):

- 🍽️ **Dinner rush** (7 PM - 9 PM): High volume, high variance
- 🍳 **Lunch peak** (12 PM - 1 PM): Variable demand

### By Day of Week

- **Weekends**: Slightly higher errors (more variable behavior)
- **Weekdays**: More stable, better predictions
- **Holidays**: Post-holiday effect is weak in data

---

## 🎯 Production Deployment

### Required Features for Inference

**Minimum features needed** (time-series only):

- Current datetime (for temporal features)
- Historical orders for past 168 hours (1 week)

**Optional features** (if available):

- Weather data for the hour
- Pollution data for the hour
- Holiday calendar

### Example Inference

```python
from src.models.inference import OrderVolumePredictor

# Load model
predictor = OrderVolumePredictor("models/xgboost_20250127.pkl")

# Predict for next hour
prediction = predictor.predict_next_hour(
    current_datetime="2024-11-09 18:00:00",
    historical_orders=[12, 15, 18, 22, 25, ...]  # Past 168 hours
)

print(f"Predicted orders for next hour: {prediction:.1f}")
```

See `docs/04_INFERENCE_GUIDE.md` for complete guide.

---

## 📚 Documentation

### For Reviewers

1. **[00_START_HERE.md](docs/00_START_HERE.md)** - Quick start guide
2. **[03_ABLATION_STUDY.md](docs/03_ABLATION_STUDY.md)** - Feature importance analysis
3. **[01_PROJECT_OVERVIEW.md](docs/01_PROJECT_OVERVIEW.md)** - Results summary

### For Production Teams

1. **[04_INFERENCE_GUIDE.md](docs/04_INFERENCE_GUIDE.md)** - Deployment guide
2. **[03_ABLATION_STUDY.md](docs/03_ABLATION_STUDY.md)** - What features to use

### For Data Scientists

Read all documents in order for complete technical understanding.

---

## 🧪 Running Experiments

### Train with Different Features

```python
from src.models.train_model import DemandModelTrainer

# Load features
df = pd.read_csv("data/processed/hourly_features.csv")

# Initialize trainer
trainer = DemandModelTrainer(df)
trainer.prepare_train_test_split(test_size=0.2)

# Train all models
trainer.train_all_models(tune_hyperparameters=False)

# Save best model
trainer.save_model(trainer.best_model, trainer.best_model_name, "models/")
```

### Run Ablation Study

```python
from src.analysis.ablation_study import AblationStudy

# Use trainer's train/test split
study = AblationStudy(
    trainer.X_train, trainer.y_train,
    trainer.X_test, trainer.y_test
)

# Run study
results = study.run_ablation_study()

# Save results and plots
study.save_results("docs/figures/ablation_study")
study.plot_ablation_results("docs/figures/ablation_study")
```

---

## 🔧 Troubleshooting

### Issue: Missing historical data for lag features

**Solution**: For first prediction, use zeros or average values:

```python
# Use average hourly orders for missing lags
avg_hourly = df['orders_per_hour'].mean()
orders_lag_24hr = avg_hourly  # if no 24hr history
```

### Issue: External data not available

**Solution**: Train model without external features:

```python
# Remove pollution/weather columns before training
external_cols = [col for col in df.columns if any(x in col for x in
                ['aqi', 'pm', 'temp', 'humidity'])]
df_no_external = df.drop(columns=external_cols)
```

### Issue: Predictions negative or unrealistic

**Solution**: Clip predictions to valid range:

```python
predictions = np.clip(predictions, 0, max_orders_seen * 1.5)
```

---

## 📊 Dataset Information

### Order Data Schema

| Column             | Type  | Description              |
| ------------------ | ----- | ------------------------ |
| `order_date`       | date  | Date of order            |
| `order_hour`       | int   | Hour of order (0-23)     |
| `order_count`      | int   | Number of items in order |
| `Restaurant name`  | str   | Restaurant identifier    |
| `Subzone`          | str   | Delivery location        |
| `Distance`         | float | Delivery distance (km)   |
| `discount_percent` | float | Discount percentage      |

### Pollution Data Schema (Optional)

| Column            | Type  | Description        |
| ----------------- | ----- | ------------------ |
| `pollution_date`  | date  | Date               |
| `pollution_hour`  | int   | Hour (0-23)        |
| `aqi`             | float | Air Quality Index  |
| `pm2_5`, `pm10`   | float | Particulate matter |
| `no2`, `o3`, `co` | float | Gas concentrations |

---

## 🎓 Key Learnings

1. **Time-series features are critical** for hourly prediction
2. **Lag features capture short-term patterns** (1hr, 24hr lags work best)
3. **Rolling windows smooth out noise** and capture trends
4. **Cyclical encoding** (sin/cos) helps model understand periodic patterns
5. **External data requires careful validation** (may add noise)

---

## 🚀 Future Improvements

1. **Deep Learning**: Try LSTM/GRU for sequence modeling
2. **Multi-step Forecasting**: Predict next 24 hours simultaneously
3. **Restaurant-Specific Models**: Train separate models per restaurant
4. **Uncertainty Quantification**: Add prediction intervals
5. **Real-time Updates**: Online learning with new data
6. **Weather Integration**: Test if weather actually helps (ablation study)

---

## 📞 Support

For questions about:

- **Model behavior**: See `docs/02_MODEL_IMPACT_ANALYSIS.md`
- **Feature importance**: See `docs/03_ABLATION_STUDY.md`
- **Production deployment**: See `docs/04_INFERENCE_GUIDE.md`
- **General overview**: See `docs/01_PROJECT_OVERVIEW.md`

---

## 📜 License

This project is for educational purposes.

---

## 👤 Author

**Saugat Shakya**  
Date: 2025-01-27

---

## 🙏 Acknowledgments

- XGBoost team for the excellent gradient boosting library
- Scikit-learn for robust ML tools
- Original notebook contributors

---

**Last Updated**: 2025-01-27  
**Model Version**: XGBoost Regressor with time-series features  
**Best Test R²**: ~0.73 (72.6% variance explained)
