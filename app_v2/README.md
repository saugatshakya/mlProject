# ML2025 Multi-Model Web Application

A unified web interface for three machine learning models focused on food delivery prediction and recommendation.

## 🎯 Models

### 1. 📊 Dish Prediction

Multi-output regression model that predicts demand for individual dishes.

**Features:**

- Historical features only (R² = 0.9545)
- Based on ablation study showing external features hurt performance
- XGBoost multi-output regressor
- Predicts all dishes simultaneously

**Input:** CSV with `timestamp` and dish columns
**Output:** Predicted orders for each dish per hour

### 2. 📈 Demand Prediction

Hourly order volume prediction using temporal features.

**Features:**

- Temporal features only (R² = 0.8647)
- Time-based patterns and lag features
- XGBoost regressor
- Forecasts multiple hours ahead

**Input:** CSV with `timestamp` and `total_orders`
**Output:** Predicted total orders for next N hours

### 3. 🎯 Dish Recommendation

Association rules-based recommendation system.

**Features:**

- Market basket analysis
- Co-occurrence matrix
- Support, confidence, and lift metrics
- Recommends dishes frequently ordered together

**Input:** CSV with `order_id` and `items` (or `dish_name`)
**Output:** Top N recommended dishes with confidence scores

## 🚀 Quick Start

### Installation

```bash
# Navigate to app directory
cd app_v2

# Install dependencies
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Access at: http://localhost:5000

## 📁 Project Structure

```
app_v2/
├── app.py                          # Main Flask application
├── models_dish_prediction.py       # Dish prediction model wrapper
├── models_demand_prediction.py     # Demand prediction model wrapper
├── models_dish_recommend.py        # Dish recommendation model wrapper
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                  # Main UI with tabs
├── static/
│   └── app.js                      # Frontend JavaScript
├── uploads/                        # Uploaded CSV files
└── models/                         # Trained model files
```

## 🎨 User Interface

The web app features a tabbed interface with:

- **Beautiful gradient design**
- **Responsive layout** (Bootstrap 5)
- **Real-time updates** (AJAX)
- **Metric visualizations**
- **Interactive predictions**

Each tab provides:

1. **Training section** - Upload CSV and train model
2. **Metrics display** - R² score, MAE, RMSE
3. **Prediction/Recommendation** - Interactive inference

## 📊 API Endpoints

### Dish Prediction

```
POST /api/dish_prediction/train
POST /api/dish_prediction/predict
GET  /api/dish_prediction/status
```

### Demand Prediction

```
POST /api/demand_prediction/train
POST /api/demand_prediction/predict
GET  /api/demand_prediction/status
```

### Dish Recommendation

```
POST /api/dish_recommend/train
POST /api/dish_recommend/recommend
POST /api/dish_recommend/search
GET  /api/dish_recommend/popular
GET  /api/dish_recommend/status
```

### General

```
GET /health
```

## 📝 Data Format

### Dish Prediction

```csv
timestamp,Dish1,Dish2,Dish3,...
2024-01-01 10:00:00,5,3,8,...
2024-01-01 11:00:00,7,4,6,...
```

### Demand Prediction

```csv
timestamp,total_orders
2024-01-01 10:00:00,25
2024-01-01 11:00:00,32
```

### Dish Recommendation

**Option 1:**

```csv
order_id,items
1,"Pizza, Pasta, Salad"
2,"Burger, Fries"
```

**Option 2:**

```csv
order_id,dish_name
1,Pizza
1,Pasta
2,Burger
```

## 🔬 Model Performance

Based on ablation studies:

- **Dish Prediction:** R² = 0.9545 (historical features only)
- **Demand Prediction:** R² = 0.8647 (temporal features only)
- **Dish Recommendation:** 120 rules, avg lift 2.5x

Key finding: **External features (weather, pollution, events) HURT performance**

See `EXTERNAL_FEATURES_ANALYSIS.md` for details.

## 🛠️ Development

### Adding New Features

1. Update model wrapper in `models_*.py`
2. Add API endpoint in `app.py`
3. Update frontend in `templates/index.html` and `static/app.js`

### Testing

```bash
# Health check
curl http://localhost:5000/health

# Train model (example)
curl -X POST -F "file=@data.csv" http://localhost:5000/api/dish_prediction/train

# Get predictions
curl -X POST -H "Content-Type: application/json" \
     -d '{"hour": 12}' \
     http://localhost:5000/api/dish_prediction/predict
```

## 📚 Credits

**Author:** Saugat Shakya  
**Course:** ML2025  
**Projects:**

- Dish Prediction (Multi-output regression)
- Demand Prediction (Time series forecasting)
- Dish Recommendation (Association rules mining)

## 📄 License

MIT License - see parent project for details.
