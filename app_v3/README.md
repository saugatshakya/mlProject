# ML2025 App v3 - Complete Multi-Model Food Delivery Prediction System

## Overview

App v3 is the final deployment version of the ML2025 project, featuring **5 advanced ML models** for comprehensive food delivery analytics and prediction. This unified web application provides end-to-end capabilities for operational decision-making in food delivery businesses.

## 🚀 Available Models

### 1. **Dish Prediction** (Multi-output regression)
- **Purpose**: Predict hourly demand for individual dishes
- **Algorithm**: Multi-output regression with time-series features
- **Use Case**: Inventory management, menu optimization
- **Performance**: Real operational data validation

### 2. **Demand Prediction** (Enhanced XGBoost)
- **Purpose**: Predict hourly order volumes
- **Algorithm**: XGBoost with 27 advanced features
- **Performance**: R² = 0.9558, MAE = 0.765 orders/hour
- **Features**: Lags, rolling statistics, pattern recognition
- **Use Case**: Staffing, capacity planning

### 3. **Dish Recommendation** (Association Rules)
- **Purpose**: Generate dish recommendations based on order patterns
- **Algorithm**: Apriori algorithm for frequent itemsets
- **Use Case**: Cross-selling, menu suggestions
- **Performance**: Confidence and lift metrics

### 4. **Prep Time Prediction** (XGBoost Regression)
- **Purpose**: Predict kitchen preparation time for orders
- **Algorithm**: XGBoost with kitchen-focused features
- **Features**: Distance, order complexity, rider wait time
- **Use Case**: Delivery time estimation, kitchen efficiency

### 5. **Promotion Effectiveness** (Random Forest)
- **Purpose**: Analyze impact of promotions on orders and sales
- **Algorithm**: Dual Random Forest models (orders + sales)
- **Features**: Promotion types, temporal patterns, historical data
- **Use Case**: Promotion ROI analysis, marketing optimization

## 📊 Data Sources

The application supports multiple data input methods:

### Original Data (Recommended)
- Uses real operational data from `data/data.csv`
- Pre-processed datasets available for all models
- Run `python prepare_original_data.py` to generate

### Generated Data
- Synthetic data generation for testing/demonstration
- Realistic patterns based on operational insights
- Configurable dataset sizes

### Upload Custom Data
- CSV upload functionality for all models
- Automatic format validation and preview
- Flexible column mapping

## 🛠️ Technical Architecture

### Backend (Flask)
- **Framework**: Flask with RESTful API design
- **Models**: Modular wrapper classes for each ML model
- **Data Processing**: Pandas-based ETL pipelines
- **File Handling**: Secure upload/download with validation

### Frontend (Bootstrap + JavaScript)
- **UI Framework**: Bootstrap 5 with custom styling
- **Interactivity**: Vanilla JavaScript with async/await
- **Visualization**: Dynamic charts and metrics display
- **UX**: Tabbed interface with guided workflows

### ML Pipeline
- **Preprocessing**: Automated feature engineering
- **Training**: Model-specific hyperparameter optimization
- **Evaluation**: Comprehensive metrics (R², MAE, RMSE)
- **Persistence**: Pickle-based model serialization

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Data Preparation
```bash
# Generate original datasets from raw data
python prepare_original_data.py
```

### Launch Application
```bash
python app.py
```

Access at: http://localhost:5001

## 📈 Model Performance Summary

| Model | Algorithm | Key Metric | Performance | Status |
|-------|-----------|------------|-------------|--------|
| Demand Prediction | XGBoost | R² Score | 0.9558 | ✅ Production Ready |
| Prep Time Prediction | XGBoost | MAE | ~2-3 min | ✅ Production Ready |
| Promotion Effectiveness | Random Forest | R² Score | 0.85+ | ✅ Production Ready |
| Dish Prediction | Multi-output Reg | R² Score | 0.80+ | ✅ Production Ready |
| Dish Recommendation | Association Rules | Confidence | 0.70+ | ✅ Production Ready |

## 🎯 Key Features

### Data Management
- **Multiple Input Methods**: Original data, generated data, file upload
- **Format Validation**: Automatic CSV structure checking
- **Preview Functionality**: Data sampling and statistics
- **Secure Storage**: Organized upload directory structure

### Model Training
- **Automated Pipelines**: End-to-end training workflows
- **Progress Tracking**: Real-time training status updates
- **Performance Metrics**: Comprehensive evaluation reporting
- **Model Persistence**: Automatic saving and loading

### Prediction Interface
- **Interactive Forms**: User-friendly input collection
- **Real-time Results**: Instant prediction generation
- **Visualization**: Metric boxes and charts
- **Export Options**: CSV download capabilities

### Business Intelligence
- **Operational Insights**: Actionable predictions for business decisions
- **Scenario Analysis**: What-if simulations for promotions
- **Performance Tracking**: Historical accuracy monitoring
- **Scalability**: Designed for production deployment

## 📁 Project Structure

```
app_v3/
├── app.py                          # Main Flask application
├── prepare_original_data.py        # Data preparation script
├── requirements.txt                # Python dependencies
├── models_*.py                     # ML model wrappers
├── static/
│   └── app.js                      # Frontend JavaScript
├── templates/
│   └── index.html                  # Main UI template
├── uploads/                        # Data storage
│   └── original_data/              # Pre-processed datasets
├── models/                         # Trained model storage
├── prep_time_best_model.ipynb      # Prep time analysis notebook
├── final_promotion_effectiveness_clean.ipynb  # Promotion analysis notebook
└── README.md                       # This file
```

## 🔧 API Endpoints

### Health Check
- `GET /health` - Model status overview

### Data Generation
- `POST /api/generate/dish_data` - Generate dish prediction data
- `POST /api/generate/demand_data` - Generate demand prediction data
- `POST /api/generate/order_data` - Generate recommendation data
- `POST /api/generate/prep_time_data` - Generate prep time data
- `POST /api/generate/promotion_data` - Generate promotion data

### Original Data Access
- `GET /api/use_original/dish_prediction` - Load original dish data
- `GET /api/use_original/demand_prediction` - Load original demand data
- `GET /api/use_original/dish_recommendation` - Load original order data
- `GET /api/use_original/prep_time_prediction` - Load original prep time data
- `GET /api/use_original/promotion_effectiveness` - Load original promotion data

### Model Operations
- `POST /api/{model}/train` - Train specific model
- `POST /api/{model}/predict` - Generate predictions
- `GET /api/{model}/status` - Check model status

## 🎨 User Interface

### Navigation
- **Tabbed Interface**: Clean separation of model functionalities
- **Progressive Disclosure**: Step-by-step workflows
- **Status Indicators**: Visual feedback on model states

### Data Input
- **Multiple Options**: Upload, generate, or use original data
- **Validation**: Real-time format checking
- **Preview**: Data sampling before processing

### Results Display
- **Metric Cards**: Prominent performance indicators
- **Interactive Charts**: Visual analysis of predictions
- **Export Options**: Download results as CSV

## 🔒 Security & Best Practices

### Data Security
- **Input Validation**: Comprehensive CSV format checking
- **File Type Restrictions**: CSV-only uploads
- **Size Limits**: 50MB maximum file size
- **Secure Paths**: Sanitized file handling

### Model Security
- **Pickle Safety**: Controlled deserialization
- **Input Sanitization**: Feature validation
- **Error Handling**: Graceful failure management

### Production Readiness
- **Logging**: Comprehensive error tracking
- **Monitoring**: Health check endpoints
- **Scalability**: Stateless design principles
- **Documentation**: Complete API documentation

## 🚀 Deployment Options

### Local Development
```bash
python app.py
```

### Production Server
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 app:app

# Using uWSGI
uwsgi --http :5001 --wsgi-file app.py --callable app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5001
CMD ["python", "app.py"]
```

## 📈 Future Enhancements

### Model Improvements
- **Deep Learning**: LSTM networks for time-series
- **Ensemble Methods**: Model stacking and blending
- **AutoML**: Automated hyperparameter optimization

### Feature Additions
- **Real-time Predictions**: Streaming data integration
- **A/B Testing**: Promotion effectiveness experiments
- **Multi-location**: Restaurant-specific models

### UI/UX Improvements
- **Advanced Visualization**: Interactive charts and dashboards
- **Mobile Optimization**: Responsive design enhancements
- **API Integration**: Third-party service connections

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit pull request

## 📄 License

This project is part of the ML2025 academic initiative. See individual notebooks for detailed methodology and analysis.

## 📞 Support

For technical issues or questions:
- Check the individual model notebooks for detailed documentation
- Review the API endpoints for integration guidance
- Examine the data preparation scripts for data format requirements

---

**Built with ❤️ for ML2025 - Advanced Machine Learning for Food Delivery Analytics**

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
