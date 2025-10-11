# 🍽️ Food Delivery Demand Prediction using Multi-Source Data Integration

## 📘 Overview
Food delivery services often struggle with fluctuating order demand affected by factors such as weather, restaurant availability, and time of day.  
This project aims to **forecast food delivery demand** by integrating multiple data sources — weather APIs, restaurant information, and historical order datasets — and training supervised ML models to predict future demand levels.

This helps businesses:
- Optimize delivery staff allocation  
- Reduce customer wait times  
- Improve inventory and logistics planning  

---

## 🎯 Business Problem
> "How can a food delivery platform predict hourly or daily order demand for each area to optimize operations and reduce delivery delays?"

By turning this **business question** into a **machine learning problem**, we can forecast expected orders based on contextual and external data (e.g., weather, ratings, time, location).

---

## 🤖 ML Problem Formulation

| Aspect | Description |
|--------|-------------|
| **Type** | Supervised Learning — Regression |
| **Objective** | Predict number of expected food orders in a given area and time |
| **Input Features** | Weather, restaurant attributes, ratings, time, and location |
| **Output Label** | Order demand (number of orders) |
| **Evaluation Metrics** | Mean Absolute Error (MAE), Root Mean Square Error (RMSE), R² score |

---

## 🧩 Data Sources

We combine **four+ APIs/datasets** to construct a real-world dataset.

| Source | Type | Description |
|---------|------|-------------|
| **1️⃣ OpenWeatherMap API** | Weather | Provides temperature, precipitation, wind, humidity, and conditions (e.g., sunny, rain) for any location and time. |
| **2️⃣ Yelp Open Dataset** | Restaurants | Contains restaurant information such as location, ratings, reviews, and cuisine type. Used to calculate restaurant density and average rating. |
| **3️⃣ Instacart Market Basket Analysis (Kaggle)** | Order History | Proxy for user purchase behavior and time-of-day patterns in ordering. Used to simulate food order demand. |
| **4️⃣ Meteostat API** | Historical Weather | Supplements weather history for missing timestamps or broader climate context. |
| **(Optional)** Google Places or Foursquare API | Location Intelligence | Adds features like popularity score, nearby restaurant count, and competitor density. |

---

## 🧮 Data Construction Pipeline

1. **Collect weather data** using OpenWeatherMap and Meteostat (temperature, humidity, rain, wind).  
2. **Fetch restaurant attributes** (ratings, categories, coordinates) using Yelp or Google Places API.  
3. **Load order data** from Instacart or a synthetic dataset (timestamp, location, order count).  
4. **Merge datasets** by timestamp and location grid (e.g., postal code or lat/lon clusters).  
5. **Feature engineering:**
   - Encode time features (hour, day of week, weekend flag).
   - Compute average restaurant rating per area.
   - Calculate restaurant density (count per area).
   - Include weather features (rain, temperature, etc.).
   - Normalize and scale numerical features.

Example dataset schema:

| time | area_id | temp | rain | wind | rating_avg | restaurant_density | is_weekend | orders |
|------|----------|------|------|------|-------------|--------------------|-------------|--------|
| 2025-10-01 12:00 | A12 | 29°C | 0 | 3.2 | 4.3 | 12 | 0 | 153 |

---

## ⚙️ Model Architecture

**Baseline Models:**
- Linear Regression  
- Random Forest Regressor  
- Gradient Boosted Trees (XGBoost, LightGBM)

**Neural Model:**
- Feed-Forward Neural Network (Multi-Layer Perceptron)
  - Input: [temp, rain, wind, restaurant_density, rating_avg, hour, day_of_week, is_weekend]
  - Hidden layers: 2–3 dense layers with ReLU activation
  - Output: single continuous value → predicted demand

**Loss Function:** Mean Squared Error (MSE)  
**Optimizer:** Adam / SGD  
**Frameworks:** PyTorch or TensorFlow  

---

## 📊 Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **MAE (Mean Absolute Error)** | Average absolute difference between predicted and actual demand. |
| **RMSE (Root Mean Square Error)** | Penalizes larger errors more heavily. |
| **R² Score** | Explains variance captured by the model. |
| **MAPE (Mean Absolute Percentage Error)** | Relative accuracy measure (use cautiously for zero values). |

---

## 🧠 Expected Insights

- High temperature + rain → lower outdoor orders, higher delivery demand.  
- Weekends & evenings → higher average order counts.  
- High restaurant density areas → smaller but more frequent orders.  
- Better-rated restaurants → stable order volumes regardless of weather.

---

## 🚀 Deployment Plan

**Goal:** Make the model accessible to users via a simple web interface or API.

### Tools:
- **Flask / FastAPI** → REST API for model prediction.  
- **Streamlit** → Interactive dashboard for visualization.

### Example Use Case:
User inputs:
```json
{
  "temperature": 26,
  "rain": 1,
  "wind": 4.2,
  "restaurant_density": 8,
  "avg_rating": 4.1,
  "hour": 19,
  "day_of_week": 5
}
````

Model output:

```json
{
  "predicted_orders": 245
}
```

Dashboard may display:

* Predicted order volume
* Confidence interval
* Weather vs demand correlation plot

---

## 🧱 System Architecture

```
[Weather APIs]      [Restaurant Data]      [Order History]
       │                   │                      │
       └──► [Data Preprocessing & Feature Engineering]
                        │
                        ▼
              [Machine Learning Model]
                        │
                        ▼
               [Prediction API / Dashboard]
```

---

## 📆 Timeline (4 Weeks)

| Week       | Task                                                          |
| ---------- | ------------------------------------------------------------- |
| **Week 1** | Collect datasets from APIs & Kaggle, preprocess and merge.    |
| **Week 2** | Feature engineering + baseline regression models.             |
| **Week 3** | Train feed-forward neural network + model tuning.             |
| **Week 4** | Build and deploy Flask/Streamlit app, finalize documentation. |

---

## 💡 Future Improvements

* Incorporate **real-time traffic or route data** for ETA prediction.
* Extend to **multi-city forecasting** using geospatial clustering.
* Add **deep time-series forecasting** (e.g., LSTM, Temporal CNNs).
* Integrate **price surge optimization** model for dynamic pricing.

---

## 📚 Tools & Technologies

| Category          | Tools                                                    |
| ----------------- | -------------------------------------------------------- |
| **Programming**   | Python 3.x                                               |
| **APIs**          | OpenWeatherMap, Yelp, Meteostat, Google Places           |
| **Libraries**     | Pandas, NumPy, Scikit-learn, XGBoost, PyTorch/TensorFlow |
| **Visualization** | Matplotlib, Seaborn, Plotly, Streamlit                   |
| **Deployment**    | Flask, FastAPI, Streamlit Cloud / Render / AWS EC2       |

---

## 🧾 References

* [OpenWeatherMap API Docs](https://openweathermap.org/api)
* [Meteostat API](https://dev.meteostat.net/)
* [Yelp Open Dataset](https://www.yelp.com/dataset)
* [Instacart Market Basket Dataset (Kaggle)](https://www.kaggle.com/competitions/instacart-market-basket-analysis)
* [Google Places API](https://developers.google.com/maps/documentation/places/web-service/overview)

---

## 👨‍💻 Author

**Saugat Shakya**
Bachelor of Computer Science, Kathmandu University
Food Delivery Demand Prediction — Machine Learning Course Project

```
