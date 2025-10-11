# 🍽️ Food Delivery Demand Prediction using Multi-Source Data Integration

## 📘 Overview
Food delivery platforms experience fluctuating demand influenced by various external factors — weather, events, holidays, and socio-economic conditions.  
This project integrates **multiple public data sources and APIs** to build a supervised ML model that predicts **food delivery demand** in real-time or for future time windows.

The final system forecasts order volume using contextual data from weather, restaurant information, holidays, events, population density, and mobility statistics.

---

## 🎯 Business Problem
> “How can a food delivery platform predict hourly or daily demand for each area, accounting for local events, population, and external conditions — to optimize delivery resources and reduce waiting times?”

---

## 🤖 ML Problem Formulation

| Aspect | Description |
|--------|-------------|
| **Type** | Supervised Learning — Regression |
| **Objective** | Predict number of expected food orders in a given area and time |
| **Inputs** | Weather, time, restaurant density, local events, population, and economic data |
| **Target Variable** | Number of orders per area per hour |
| **Evaluation Metrics** | MAE, RMSE, R², MAPE |

---

## 🧩 Data Sources

We combine **multiple APIs and datasets** across environmental, demographic, and behavioral factors.

| # | Source | Type | Description / Features Provided |
|---|---------|------|----------------------------------|
| **1️⃣ OpenWeatherMap API** | Weather | Temperature, humidity, wind, and precipitation. Weather affects delivery times and demand. |
| **2️⃣ Meteostat API** | Historical Weather | Long-term weather records to augment historical datasets. |
| **3️⃣ Yelp Open Dataset** | Restaurant Data | Business names, categories, ratings, and reviews. Used to compute restaurant density, cuisine diversity, and average rating per region. |
| **4️⃣ Instacart Market Basket Dataset (Kaggle)** | Order History | Simulates historical customer purchase behavior and ordering frequency. |
| **5️⃣ Google Places / Foursquare API** | Location Intelligence | Popularity, check-ins, and competitor density per area. Used to estimate demand hotspots. |
| **6️⃣ Google Mobility Reports (COVID-19 Mobility / Community Mobility)** | Human Mobility | Tracks how foot traffic changes in grocery, retail, and residential areas — useful to estimate demand shifts on weekends or events. |
| **7️⃣ Public Holidays API** | Calendar Events | Returns local holidays, festivals, and cultural events by country/region. Used as categorical time features. [https://date.nager.at](https://date.nager.at) |
| **8️⃣ Ticketmaster / Eventbrite API** | Local Events | Provides concerts, sports, and public gatherings in an area. High-impact predictor for sudden demand spikes. |
| **9️⃣ WorldPop / UN Data / OpenStreetMap Population Grid** | Demographics | Population density per area, urban vs rural classification, age groups. Helps scale potential order volume. |
| **🔟 World Bank Open Data API** | Economic Indicators | GDP per capita, income level, urbanization rate — proxy for affordability and ordering frequency. |
| **1️⃣1️⃣ City-level Open Data Portals** | Local Statistics | Traffic volume, commute patterns, and restaurant inspections (varies by region). |
| **1️⃣2️⃣ Google Trends API** | Search Interest | Popular search queries like “pizza delivery near me” can proxy short-term demand surges. |

---

## 🧮 Data Construction Pipeline

1. **Collect Weather Data**  
   - From OpenWeatherMap & Meteostat (temperature, rain, humidity, etc.)
2. **Extract Restaurant Attributes**  
   - Yelp / Google Places (ratings, cuisines, restaurant count)
3. **Load Order Data**  
   - Instacart orders or synthetic dataset (orders per time/location)
4. **Integrate Events and Holidays**  
   - Public Holidays API and Ticketmaster/Eventbrite API to mark special occasions.
5. **Add Socioeconomic Features**  
   - Population density, GDP, and mobility data (WorldPop, World Bank, Google Mobility).
6. **Feature Engineering**  
   - Time features: hour, day of week, weekend flag  
   - Weather features: temperature, rain, wind  
   - Demographics: population_density, avg_income  
   - Events: binary “event_today” flag, “holiday” flag  
   - Restaurant features: density, average rating  
   - Normalize and encode categorical values

**Sample Feature Schema:**

| timestamp | area_id | temp | rain | event_today | holiday | population_density | avg_income | rest_density | rating_avg | orders |
|------------|----------|------|------|--------------|----------|--------------------|-------------|----------------|-------------|--------|
| 2025-10-01 18:00 | A12 | 28.5 | 1 | 0 | 1 | 3500 | 42000 | 14 | 4.2 | 248 |

---

## ⚙️ Model Architecture

**Baseline Models**
- Linear Regression  
- Random Forest Regressor  
- Gradient Boosting (XGBoost, LightGBM)

**Neural Models**
- Feed-Forward Neural Network (MLP)  
  - Input: concatenated numeric & encoded categorical features  
  - Hidden Layers: 2–3 dense layers (64–128 neurons) with ReLU  
  - Output: continuous value (predicted demand)

**Loss Function:** MSE  
**Optimizer:** Adam  
**Framework:** PyTorch / TensorFlow  

---

## 📊 Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **MAE** | Measures average magnitude of prediction errors |
| **RMSE** | Penalizes large deviations more than MAE |
| **R² Score** | Proportion of variance explained by the model |
| **MAPE** | Measures relative prediction error (for business interpretability) |

---

## 🧠 Expected Insights

| Factor | Insight |
|---------|----------|
| 🌧️ **Weather** | Rainy evenings → more delivery orders |
| 📅 **Holidays** | Orders spike during long weekends and festivals |
| 🎉 **Events** | Stadium concerts or sports events nearby → demand surge |
| 👨‍👩‍👧 **Population** | Denser, younger areas have consistently higher order counts |
| 💰 **Economy** | Higher GDP/income correlates with more premium food orders |
| 🍕 **Restaurant Density** | High density leads to greater competition but higher total volume |

---

## 🚀 Deployment Plan

**Goal:** Make the model available via a REST API and interactive dashboard.

- **Backend:** Flask / FastAPI  
- **Frontend:** Streamlit Dashboard (interactive map of predicted demand)  
- **Hosting:** Render / AWS / Streamlit Cloud  

**Example Input:**
```json
{
  "temperature": 27,
  "rain": 0,
  "hour": 20,
  "day_of_week": 5,
  "holiday": 1,
  "event_today": 0,
  "population_density": 4200,
  "avg_income": 40000,
  "restaurant_density": 12,
  "avg_rating": 4.3
}
````

**Example Output:**

```json
{
  "predicted_orders": 318
}
```

---

## 🧱 System Architecture

```
[Weather APIs] ─┐
[Restaurant APIs] ─┤
[Event + Holiday APIs] ─┤
[Demographics & Economy] ─┤
[Order History Dataset] ─┘
            │
            ▼
 [Data Cleaning + Feature Engineering]
            │
            ▼
     [Machine Learning Model]
            │
            ▼
 [Flask / Streamlit Deployment Layer]
```

---

## 📆 Project Timeline (4 Weeks)

| Week       | Deliverables                                                                                         |
| ---------- | ---------------------------------------------------------------------------------------------------- |
| **Week 1** | Collect datasets (weather, events, population, restaurant, orders). Data cleaning and schema design. |
| **Week 2** | Merge and preprocess multi-source data. Feature engineering and baseline regression models.          |
| **Week 3** | Train MLP and gradient boosting models. Hyperparameter tuning and model evaluation.                  |
| **Week 4** | Build and deploy Flask API / Streamlit app. Final report and presentation.                           |

---

## 💡 Future Work

* Incorporate **real-time traffic / route ETA** data for delivery-time prediction.
* Expand to **multi-city forecasting** using geospatial clustering.
* Use **LSTM / Temporal CNN** for sequential modeling of hourly trends.
* Integrate **price-surge optimization** and **driver allocation** simulations.
* Add **demand heatmap visualization** using Mapbox or Folium.

---

## 📚 Tools & Technologies

| Category            | Tools                                                                      |
| ------------------- | -------------------------------------------------------------------------- |
| **Programming**     | Python 3.x                                                                 |
| **APIs**            | OpenWeatherMap, Yelp, Meteostat, Public Holidays, Ticketmaster, World Bank |
| **Libraries**       | Pandas, NumPy, Scikit-learn, PyTorch, XGBoost                              |
| **Visualization**   | Matplotlib, Seaborn, Plotly, Streamlit                                     |
| **Deployment**      | Flask, Streamlit Cloud, Render, AWS EC2                                    |
| **Version Control** | GitHub, GitHub Actions (for CI/CD)                                         |

---

## 📎 References

* [OpenWeatherMap API](https://openweathermap.org/api)
* [Meteostat API](https://dev.meteostat.net/)
* [Yelp Open Dataset](https://www.yelp.com/dataset)
* [Instacart Dataset (Kaggle)](https://www.kaggle.com/competitions/instacart-market-basket-analysis)
* [Public Holidays API](https://date.nager.at)
* [Ticketmaster Developer API](https://developer.ticketmaster.com/products-and-docs/apis/getting-started/)
* [WorldPop Population Data](https://www.worldpop.org/)
* [World Bank Open Data](https://data.worldbank.org/)
* [Google Mobility Reports](https://www.google.com/covid19/mobility/)
* [Google Places API](https://developers.google.com/maps/documentation/places/web-service/overview)

---

## 👨‍💻 Author

**Saugat Shakya**
Masters in Data Science and Artificial Intelligence
**Food Delivery Demand Prediction — ML Course Project**

```
