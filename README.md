# 🍽️ Food Delivery Demand Prediction — ML Course Project

## 📘 Overview

Food delivery platforms experience **fluctuating demand** influenced by various factors — weather, events, holidays, and local demographics.
This project integrates **multi-source data** to build supervised ML models that predict **food delivery demand** at the subzone level for a small city.

The final system forecasts **hourly and daily order volumes** for each subzone, using historical orders and contextual features from weather, restaurants, holidays, events, and demographics.

---

## 🎯 Business Problem

> “How can a food delivery platform predict hourly or daily demand for each subzone in a city, accounting for historical trends, local events, weather, and population — to optimize delivery resources and reduce waiting times?”

---

## 🤖 ML Problem Formulation

| Aspect                 | Description                                                                                               |
| ---------------------- | --------------------------------------------------------------------------------------------------------- |
| **Type**               | Supervised Learning — Regression                                                                          |
| **Objective**          | Predict number of expected food orders per subzone per hour or per day                                    |
| **Inputs**             | Historical orders, subzone, time features, weather, restaurant data, events, holidays, population density |
| **Target Variable**    | Number of orders per subzone per hour                                                                     |
| **Evaluation Metrics** | MAE, RMSE, R², MAPE                                                                                       |

**Notes:**

* We will train **separate models for hourly and daily predictions**.
* Daily demand can also be derived by **aggregating hourly predictions**, but we will analyze both approaches for comparison.
* Multiple models (Linear Regression, Random Forest, Gradient Boosting, MLP) will be trained and compared.

---

## 🧩 Data Sources

### 1. Historical Orders

* **Dataset:** [Food Delivery Order History](https://www.kaggle.com/datasets/sujalsuthar/food-delivery-order-history-data)
* **Key Columns:**

  * `Restaurant ID`, `Restaurant Name`
  * `Subzone` (used to define areas)
  * `Order Placed At` (timestamp)
  * `Distance`, `Items in order`, `Order Status`, `Total`
  * `Customer Rating`
* **Purpose:** Provides past order volume per subzone and time — the **core target variable**.

### 2. Weather

* **APIs:** OpenWeatherMap, Meteostat
* **Features:** Temperature, humidity, precipitation, wind
* **Derived Features:**

  * `temp`, `rain`, `humidity`
  * Can create flags like `rainy_day` or `hot_day`

### 3. Restaurant Data

* **APIs/Datasets:** Yelp Open Dataset, Google Places API
* **Features:** Restaurant count, ratings, cuisines per subzone
* **Derived Features:**

  * `restaurant_density` (restaurants per subzone)
  * `avg_rating`, `cuisine_diversity`

### 4. Events and Holidays

* **APIs:** Public Holidays API, Ticketmaster/Eventbrite
* **Features:** Local holidays, festivals, concerts, sports events
* **Derived Features:**

  * `event_today` (binary)
  * `holiday` (binary)

### 5. Demographics & Population

* **Datasets:** WorldPop, City Open Data portals
* **Features:** Population density, urban/rural classification, age distribution
* **Derived Features:**

  * `population_density`, `young_population_fraction`

### 6. Optional Features

* Google Mobility Reports, Google Trends (for search interest like “pizza delivery near me”)

---

## 🧮 Feature Engineering

| Feature                                                   | Type        | Source            | Notes                               |
| --------------------------------------------------------- | ----------- | ----------------- | ----------------------------------- |
| `subzone`                                                 | Categorical | Orders dataset    | Defines area for prediction         |
| `hour_of_day`                                             | Numerical   | Orders timestamp  | Hourly trend analysis               |
| `day_of_week`                                             | Categorical | Orders timestamp  | Weekend vs weekday patterns         |
| `temp`, `humidity`, `rain`                                | Numerical   | Weather API       | Environmental factors               |
| `restaurant_density`, `avg_rating`                        | Numerical   | Restaurant API    | Supply-side features                |
| `event_today`, `holiday`                                  | Binary      | Event/Holiday API | External demand drivers             |
| `population_density`                                      | Numerical   | Demographics      | Scaling factor for potential demand |
| `historical_orders_last_1h`, `historical_orders_last_24h` | Numerical   | Orders dataset    | Lag features for temporal trends    |
| `discount_used`, `avg_total_bill`                         | Numerical   | Orders dataset    | Pricing/promotions impact           |

**Sample Feature Table (per subzone, per hour):**

| timestamp        | subzone | temp | rain | event_today | holiday | population_density | restaurant_density | avg_rating | historical_orders_last_1h | total_orders |
| ---------------- | ------- | ---- | ---- | ----------- | ------- | ------------------ | ------------------ | ---------- | ------------------------- | ------------ |
| 2024-09-01 18:00 | GK2     | 28.5 | 1    | 0           | 1       | 3500               | 14                 | 4.2        | 120                       | 248          |

---

## ⚙️ Model Architecture

**Baseline Models**

* Linear Regression
* Random Forest Regressor
* Gradient Boosting (XGBoost, LightGBM)

**Neural Models**

* Feed-Forward Neural Network (MLP)

  * Input: concatenated numeric & encoded categorical features
  * Hidden Layers: 2–3 dense layers (64–128 neurons) with ReLU
  * Output: continuous value (predicted orders per subzone)

**Loss Function:** MSE
**Optimizer:** Adam
**Framework:** PyTorch / TensorFlow

**Approach:**

* Train separate models for **hourly** and **daily** prediction.
* Compare baseline vs neural models.
* Analyze performance at subzone and city aggregate levels.

---

## 📊 Evaluation Metrics

| Metric   | Purpose                                   |
| -------- | ----------------------------------------- |
| MAE      | Average magnitude of errors               |
| RMSE     | Penalizes large deviations                |
| R² Score | Fraction of variance explained            |
| MAPE     | Relative error, business interpretability |

---

## 🧠 Expected Insights

| Factor                | Insight                                               |
| --------------------- | ----------------------------------------------------- |
| 🌧️ Weather           | Rain → higher order volume, hot days → lower delivery |
| 📅 Holidays           | Festivals, public holidays → spikes                   |
| 🎉 Events             | Sports, concerts → localized demand surge             |
| 👨‍👩‍👧 Population   | Dense or young subzones → consistently higher orders  |
| 🍕 Restaurant Density | High density → more orders but competition too        |
| 💰 Discounts/Promos   | Promotions → increased demand in short periods        |

---

## 🗺️ Area Definition

* **Area = Subzone** (from dataset, e.g., GK2, Sector 4)
* City-level aggregation unnecessary; city = single value (`Delhi NCR`)
* Subzones provide sufficient granularity for prediction and feature mapping

---

## 📆 Project Timeline (4 Weeks)

| Week       | Tasks                                                                                                               |
| ---------- | ------------------------------------------------------------------------------------------------------------------- |
| **Week 1** | Collect datasets: orders, weather, restaurants, events, population. Clean & integrate.                              |
| **Week 2** | Feature engineering: time features, lag features, derived variables. Train baseline regression models.              |
| **Week 3** | Train MLP & gradient boosting models. Hyperparameter tuning. Evaluate hourly vs daily predictions.                  |
| **Week 4** | Compare models, finalize evaluation. Prepare report & documentation. Optional: deploy simple REST API or dashboard. |

---

## 📚 Tools & Technologies

| Category        | Tools                                                          |
| --------------- | -------------------------------------------------------------- |
| Programming     | Python 3.x                                                     |
| APIs            | OpenWeatherMap, Yelp, Meteostat, Ticketmaster, Public Holidays |
| Libraries       | Pandas, NumPy, Scikit-learn, PyTorch, XGBoost, LightGBM        |
| Visualization   | Matplotlib, Seaborn, Plotly, Streamlit                         |
| Version Control | GitHub                                                         |

---

## 📎 References

* [Food Delivery Order History Dataset](https://www.kaggle.com/datasets/sujalsuthar/food-delivery-order-history-data)
* [OpenWeatherMap API](https://openweathermap.org/api)
* [Meteostat API](https://dev.meteostat.net/)
* [Yelp Open Dataset](https://www.yelp.com/dataset)
* [Ticketmaster Developer API](https://developer.ticketmaster.com/products-and-docs/apis/getting-started/)
* [Public Holidays API](https://date.nager.at)
* [WorldPop Population Data](https://www.worldpop.org/)
