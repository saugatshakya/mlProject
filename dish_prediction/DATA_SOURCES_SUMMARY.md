# 📡 DATA SOURCES FOR INFERENCE

## Summary
The model needs **52 features** to make predictions. Here's where each comes from:

---

## 1. TEMPORAL FEATURES (5) - AUTOMATIC ✅
**Source**: System clock / current timestamp

These are automatically calculated from the current time:
- `hour` (0-23)
- `day_of_week` (0=Mon, 6=Sun)
- `is_weekend` (0 or 1)
- `sin_hour` (cyclical encoding)
- `cos_hour` (cyclical encoding)

**Implementation**: Built into inference script, no external data needed.

---

## 2. HISTORICAL ORDER DATA (40) - YOUR DATABASE 📊
**Source**: Your restaurant order database

You need the **last 3 hours** of orders for each of the 10 dishes:
- 30 lag features (10 dishes × 3 lags)
- 10 smoothed features (3-hour rolling average)

**Example Database Query**:
```sql
SELECT 
    timestamp,
    dish_name,
    SUM(quantity) as quantity
FROM orders
WHERE timestamp >= NOW() - INTERVAL 3 HOUR
GROUP BY timestamp, dish_name
ORDER BY timestamp DESC
```

**Required Dishes** (Top 10):
1. Bageecha Pizza
2. Chilli Cheese Garlic Bread
3. Bone in Jamaican Grilled Chicken
4. All About Chicken Pizza
5. Makhani Paneer Pizza
6. Margherita Pizza
7. Cheesy Garlic Bread
8. Jamaican Chicken Melt
9. Herbed Potato
10. Tripple Cheese Pizza

---

## 3. WEATHER DATA (4) - EXTERNAL API 🌤️
**Source**: Weather API (real-time)

### RECOMMENDED: WeatherAPI.com
- **Free Tier**: 1 million calls/month
- **Cost**: FREE
- **Sign up**: https://www.weatherapi.com/signup.aspx
- **Features**: Weather + Air Quality in one call
- **Response time**: < 1 second

**Setup**:
```bash
# Get API key from dashboard
export WEATHERAPI_KEY="your_api_key_here"
```

**Fields Needed**:
- `env_temp`: Temperature (°C)
- `env_rhum`: Humidity (%)
- `env_precip`: Precipitation (mm)
- `env_wspd`: Wind speed (km/h)

**Code**:
```python
from api_integration import get_weather_weatherapi
weather, pollution = get_weather_weatherapi("Delhi")
```

### ALTERNATIVE: OpenWeatherMap
- **Free Tier**: 60 calls/min, 1M calls/month
- **Cost**: FREE
- **Sign up**: https://openweathermap.org/api
- **Setup**:
```bash
export OPENWEATHER_API_KEY="your_key"
```

**Fallback**: If API fails, uses defaults (temp=25°C, humidity=60%)

---

## 4. POLLUTION DATA (1) - EXTERNAL API 🏭
**Source**: Air Quality API (real-time)

### RECOMMENDED: WeatherAPI.com (same as weather)
Includes AQI in the weather call - no extra request needed!

### ALTERNATIVE: OpenWeatherMap Air Pollution API
```bash
export OPENWEATHER_API_KEY="your_key"
```

**Field Needed**:
- `aqi`: Air Quality Index (0-500)

**Fallback**: If API fails, uses default (aqi=100)

---

## 5. EVENT/HOLIDAY DATA (2) - CALENDAR 📅
**Source**: Static calendar data or holiday API

**Fields**:
- `has_event`: 1 if major event in Delhi, 0 otherwise
- `holiday`: 1 if public holiday, 0 otherwise

**Implementation Options**:

### Option A: Static List (Recommended)
```python
DELHI_HOLIDAYS_2025 = [
    "2025-01-26",  # Republic Day
    "2025-03-08",  # Holi
    "2025-08-15",  # Independence Day
    "2025-10-02",  # Gandhi Jayanti
    "2025-10-24",  # Diwali
    # ... add more
]

def is_holiday(date):
    return date.strftime("%Y-%m-%d") in DELHI_HOLIDAYS_2025
```

### Option B: Holiday API
- **Calendarific**: https://calendarific.com/
- **Abstract API**: https://www.abstractapi.com/holidays-api

**Fallback**: If unknown, set both to 0 (regular day)

---

## 📊 DATA FLOW DIAGRAM

```
┌─────────────────────┐
│   INFERENCE TIME    │
│   (Every Hour)      │
└──────────┬──────────┘
           │
           ├─────────────────────────────────────────┐
           │                                         │
           ▼                                         ▼
┌──────────────────────┐                  ┌──────────────────────┐
│  AUTOMATIC FEATURES  │                  │   DATA COLLECTION    │
│  (System Clock)      │                  │   (External Sources) │
└──────────────────────┘                  └──────────┬───────────┘
           │                                         │
           │                              ┌──────────┴──────────┐
           │                              │                     │
           │                              ▼                     ▼
           │                   ┌─────────────────┐   ┌─────────────────┐
           │                   │  YOUR DATABASE  │   │  EXTERNAL APIs  │
           │                   │  (Last 3 hrs    │   │  - Weather API  │
           │                   │   of orders)    │   │  - AQI API      │
           │                   └────────┬────────┘   │  - Calendar     │
           │                            │            └────────┬────────┘
           │                            │                     │
           ├────────────────────────────┴─────────────────────┤
           │                                                  │
           ▼                                                  ▼
┌──────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                             │
│  - Calculate lags (1h, 2h, 3h)                              │
│  - Calculate rolling means                                   │
│  - Cyclical encoding of time                                │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              TRAINED MODEL                                   │
│  (CatBoost Multi-Output Regressor)                          │
│  52 features → 10 predictions                                │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              PREDICTIONS                                     │
│  Expected orders for each of 10 dishes in next hour         │
└──────────────────────────────────────────────────────────────┘
```

---

## �� COST ANALYSIS

| Data Source | Cost | Frequency | Total/Month |
|-------------|------|-----------|-------------|
| **System Time** | FREE | Every hour | FREE |
| **Your Database** | FREE* | Every hour | FREE* |
| **WeatherAPI** | FREE | Every hour | 720 calls/month |
| **Holiday Calendar** | FREE | Once/year | FREE |

*Assuming you already have order data

**Total Cost**: **$0** (using free tiers)

**At Scale** (1000 API calls/day):
- WeatherAPI.com: Still FREE (1M/month limit)
- OpenWeatherMap: Still FREE (60 calls/min limit)

---

## ⚡ PERFORMANCE

| Component | Latency |
|-----------|---------|
| Database query | < 100ms |
| Weather API call | < 1000ms |
| Feature engineering | < 10ms |
| Model inference | < 50ms |
| **TOTAL** | **< 1.2 seconds** |

Fast enough for real-time predictions!

---

## 🔧 SETUP CHECKLIST

- [ ] Sign up for WeatherAPI.com (or OpenWeatherMap)
- [ ] Get API key and set environment variable
- [ ] Set up database query to fetch last 3 hours of orders
- [ ] Create holiday calendar (optional, use defaults if not available)
- [ ] Test `inference_production.py` with dummy data
- [ ] Integrate with real database
- [ ] Schedule hourly execution (cron job)
- [ ] Set up logging and monitoring

---

## 📝 SAMPLE CRON JOB

```bash
# Run prediction every hour at minute 0
0 * * * * cd /path/to/dish_prediction && /usr/bin/python3 inference_production.py >> logs/predictions.log 2>&1
```

---

## 🆘 TROUBLESHOOTING

**Q: What if API is down?**
A: Model uses default values (temp=25°C, humidity=60%, aqi=100). Predictions still work but may be slightly less accurate.

**Q: What if I don't have 3 hours of history?**
A: Use zeros for missing hours. First few predictions will be less accurate until history builds up.

**Q: Can I use different dishes?**
A: Yes, but you need to retrain the model with your dishes. The model is trained on these specific 10 dishes.

**Q: How often should I update weather data?**
A: Once per hour is sufficient. Weather doesn't change drastically minute-to-minute.

---

*See `inference_production.py` for complete implementation*
*See `api_integration.py` for API integration code*
*See `INFERENCE_GUIDE.md` for detailed feature documentation*
