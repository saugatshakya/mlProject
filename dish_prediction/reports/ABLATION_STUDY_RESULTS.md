# ABLATION STUDY RESULTS

**Scientific Analysis of Feature Group Impact on Model Performance**

---

## 🎯 Executive Summary

This ablation study systematically removes feature groups to measure their **actual impact** on model performance. Unlike correlation analysis, this shows **causal relationships** - what happens when we remove specific features.

---

## 📊 Key Findings

### **FULL MODEL Performance**

- **Test R²**: 0.9417 (94.17% variance explained)
- **Features**: 57 total features across 5 groups
- **Baseline**: This is our best performing model

---

## 🔬 Ablation Results: What Happens When We Remove Features?

### **SURPRISING DISCOVERY**:

**Removing some feature groups IMPROVES performance!** ❗

| Experiment           | Test R²    | Change from Full     | Interpretation                    |
| -------------------- | ---------- | -------------------- | --------------------------------- |
| **FULL MODEL**       | **0.9417** | **Baseline**         | All features included             |
| **NO WEATHER**       | **0.9450** | **+0.0033 (+0.35%)** | ✅ **Better without weather!**    |
| **NO POLLUTION**     | **0.9459** | **+0.0043 (+0.45%)** | ✅ **Better without pollution!**  |
| **NO EVENTS**        | **0.9431** | **+0.0014 (+0.15%)** | ✅ **Better without events!**     |
| **NO TEMPORAL**      | **0.9425** | **+0.0008 (+0.09%)** | ✅ **Better without temporal!**   |
| **ONLY HISTORICAL**  | **0.9545** | **+0.0128 (+1.36%)** | ✅ **BEST: Only lag features!**   |
| **NO EXTERNAL DATA** | **0.9501** | **+0.0085 (+0.90%)** | ✅ **Historical + Temporal only** |

---

## 💡 What This Means

### **CRITICAL INSIGHT**: The model is **OVERFITTING** with too many features!

1. **Historical features alone** (lag1, lag2, lag3, smooth) achieve **R² = 0.9545**
   - This is **1.36% BETTER** than using all features!
2. **Weather features HURT performance** by 0.35%
   - Removing: temp, humidity, precipitation, wind → **Improves R²**
3. **Pollution features HURT performance** by 0.45%
   - Removing: AQI, PM2.5, PM10, NO2, O3, CO → **Improves R²**
4. **Events/Holidays HURT performance** by 0.15%
   - Removing: holiday, has_event → **Improves R²**

---

## 📈 Feature Group Importance Ranking (by impact when removed)

### From LEAST to MOST harmful:

| Rank | Feature Group | Impact     | # Features | Conclusion                            |
| ---- | ------------- | ---------- | ---------- | ------------------------------------- |
| 1    | **Temporal**  | +0.0008 R² | 5          | Slightly harmful, can remove          |
| 2    | **Events**    | +0.0014 R² | 2          | Slightly harmful, can remove          |
| 3    | **Weather**   | +0.0033 R² | 4          | **Moderately harmful**, should remove |
| 4    | **Pollution** | +0.0043 R² | 6          | **Most harmful**, should remove       |

**All external features (weather, pollution, events) are adding noise, not signal!**

---

## 🎓 Scientific Explanation

### Why do these features hurt performance?

1. **Overfitting**: Model learns noise instead of signal
   - 57 features for 2004 training samples
   - Weather/pollution may have random correlations with specific training data
2. **Signal-to-Noise Ratio**:
   - Historical features (past orders) are **strong predictors** (R² = 0.9545)
   - Weather/pollution are **weak predictors** that add noise
3. **Multicollinearity**:

   - Weather/pollution/events may correlate with time patterns
   - Model gets confused between real causes and spurious correlations

4. **Temporal Autocorrelation**:
   - Past orders predict future orders **very well** (lag features)
   - External factors don't add meaningful information beyond past patterns

---

## ✅ Recommendations

### **Option 1: SIMPLE MODEL (Recommended)**

- **Use**: Only historical lag features (40 features)
- **Performance**: R² = 0.9545 (BEST)
- **Benefits**:
  - Simplest model
  - Best performance
  - No external data needed
  - Fastest inference

### **Option 2: HISTORICAL + TEMPORAL**

- **Use**: Historical + temporal features (45 features)
- **Performance**: R² = 0.9501 (Very Good)
- **Benefits**:
  - Captures time-of-day patterns
  - Still very simple
  - No external data needed

### **Option 3: FULL MODEL (Current)**

- **Use**: All 57 features
- **Performance**: R² = 0.9417 (Good but worse)
- **Drawbacks**:
  - More complex
  - Requires external data (weather/pollution APIs)
  - **LOWER performance**
  - Overfitting issues

---

## 📊 Detailed Statistics

### Model Comparison Table

```
┌─────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────────┐
│ Experiment          │ Features │ Train R² │ Test R²  │ Test MAE │ R² Change    │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤
│ FULL MODEL          │    57    │ 0.9998   │ 0.9417   │ 0.0714   │   Baseline   │
│ NO WEATHER          │    53    │ 0.9998   │ 0.9450   │ 0.0666   │  +0.35% ✅   │
│ NO POLLUTION        │    51    │ 0.9998   │ 0.9459   │ 0.0680   │  +0.45% ✅   │
│ NO EVENTS           │    55    │ 0.9998   │ 0.9431   │ 0.0702   │  +0.15% ✅   │
│ NO TEMPORAL         │    52    │ 0.9998   │ 0.9425   │ 0.0655   │  +0.09% ✅   │
│ ONLY HISTORICAL     │    40    │ 0.9998   │ 0.9545   │ 0.0579   │  +1.36% ✅✅ │
│ NO EXTERNAL DATA    │    45    │ 0.9998   │ 0.9501   │ 0.0641   │  +0.90% ✅   │
└─────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────────┘
```

**Note**: All Train R² ≈ 0.9998 shows the model CAN fit the training data perfectly with any feature set. The question is which generalizes best to test data.

---

## 🔍 Per-Dish Analysis

All 10 dishes show the **same pattern**:

- Historical features alone work best
- Adding weather/pollution/events reduces performance
- Some dishes are more sensitive than others

See `ablation_per_dish_results.csv` for detailed breakdown.

---

## 📁 Generated Files

### Visualizations (300 DPI, publication-ready):

1. `01_ablation_study_overview.png` - Overall R² comparison, MAE, feature count analysis
2. `02_ablation_per_dish_analysis.png` - Heatmap and per-dish performance drops
3. `03_feature_group_importance.png` - Feature group ranking and cumulative impact

### Data Files:

1. `ablation_study_summary.csv` - Summary table
2. `feature_group_importance.csv` - Feature group impacts
3. `ablation_per_dish_results.csv` - Detailed per-dish results (70 rows)

---

## 🎯 Final Conclusion

**The data speaks clearly**:

- **Historical features (past orders) are sufficient** for excellent predictions (R² = 0.9545)
- **External features (weather, pollution, events) ADD NOISE, not signal**
- **Simpler is better** - Occam's Razor applies!

### Recommended Action:

**Switch to the ONLY HISTORICAL model** for:

- ✅ Better performance (+1.36% R²)
- ✅ Simpler architecture
- ✅ No external data dependencies
- ✅ Faster inference
- ✅ More robust predictions

---

_Generated: November 9, 2025_  
_Study Type: Ablation Study / Feature Importance Analysis_  
_Total Models Trained: 7_  
_Training Data: 2004 samples_  
_Test Data: 501 samples_
