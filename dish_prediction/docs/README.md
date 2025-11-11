# Documentation Index

Welcome to the Dish Order Prediction project documentation!

---

## � **START HERE** → [00_START_HERE.md](00_START_HERE.md)

**New to this project?** Read the quick start guide above for:

- Recommended reading order for your role
- Key findings summary
- Documentation structure overview
- FAQ and next steps

---

## �📚 Documentation Overview

This folder contains comprehensive documentation for the dish order prediction system, including model analysis, ablation studies, and inference guides.

### Quick Navigation

| Document                                                       | Description                      | Key Topics                                   |
| -------------------------------------------------------------- | -------------------------------- | -------------------------------------------- |
| **[00_START_HERE.md](00_START_HERE.md)**                       | ⭐ Quick start guide             | Reading order, key findings, FAQ             |
| **[01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md)**           | Project summary and quick start  | Performance metrics, dataset info, results   |
| **[02_MODEL_IMPACT_ANALYSIS.md](02_MODEL_IMPACT_ANALYSIS.md)** | What the model learned           | Weather, pollution, events impact            |
| **[03_ABLATION_STUDY.md](03_ABLATION_STUDY.md)**               | ⚠️ Scientific feature importance | Feature removal experiments (MOST IMPORTANT) |
| **[04_INFERENCE_GUIDE.md](04_INFERENCE_GUIDE.md)**             | How to make predictions          | Features required, code examples             |

---

## 🎯 Reading Guides by Role

### For Project Reviewers (30 min total)

1. **[00_START_HERE.md](00_START_HERE.md)** (5 min) - Overview and key findings
2. **[03_ABLATION_STUDY.md](03_ABLATION_STUDY.md)** (10 min) - The shocking discovery
3. **[01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md)** (5 min) - Model comparison
4. **[02_MODEL_IMPACT_ANALYSIS.md](02_MODEL_IMPACT_ANALYSIS.md)** (10 min) - Detailed analysis

### For Production Teams (20 min total)

1. **[00_START_HERE.md](00_START_HERE.md)** (5 min) - Quick start and recommendations
2. **[04_INFERENCE_GUIDE.md](04_INFERENCE_GUIDE.md)** (15 min) - Production deployment

### For Data Scientists (60 min deep dive)

👉 **Read all documents in order** for complete technical understanding

---

## 🔬 The Key Discovery

**External features (weather, pollution, events) HURT model performance!**

| Model               | R²         | Change     | Recommendation   |
| ------------------- | ---------- | ---------- | ---------------- |
| FULL MODEL          | 0.9417     | Baseline   | ❌ Don't use     |
| **ONLY HISTORICAL** | **0.9545** | **+1.36%** | ✅ **Use this!** |

See [03_ABLATION_STUDY.md](03_ABLATION_STUDY.md) for complete analysis.

---

## 🎨 Visual Analysis

Every figure includes:

- ✅ Full image display (embedded in markdown)
- ✅ Panel-by-panel breakdown (for multi-panel charts)
- ✅ Detailed interpretation (4-6 paragraphs per figure)
- ✅ Critical observations (key takeaways highlighted)
- ✅ Connections to other findings

Example: The ablation study overview has 14 paragraphs explaining all 4 panels!

---

## 📊 Need Quick Answers?

### Want to understand what features matter?

👉 **Read**: [04_INFERENCE_GUIDE.md](04_INFERENCE_GUIDE.md)

### Curious about model behavior?

👉 **Read**: [02_MODEL_IMPACT_ANALYSIS.md](02_MODEL_IMPACT_ANALYSIS.md)

---

## 🔑 Key Findings Summary

### 🏆 Best Model Performance

- **Algorithm**: CatBoost Multi-Output Regressor
- **Test R²**: 0.9494 (94.94% variance explained)
- **Mean Absolute Error**: 0.657 orders
- **Best Dish**: Tripple Cheese Pizza (R² = 0.9913)

### 🚨 CRITICAL DISCOVERY (Ablation Study)

**Weather, pollution, and events are HURTING performance!**

| Configuration   | Features | Test R²    | vs Full Model |
| --------------- | -------- | ---------- | ------------- |
| FULL MODEL      | 57       | 0.9417     | Baseline      |
| ONLY HISTORICAL | 40       | **0.9545** | **+1.36% ✅** |

**Recommendation**: Use **ONLY historical features** (past orders) for best performance!

---

## 📊 Visualizations

All analysis figures are located in `figures/` subdirectories:

### Model Impact Analysis

- `figures/model_impact/01_model_weather_impact.png`
- `figures/model_impact/02_model_pollution_impact.png`
- `figures/model_impact/03_model_event_holiday_impact.png`
- `figures/model_impact/04_model_feature_importance.png`

### Ablation Study

- `figures/ablation_study/01_ablation_study_overview.png` ⭐
- `figures/ablation_study/02_ablation_per_dish_analysis.png`
- `figures/ablation_study/03_feature_group_importance.png`

### Comprehensive Analysis

- `figures/comprehensive/01_model_comparison.png`
- `figures/comprehensive/02_weather_impact.png`
- `figures/comprehensive/03_pollution_impact.png`
- And more...

---

## 📖 Document Details

### 1. Project Overview

**File**: [01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md)

**Contents**:

- Executive summary
- Performance metrics
- Dataset information
- Quick start guide
- Model comparison (CatBoost vs XGBoost)

**Read this if**: You're new to the project or need a high-level overview.

---

### 2. Model Impact Analysis

**File**: [02_MODEL_IMPACT_ANALYSIS.md](02_MODEL_IMPACT_ANALYSIS.md)

**Contents**:

- What the model learned about weather impact
- What the model learned about pollution impact
- What the model learned about events/holidays
- Feature importance from the model's perspective
- Controlled experiment methodology

**Read this if**: You want to understand how the trained model responds to feature changes.

**Key Insight**: Shows the model's learned relationships, not necessarily what's true!

---

### 3. Ablation Study ⭐ MOST IMPORTANT

**File**: [03_ABLATION_STUDY.md](03_ABLATION_STUDY.md)

**Contents**:

- Systematic feature removal experiments
- 7 different model configurations tested
- Performance comparison across all configurations
- Scientific analysis of which features help vs hurt
- Detailed recommendations

**Read this if**: You want to know which features **actually matter** for performance.

**Key Finding**:

- Removing weather, pollution, and events **IMPROVES** performance!
- Historical features alone achieve the best R² (0.9545)

**This is the most important document** - it proves which features to use!

---

### 4. Inference Guide

**File**: [04_INFERENCE_GUIDE.md](04_INFERENCE_GUIDE.md)

**Contents**:

- Required features for prediction (52 total)
- Data sources (your existing CSV files)
- Code examples for inference
- Simplified model recommendation
- Troubleshooting guide
- Best practices

**Read this if**: You need to make predictions with the model.

**Quick Start**:

```bash
python inference_simple.py
```

---

## 🔬 Analysis Types Explained

### Model Impact Analysis vs Ablation Study

| Analysis Type      | Question                  | Method                               | Usefulness               |
| ------------------ | ------------------------- | ------------------------------------ | ------------------------ |
| **Model Impact**   | What did the model learn? | Vary features, observe predictions   | Shows model behavior     |
| **Ablation Study** | Which features help?      | Remove features, measure performance | Shows feature importance |

**Key Difference**:

- Model Impact: "The model thinks weather matters"
- Ablation Study: "But removing weather improves performance!"

**Trust the Ablation Study** - it's based on actual performance metrics, not model internals.

---

## 📁 Directory Structure

```
docs/
├── README.md                          ← You are here
├── 01_PROJECT_OVERVIEW.md             ← Start here for overview
├── 02_MODEL_IMPACT_ANALYSIS.md        ← Model's learned behavior
├── 03_ABLATION_STUDY.md              ← ⭐ Feature importance (CRITICAL)
├── 04_INFERENCE_GUIDE.md             ← How to make predictions
└── figures/                           ← All analysis visualizations
    ├── model_impact/
    │   ├── 01_model_weather_impact.png
    │   ├── 02_model_pollution_impact.png
    │   ├── 03_model_event_holiday_impact.png
    │   └── 04_model_feature_importance.png
    ├── ablation_study/
    │   ├── 01_ablation_study_overview.png
    │   ├── 02_ablation_per_dish_analysis.png
    │   ├── 03_feature_group_importance.png
    │   ├── ablation_study_summary.csv
    │   ├── feature_group_importance.csv
    │   └── ablation_per_dish_results.csv
    └── comprehensive/
        ├── 01_model_comparison.png
        ├── 02_weather_impact.png
        ├── 03_pollution_impact.png
        ├── 04_temporal_patterns.png
        ├── 05_events_holidays.png
        ├── 06_dish_popularity.png
        └── 07_correlation_matrix.png
```

---

## 🎓 Recommended Reading Order

### For Data Scientists / ML Engineers

1. **[03_ABLATION_STUDY.md](03_ABLATION_STUDY.md)** ⭐ (Start here!)

   - Understand which features matter
   - See the shocking discovery about external features

2. **[01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md)**

   - Get context on the problem
   - See performance metrics

3. **[02_MODEL_IMPACT_ANALYSIS.md](02_MODEL_IMPACT_ANALYSIS.md)**

   - Understand model internals
   - See what the model learned (even if it's wrong!)

4. **[04_INFERENCE_GUIDE.md](04_INFERENCE_GUIDE.md)**
   - Learn how to deploy
   - Get practical code examples

---

### For Business Stakeholders

1. **[01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md)**

   - High-level summary
   - Key metrics and performance

2. **[03_ABLATION_STUDY.md](03_ABLATION_STUDY.md)** (Focus on Executive Summary)

   - Understand the main finding
   - See why simpler is better

3. **[04_INFERENCE_GUIDE.md](04_INFERENCE_GUIDE.md)** (Focus on Quick Start)
   - How to use the system
   - What data is needed

---

### For Developers/Engineers

1. **[04_INFERENCE_GUIDE.md](04_INFERENCE_GUIDE.md)**

   - Integration guide
   - Code examples
   - API requirements

2. **[03_ABLATION_STUDY.md](03_ABLATION_STUDY.md)** (Focus on Recommendations)

   - Which model to deploy
   - Infrastructure requirements

3. **[01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md)**
   - Project structure
   - File locations

---

## 💡 Key Takeaways

### Top 3 Most Important Findings

1. **Historical features alone are best**

   - R² = 0.9545 with only lag features
   - Beats all other configurations

2. **External features hurt performance**

   - Weather: -0.35% performance
   - Pollution: -0.45% performance
   - Events: -0.15% performance

3. **Simpler is better**
   - 40 features > 57 features
   - No external data needed
   - Faster, more robust predictions

### Recommended Production Model

✅ **ONLY HISTORICAL MODEL**

- 40 features (lag1, lag2, lag3, smooth for each dish)
- R² = 0.9545
- MAE = 0.0579
- No external APIs needed
- Simple and fast

---

## 🔗 External Resources

### Project Files

- **Training Script**: `../src/models/final_model.py`
- **Inference Script**: `../inference_simple.py`
- **Data Files**: `../data/`
- **Saved Models**: `../models/final/`

### Data Sources

- Order history: `../data/processed/hourly_data_with_features.csv`
- Weather data: `../data/hourly_orders_weather.csv`
- Pollution data: `../data/pollution.csv`
- Events: `../data/events.csv`

---

## ❓ FAQ

**Q: Which document should I read first?**  
A: For technical audience: [03_ABLATION_STUDY.md](03_ABLATION_STUDY.md). For general overview: [01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md)

**Q: Do I need weather and pollution data?**  
A: **No!** The ablation study proved they hurt performance. Use only historical features.

**Q: What's the best R² score achieved?**  
A: **0.9545** using only historical features (40 features)

**Q: How many models were tested?**  
A: 7 different configurations in the ablation study, plus CatBoost vs XGBoost comparison.

**Q: Can I use this in production?**  
A: Yes! See [04_INFERENCE_GUIDE.md](04_INFERENCE_GUIDE.md) for deployment guide.

**Q: What if I don't have historical data?**  
A: The model requires at least 3 hours of past orders. For cold start, use average orders from training data.

---

## 📞 Support

For questions or issues:

1. Check the relevant documentation above
2. Review code comments in source files
3. Check ablation study results for feature selection questions

---

_Documentation Version: 1.0_  
_Last Updated: November 9, 2025_  
_Project: Dish Order Prediction_
