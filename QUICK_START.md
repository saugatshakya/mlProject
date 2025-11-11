# 🚀 Quick Start Guide

## Delivery Time Prediction

```bash
cd delivery_time_prediction
pip install -r requirements.txt
# Place data.csv in data/raw/
python run_training.py
python inference.py --model models/baseline/xgboost_model.pkl --data new_data.csv
```

**Key Outputs**:

- `models/baseline/model_comparison.csv` - Model performance
- `models/baseline/ablation_study.csv` - Feature importance
- `reports/figures/` - Visualizations

## Promotion Effectiveness

```bash
cd promotion_effectiveness
pip install -r requirements.txt
# Place data_4.csv in data/raw/
python run_training.py
```

**Key Outputs**:

- `reports/promotion_effectiveness_report.txt` ⭐ **MAIN DELIVERABLE**
- `models/baseline/model_comparison.csv` - Model performance
- `reports/figures/shap_summary.png` - Feature importance

## What Makes These Special

### Delivery Time Prediction

- **Temporal Features**: Lag (1-24 periods) + Rolling windows
- **Restaurant Patterns**: Per-restaurant historical averages
- **External Data**: Pollution, weather integration
- **Ablation Study**: Proves which features actually help

### Promotion Effectiveness

- **SHAP Analysis**: Exact Rupee impact of each promotion
- **Business Insights**: Which promotions increase revenue
- **Interpretability**: Understand WHY model makes predictions
- **Marketing ROI**: Optimize promotion budget

## Files Created

**Both Projects**:

- ✅ Complete src/ package (data, features, models, analysis)
- ✅ Training scripts with logging
- ✅ Comprehensive README.md
- ✅ PROJECT_STATUS.md
- ✅ requirements.txt

**Total**: ~30 files, ~3000 lines of production-ready code

Ready to run! 🎉
