# Feature Selection Strategy

## Data Science Best Practices for Restaurant Demand Prediction

**Author:** Data Scientist  
**Date:** November 2025  
**Context:** Avoiding overfitting, ensuring model generalization

---

## ⚠️ THE PROBLEM: Too Many Features

### Current Situation:

- **Potential features:** ~888 (if we create all possible combinations)
- **Data points:** 2,555 hours × 30 dishes = 76,650 dish-hour records
- **Ratio:** ~11.6 samples per feature per dish

### Why This is BAD:

```
Rule of Thumb: Need at least 10-20 samples per feature
Current: 76,650 / 888 = 86 samples per feature (borderline)
Per dish: 2,555 / 888 = 2.9 samples per feature (TERRIBLE!)
```

**Verdict:** MUST do aggressive feature selection

---

## 🎯 FEATURE SELECTION STRATEGY (Multi-Stage)

### Stage 1: DOMAIN-DRIVEN REDUCTION (Before Modeling)

**Eliminate obviously redundant features:**

#### 1.1 Remove Highly Correlated Features

- If correlation > 0.95 → Keep only one
- Example: `temp` and `temp_squared` if r > 0.95
- Example: `rolling_mean_6h` and `rolling_mean_12h` if very similar

#### 1.2 Remove Zero/Low Variance Features

- If feature has same value 95%+ of the time → Remove
- Example: If it never rains → Remove rain-related features

#### 1.3 Remove Leakage Features

- Features that wouldn't be available at prediction time
- Example: Revenue (we're predicting orders, not revenue)

**Expected reduction:** 888 → ~400 features

---

### Stage 2: STATISTICAL FEATURE SELECTION

#### 2.1 Variance Threshold

```python
from sklearn.feature_selection import VarianceThreshold

# Remove features with very low variance
selector = VarianceThreshold(threshold=0.01)
X_reduced = selector.fit_transform(X)
```

#### 2.2 Correlation with Target

```python
# Calculate correlation with target (order_count)
correlations = X.corrwith(y).abs().sort_values(ascending=False)

# Keep top N most correlated
top_features = correlations.head(200).index
```

#### 2.3 Mutual Information

```python
from sklearn.feature_selection import mutual_info_regression

# Non-linear relationships
mi_scores = mutual_info_regression(X, y)
top_mi_features = mi_scores.argsort()[-200:]
```

**Expected reduction:** 400 → ~200 features

---

### Stage 3: MODEL-BASED FEATURE SELECTION

#### 3.1 Tree-Based Feature Importance

```python
from sklearn.ensemble import RandomForestRegressor

# Train RF to get feature importances
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Keep features with importance > threshold
importances = rf.feature_importances_
important_features = X.columns[importances > 0.001]
```

#### 3.2 L1 Regularization (Lasso)

```python
from sklearn.linear_model import LassoCV

# Lasso automatically zeros out weak features
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)

# Keep non-zero coefficients
selected_features = X.columns[lasso.coef_ != 0]
```

#### 3.3 Recursive Feature Elimination (RFE)

```python
from sklearn.feature_selection import RFE

# Iteratively remove weakest features
rfe = RFE(estimator=rf, n_features_to_select=100)
rfe.fit(X_train, y_train)

# Get selected features
rfe_features = X.columns[rfe.support_]
```

**Expected reduction:** 200 → ~100-150 features

---

### Stage 4: CROSS-VALIDATION VERIFICATION

#### 4.1 Forward/Backward Selection with CV

```python
# Start with best feature, keep adding if CV score improves
# Or start with all, keep removing if CV score doesn't drop

best_score = 0
selected = []

for feature in candidate_features:
    temp_features = selected + [feature]
    score = cross_val_score(model, X[temp_features], y, cv=5).mean()

    if score > best_score:
        best_score = score
        selected.append(feature)
```

#### 4.2 Permutation Importance

```python
from sklearn.inspection import permutation_importance

# Which features, when shuffled, hurt performance most?
perm_importance = permutation_importance(
    model, X_val, y_val, n_repeats=10, random_state=42
)

# Keep features with positive importance
important = perm_importance.importances_mean > 0
```

**Final reduction:** 100-150 → ~50-100 BEST features

---

## 📊 PROPOSED FINAL FEATURE SET (Tiered)

### TIER 1: GUARANTEED INCLUSION (~30 features)

**Temporal (10):**

- `hour` - PRIMARY FEATURE (EDA showed r=0.7+ with orders)
- `day_of_week`
- `is_weekend`
- `is_peak_hour` (19-21)
- `meal_period` (categorical)
- `hour_sin`, `hour_cos` (cyclical)
- `day_sin`, `day_cos` (cyclical)
- `week_of_year`

**Lag Features (10):**

- `orders_lag_1h` - Most recent
- `orders_lag_2h`
- `orders_lag_3h`
- `orders_lag_24h` - Same hour yesterday
- `orders_lag_168h` - Same hour last week
- `revenue_lag_24h`
- `trend_last_3h` (slope)
- `orders_lag_6h`
- `orders_lag_12h`
- `orders_same_hour_last_week_avg` (avg of last 4 weeks)

**Rolling Stats (7):**

- `orders_rolling_mean_24h`
- `orders_rolling_std_24h`
- `orders_rolling_mean_168h` (7-day avg)
- `orders_rolling_max_24h`
- `orders_rolling_min_24h`
- `orders_cv_24h` (coefficient of variation)
- `orders_trend_7d` (weekly trend)

**Weather (3):**

- `temperature` - Significant correlation
- `humidity` - Strongest correlation (+0.205)
- `is_raining` - Significant impact (-15.2%)

---

### TIER 2: LIKELY INCLUSION (~30 features)

**Temporal Extended (5):**

- `is_lunch_rush` (12-14)
- `is_dinner_rush` (19-22)
- `is_late_night` (0-4)
- `is_friday` (pre-weekend)
- `month` (seasonal)

**Lag Extended (8):**

- `orders_lag_48h`
- `orders_lag_72h`
- Dish-specific lags for TOP 5 dishes only
- Category lags (pizza, chicken, etc.)

**Rolling Extended (8):**

- `orders_rolling_mean_6h`
- `orders_rolling_mean_12h`
- `orders_rolling_std_6h`
- Dish-specific rolling for TOP 5

**Weather/Pollution (4):**

- `wind_speed`
- `aqi` - Weak but significant
- `temp_category` (cold/moderate/warm)
- `weather_condition` (categorical)

**Events (3):**

- `has_event` - +9.6% impact
- `days_to_next_event`
- `is_holiday`

**Delhi-Specific (2):**

- `is_smog_season` (Oct-Feb)
- `delhi_season` (summer/monsoon/winter)

---

### TIER 3: CONDITIONAL INCLUSION (~20-40 features)

**Include ONLY if feature importance > threshold:**

**Dish Features (per dish - selective):**

- `dish_category` (for top 10 dishes)
- `dish_popularity_rank`
- `is_vegetarian`
- `protein_type`

**Interaction Features (selective):**

- `hour_weekend_interaction` (if important)
- `temp_rain_interaction`
- `hour_weather_interaction`

**Advanced Temporal:**

- `hour_squared`
- `fourier_features` (sin/cos of multiple periods)

---

## 🔬 FEATURE SELECTION PIPELINE

### Implementation Plan:

```python
# src/features/selector.py

class FeatureSelector:
    def __init__(self, strategy='auto'):
        self.strategy = strategy
        self.selected_features = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Multi-stage feature selection
        """
        # Stage 1: Remove low variance
        X = self._remove_low_variance(X)

        # Stage 2: Remove highly correlated
        X = self._remove_correlated(X, threshold=0.95)

        # Stage 3: Select by importance
        if self.strategy == 'tree':
            selected = self._tree_importance(X, y, top_k=100)
        elif self.strategy == 'lasso':
            selected = self._lasso_selection(X, y)
        elif self.strategy == 'rfe':
            selected = self._rfe_selection(X, y, n_features=100)
        else:
            # Auto: Combine multiple methods
            selected = self._auto_select(X, y)

        # Stage 4: Validate with CV
        if X_val is not None:
            selected = self._validate_features(X[selected], y, X_val[selected], y_val)

        self.selected_features = selected
        return self

    def transform(self, X):
        return X[self.selected_features]
```

---

## 📈 EVALUATION METRICS FOR FEATURE SETS

### Compare different feature sets:

| Feature Set     | # Features | CV R² | CV MAE | CV RMSE | Training Time | Overfitting Gap |
| --------------- | ---------- | ----- | ------ | ------- | ------------- | --------------- |
| All (888)       | 888        | ?     | ?      | ?       | High          | High            |
| Tier 1 only     | 30         | ?     | ?      | ?       | Low           | Low             |
| Tier 1+2        | 60         | ?     | ?      | ?       | Medium        | Medium          |
| Selected (RFE)  | 100        | ?     | ?      | ?       | Medium        | Low             |
| Selected (Auto) | 80         | ?     | ?      | ?       | Medium        | Low             |

**Best = Highest CV score + Lowest overfitting gap**

---

## 💡 DATA SCIENTIST RECOMMENDATIONS

### 1. Start Small, Add Carefully

```
❌ Don't: Create all 888 features, then select
✅ Do: Start with 30 TIER 1, measure, add TIER 2 if helps
```

### 2. Use Multiple Selection Methods

```
✅ Tree importance (RandomForest)
✅ L1 regularization (Lasso)
✅ Permutation importance
✅ Cross-validation
→ Keep features selected by 2+ methods
```

### 3. Monitor Overfitting

```python
train_score = model.score(X_train, y_train)
val_score = model.score(X_val, y_val)

overfitting_gap = train_score - val_score

if overfitting_gap > 0.1:  # 10% gap
    # Reduce features OR increase regularization
```

### 4. Dish-Specific vs Global Features

```
Global features (30): Same for all dishes
  → hour, day, weather, events

Dish-specific features (select per dish):
  → Only for TOP 10-15 dishes
  → Others use global features only
```

### 5. Feature Engineering ≠ Feature Selection

```
Feature Engineering: Create smart features (creative)
Feature Selection: Choose which to keep (statistical)

Do BOTH!
```

---

## 🎯 FINAL RECOMMENDATION

### Conservative Approach (Recommended for Academic Project):

**Phase 1: Baseline (30 features)**

- 10 temporal + 10 lag + 7 rolling + 3 weather
- Train baseline models
- Establish performance floor

**Phase 2: Expanded (60 features)**

- Add TIER 2 features
- Measure improvement
- Keep if CV score improves by >2%

**Phase 3: Optimized (80-100 features)**

- Use RFE + Tree importance
- Select best 80-100 features
- Final model training

**Phase 4: Validation**

- Test on holdout set
- Check overfitting gap
- Document feature importance

### Aggressive Approach (If Pursuing Highest Accuracy):

**Create 888 → Select top 150 using:**

1. Random Forest importance (top 200)
2. Lasso selection (top 200)
3. Intersection (keep features in both)
4. RFE to 150
5. CV validation

---

## 📊 EXPECTED OUTCOMES

### With Proper Feature Selection:

✅ **Better Generalization** - Model works on new data
✅ **Faster Training** - 100 features vs 888 = 8x faster
✅ **Interpretability** - Can explain what drives orders
✅ **Lower Variance** - More stable predictions
✅ **Avoid Overfitting** - Train/Val scores closer

### Likely Final Feature Count:

```
Conservative: 60-80 features (SAFER for academic project)
Aggressive: 100-150 features (Higher risk, potential reward)

Recommended: 80 features (sweet spot)
```

---

## ✅ IMPLEMENTATION CHECKLIST

- [ ] Create 888 candidate features
- [ ] Remove low variance features (<0.01)
- [ ] Remove highly correlated (>0.95)
- [ ] Calculate feature importances (RF)
- [ ] Run Lasso selection
- [ ] Run RFE with top model
- [ ] Calculate permutation importance
- [ ] Create feature importance visualization
- [ ] Test multiple feature counts (30, 60, 80, 100, 150)
- [ ] Cross-validate each set
- [ ] Compare overfitting gaps
- [ ] Select final feature set
- [ ] Document selected features and why
- [ ] Re-train final models with selected features
- [ ] Validate on holdout test set

---

**Bottom Line:** Start with 30 core features (TIER 1), carefully add more based on validation performance. Target 60-100 features, NOT all 888!

**Status:** Feature selection is CRITICAL - will implement multi-stage pipeline ✅
