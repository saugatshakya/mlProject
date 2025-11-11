# 🔬 Algorithm Comparison Results

**Generated:** November 10, 2025

This document compares the two recommendation algorithms implemented in the dish recommendation system.

---

## 🎯 Algorithms Compared

### 1. Association Rules (Apriori-based)

- **Method:** Analyzes frequent itemsets and generates rules
- **Metrics:** Support, Confidence, Lift
- **Ranking:** By Lift (then Confidence)
- **Filtering:** Min support = 0.1%, Min confidence = 10%

### 2. Co-occurrence Matrix

- **Method:** Simple count-based approach
- **Metrics:** Times ordered together, Co-occurrence rate
- **Ranking:** By count (descending)
- **Filtering:** None (all co-occurrences included)

---

## 📊 Performance Comparison (Top 20 Popular Dishes)

| Metric                  | Association Rules         | Co-occurrence         | Observation                     |
| ----------------------- | ------------------------- | --------------------- | ------------------------------- |
| **Avg Recommendations** | 1.4 per dish              | 10.0 per dish         | Co-occurrence returns more      |
| **Avg Overlap**         | -                         | -                     | 14.5% agreement                 |
| **Coverage**            | Lower (strict thresholds) | Higher (no filtering) | Trade-off: quality vs quantity  |
| **Total Rules**         | 120 high-quality rules    | ~6,678 pairs          | Association rules are selective |

### Key Finding: Low Overlap (14.5%)

**Why?**

- Association Rules are **very selective** (only 120 rules from thousands of pairs)
- Filters out weak associations using confidence threshold (10%)
- Co-occurrence includes **all** pairs regardless of strength

**Example:**

- **Association Rules:** Only recommends if confidence ≥ 10% AND support ≥ 0.1%
- **Co-occurrence:** Recommends any dish that co-occurs, even once

---

## 🔍 Algorithm Characteristics

### Association Rules ✨

**Strengths:**

- ✅ **Quality filtering** - Only strong associations (lift, confidence)
- ✅ **Interpretable metrics** - Clear meaning (e.g., "20% of customers who buy A also buy B")
- ✅ **Handles rare items well** - Support threshold prevents noise
- ✅ **Better ranking** - Lift measures how much more likely vs random

**Weaknesses:**

- ⚠️ **Fewer recommendations** - Strict thresholds mean fewer results
- ⚠️ **Slower computation** - O(n²) with rule generation
- ⚠️ **Cold start issues** - New dishes need minimum support
- ⚠️ **May miss obvious pairs** - If they don't meet thresholds

**Best For:**

- Quality over quantity
- When you want **strong** recommendations
- Production systems where accuracy matters
- Understanding **why** recommendations are made

### Co-occurrence 🎲

**Strengths:**

- ✅ **More recommendations** - No quality filtering
- ✅ **Faster computation** - Simple O(n) counting
- ✅ **No cold start** - Works with any co-occurrence
- ✅ **Very intuitive** - "These were ordered together X times"

**Weaknesses:**

- ❌ **No quality metrics** - Can't distinguish strong vs weak
- ❌ **Popularity bias** - Popular items dominate
- ❌ **Noise included** - Random co-occurrences treated equally
- ❌ **Poor ranking** - Just by count, not relationship strength

**Best For:**

- Quick/simple recommendations
- When you want **more** options
- Exploratory analysis
- Baseline/fallback method

---

## 📈 Detailed Findings

### Dishes with Largest Algorithm Differences

| Dish                             | Assoc Rules | Co-occur | Difference | Why?                            |
| -------------------------------- | ----------- | -------- | ---------- | ------------------------------- |
| Bone in Jamaican Grilled Chicken | 0           | 10       | 10         | No associations meet thresholds |
| All About Chicken Pizza          | 1           | 10       | 9          | Only 1 strong association found |
| Margherita Pizza                 | 1           | 10       | 9          | Only 1 strong association found |
| Jamaican Chicken Melt            | 1           | 10       | 9          | Only 1 strong association found |
| Tripple Cheese Pizza             | 1           | 10       | 9          | Only 1 strong association found |

**Insight:** Popular dishes often co-occur with many items, but most pairs are **weak** (random co-occurrence). Association rules filter these out, co-occurrence includes everything.

---

## 🎯 When to Use Which Algorithm?

### Use Association Rules When:

1. ✅ **Quality matters** - You want high-confidence recommendations
2. ✅ **Explainability needed** - You need to justify recommendations with metrics
3. ✅ **Limited UI space** - Can only show top 3-5 recommendations
4. ✅ **Production deployment** - Want reliable, tested rules

### Use Co-occurrence When:

1. ✅ **Exploration** - Analyzing all possible dish combinations
2. ✅ **Backup/fallback** - When association rules return nothing
3. ✅ **Speed critical** - Need fastest possible lookup
4. ✅ **Many recommendations** - Want to show 10+ options

### Hybrid Approach (Recommended! 🌟)

```python
# Try association rules first (quality)
recs = recommend_by_association(dish, top_n=5)

# If not enough, supplement with co-occurrence
if len(recs) < 5:
    cooccur_recs = recommend_by_cooccurrence(dish, top_n=10)
    # Add co-occurrence recs not already in association rules
    recs = combine_unique(recs, cooccur_recs, max_total=5)
```

---

## 📊 Generated Visualizations

### 1. `06_algorithm_comparison_examples.png`

Side-by-side comparison for 4 example dishes:

- Bageecha Pizza
- Chilli Cheese Garlic Bread
- Peri Peri Fries
- Animal Fries

Shows both algorithms' recommendations with:

- Blue bars: Association Rules (confidence %)
- Orange bars: Co-occurrence (normalized count)

### 2. `07_algorithm_performance_comparison.png`

4-panel analysis:

- **Scatter plot:** Association count vs Co-occurrence count
- **Histogram:** Overlap percentage distribution
- **Scatter plot:** Dish support vs algorithm agreement
- **Summary table:** Key metrics comparison

### 3. `08_algorithm_pros_cons.png`

Comprehensive comparison table covering:

- Algorithm characteristics
- Recommendation quality
- Practical usage considerations
- Results on top 20 dishes

---

## 💡 Recommendations

### For Current Implementation (App V2)

**Status:** Currently uses Association Rules only ✅

**Suggestions:**

1. ✅ **Keep association rules as primary** - Quality is good
2. ⚠️ **Add co-occurrence as fallback** - When association rules return < 3 recommendations
3. ⚠️ **Add "method" indicator** - Show users which algorithm was used
4. ⚠️ **Consider hybrid scoring** - Combine both metrics for ranking

### Example Hybrid Implementation:

```python
def get_recommendations(dish, top_n=5):
    # Try association rules first
    assoc = recommend_by_association(dish, top_n=top_n)

    if len(assoc) >= 3:
        # Enough high-quality recommendations
        return assoc
    else:
        # Supplement with co-occurrence
        cooccur = recommend_by_cooccurrence(dish, top_n=top_n)

        # Combine (prioritize association rules)
        combined = []
        combined.extend(assoc)

        # Add co-occurrence recs not in association
        assoc_dishes = set(assoc['dish'])
        for _, row in cooccur.iterrows():
            if row['dish'] not in assoc_dishes and len(combined) < top_n:
                combined.append(row)

        return combined
```

---

## 📁 Files Generated

### In `dish_recommend/docs/figures/`:

- `06_algorithm_comparison_examples.png` - Side-by-side examples
- `07_algorithm_performance_comparison.png` - Performance metrics
- `08_algorithm_pros_cons.png` - Detailed comparison table

### Scripts:

- `algorithm_comparison.py` - Full comparison analysis

---

## 🎓 Key Takeaways

1. **Different Use Cases:** Association rules for quality, co-occurrence for coverage
2. **Low Overlap (14.5%):** Algorithms prioritize differently - not a problem!
3. **Association Rules Selective:** 120 rules from 6,678 pairs (1.8%) - very picky
4. **Hybrid is Best:** Combine both for optimal results
5. **Current Implementation Good:** Using association rules provides quality recommendations

---

## 📊 Summary Statistics

| Aspect                          | Value                  |
| ------------------------------- | ---------------------- |
| **Association Rules Generated** | 120 rules              |
| **Co-occurrence Pairs**         | 6,678 pairs            |
| **Selectivity**                 | 1.8% (rules/pairs)     |
| **Avg Overlap**                 | 14.5%                  |
| **Avg Assoc Recs**              | 1.4 per dish (top 20)  |
| **Avg Cooccur Recs**            | 10.0 per dish (top 20) |

**Conclusion:** Both algorithms serve different purposes. Association rules provide **quality**, co-occurrence provides **coverage**. Current implementation using association rules is appropriate for production use. Consider adding co-occurrence as a fallback for dishes with few strong associations.

---

**Analysis completed:** November 10, 2025  
**Script:** `algorithm_comparison.py`  
**Algorithms:** Association Rules (Apriori) vs Co-occurrence Matrix
