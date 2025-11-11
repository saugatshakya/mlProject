# 🍕 Dish Recommendation System

**Intelligent dish recommendations based on order history**

When a customer selects a dish, this system recommends other dishes that are frequently ordered together, helping restaurants increase order value and improve customer experience.

---

## 📚 Documentation

- **[ANALYSIS_RESULTS.md](./ANALYSIS_RESULTS.md)** - Complete analysis with visualizations and implementation comparison
- **[ALGORITHM_COMPARISON.md](./ALGORITHM_COMPARISON.md)** - Detailed comparison of Association Rules vs Co-occurrence algorithms

---

## 🎯 Overview

This project implements a **dish recommendation engine** using **association rules** and **co-occurrence analysis**. By analyzing historical order data, the system identifies which dishes are commonly ordered together and provides intelligent recommendations.

### Key Features

- ✅ **Association Rules Mining** - Discovers dish combinations using Apriori-like algorithm
- ✅ **Co-occurrence Analysis** - Counts how often dishes appear together
- ✅ **Confidence & Lift Metrics** - Measures recommendation strength
- ✅ **Simple Inference API** - Easy-to-use production interface
- ✅ **Batch Processing** - Get recommendations for multiple dishes at once
- ✅ **Dish Search** - Find dishes by name/keyword

---

## 📊 Results Summary

### Dataset Statistics

- **Total Orders**: 21,131 delivered orders
- **Unique Dishes**: 243 dishes (142 after filtering rare items)
- **Avg Items/Order**: 1.79 dishes
- **Multi-item Orders**: 11,607 (55%)

### Model Performance

- **Association Rules Generated**: 120 high-quality rules
- **Min Support**: 0.1% (appears in at least 21 orders)
- **Min Confidence**: 10% (recommended in 10% of cases when antecedent is present)
- **Top Lift Score**: 19.0x (Tipsy Tiger Ginger Ale → Fresh Lime Soda)

### Top Recommendations Examples

**For "Bageecha Pizza":**

1. Chilli Cheese Garlic Bread (478 co-orders, 27.7% confidence)
2. Makhani Paneer Pizza (368 co-orders, 11.9% confidence)
3. All About Chicken Pizza (222 co-orders)

**For "Bone in Jamaican Grilled Chicken":**

1. Bone in Peri Peri Grilled Chicken (160 co-orders)
2. Bone in Smoky BBQ Grilled Chicken (155 co-orders)
3. Animal Fries (98 co-orders)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data

```bash
python src/data/preprocessing.py
```

This will:

- Load order data from `../data/data.csv`
- Parse "Items in order" column
- Create transaction baskets
- Filter rare dishes (< 10 orders)
- Save processed transactions

### 3. Train Recommendation Engine

```bash
python src/models/recommender.py
```

This will:

- Build co-occurrence matrix
- Generate association rules
- Calculate support, confidence, and lift metrics
- Save model files to `models/`

### 4. Use Inference API

```bash
python inference.py
```

Or in your code:

```python
from inference import DishRecommendationAPI

# Initialize and load model
api = DishRecommendationAPI()
api.load_model('models/')

# Get recommendations
recs = api.recommend('bageecha pizza', top_n=5)
print(recs)

# Search for dishes
matches = api.search_dishes('chicken')
print(matches)

# Get popular dishes
popular = api.get_popular_dishes(top_n=20)
print(popular)
```

---

## 📁 Project Structure

```
dish_recommend/
│
├── src/
│   ├── data/
│   │   └── preprocessing.py      # Data loading and transaction creation
│   └── models/
│       └── recommender.py        # Recommendation engine (association rules)
│
├── data/
│   ├── raw/                      # (empty - uses ../data/data.csv)
│   └── processed/
│       └── transactions.csv      # Processed transaction baskets
│
├── models/                       # Trained model files
│   ├── association_rules.csv    # 120 rules with confidence/lift
│   ├── dish_support.csv         # Popularity of each dish
│   └── cooccurrence_matrix.csv  # Dish co-occurrence counts
│
├── inference.py                  # Simple inference API
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🔬 How It Works

### 1. Data Preprocessing

**Input**: Order data with "Items in order" column

```
"1 x Grilled Chicken Jamaican Tender, 1 x Grilled Chicken Peri Peri Tangdi"
```

**Processing**:

1. Parse items using regex: `\d+ x (.+)`
2. Normalize dish names (lowercase, trim whitespace)
3. Create transaction baskets: `[['dish1', 'dish2'], ['dish3', 'dish4'], ...]`
4. Filter rare dishes (min 10 occurrences)

**Output**: Transaction baskets ready for association rule mining

### 2. Association Rules Mining

For each dish pair (A, B), calculate:

**Support**: How often A and B appear together

```
support(A → B) = count(A ∩ B) / total_transactions
```

**Confidence**: When A is ordered, how often is B also ordered?

```
confidence(A → B) = count(A ∩ B) / count(A)
```

**Lift**: How much more likely is B when A is present vs. baseline?

```
lift(A → B) = confidence(A → B) / support(B)
```

**Lift Interpretation**:

- Lift > 1: Positive association (B is more likely when A is present)
- Lift = 1: No association (independent)
- Lift < 1: Negative association (B is less likely when A is present)

### 3. Recommendations

Given a dish, we:

1. Find all association rules where this dish is the antecedent
2. Sort by lift (strength of association)
3. Return top N recommendations with confidence scores

**Fallback**: If no association rules exist, use simple co-occurrence counts.

---

## 📊 Key Metrics Explained

### Support

- **Definition**: Frequency of itemset in transactions
- **Range**: 0 to 1 (0% to 100%)
- **Example**: Support = 0.02 means the pair appears in 2% of orders
- **Use**: Filters out rare combinations

### Confidence

- **Definition**: Probability of consequent given antecedent
- **Range**: 0 to 1
- **Example**: Confidence = 0.30 means 30% of customers who ordered A also ordered B
- **Use**: Measures recommendation reliability

### Lift

- **Definition**: Strength of association vs. baseline
- **Range**: 0 to ∞ (typically 0.5 to 20)
- **Example**: Lift = 5.0 means B is 5x more likely when A is ordered
- **Use**: Identifies strongest associations

---

## 💡 Usage Examples

### Example 1: Simple Recommendations

```python
from inference import DishRecommendationAPI

api = DishRecommendationAPI()
api.load_model('models/')

# Get top 5 recommendations
recs = api.recommend('bageecha pizza', top_n=5)
print(recs)
```

Output:

```
                         dish  times_ordered_together
1  chilli cheese garlic bread                     478
2        makhani paneer pizza                     368
3     all about chicken pizza                     222
```

### Example 2: Search Dishes

```python
# Find dishes containing "chicken"
matches = api.search_dishes('chicken')
print(matches[:5])
```

Output:

```
['all about chicken pizza',
 'bone in jamaican grilled chicken',
 'jamaican chicken melt',
 'bone in smoky bbq grilled chicken',
 'bone in peri peri grilled chicken']
```

### Example 3: Batch Recommendations

```python
# Get recommendations for multiple dishes at once
dishes = ['bageecha pizza', 'animal fries']
results = api.batch_recommend(dishes, top_n=3)

for dish, recs in results.items():
    print(f"\n{dish}:")
    print(recs)
```

---

## 🎯 API Reference

### `DishRecommendationAPI`

#### Methods

**`load_model(model_dir)`**

- Load trained model from directory
- Args: `model_dir` (str) - Path to model directory
- Returns: None

**`recommend(dish_name, top_n=5, method='both')`**

- Get recommendations for a dish
- Args:
  - `dish_name` (str) - Dish name (will be normalized)
  - `top_n` (int) - Number of recommendations
  - `method` (str) - 'association', 'cooccurrence', or 'both'
- Returns: DataFrame with recommendations

**`get_popular_dishes(top_n=20)`**

- Get most popular dishes
- Args: `top_n` (int) - Number of dishes
- Returns: DataFrame with dish names and popularity scores

**`search_dishes(query)`**

- Search for dishes matching a query
- Args: `query` (str) - Search term
- Returns: List of matching dish names (sorted by popularity)

**`batch_recommend(dishes, top_n=5)`**

- Get recommendations for multiple dishes
- Args:
  - `dishes` (List[str]) - List of dish names
  - `top_n` (int) - Recommendations per dish
- Returns: Dict mapping dish names to recommendation DataFrames

---

## 📈 Model Files

### `association_rules.csv`

Contains 120 rules with columns:

- `antecedent` - The dish that triggers recommendation
- `consequent` - The recommended dish
- `support` - How often they appear together (0-1)
- `confidence` - P(consequent | antecedent) (0-1)
- `lift` - Strength of association (0-∞)
- `count` - Number of times they appeared together

### `dish_support.csv`

Contains popularity for each dish:

- `dish` - Dish name
- `support` - Fraction of orders containing this dish (0-1)

### `cooccurrence_matrix.csv`

Contains pairwise co-occurrence counts:

- `dish1`, `dish2` - Pair of dishes
- `count` - Number of orders containing both

---

## 🔍 Interesting Findings

### Top Association Rules (by Lift)

1. **Tipsy Tiger Ginger Ale → Fresh Lime Soda** (Lift: 19.0)

   - When customers order ginger ale, they're 19x more likely to also order lime soda
   - Beverage pairing strategy!

2. **Fried Chicken Kabuli Tender → Angara Tender** (Lift: 12.4)

   - Customers trying different chicken flavors together
   - Cross-selling opportunity

3. **Just Pepperoni Pide ↔ Mutton Seekh Pide** (Lift: 10.7)
   - Bidirectional strong association
   - Customers order multiple pides

### Most Popular Dishes

1. **Bageecha Pizza** - 14.7% of all orders
2. **Chilli Cheese Garlic Bread** - 8.2%
3. **All About Chicken Pizza** - 7.7%
4. **Bone in Jamaican Grilled Chicken** - 7.7%
5. **Makhani Paneer Pizza** - 6.9%

### Strongest Co-occurrences

1. **Bageecha Pizza + Chilli Cheese Garlic Bread** - 478 orders
2. **Bageecha Pizza + Makhani Paneer Pizza** - 368 orders
3. **Bone in Jamaican + Peri Peri Chicken** - 160 orders

---

## 🚧 Future Improvements

- [ ] **Personalized Recommendations** - Consider user history
- [ ] **Time-based Patterns** - Lunch vs. dinner recommendations
- [ ] **Seasonal Trends** - Adjust for seasonal preferences
- [ ] **Collaborative Filtering** - User-based recommendations
- [ ] **Deep Learning** - Neural collaborative filtering
- [ ] **A/B Testing Framework** - Measure recommendation impact
- [ ] **Real-time Updates** - Incremental model updates
- [ ] **UI Integration** - Web interface for recommendations

---

## 📚 Technical Details

### Algorithms Used

1. **Association Rules Mining** (Apriori-like)

   - Identifies frequent itemsets
   - Generates rules with support/confidence thresholds
   - Calculates lift to measure association strength

2. **Co-occurrence Matrix**
   - Simple count-based approach
   - Faster than association rules
   - Good for fallback recommendations

### Performance

- **Training Time**: ~2 seconds (21K orders, 142 dishes)
- **Inference Time**: <1ms per recommendation
- **Memory Usage**: ~5MB (model files)
- **Scalability**: O(n²) for n dishes (acceptable for <1000 dishes)

---

## 🙋 FAQ

**Q: How are dish names normalized?**
A: Converted to lowercase, extra whitespace removed, special characters stripped.

**Q: What if a dish has no recommendations?**
A: Returns empty DataFrame. Consider showing popular dishes as fallback.

**Q: Can I adjust the confidence/lift thresholds?**
A: Yes! Edit `src/models/recommender.py` and retrain with `min_support` and `min_confidence` parameters.

**Q: How often should I retrain the model?**
A: Weekly or monthly, depending on menu changes and new order volume.

**Q: Does it work for new dishes?**
A: No - dishes need at least 10 orders to appear in recommendations. Use popularity-based fallback for new items.

---

## 📝 License

This project is part of ML2025 coursework.

---

## 👨‍💻 Author

**Saugat Shakya**  
Date: November 9, 2025

---

## 🎉 Summary

This dish recommendation system successfully:

- ✅ Processes 21K+ orders into transaction baskets
- ✅ Generates 120 high-quality association rules
- ✅ Provides instant recommendations with confidence scores
- ✅ Offers simple, production-ready inference API
- ✅ Identifies strong dish associations (lift up to 19x)

**Ready for production deployment!** 🚀

Use `inference.py` to integrate recommendations into your restaurant ordering system.
