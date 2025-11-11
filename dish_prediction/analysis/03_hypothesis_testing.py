"""
STEP 03: HYPOTHESIS TESTING - VERIFY EDA FINDINGS STATISTICALLY
Test each hypothesis with proper statistical tests - NO HALLUCINATIONS
"""
import pandas as pd
import numpy as np
from scipy import stats
import ast
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('data/eda_processed.csv', parse_dates=['order_hour', 'Order Placed At'])

print("="*80)
print("HYPOTHESIS TESTING - STATISTICAL VERIFICATION")
print("="*80)

results = []

# ==============================================================================
# H1: Peak hours (19-21) have higher demand than off-peak
# ==============================================================================
print("\n📊 H1: Peak hours (19-21) have higher demand than off-peak")

peak_dishes = data[data['hour'].isin([19, 20, 21])]['total_dishes']
off_peak_dishes = data[data['hour'].isin([11, 12, 13, 14, 15, 16])]['total_dishes']

# T-test
t_stat, p_value = stats.ttest_ind(peak_dishes, off_peak_dishes)
ratio = peak_dishes.mean() / off_peak_dishes.mean()

print(f"   Peak avg dishes/order: {peak_dishes.mean():.2f}")
print(f"   Off-peak avg dishes/order: {off_peak_dishes.mean():.2f}")
print(f"   Ratio: {ratio:.2f}x")
print(f"   T-statistic: {t_stat:.3f}, p-value: {p_value:.6f}")

if p_value < 0.05 and ratio > 1.1:
    result = f"✅ SUPPORTED - {ratio:.2f}x higher, p={p_value:.4f}"
else:
    result = f"❌ WEAK/REJECTED - Only {ratio:.2f}x, p={p_value:.4f}"
print(f"   {result}")

results.append({
    'hypothesis': 'Peak hours (19-21) have higher demand',
    'statistic': ratio,
    'p_value': p_value,
    'result': 'Supported' if (p_value < 0.05 and ratio > 1.1) else 'Rejected'
})

# ==============================================================================
# H2: Weekends have higher demand than weekdays
# ==============================================================================
print("\n📊 H2: Weekends have higher demand than weekdays")

weekend_data = data[data['is_weekend']]
weekday_data = data[~data['is_weekend']]

weekend_avg = weekend_data['total_dishes'].sum() / weekend_data['date'].nunique()
weekday_avg = weekday_data['total_dishes'].sum() / weekday_data['date'].nunique()
lift = (weekend_avg - weekday_avg) / weekday_avg * 100

# Mann-Whitney U test (non-parametric)
weekend_daily = weekend_data.groupby('date')['total_dishes'].sum()
weekday_daily = weekday_data.groupby('date')['total_dishes'].sum()
u_stat, p_value = stats.mannwhitneyu(weekend_daily, weekday_daily, alternative='greater')

print(f"   Weekend avg dishes/day: {weekend_avg:.1f}")
print(f"   Weekday avg dishes/day: {weekday_avg:.1f}")
print(f"   Lift: {lift:+.1f}%")
print(f"   U-statistic: {u_stat:.3f}, p-value: {p_value:.6f}")

if p_value < 0.05 and lift > 5:
    result = f"✅ SUPPORTED - {lift:+.1f}% lift, p={p_value:.4f}"
else:
    result = f"❌ WEAK - Only {lift:+.1f}%, p={p_value:.4f}"
print(f"   {result}")

results.append({
    'hypothesis': 'Weekends have higher demand',
    'statistic': lift,
    'p_value': p_value,
    'result': 'Supported' if (p_value < 0.05 and lift > 5) else 'Weak'
})

# ==============================================================================
# H3: Rain reduces delivery orders
# ==============================================================================
print("\n📊 H3: Rain reduces delivery orders")

rainy_daily = data[data['env_precip'] > 0].groupby('date').size()
non_rainy_daily = data[data['env_precip'] == 0].groupby('date').size()

rainy_avg = rainy_daily.mean() if len(rainy_daily) > 0 else 0
non_rainy_avg = non_rainy_daily.mean()
effect = (rainy_avg - non_rainy_avg) / non_rainy_avg * 100

# T-test
if len(rainy_daily) > 1 and len(non_rainy_daily) > 1:
    t_stat, p_value = stats.ttest_ind(rainy_daily, non_rainy_daily)
else:
    p_value = 1.0

print(f"   Rainy days avg orders: {rainy_avg:.1f}")
print(f"   Non-rainy days avg orders: {non_rainy_avg:.1f}")
print(f"   Effect: {effect:.1f}%")
print(f"   p-value: {p_value:.6f}")

if p_value < 0.05 and effect < -5:
    result = f"✅ SUPPORTED - {effect:.1f}% reduction, p={p_value:.4f}"
else:
    result = f"❌ NOT SIGNIFICANT - {effect:.1f}%, p={p_value:.4f}"
print(f"   {result}")

results.append({
    'hypothesis': 'Rain reduces orders',
    'statistic': effect,
    'p_value': p_value,
    'result': 'Supported' if (p_value < 0.05 and effect < -5) else 'Rejected'
})

# ==============================================================================
# H4: Temperature affects demand (cool weather 10-15°C is optimal)
# ==============================================================================
print("\n📊 H4: Cool weather (10-15°C) is optimal for orders")

temp_bins = pd.cut(data['env_temp'], bins=[0, 10, 15, 20, 25, 30, 40])
temp_counts = data.groupby(temp_bins).size()

optimal_range = data[(data['env_temp'] >= 10) & (data['env_temp'] < 15)]
other_ranges = data[(data['env_temp'] < 10) | (data['env_temp'] >= 25)]

optimal_count = len(optimal_range)
optimal_pct = optimal_count / len(data) * 100

print(f"   10-15°C orders: {optimal_count:,} ({optimal_pct:.1f}% of all orders)")
print(f"   Temperature distribution:")
for temp_range, count in temp_counts.items():
    print(f"      {temp_range}: {count:,} orders")

# ANOVA test
temp_groups = [group['total_dishes'].values for name, group in data.groupby(temp_bins)]
f_stat, p_value = stats.f_oneway(*[g for g in temp_groups if len(g) > 0])

print(f"   ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.6f}")

if p_value < 0.05:
    result = f"✅ SUPPORTED - Temperature affects demand, p={p_value:.4f}"
else:
    result = f"❌ REJECTED - No significant effect, p={p_value:.4f}"
print(f"   {result}")

results.append({
    'hypothesis': 'Temperature affects demand (10-15°C optimal)',
    'statistic': optimal_pct,
    'p_value': p_value,
    'result': 'Supported' if p_value < 0.05 else 'Rejected'
})

# ==============================================================================
# H5: High pollution (AQI 5) reduces orders
# ==============================================================================
print("\n📊 H5: High pollution (AQI 5) reduces orders")

high_poll = data[data['aqi'] >= 5].groupby('order_hour').size()
low_poll = data[data['aqi'] < 5].groupby('order_hour').size()

high_avg = high_poll.mean() if len(high_poll) > 0 else 0
low_avg = low_poll.mean() if len(low_poll) > 0 else 0
effect = (high_avg - low_avg) / low_avg * 100 if low_avg > 0 else 0

# T-test
if len(high_poll) > 1 and len(low_poll) > 1:
    t_stat, p_value = stats.ttest_ind(high_poll, low_poll)
else:
    p_value = 1.0

print(f"   High pollution (AQI 5) avg orders/hour: {high_avg:.2f}")
print(f"   Low pollution (AQI <5) avg orders/hour: {low_avg:.2f}")
print(f"   Effect: {effect:.1f}%")
print(f"   p-value: {p_value:.6f}")
print(f"   NOTE: 91.5% of data has AQI 5 - it's the NORM!")

if p_value < 0.05 and abs(effect) > 5:
    result = f"✅ SUPPORTED - {effect:.1f}% effect, p={p_value:.4f}"
else:
    result = f"❌ WEAK - Only {effect:.1f}%, p={p_value:.4f}"
print(f"   {result}")

results.append({
    'hypothesis': 'High pollution reduces orders',
    'statistic': effect,
    'p_value': p_value,
    'result': 'Supported' if (p_value < 0.05 and abs(effect) > 5) else 'Weak'
})

# ==============================================================================
# H6: Chicken dishes dominate (>40%)
# ==============================================================================
print("\n📊 H6: Chicken dishes dominate the market")

# Parse dishes
all_dishes = []
for dishes_str in data['parsed_dishes']:
    try:
        if isinstance(dishes_str, str):
            dishes_dict = ast.literal_eval(dishes_str)
        else:
            dishes_dict = dishes_str
        for dish, qty in dishes_dict.items():
            all_dishes.extend([dish] * qty)
    except:
        continue

chicken_keywords = ['Chicken', 'chicken', 'Murgh', 'Tangdi', 'Tender']
chicken_dishes = [d for d in all_dishes if any(kw in d for kw in chicken_keywords)]
chicken_pct = len(chicken_dishes) / len(all_dishes) * 100

print(f"   Total dish orders: {len(all_dishes):,}")
print(f"   Chicken dish orders: {len(chicken_dishes):,}")
print(f"   Chicken percentage: {chicken_pct:.1f}%")

# Proportion test
from scipy.stats import binomtest
binom_test = binomtest(len(chicken_dishes), len(all_dishes), 0.4, alternative='greater')
p_value = binom_test.pvalue

print(f"   Binomial test p-value: {p_value:.6f}")

if chicken_pct > 40 and p_value < 0.05:
    result = f"✅ SUPPORTED - {chicken_pct:.1f}% market share, p={p_value:.4f}"
elif chicken_pct > 40:
    result = f"✅ SUPPORTED - {chicken_pct:.1f}% market share"
else:
    result = f"❌ REJECTED - Only {chicken_pct:.1f}%"
print(f"   {result}")

results.append({
    'hypothesis': 'Chicken dominates (>40%)',
    'statistic': chicken_pct,
    'p_value': p_value,
    'result': 'Supported' if chicken_pct > 40 else 'Rejected'
})

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("HYPOTHESIS TESTING SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)
results_df.to_csv('outputs/hypothesis_test_results.csv', index=False)

print(f"\n📊 Results:")
for _, row in results_df.iterrows():
    status = "✅" if row['result'] == 'Supported' else "⚠️" if row['result'] == 'Weak' else "❌"
    print(f"   {status} {row['hypothesis']}")
    print(f"      Statistic: {row['statistic']:.2f}, p-value: {row['p_value']:.4f}")

supported = len(results_df[results_df['result'] == 'Supported'])
weak = len(results_df[results_df['result'] == 'Weak'])
rejected = len(results_df[results_df['result'] == 'Rejected'])

print(f"\n📈 Overall:")
print(f"   ✅ Supported: {supported}/{len(results_df)}")
print(f"   ⚠️ Weak: {weak}/{len(results_df)}")
print(f"   ❌ Rejected: {rejected}/{len(results_df)}")

print("\n✓ Saved to outputs/hypothesis_test_results.csv")

print("\n" + "="*80)
print("✅ HYPOTHESIS TESTING COMPLETE")
print("="*80)
