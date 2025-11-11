"""
ABLATION STUDY - Feature Importance Analysis
Shows the impact of removing feature groups on model performance

Compares models with:
1. ALL features (baseline)
2. WITHOUT weather features
3. WITHOUT pollution features  
4. WITHOUT event/holiday features
5. WITHOUT temporal features
6. ONLY historical features (minimal baseline)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("ABLATION STUDY - FEATURE GROUP IMPACT ANALYSIS")
print("="*80)

# Base directory
base_dir = Path('/Users/saugatshakya/Projects/ML2025/project/dish_prediction')

# Create output directory
output_dir = base_dir / 'reports/figures/ablation_study'
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
print("\nLoading data...")
df = pd.read_csv(base_dir / 'data/processed/hourly_data_with_features.csv')
df['hour'] = pd.to_datetime(df['hour'])
df = df.sort_values('hour').reset_index(drop=True)
print(f"✓ Loaded {len(df)} samples")

# Top 10 dishes
TOP_DISHES = [
    'Bageecha Pizza',
    'Chilli Cheese Garlic Bread',
    'Bone in Jamaican Grilled Chicken',
    'All About Chicken Pizza',
    'Makhani Paneer Pizza',
    'Margherita Pizza',
    'Cheesy Garlic Bread',
    'Jamaican Chicken Melt',
    'Herbed Potato',
    'Tripple Cheese Pizza'
]

# Create lag features
print("\nCreating lag features...")
for dish in TOP_DISHES:
    df[f'{dish}_lag1'] = df[dish].shift(1).fillna(0)
    df[f'{dish}_lag2'] = df[dish].shift(2).fillna(0)
    df[f'{dish}_lag3'] = df[dish].shift(3).fillna(0)
    df[f'{dish}_smooth'] = df[dish].rolling(window=3, min_periods=1).mean()

# Create cyclical features
df['sin_hour'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['cos_hour'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

print("✓ All features created")

# Define feature groups
FEATURE_GROUPS = {
    'temporal': ['hour_of_day', 'day_of_week', 'is_weekend', 'sin_hour', 'cos_hour'],
    'weather': ['env_temp', 'env_rhum', 'env_precip', 'env_wspd'],
    'pollution': ['aqi', 'pm2_5', 'pm10', 'no2', 'o3', 'co'],
    'events': ['holiday', 'has_event'],
    'historical': []
}

# Add historical features
for dish in TOP_DISHES:
    FEATURE_GROUPS['historical'].extend([
        f'{dish}_lag1', f'{dish}_lag2', f'{dish}_lag3', f'{dish}_smooth'
    ])

print("\n" + "="*80)
print("FEATURE GROUPS")
print("="*80)
for group, features in FEATURE_GROUPS.items():
    print(f"{group.upper():15s}: {len(features):2d} features")
print("="*80)

# Prepare target
y = df[TOP_DISHES].values

# Train-test split (same as final model)
train_size = int(len(df) * 0.8)
train_idx = range(train_size)
test_idx = range(train_size, len(df))

y_train = y[train_idx]
y_test = y[test_idx]

print(f"\nTrain size: {len(train_idx)}")
print(f"Test size:  {len(test_idx)}")

# ============================================================================
# ABLATION EXPERIMENTS
# ============================================================================

ablation_results = []

def train_and_evaluate(feature_list, experiment_name, description):
    """Train model with specific features and evaluate"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Description: {description}")
    print(f"Features: {len(feature_list)}")
    print(f"{'='*80}")
    
    # Prepare data
    X = df[feature_list].values
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    # Train model (same settings as final model)
    print("Training model...")
    model = MultiOutputRegressor(
        CatBoostRegressor(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            verbose=0,
            random_state=42
        ),
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predict
    print("Evaluating...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics for each dish
    dish_results = []
    for i, dish in enumerate(TOP_DISHES):
        r2_train = r2_score(y_train[:, i], y_pred_train[:, i])
        r2_test = r2_score(y_test[:, i], y_pred_test[:, i])
        mae_test = mean_absolute_error(y_test[:, i], y_pred_test[:, i])
        rmse_test = np.sqrt(mean_squared_error(y_test[:, i], y_pred_test[:, i]))
        
        dish_results.append({
            'experiment': experiment_name,
            'dish': dish,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mae_test': mae_test,
            'rmse_test': rmse_test
        })
    
    # Overall metrics
    r2_train_mean = np.mean([r['r2_train'] for r in dish_results])
    r2_test_mean = np.mean([r['r2_test'] for r in dish_results])
    mae_test_mean = np.mean([r['mae_test'] for r in dish_results])
    
    print(f"\n✓ Results:")
    print(f"  Mean R² (train): {r2_train_mean:.4f}")
    print(f"  Mean R² (test):  {r2_test_mean:.4f}")
    print(f"  Mean MAE (test): {mae_test_mean:.4f}")
    
    # Store summary
    ablation_results.append({
        'experiment': experiment_name,
        'description': description,
        'num_features': len(feature_list),
        'r2_train_mean': r2_train_mean,
        'r2_test_mean': r2_test_mean,
        'mae_test_mean': mae_test_mean,
        'dish_results': dish_results
    })
    
    return model, dish_results

# ============================================================================
# RUN ALL ABLATION EXPERIMENTS
# ============================================================================

print("\n" + "="*80)
print("RUNNING ABLATION EXPERIMENTS")
print("="*80)

# 1. FULL MODEL (Baseline)
all_features = []
for group in FEATURE_GROUPS.values():
    all_features.extend(group)
all_features = list(set(all_features))  # Remove duplicates

train_and_evaluate(
    all_features,
    "FULL MODEL",
    "All features: temporal + weather + pollution + events + historical"
)

# 2. WITHOUT WEATHER
features_no_weather = [f for f in all_features if f not in FEATURE_GROUPS['weather']]
train_and_evaluate(
    features_no_weather,
    "NO WEATHER",
    "All features EXCEPT weather (temp, humidity, precip, wind)"
)

# 3. WITHOUT POLLUTION
features_no_pollution = [f for f in all_features if f not in FEATURE_GROUPS['pollution']]
train_and_evaluate(
    features_no_pollution,
    "NO POLLUTION",
    "All features EXCEPT pollution (AQI, PM2.5, PM10, NO2, O3, CO)"
)

# 4. WITHOUT EVENTS
features_no_events = [f for f in all_features if f not in FEATURE_GROUPS['events']]
train_and_evaluate(
    features_no_events,
    "NO EVENTS",
    "All features EXCEPT events/holidays"
)

# 5. WITHOUT TEMPORAL
features_no_temporal = [f for f in all_features if f not in FEATURE_GROUPS['temporal']]
train_and_evaluate(
    features_no_temporal,
    "NO TEMPORAL",
    "All features EXCEPT temporal (hour, day_of_week, etc.)"
)

# 6. ONLY HISTORICAL (Minimal baseline)
train_and_evaluate(
    FEATURE_GROUPS['historical'],
    "ONLY HISTORICAL",
    "Only historical lag features (no weather, pollution, events, or temporal)"
)

# 7. NO EXTERNAL DATA (Historical + Temporal only)
features_no_external = FEATURE_GROUPS['historical'] + FEATURE_GROUPS['temporal']
train_and_evaluate(
    features_no_external,
    "NO EXTERNAL DATA",
    "Only historical + temporal (NO weather, pollution, or events)"
)

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Create summary DataFrame
summary_df = pd.DataFrame([
    {
        'Experiment': r['experiment'],
        'Description': r['description'],
        'Features': r['num_features'],
        'Train R²': r['r2_train_mean'],
        'Test R²': r['r2_test_mean'],
        'Test MAE': r['mae_test_mean']
    }
    for r in ablation_results
])

# Calculate performance drop from FULL MODEL
baseline_r2 = summary_df[summary_df['Experiment'] == 'FULL MODEL']['Test R²'].values[0]
summary_df['R² Drop'] = baseline_r2 - summary_df['Test R²']
summary_df['R² Drop %'] = (summary_df['R² Drop'] / baseline_r2) * 100

print("\n" + "="*80)
print("ABLATION STUDY RESULTS")
print("="*80)
print(summary_df.to_string(index=False))
print("="*80)

# ============================================================================
# FIGURE 1: R² Comparison Across Experiments
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1.1 Mean R² by experiment
ax1 = axes[0, 0]
x = np.arange(len(summary_df))
width = 0.35

bars1 = ax1.bar(x - width/2, summary_df['Train R²'], width, label='Train R²', 
                color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, summary_df['Test R²'], width, label='Test R²', 
                color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Experiment', fontsize=12, fontweight='bold')
ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('ABLATION STUDY: R² Performance by Feature Group\n(Higher is Better)', 
              fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(summary_df['Experiment'], rotation=45, ha='right', fontsize=9)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=baseline_r2, color='green', linestyle='--', linewidth=2, 
            alpha=0.7, label=f'Baseline: {baseline_r2:.4f}')

# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 1.2 Performance drop from baseline
ax2 = axes[0, 1]
colors_drop = ['green' if x == 0 else 'red' for x in summary_df['R² Drop']]
bars = ax2.barh(range(len(summary_df)), summary_df['R² Drop %'], 
                color=colors_drop, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_yticks(range(len(summary_df)))
ax2.set_yticklabels(summary_df['Experiment'], fontsize=10)
ax2.set_xlabel('R² Drop from Baseline (%)', fontsize=12, fontweight='bold')
ax2.set_title('IMPACT OF REMOVING FEATURE GROUPS\n(Lower is Better - Less Impact)', 
              fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

# Add value labels
for i, (bar, drop, drop_pct) in enumerate(zip(bars, summary_df['R² Drop'], summary_df['R² Drop %'])):
    width_val = bar.get_width()
    ax2.text(width_val, bar.get_y() + bar.get_height()/2.,
            f' {drop:.4f} ({drop_pct:.1f}%)', 
            ha='left', va='center', fontsize=9, fontweight='bold')

# 1.3 Feature count vs Performance
ax3 = axes[1, 0]
ax3.scatter(summary_df['Features'], summary_df['Test R²'], 
           s=200, alpha=0.7, color='purple', edgecolors='black', linewidth=2)

# Add labels
for _, row in summary_df.iterrows():
    ax3.annotate(row['Experiment'], 
                (row['Features'], row['Test R²']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

ax3.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
ax3.set_ylabel('Test R² Score', fontsize=12, fontweight='bold')
ax3.set_title('Feature Count vs Model Performance\n(Diminishing Returns Analysis)', 
              fontsize=14, fontweight='bold')
ax3.grid(alpha=0.3)

# Add trend line
z = np.polyfit(summary_df['Features'], summary_df['Test R²'], 2)
p = np.poly1d(z)
x_trend = np.linspace(summary_df['Features'].min(), summary_df['Features'].max(), 100)
ax3.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8, label='Trend')
ax3.legend()

# 1.4 MAE comparison
ax4 = axes[1, 1]
bars = ax4.bar(range(len(summary_df)), summary_df['Test MAE'], 
              color='orange', alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.set_xticks(range(len(summary_df)))
ax4.set_xticklabels(summary_df['Experiment'], rotation=45, ha='right', fontsize=9)
ax4.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
ax4.set_title('ABLATION STUDY: MAE Performance\n(Lower is Better)', 
              fontsize=14, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '01_ablation_study_overview.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / '01_ablation_study_overview.png'}")
plt.close()

# ============================================================================
# FIGURE 2: Detailed Per-Dish Performance
# ============================================================================

fig, axes = plt.subplots(2, 1, figsize=(18, 12))

# Prepare data for per-dish comparison
dish_comparison = []
for result in ablation_results:
    for dish_result in result['dish_results']:
        dish_comparison.append({
            'Experiment': result['experiment'],
            'Dish': dish_result['dish'],
            'R² Test': dish_result['r2_test']
        })

dish_df = pd.DataFrame(dish_comparison)
dish_pivot = dish_df.pivot(index='Dish', columns='Experiment', values='R² Test')

# 2.1 Heatmap of R² by dish and experiment
ax1 = axes[0]
sns.heatmap(dish_pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
           vmin=0.5, vmax=1.0, center=0.85,
           cbar_kws={'label': 'R² Score'},
           linewidths=0.5, ax=ax1)
ax1.set_title('ABLATION STUDY: Per-Dish R² Performance Across Experiments\n(Green = Better)', 
             fontsize=14, fontweight='bold')
ax1.set_xlabel('Experiment', fontsize=12, fontweight='bold')
ax1.set_ylabel('Dish', fontsize=12, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# 2.2 Performance drop per dish
ax2 = axes[1]
baseline_dish = dish_pivot['FULL MODEL']
drop_df = dish_pivot.subtract(baseline_dish, axis=0)

# Plot for each experiment (except FULL MODEL)
experiments_to_plot = [col for col in drop_df.columns if col != 'FULL MODEL']
x = np.arange(len(TOP_DISHES))
width = 0.12

for i, exp in enumerate(experiments_to_plot):
    offset = (i - len(experiments_to_plot)/2) * width
    bars = ax2.bar(x + offset, drop_df[exp], width, label=exp, alpha=0.8, edgecolor='black', linewidth=0.5)

ax2.set_xlabel('Dish', fontsize=12, fontweight='bold')
ax2.set_ylabel('R² Drop from FULL MODEL', fontsize=12, fontweight='bold')
ax2.set_title('ABLATION STUDY: R² Performance Drop per Dish\n(Negative = Worse than FULL MODEL)', 
             fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(TOP_DISHES, rotation=45, ha='right', fontsize=9)
ax2.legend(fontsize=9, loc='upper left', ncol=2)
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

plt.tight_layout()
plt.savefig(output_dir / '02_ablation_per_dish_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '02_ablation_per_dish_analysis.png'}")
plt.close()

# ============================================================================
# FIGURE 3: Feature Group Importance Ranking
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Calculate importance by comparing with/without each group
feature_group_impact = []

full_r2 = summary_df[summary_df['Experiment'] == 'FULL MODEL']['Test R²'].values[0]

for group in ['weather', 'pollution', 'events', 'temporal']:
    exp_name = f"NO {group.upper()}"
    if exp_name in summary_df['Experiment'].values:
        no_group_r2 = summary_df[summary_df['Experiment'] == exp_name]['Test R²'].values[0]
        impact = full_r2 - no_group_r2
        impact_pct = (impact / full_r2) * 100
        
        feature_group_impact.append({
            'Group': group.title(),
            'R² Drop': impact,
            'R² Drop %': impact_pct,
            'Num Features': len(FEATURE_GROUPS[group])
        })

impact_df = pd.DataFrame(feature_group_impact).sort_values('R² Drop', ascending=False)

# 3.1 Feature group importance
ax1 = axes[0, 0]
colors_impact = ['red', 'orange', 'yellow', 'lightgreen']
bars = ax1.barh(range(len(impact_df)), impact_df['R² Drop'], 
               color=colors_impact, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_yticks(range(len(impact_df)))
ax1.set_yticklabels(impact_df['Group'], fontsize=11, fontweight='bold')
ax1.set_xlabel('R² Drop When Removed', fontsize=12, fontweight='bold')
ax1.set_title('FEATURE GROUP IMPORTANCE RANKING\n(Higher = More Important)', 
             fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# Add value labels
for i, (bar, drop, drop_pct) in enumerate(zip(bars, impact_df['R² Drop'], impact_df['R² Drop %'])):
    width_val = bar.get_width()
    ax1.text(width_val, bar.get_y() + bar.get_height()/2.,
            f' {drop:.4f} ({drop_pct:.2f}%)', 
            ha='left', va='center', fontsize=10, fontweight='bold')

# 3.2 Importance per feature (efficiency)
ax2 = axes[0, 1]
impact_df['Impact per Feature'] = impact_df['R² Drop'] / impact_df['Num Features']
colors_eff = ['darkred', 'darkorange', 'gold', 'lightgreen']
bars = ax2.barh(range(len(impact_df)), impact_df['Impact per Feature'], 
               color=colors_eff, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_yticks(range(len(impact_df)))
ax2.set_yticklabels(impact_df['Group'], fontsize=11, fontweight='bold')
ax2.set_xlabel('R² Drop per Feature', fontsize=12, fontweight='bold')
ax2.set_title('FEATURE EFFICIENCY RANKING\n(Impact per Feature - Higher = More Efficient)', 
             fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

# Add value labels
for i, (bar, val, num_feat) in enumerate(zip(bars, impact_df['Impact per Feature'], impact_df['Num Features'])):
    width_val = bar.get_width()
    ax2.text(width_val, bar.get_y() + bar.get_height()/2.,
            f' {val:.5f} ({num_feat} feat)', 
            ha='left', va='center', fontsize=10, fontweight='bold')

# 3.3 Cumulative importance
ax3 = axes[1, 0]
cumulative_impact = np.cumsum(impact_df['R² Drop'].values)
cumulative_pct = (cumulative_impact / full_r2) * 100

ax3.plot(range(1, len(impact_df)+1), cumulative_impact, 'o-', 
        color='darkblue', linewidth=3, markersize=10, label='Cumulative R² Drop')
ax3.fill_between(range(1, len(impact_df)+1), cumulative_impact, 
                alpha=0.3, color='darkblue')

ax3_twin = ax3.twinx()
ax3_twin.plot(range(1, len(impact_df)+1), cumulative_pct, 's--', 
             color='red', linewidth=2, markersize=8, label='Cumulative %')

ax3.set_xlabel('Number of Feature Groups Removed (by importance)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Cumulative R² Drop', fontsize=12, fontweight='bold', color='darkblue')
ax3_twin.set_ylabel('Cumulative Drop %', fontsize=12, fontweight='bold', color='red')
ax3.set_title('CUMULATIVE FEATURE GROUP IMPACT\n(Removing Groups in Order of Importance)', 
             fontsize=14, fontweight='bold')
ax3.set_xticks(range(1, len(impact_df)+1))
ax3.set_xticklabels([f"{i}. {row}" for i, row in enumerate(impact_df['Group'], 1)], 
                    rotation=45, ha='right', fontsize=9)
ax3.grid(alpha=0.3)
ax3.legend(loc='upper left', fontsize=10)
ax3_twin.legend(loc='upper right', fontsize=10)

# 3.4 Summary table
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

table_data = []
table_data.append(['Experiment', 'Features', 'Test R²', 'R² Drop', 'Drop %'])
table_data.append(['─'*20, '─'*8, '─'*8, '─'*8, '─'*8])

for _, row in summary_df.iterrows():
    table_data.append([
        row['Experiment'][:20],
        f"{row['Features']}",
        f"{row['Test R²']:.4f}",
        f"{row['R² Drop']:.4f}",
        f"{row['R² Drop %']:.1f}%"
    ])

table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                 colWidths=[0.4, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(5):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code rows
for i in range(2, len(table_data)):
    r2_drop_pct = float(table_data[i][4].strip('%'))
    if r2_drop_pct < 1:
        color = '#90EE90'  # Light green
    elif r2_drop_pct < 5:
        color = '#FFFF99'  # Light yellow
    elif r2_drop_pct < 10:
        color = '#FFB366'  # Light orange
    else:
        color = '#FF9999'  # Light red
    
    for j in range(5):
        table[(i, j)].set_facecolor(color)

ax4.set_title('ABLATION STUDY SUMMARY TABLE\n(Color coded by performance drop)', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir / '03_feature_group_importance.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '03_feature_group_importance.png'}")
plt.close()

# ============================================================================
# Save Results to CSV
# ============================================================================

summary_df.to_csv(output_dir / 'ablation_study_summary.csv', index=False)
print(f"✓ Saved: {output_dir / 'ablation_study_summary.csv'}")

impact_df.to_csv(output_dir / 'feature_group_importance.csv', index=False)
print(f"✓ Saved: {output_dir / 'feature_group_importance.csv'}")

# Save detailed per-dish results
all_dish_results = []
for result in ablation_results:
    for dish_result in result['dish_results']:
        all_dish_results.append(dish_result)
pd.DataFrame(all_dish_results).to_csv(output_dir / 'ablation_per_dish_results.csv', index=False)
print(f"✓ Saved: {output_dir / 'ablation_per_dish_results.csv'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ABLATION STUDY COMPLETE!")
print("="*80)
print(f"""
Generated 3 comprehensive ablation study figures:

1. ✓ Ablation Study Overview (R², MAE, Performance drops)
2. ✓ Per-Dish Analysis (Heatmap and detailed drops)
3. ✓ Feature Group Importance Ranking

Key Findings:
-------------
FULL MODEL:      {summary_df[summary_df['Experiment'] == 'FULL MODEL']['Test R²'].values[0]:.4f} R²

Feature Group Importance (by R² drop when removed):
""")

for _, row in impact_df.iterrows():
    print(f"  {row['Group']:12s}: {row['R² Drop']:+.4f} R² ({row['R² Drop %']:+5.2f}%) - {int(row['Num Features'])} features")

print(f"""
Most Important:  {impact_df.iloc[0]['Group']} (removes {impact_df.iloc[0]['R² Drop']:.4f} R²)
Least Important: {impact_df.iloc[-1]['Group']} (removes {impact_df.iloc[-1]['R² Drop']:.4f} R²)

All results saved to: {output_dir}
""")
print("="*80)
