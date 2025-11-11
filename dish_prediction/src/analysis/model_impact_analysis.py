"""
MODEL-BASED IMPACT ANALYSIS
Shows what the TRAINED MODEL learned about weather, pollution, and events
NOT just correlation in raw data, but actual model predictions!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("="*80)
print("MODEL-BASED IMPACT ANALYSIS")
print("Using the trained model to understand feature impacts")
print("="*80)

# Base directory
base_dir = Path('/Users/saugatshakya/Projects/ML2025/project/dish_prediction')

# Create output directory
output_dir = base_dir / 'reports/figures/model_impact'
output_dir.mkdir(parents=True, exist_ok=True)

# Load trained model
print("\nLoading trained model...")
model = joblib.load(base_dir / 'models/final/catboost_model.pkl')
print("✓ CatBoost model loaded")

# Get the exact feature order the model expects
feature_names = model.estimators_[0].feature_names_
print(f"✓ Model expects {len(feature_names)} features in specific order")

# Load test data
print("\nLoading test data...")
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

# Create lag features if they don't exist
print("\nCreating lag features...")
for dish in TOP_DISHES:
    if f'{dish}_lag1' not in df.columns:
        df[f'{dish}_lag1'] = df[dish].shift(1).fillna(0)
        df[f'{dish}_lag2'] = df[dish].shift(2).fillna(0)
        df[f'{dish}_lag3'] = df[dish].shift(3).fillna(0)
        df[f'{dish}_smooth'] = df[dish].rolling(window=3, min_periods=1).mean()
print(f"✓ Created lag features for {len(TOP_DISHES)} dishes")

# Use the exact features the model expects (all 52)
feature_cols = feature_names  # Use model's exact feature list
print(f"\n✓ Using all {len(feature_cols)} features (in model's expected order)")

# Verify all features exist
missing_features = [f for f in feature_cols if f not in df.columns]
if missing_features:
    print(f"⚠️  WARNING: {len(missing_features)} features still missing:")
    for f in missing_features[:5]:
        print(f"    - {f}")
    # Fill missing with 0
    for f in missing_features:
        df[f] = 0
    print("  ✓ Filled missing features with 0")

# Prepare baseline sample (median values for last 1000 hours)
baseline = df[feature_cols].tail(1000).median().to_dict()
print("✓ Created baseline scenario (median of last 1000 hours)")

# ============================================================================
# HELPER FUNCTION: Predict with modified features
# ============================================================================

def predict_with_change(baseline_dict, feature_name, new_value):
    """Make prediction with a single feature changed"""
    modified = baseline_dict.copy()
    modified[feature_name] = new_value
    # Create DataFrame with features in the exact order model expects
    X = pd.DataFrame([modified])[feature_cols]
    prediction = model.predict(X)[0]  # Returns predictions for all 10 dishes
    return prediction.sum()  # Total orders across all dishes

# ============================================================================
# 1. WEATHER IMPACT - WHAT THE MODEL LEARNED
# ============================================================================
print("\n" + "="*80)
print("1. ANALYZING WEATHER IMPACT (MODEL PREDICTIONS)")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Temperature impact
print("\n  Analyzing temperature impact...")
ax1 = axes[0, 0]
temps = np.linspace(df['env_temp'].min(), df['env_temp'].max(), 30)
temp_predictions = []
for temp in temps:
    pred = predict_with_change(baseline, 'env_temp', temp)
    temp_predictions.append(pred)

ax1.plot(temps, temp_predictions, 'o-', color='orangered', linewidth=3, markersize=8)
ax1.fill_between(temps, temp_predictions, alpha=0.2, color='orangered')
ax1.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Model Predicted Orders', fontsize=12, fontweight='bold')
ax1.set_title('MODEL LEARNED: Temperature Impact on Orders\n(All other features held constant)', 
              fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3)

# Calculate impact
temp_range = temps.max() - temps.min()
pred_range = max(temp_predictions) - min(temp_predictions)
impact_pct = (pred_range / np.mean(temp_predictions)) * 100
ax1.text(0.05, 0.95, 
         f'Impact: {pred_range:.1f} orders\n({impact_pct:.1f}% change)\nover {temp_range:.1f}°C range',
         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Humidity impact
print("  Analyzing humidity impact...")
ax2 = axes[0, 1]
humidities = np.linspace(df['env_rhum'].min(), df['env_rhum'].max(), 30)
humidity_predictions = []
for hum in humidities:
    pred = predict_with_change(baseline, 'env_rhum', hum)
    humidity_predictions.append(pred)

ax2.plot(humidities, humidity_predictions, 'o-', color='steelblue', linewidth=3, markersize=8)
ax2.fill_between(humidities, humidity_predictions, alpha=0.2, color='steelblue')
ax2.set_xlabel('Humidity (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Model Predicted Orders', fontsize=12, fontweight='bold')
ax2.set_title('MODEL LEARNED: Humidity Impact on Orders\n(All other features held constant)', 
              fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3)

hum_range = humidities.max() - humidities.min()
pred_range_h = max(humidity_predictions) - min(humidity_predictions)
impact_pct_h = (pred_range_h / np.mean(humidity_predictions)) * 100
ax2.text(0.05, 0.95,
         f'Impact: {pred_range_h:.1f} orders\n({impact_pct_h:.1f}% change)\nover {hum_range:.1f}% range',
         transform=ax2.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Precipitation impact
print("  Analyzing precipitation impact...")
ax3 = axes[1, 0]
precips = np.linspace(df['env_precip'].min(), df['env_precip'].max(), 20)
precip_predictions = []
for precip in precips:
    pred = predict_with_change(baseline, 'env_precip', precip)
    precip_predictions.append(pred)

ax3.plot(precips, precip_predictions, 'o-', color='purple', linewidth=3, markersize=8)
ax3.fill_between(precips, precip_predictions, alpha=0.2, color='purple')
ax3.set_xlabel('Precipitation (mm)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Model Predicted Orders', fontsize=12, fontweight='bold')
ax3.set_title('MODEL LEARNED: Precipitation Impact on Orders\n(All other features held constant)', 
              fontsize=13, fontweight='bold')
ax3.grid(alpha=0.3)

precip_range = precips.max() - precips.min()
pred_range_p = max(precip_predictions) - min(precip_predictions)
impact_pct_p = (pred_range_p / np.mean(precip_predictions)) * 100
ax3.text(0.05, 0.95,
         f'Impact: {pred_range_p:.1f} orders\n({impact_pct_p:.1f}% change)\nover {precip_range:.2f}mm range',
         transform=ax3.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Wind speed impact
print("  Analyzing wind speed impact...")
ax4 = axes[1, 1]
winds = np.linspace(df['env_wspd'].min(), df['env_wspd'].max(), 20)
wind_predictions = []
for wind in winds:
    pred = predict_with_change(baseline, 'env_wspd', wind)
    wind_predictions.append(pred)

ax4.plot(winds, wind_predictions, 'o-', color='green', linewidth=3, markersize=8)
ax4.fill_between(winds, wind_predictions, alpha=0.2, color='green')
ax4.set_xlabel('Wind Speed (km/h)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Model Predicted Orders', fontsize=12, fontweight='bold')
ax4.set_title('MODEL LEARNED: Wind Speed Impact on Orders\n(All other features held constant)', 
              fontsize=13, fontweight='bold')
ax4.grid(alpha=0.3)

wind_range = winds.max() - winds.min()
pred_range_w = max(wind_predictions) - min(wind_predictions)
impact_pct_w = (pred_range_w / np.mean(wind_predictions)) * 100
ax4.text(0.05, 0.95,
         f'Impact: {pred_range_w:.1f} orders\n({impact_pct_w:.1f}% change)\nover {wind_range:.1f} km/h range',
         transform=ax4.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.suptitle('WEATHER IMPACT - ACCORDING TO TRAINED MODEL', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(output_dir / '01_model_weather_impact.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / '01_model_weather_impact.png'}")
plt.close()

# ============================================================================
# 2. POLLUTION IMPACT - WHAT THE MODEL LEARNED
# ============================================================================
print("\n" + "="*80)
print("2. ANALYZING POLLUTION IMPACT (MODEL PREDICTIONS)")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

pollutants = ['aqi', 'pm2_5', 'pm10', 'no2', 'o3', 'co']
titles = ['Air Quality Index (AQI)', 'PM2.5', 'PM10', 'NO2', 'O3', 'CO']
colors = ['red', 'orange', 'brown', 'purple', 'blue', 'green']

for idx, (pollutant, title, color) in enumerate(zip(pollutants, titles, colors)):
    print(f"  Analyzing {pollutant} impact...")
    ax = axes[idx // 3, idx % 3]
    
    # Test different levels of this pollutant
    levels = np.linspace(df[pollutant].min(), df[pollutant].max(), 25)
    predictions = []
    for level in levels:
        pred = predict_with_change(baseline, pollutant, level)
        predictions.append(pred)
    
    ax.plot(levels, predictions, 'o-', color=color, linewidth=3, markersize=7, alpha=0.8)
    ax.fill_between(levels, predictions, alpha=0.2, color=color)
    
    ax.set_xlabel(f'{title} Level', fontsize=11, fontweight='bold')
    ax.set_ylabel('Model Predicted Orders', fontsize=11, fontweight='bold')
    ax.set_title(f'MODEL LEARNED: {title} Impact\n(All other features held constant)', 
                fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Calculate impact
    level_range = levels.max() - levels.min()
    pred_range = max(predictions) - min(predictions)
    impact_pct = (pred_range / np.mean(predictions)) * 100
    
    ax.text(0.05, 0.95,
           f'Impact: {pred_range:.1f} orders\n({impact_pct:.1f}% change)\nRange: {level_range:.1f}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.suptitle('POLLUTION IMPACT - ACCORDING TO TRAINED MODEL', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / '02_model_pollution_impact.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / '02_model_pollution_impact.png'}")
plt.close()

# ============================================================================
# 3. EVENT/HOLIDAY IMPACT - WHAT THE MODEL LEARNED
# ============================================================================
print("\n" + "="*80)
print("3. ANALYZING EVENT/HOLIDAY IMPACT (MODEL PREDICTIONS)")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Holiday impact
print("  Analyzing holiday impact...")
ax1 = axes[0]
scenarios = ['Regular Day', 'Holiday']
predictions_holiday = []

# Regular day
pred_regular = predict_with_change(baseline, 'holiday', 0)
predictions_holiday.append(pred_regular)

# Holiday
pred_holiday = predict_with_change(baseline, 'holiday', 1)
predictions_holiday.append(pred_holiday)

bars = ax1.bar(scenarios, predictions_holiday, color=['steelblue', 'coral'], 
               alpha=0.7, edgecolor='black', linewidth=2, width=0.6)
ax1.set_ylabel('Model Predicted Orders', fontsize=12, fontweight='bold')
ax1.set_title('MODEL LEARNED: Holiday Impact on Orders\n(All other features held constant)', 
              fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, predictions_holiday):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Calculate impact
holiday_impact = pred_holiday - pred_regular
holiday_impact_pct = (holiday_impact / pred_regular) * 100
ax1.text(0.5, 0.95,
         f'Holiday Effect: {holiday_impact:+.1f} orders ({holiday_impact_pct:+.1f}%)',
         transform=ax1.transAxes, fontsize=12, verticalalignment='top', ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Event impact
print("  Analyzing event impact...")
ax2 = axes[1]
scenarios_event = ['Regular Day', 'Event Day']
predictions_event = []

# Regular day
pred_no_event = predict_with_change(baseline, 'has_event', 0)
predictions_event.append(pred_no_event)

# Event day
pred_event = predict_with_change(baseline, 'has_event', 1)
predictions_event.append(pred_event)

bars2 = ax2.bar(scenarios_event, predictions_event, color=['steelblue', 'orange'], 
                alpha=0.7, edgecolor='black', linewidth=2, width=0.6)
ax2.set_ylabel('Model Predicted Orders', fontsize=12, fontweight='bold')
ax2.set_title('MODEL LEARNED: Event Impact on Orders\n(All other features held constant)', 
              fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars2, predictions_event):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Calculate impact
event_impact = pred_event - pred_no_event
event_impact_pct = (event_impact / pred_no_event) * 100
ax2.text(0.5, 0.95,
         f'Event Effect: {event_impact:+.1f} orders ({event_impact_pct:+.1f}%)',
         transform=ax2.transAxes, fontsize=12, verticalalignment='top', ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.suptitle('EVENT/HOLIDAY IMPACT - ACCORDING TO TRAINED MODEL', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / '03_model_event_holiday_impact.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / '03_model_event_holiday_impact.png'}")
plt.close()

# ============================================================================
# 4. FEATURE IMPORTANCE - FROM THE MODEL
# ============================================================================
print("\n" + "="*80)
print("4. EXTRACTING FEATURE IMPORTANCE FROM MODEL")
print("="*80)

# Get feature importances from CatBoost (from first estimator)
# MultiOutputRegressor wraps multiple estimators, take average
all_importances = []
for estimator in model.estimators_:
    all_importances.append(estimator.feature_importances_)

# Average importance across all dishes
feature_importance = np.mean(all_importances, axis=0)
feature_names_imp = model.estimators_[0].feature_names_

# Create DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names_imp,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(importance_df.head(20).to_string(index=False))

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top 20 features
ax1 = axes[0, 0]
top_20 = importance_df.head(20)
ax1.barh(range(len(top_20)), top_20['importance'], color='steelblue', alpha=0.7)
ax1.set_yticks(range(len(top_20)))
ax1.set_yticklabels(top_20['feature'], fontsize=9)
ax1.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax1.set_title('Top 20 Most Important Features\n(According to CatBoost Model)', 
              fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Feature category breakdown
ax2 = axes[0, 1]
# Categorize features
categories = []
for feat in importance_df['feature']:
    if feat in ['hour_of_day', 'day_of_week', 'is_weekend', 'sin_hour', 'cos_hour', 'month']:
        categories.append('Temporal')
    elif feat in ['env_temp', 'env_rhum', 'env_precip', 'env_wspd']:
        categories.append('Weather')
    elif feat in ['aqi', 'pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2', 'nh3']:
        categories.append('Pollution')
    elif feat in ['holiday', 'has_event']:
        categories.append('Events')
    elif '_lag' in feat or '_smooth' in feat:
        categories.append('Historical')
    else:
        categories.append('Other')

importance_df['category'] = categories
category_importance = importance_df.groupby('category')['importance'].sum().sort_values(ascending=False)

colors_cat = ['steelblue', 'coral', 'green', 'purple', 'orange']
wedges, texts, autotexts = ax2.pie(category_importance.values, 
                                     labels=category_importance.index,
                                     autopct='%1.1f%%',
                                     colors=colors_cat,
                                     startangle=90,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('Feature Importance by Category\n(Total Importance Distribution)', 
              fontsize=13, fontweight='bold')

# Weather features detail
ax3 = axes[1, 0]
weather_features = importance_df[importance_df['category'] == 'Weather'].sort_values('importance', ascending=False)
if len(weather_features) > 0:
    ax3.bar(weather_features['feature'], weather_features['importance'], 
           color='orangered', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Importance Score', fontsize=11, fontweight='bold')
    ax3.set_title('Weather Features Importance\n(Detailed Breakdown)', 
                  fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, (feat, imp) in enumerate(zip(weather_features['feature'], weather_features['importance'])):
        ax3.text(i, imp, f'{imp:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Pollution features detail
ax4 = axes[1, 1]
pollution_features = importance_df[importance_df['category'] == 'Pollution'].sort_values('importance', ascending=False)
if len(pollution_features) > 0:
    ax4.bar(pollution_features['feature'], pollution_features['importance'],
           color='red', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Importance Score', fontsize=11, fontweight='bold')
    ax4.set_title('Pollution Features Importance\n(Detailed Breakdown)', 
                  fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, (feat, imp) in enumerate(zip(pollution_features['feature'], pollution_features['importance'])):
        ax4.text(i, imp, f'{imp:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('FEATURE IMPORTANCE - WHAT THE MODEL ACTUALLY USES', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(output_dir / '04_model_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / '04_model_feature_importance.png'}")
plt.close()

# Save feature importance to CSV
importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
print(f"✓ Saved: {output_dir / 'feature_importance.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"""
Generated {4} model-based analysis figures:

1. ✓ Weather Impact (Temperature, Humidity, Precipitation, Wind)
2. ✓ Pollution Impact (AQI, PM2.5, PM10, NO2, O3, CO)
3. ✓ Event/Holiday Impact
4. ✓ Feature Importance

All figures saved to: {output_dir}

KEY DIFFERENCE FROM PREVIOUS ANALYSIS:
- Previous: "When X happens, we see Y in the data" (correlation)
- This: "According to the model, changing X causes Y" (learned causation)

The model holds all other features constant and tests each feature's impact.
This shows what the model ACTUALLY LEARNED during training!
""")
