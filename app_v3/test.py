# Load and process raw data from data.csv
def load_and_process_raw_data():
    """
    Load raw data.csv and create hourly demand data.
    
    Returns:
        DataFrame with timestamp and total_orders columns
    """
    print("📊 Loading and processing raw data from data.csv...")
    
    # Load raw data
    df = pd.read_csv('/Users/saugatshakya/Projects/ML2025/project/data/data.csv')
    print(f"✅ Loaded {len(df)} raw orders")
    
    # Filter only delivered orders
    df = df[df['Order Status'] == 'Delivered']
    print(f"   After filtering delivered orders: {len(df)} orders")
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['Order Placed At'], format='%I:%M %p, %B %d %Y')
    
    # Create hourly aggregation
    df['hour_timestamp'] = df['timestamp'].dt.floor('H')
    
    # Count orders per hour
    hourly_orders = df.groupby('hour_timestamp').size().reset_index(name='total_orders')
    hourly_orders = hourly_orders.sort_values('hour_timestamp').reset_index(drop=True)
    hourly_orders.rename(columns={'hour_timestamp': 'timestamp'}, inplace=True)
    
    print(f"✅ Created hourly data: {len(hourly_orders)} hours")
    print(f"   Date range: {hourly_orders['timestamp'].min()} to {hourly_orders['timestamp'].max()}")
    print(f"   Total orders range: {hourly_orders['total_orders'].min()} to {hourly_orders['total_orders'].max()}")
    print(f"   Average orders per hour: {hourly_orders['total_orders'].mean():.2f}")
    
    return hourly_orders

# Load and process raw data
df = load_and_process_raw_data()

print("Raw data processed successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Quick overview
df.head()


def preprocess_data(df):
    """Complete preprocessing pipeline for real demand prediction data."""
    
    print("Starting preprocessing...")
    original_shape = df.shape
    
    # Convert timestamp to datetime
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Extract temporal components
    df['order_date'] = df['timestamp'].dt.date
    df['order_hour'] = df['timestamp'].dt.hour
    
    # Rename total_orders to orders_per_hour for consistency
    df['orders_per_hour'] = df['total_orders']
    
    print(f"Preprocessing complete! Shape: {df.shape}")
    return df

# Run preprocessing
processed_df = preprocess_data(df)
processed_df.head()


def create_hourly_features(df):
    """Aggregate to hourly level and create features (already hourly for real data)."""

    print("Creating hourly features...")

    # For real data, we already have hourly data
    hourly_df = df.copy()
    hourly_df = hourly_df.sort_values(by=['timestamp'])

    print(f"Hourly data: {len(hourly_df)} hour-blocks")
    return hourly_df


def create_temporal_features(df):
    """Create temporal features matching app_v2 implementation."""

    df = df.copy()

    # Basic temporal features (matching app_v2)
    df['hour'] = df['order_hour']
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Cyclic encoding (matching app_v2)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Time-based features (matching app_v2)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 23)).astype(int)

    return df


def create_lag_features(df):
    """Create lag features matching app_v2."""

    df = df.copy()
    df = df.sort_values(by=['timestamp'])

    # Lag features (matching app_v2 exactly)
    df['total_orders_lag_1h'] = df['orders_per_hour'].shift(1)
    df['total_orders_lag_24h'] = df['orders_per_hour'].shift(24)

    # Additional lags for better performance
    df['total_orders_lag_2h'] = df['orders_per_hour'].shift(2)
    df['total_orders_lag_3h'] = df['orders_per_hour'].shift(3)
    df['total_orders_lag_48h'] = df['orders_per_hour'].shift(48)
    df['total_orders_lag_168h'] = df['orders_per_hour'].shift(168)  # 1 week

    return df


def create_rolling_features(df):
    """Create rolling window features while avoiding leakage by using only past values.

    Important: rolling windows must not include the current row's target. We compute
    rolling statistics on the 1-step-shifted series so that each rolling value only
    depends on past observations.
    """

    df = df.copy()
    df = df.sort_values(by=['timestamp'])

    # Use a shifted series so rolling stats use only historical values
    s = df['orders_per_hour'].shift(1)

    # Rolling features (safe - don't include current)
    df['total_orders_rolling_6h'] = s.rolling(window=6, min_periods=1).mean()
    df['total_orders_rolling_24h'] = s.rolling(window=24, min_periods=1).mean()

    # Additional rolling features for better performance
    df['total_orders_rolling_3h'] = s.rolling(window=3, min_periods=1).mean()
    df['total_orders_rolling_12h'] = s.rolling(window=12, min_periods=1).mean()

    # Rolling stds (safe)
    df['total_orders_rolling_6h_std'] = s.rolling(window=6, min_periods=1).std()
    df['total_orders_rolling_24h_std'] = s.rolling(window=24, min_periods=1).std()

    # If std is NaN (e.g., single observation), replace with 0
    df['total_orders_rolling_6h_std'] = df['total_orders_rolling_6h_std'].fillna(0)
    df['total_orders_rolling_24h_std'] = df['total_orders_rolling_24h_std'].fillna(0)

    return df


def create_pattern_features(df):
    """Create hourly pattern features WITHOUT data leakage.

    ⚠️  CRITICAL: Pattern features must be calculated only on training data
    to avoid data leakage. This function should be called AFTER train/test split.
    """

    df = df.copy()

    # ⚠️  WARNING: These features should NOT be used directly
    # They must be calculated separately on training data only
    # and then applied to both train and test sets

    print("⚠️  WARNING: Pattern features should be calculated on training data only!")
    print("   This function is for reference - use create_train_pattern_features() instead")

    return df


def create_train_pattern_features(train_df):
    """Create pattern features using ONLY training data to avoid leakage."""

    # Calculate patterns ONLY from training data
    hourly_avg = train_df.groupby('order_hour')['orders_per_hour'].mean().to_dict()
    dow_avg = train_df.groupby('day_of_week')['orders_per_hour'].mean().to_dict()

    # Create hour-day combination patterns
    hour_dow_avg = train_df.groupby(['order_hour', 'day_of_week'])['orders_per_hour'].mean().to_dict()

    return {
        'hourly_avg': hourly_avg,
        'dow_avg': dow_avg,
        'hour_dow_avg': hour_dow_avg
    }


def apply_pattern_features(df, pattern_stats):
    """Apply pre-calculated pattern features to any dataset (train or test)."""

    df = df.copy()

    # Apply patterns calculated from training data only
    df['hour_avg_orders'] = df['order_hour'].map(pattern_stats['hourly_avg'])
    df['dow_avg_orders'] = df['day_of_week'].map(pattern_stats['dow_avg'])

    # Hour + day of week combination
    df['hour_dow_avg_orders'] = df.apply(
        lambda row: pattern_stats['hour_dow_avg'].get(
            (row['order_hour'], row['day_of_week']), 
            pattern_stats['hourly_avg'].get(row['order_hour'], 0)
        ), axis=1
    )

    return df


def create_restaurant_features(df):
    """Create restaurant-specific features for demand prediction.

    IMPORTANT: Do NOT derive features using the current target value 'orders_per_hour'.
    Use lag/rolling statistics computed from historical data only.
    """

    df = df.copy()

    # Meal time categorization (restaurant-focused)
    df['meal_breakfast'] = ((df['order_hour'] >= 6) & (df['order_hour'] <= 10)).astype(int)
    df['meal_lunch'] = ((df['order_hour'] >= 11) & (df['order_hour'] <= 15)).astype(int)
    df['meal_dinner'] = ((df['order_hour'] >= 17) & (df['order_hour'] <= 22)).astype(int)
    df['meal_late_night'] = ((df['order_hour'] >= 23) | (df['order_hour'] <= 5)).astype(int)

    # Peak hours (based on typical restaurant patterns)
    df['is_peak_hour'] = (((df['order_hour'] >= 12) & (df['order_hour'] <= 14)) |
                         ((df['order_hour'] >= 19) & (df['order_hour'] <= 21))).astype(int)

    # Weekend vs weekday patterns
    df['is_friday_evening'] = ((df['day_of_week'] == 4) & (df['order_hour'] >= 18)).astype(int)
    df['is_saturday'] = (df['day_of_week'] == 5).astype(int)
    df['is_sunday'] = (df['day_of_week'] == 6).astype(int)

    # Seasonal patterns (Delhi weather consideration)
    month = df['timestamp'].dt.month
    df['season_winter'] = month.isin([12, 1, 2]).astype(int)  # Cold months
    df['season_summer'] = month.isin([4, 5, 6]).astype(int)   # Hot months
    df['season_monsoon'] = month.isin([7, 8, 9]).astype(int)  # Rainy months

    # Business day patterns
    df['is_business_day'] = ((df['day_of_week'] >= 0) & (df['day_of_week'] <= 4)).astype(int)

    # Time since start of week/month
    df['week_of_month'] = ((df['timestamp'].dt.day - 1) // 7) + 1
    df['is_month_end'] = (df['timestamp'].dt.day >= 25).astype(int)
    df['is_month_start'] = (df['timestamp'].dt.day <= 7).astype(int)

    # Use safe rolling-based momentum (rolling features are computed on past values only)
    # These momentum features do NOT use the current 'orders_per_hour' directly.
    if 'total_orders_rolling_6h' in df.columns:
        df['demand_momentum_6h'] = df['total_orders_rolling_6h'] - df['total_orders_rolling_6h'].shift(6)
    else:
        df['demand_momentum_6h'] = np.nan

    if 'total_orders_rolling_24h' in df.columns:
        df['demand_momentum_24h'] = df['total_orders_rolling_24h'] - df['total_orders_rolling_24h'].shift(24)
    else:
        df['demand_momentum_24h'] = np.nan

    return df


def create_business_impact_features(df):
    """Create business-impact features derived from historically-based statistics (no target leakage).

    We compute staffing and revenue estimates from lagged/rolling demand, not current target.
    """

    df = df.copy()

    # Use rolling 6h mean as the basis for short-term staffing estimates (safe)
    if 'total_orders_rolling_6h' in df.columns:
        df['staffing_load_est'] = df['total_orders_rolling_6h'] / 5.0  # expected orders per staff
        avg_order_value = 250
        df['revenue_potential_est'] = df['total_orders_rolling_6h'] * avg_order_value
        df['orders_per_staff_hour_est'] = df['total_orders_rolling_6h'] / 3.0
    else:
        df['staffing_load_est'] = np.nan
        df['revenue_potential_est'] = np.nan
        df['orders_per_staff_hour_est'] = np.nan

    # Keep alerts/higher-level flags to be computed from training quantiles elsewhere
    return df


def apply_safe_ts_features(df):
    """Apply time-series features safely after train/test split."""
    df = create_lag_features(df)
    df = create_rolling_features(df)
    return df


def apply_safe_restaurant_features(df):
    """Apply restaurant features safely."""
    return create_restaurant_features(df)


def apply_safe_business_features(train_df, test_df):
    """Apply business features with training-based quantiles."""
    # For now, just apply the basic business features
    # In a more sophisticated implementation, we could compute quantiles from train
    # and apply alerts to both splits
    train_df = create_business_impact_features(train_df)
    test_df = create_business_impact_features(test_df)
    return train_df, test_df


# Execute the feature engineering pipeline
print("🚀 Starting Feature Engineering Pipeline...")

# Step 1: Create hourly features
hourly_df = create_hourly_features(processed_df)
print(f"✅ Hourly features created: {hourly_df.shape}")

# Step 2: Add temporal features
temporal_df = create_temporal_features(hourly_df)
print(f"✅ Temporal features added: {temporal_df.shape}")

# Step 3: Add lag features
lag_df = create_lag_features(temporal_df)
print(f"✅ Lag features added: {lag_df.shape}")

# Step 4: Add rolling features (safe - no leakage)
rolling_df = create_rolling_features(lag_df)
print(f"✅ Rolling features added: {rolling_df.shape}")

# Step 5: Train/Test Split (time-aware - no future data in training)
# Use 80% for training, 20% for testing (chronological split)
split_idx = int(len(rolling_df) * 0.8)
train_df = rolling_df.iloc[:split_idx].copy()
test_df = rolling_df.iloc[split_idx:].copy()

print(f"✅ Train/Test split: Train {len(train_df)} hours, Test {len(test_df)} hours")
print(f"   Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
print(f"   Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

# Step 6: Compute pattern features ONLY on training data (to avoid leakage)
pattern_stats = create_train_pattern_features(train_df)
print("✅ Pattern statistics computed from training data only")

# Step 7: Apply pattern features to both train and test
train_df = apply_pattern_features(train_df, pattern_stats)
test_df = apply_pattern_features(test_df, pattern_stats)
print("✅ Pattern features applied to train and test sets")

# Step 8: Apply safe time-series features (lags and rolling after split)
train_df = apply_safe_ts_features(train_df)
test_df = apply_safe_ts_features(test_df)
print("✅ Safe time-series features applied")

# Step 9: Apply restaurant features
train_df = create_restaurant_features(train_df)
test_df = create_restaurant_features(test_df)
print("✅ Restaurant features applied")

# Step 10: Apply business impact features (safe - based on rolling stats)
train_df = create_business_impact_features(train_df)
test_df = create_business_impact_features(test_df)
print("✅ Business impact features applied")

# Step 11: Fill missing values
def fill_missing_values(df):
    """Fill missing values with forward/backward fill, then zero for remaining."""
    df = df.copy()
    # Sort by timestamp to ensure proper filling
    df = df.sort_values('timestamp')
    # Forward fill, then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    # Fill any remaining NaNs with 0 (for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df

train_df = fill_missing_values(train_df)
test_df = fill_missing_values(test_df)
print("✅ Missing values filled")

# Step 12: Define feature sets
# App_v2 features (matching the original implementation)
app_v2_features = [
    'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'is_morning', 'is_afternoon', 'is_evening',
    'total_orders_lag_1h', 'total_orders_lag_24h',
    'total_orders_rolling_6h', 'total_orders_rolling_24h'
]

# Enhanced features (app_v2 + additional features)
enhanced_features = app_v2_features + [
    # Additional lags
    'total_orders_lag_2h', 'total_orders_lag_3h', 'total_orders_lag_48h', 'total_orders_lag_168h',
    # Additional rolling means
    'total_orders_rolling_3h', 'total_orders_rolling_12h',
    # Rolling stds
    'total_orders_rolling_6h_std', 'total_orders_rolling_24h_std',
    # Pattern features
    'hour_avg_orders', 'dow_avg_orders', 'hour_dow_avg_orders'
]

# Restaurant features (for separate analysis)
restaurant_features = enhanced_features + [
    'meal_breakfast', 'meal_lunch', 'meal_dinner', 'meal_late_night',
    'is_peak_hour', 'is_friday_evening', 'is_saturday', 'is_sunday',
    'season_winter', 'season_summer', 'season_monsoon',
    'is_business_day', 'week_of_month', 'is_month_end', 'is_month_start',
    'demand_momentum_6h', 'demand_momentum_24h'
]

# Business features (including staffing/revenue estimates)
business_features = restaurant_features + [
    'staffing_load_est', 'revenue_potential_est', 'orders_per_staff_hour_est'
]

print(f"✅ Feature sets defined:")
print(f"   App_v2: {len(app_v2_features)} features")
print(f"   Enhanced: {len(enhanced_features)} features")
print(f"   Restaurant: {len(restaurant_features)} features")
print(f"   Business: {len(business_features)} features")

# Step 13: Create feature matrices
X_train_app_v2 = train_df[app_v2_features]
X_test_app_v2 = test_df[app_v2_features]
X_train_enhanced = train_df[enhanced_features]
X_test_enhanced = test_df[enhanced_features]
X_train_restaurant = train_df[restaurant_features]
X_test_restaurant = test_df[restaurant_features]
X_train_business = train_df[business_features]
X_test_business = test_df[business_features]

y_train = train_df['orders_per_hour']
y_test = test_df['orders_per_hour']

print(f"✅ Feature matrices created:")
print(f"   X_train_app_v2: {X_train_app_v2.shape}, X_test_app_v2: {X_test_app_v2.shape}")
print(f"   X_train_enhanced: {X_train_enhanced.shape}, X_test_enhanced: {X_test_enhanced.shape}")
print(f"   y_train: {len(y_train)}, y_test: {len(y_test)}")

print("🎉 Feature Engineering Pipeline Complete!")
print("   Ready for model training.")


# Summary of the already-applied, safe train/test split
# This cell used to re-create the split and overwrite the safe split.
# Now it reports the current split if available and gives clear instructions otherwise.
if 'train_df' in globals() and 'test_df' in globals():
    print(f"Train shape (already split): {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

    # Ensure feature matrices exist (created in previous cell). If not, create them.
    if 'X_train_app_v2' not in globals():
        X_train_app_v2 = train_df[app_v2_features]
        X_test_app_v2 = test_df[app_v2_features]
        X_train_enhanced = train_df[enhanced_features]
        X_test_enhanced = test_df[enhanced_features]
        X_train_restaurant = train_df[restaurant_features]
        X_test_restaurant = test_df[restaurant_features]
        X_train_business = train_df[business_features]
        X_test_business = test_df[business_features]
        y_train = train_df['orders_per_hour']
        y_test = test_df['orders_per_hour']

    print('Feature matrices ready (not re-created).')
else:
    print("train_df / test_df not found in the notebook namespace.")
    print("Please run the earlier feature-engineering cell that creates 'train_df' and 'test_df' (the cell under '## 3. Feature Engineering').")
    print("If you ran that cell but still see this message, run the whole notebook from the top to ensure definitions and functions are executed in order.")
