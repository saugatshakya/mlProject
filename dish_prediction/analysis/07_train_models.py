"""07_train_models.py
Simple baseline model training (multi-output) for dish demand.

Trains three baselines:
 - Ridge (linear, closed-form)
 - RandomForest (tree-based)
 - LightGBM (if available)

Saves per-dish and aggregate MAE/RMSE to `analysis/outputs/model_performance.csv` and
persists models to `analysis/models/`.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


OUT = Path('analysis/outputs')
MODELDIR = Path('analysis/models')
OUT.mkdir(parents=True, exist_ok=True)
MODELDIR.mkdir(parents=True, exist_ok=True)


def load_splits():
	train = pd.read_csv('data/train_full.csv')
	test = pd.read_csv('data/test_full.csv')
	# split metadata is saved under analysis/outputs relative to this script
	meta = json.load(open('outputs/split_metadata.json'))
	dish_cols = meta['dish_columns']
	feature_cols = meta['feature_columns']
	return train, test, feature_cols, dish_cols


def preprocess(train, test, feature_cols):
	# Keep only feature columns (if present) and align dtypes
	X_train = train.copy()
	X_test = test.copy()

	# Some metadata may include engineered features not present as columns; intersect
	feat_use = [c for c in feature_cols if c in X_train.columns]

	X_train = X_train[feat_use]
	X_test = X_test[feat_use]

	# Simple imputation: numeric median, categorical fill with mode
	for col in X_train.columns:
		if X_train[col].dtype.kind in 'biufc':
			med = X_train[col].median()
			X_train[col] = X_train[col].fillna(med)
			X_test[col] = X_test[col].fillna(med)
		else:
			mode = X_train[col].mode().iloc[0] if not X_train[col].mode().empty else ''
			X_train[col] = X_train[col].fillna(mode)
			X_test[col] = X_test[col].fillna(mode)

	# One-hot encode object columns consistently across train+test
	combined = pd.concat([X_train, X_test], axis=0)
	combined = pd.get_dummies(combined, drop_first=False)
	X_train_enc = combined.iloc[:len(X_train), :].reset_index(drop=True)
	X_test_enc = combined.iloc[len(X_train):, :].reset_index(drop=True)

	return X_train_enc, X_test_enc


def train_and_eval():
	train, test, feature_cols, dish_cols = load_splits()

	X_train, X_test = preprocess(train, test, feature_cols)
	y_train = train[dish_cols].reset_index(drop=True)
	y_test = test[dish_cols].reset_index(drop=True)

	results = []

	# Baseline: mean predictor (per-dish) as a reference
	y_pred_mean = np.tile(y_train.mean().values, (len(y_test), 1))
	mae_mean = mean_absolute_error(y_test, y_pred_mean)
	rmse_mean = np.sqrt(mean_squared_error(y_test, y_pred_mean))
	results.append({'model': 'MeanBaseline', 'mae': mae_mean, 'rmse': rmse_mean})

	# 1) Ridge (multioutput)
	ridge = MultiOutputRegressor(Ridge(alpha=1.0))
	ridge.fit(X_train, y_train)
	y_pred_ridge = ridge.predict(X_test)
	mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
	rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
	results.append({'model': 'Ridge', 'mae': mae_ridge, 'rmse': rmse_ridge})
	joblib.dump(ridge, MODELDIR / 'ridge_multioutput.pkl')

	# 2) RandomForest
	rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42))
	rf.fit(X_train, y_train)
	y_pred_rf = rf.predict(X_test)
	mae_rf = mean_absolute_error(y_test, y_pred_rf)
	rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
	results.append({'model': 'RandomForest', 'mae': mae_rf, 'rmse': rmse_rf})
	joblib.dump(rf, MODELDIR / 'rf_multioutput.pkl')

	# 3) LightGBM (wrapped) if available
	try:
		from lightgbm import LGBMRegressor
		lgb = MultiOutputRegressor(LGBMRegressor(n_estimators=100, random_state=42))
		lgb.fit(X_train, y_train)
		y_pred_lgb = lgb.predict(X_test)
		mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
		rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
		results.append({'model': 'LightGBM', 'mae': mae_lgb, 'rmse': rmse_lgb})
		joblib.dump(lgb, MODELDIR / 'lgb_multioutput.pkl')
	except Exception as e:
		print('LightGBM not available or failed to run:', e)

	# Per-dish errors (mean across dishes) for the best model (use RF predictions)
	per_dish = []
	preds = y_pred_rf
	for i, dish in enumerate(dish_cols):
		mae_d = mean_absolute_error(y_test.iloc[:, i], preds[:, i])
		rmse_d = np.sqrt(mean_squared_error(y_test.iloc[:, i], preds[:, i]))
		per_dish.append({'dish': dish, 'mae': mae_d, 'rmse': rmse_d})

	perf_df = pd.DataFrame(results)
	per_dish_df = pd.DataFrame(per_dish)

	perf_df.to_csv(OUT / 'model_performance_summary.csv', index=False)
	per_dish_df.to_csv(OUT / 'model_performance_per_dish_rf.csv', index=False)

	print('\nModel performance summary:')
	print(perf_df)
	print('\nPer-dish RF errors saved to analysis/outputs/model_performance_per_dish_rf.csv')


if __name__ == '__main__':
	train_and_eval()

