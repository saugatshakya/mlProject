"""
Inference Module - Kitchen Prep Time Prediction

This module provides a production-ready API for predicting kitchen preparation times.
It loads a trained model and makes predictions on new orders.

Features:
- Load trained model and metadata
- Validate input features
- Make predictions (single or batch)
- Return predictions in minutes (inverse log transform)

Author: AI Assistant
Date: November 2025
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Union, Dict, List
import warnings
warnings.filterwarnings('ignore')


class PrepTimePredictionAPI:
    """
    Production API for kitchen prep time predictions.
    """
    
    def __init__(self, model_dir: str = "models/final"):
        """
        Initialize the prediction API.
        
        Parameters:
        -----------
        model_dir : str
            Directory containing trained model files
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.feature_names = None
        self.model_config = None
        
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """
        Load model, feature names, and configuration from disk.
        """
        print(f"Loading model artifacts from: {self.model_dir}")
        
        # Load model
        model_path = self.model_dir / "best_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Model loaded: {model_path}")
        
        # Load feature names
        features_path = self.model_dir / "feature_names.txt"
        if not features_path.exists():
            raise FileNotFoundError(f"Feature names file not found: {features_path}")
        
        with open(features_path, 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        print(f"✓ Feature names loaded: {len(self.feature_names)} features")
        
        # Load config
        config_path = self.model_dir / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.model_config = json.load(f)
        print(f"✓ Model config loaded")
        print(f"  Model: {self.model_config['model_name']}")
        print(f"  Test MAE: {self.model_config['test_mae_minutes']:.3f} minutes")
        print(f"  Test R²: {self.model_config['test_r2']:.4f}")
    
    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that input features match expected features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        pd.DataFrame : Validated and ordered features
        """
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select and order features to match training
        return X[self.feature_names]
    
    def predict(
        self,
        X: Union[pd.DataFrame, Dict, List[Dict]],
        return_confidence: bool = False
    ) -> Union[float, np.ndarray, Dict]:
        """
        Make prep time predictions.
        
        Parameters:
        -----------
        X : pd.DataFrame, dict, or list of dicts
            Input features
        return_confidence : bool
            If True, return confidence interval estimates
            
        Returns:
        --------
        Predictions in minutes (float or array)
        If return_confidence=True, returns dict with predictions and intervals
        """
        # Convert input to DataFrame if needed
        if isinstance(X, dict):
            X = pd.DataFrame([X])
            single_prediction = True
        elif isinstance(X, list):
            X = pd.DataFrame(X)
            single_prediction = False
        else:
            single_prediction = False
        
        # Validate features
        X_validated = self._validate_features(X)
        
        # Make prediction (model outputs log-transformed values)
        y_log_pred = self.model.predict(X_validated)
        
        # Inverse transform to get minutes
        y_pred_minutes = np.expm1(y_log_pred)
        
        if single_prediction:
            result = float(y_pred_minutes[0])
        else:
            result = y_pred_minutes
        
        if return_confidence:
            # Estimate confidence interval based on test MAE
            # ±2 * MAE gives approximately 95% confidence
            mae = self.model_config['test_mae_minutes']
            margin = 2 * mae
            
            if single_prediction:
                return {
                    "prediction_minutes": result,
                    "lower_bound": max(0, result - margin),
                    "upper_bound": result + margin,
                    "confidence_level": 0.95
                }
            else:
                return {
                    "predictions_minutes": result,
                    "lower_bounds": np.maximum(0, result - margin),
                    "upper_bounds": result + margin,
                    "confidence_level": 0.95
                }
        
        return result
    
    def predict_batch(
        self,
        df: pd.DataFrame,
        add_to_dataframe: bool = True
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Make predictions for a batch of orders.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with features
        add_to_dataframe : bool
            If True, add predictions to dataframe
            
        Returns:
        --------
        np.ndarray or pd.DataFrame : Predictions or dataframe with predictions
        """
        predictions = self.predict(df)
        
        if add_to_dataframe:
            df_result = df.copy()
            df_result['predicted_KPT_minutes'] = predictions
            return df_result
        else:
            return predictions
    
    def explain_prediction(
        self,
        X: Union[pd.DataFrame, Dict],
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Explain a prediction by showing feature contributions.
        
        Note: This uses feature values as a proxy for contribution.
        For more accurate explanations, use SHAP or LIME.
        
        Parameters:
        -----------
        X : pd.DataFrame or dict
            Input features
        top_n : int
            Number of top features to show
            
        Returns:
        --------
        pd.DataFrame : Feature contributions
        """
        # Convert to DataFrame if needed
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        # Validate features
        X_validated = self._validate_features(X)
        
        # Get feature values
        feature_values = X_validated.iloc[0]
        
        # Create explanation dataframe
        explanation_df = pd.DataFrame({
            'feature': feature_values.index,
            'value': feature_values.values
        })
        
        # Sort by absolute value
        explanation_df['abs_value'] = explanation_df['value'].abs()
        explanation_df = explanation_df.sort_values('abs_value', ascending=False)
        
        return explanation_df.head(top_n)[['feature', 'value']]
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
        --------
        Dict : Model metadata
        """
        return self.model_config.copy()
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of required feature names.
        
        Returns:
        --------
        List[str] : Feature names
        """
        return self.feature_names.copy()


def predict_single_order(
    order_features: Dict,
    model_dir: str = "models/final"
) -> Dict:
    """
    Convenience function to predict prep time for a single order.
    
    Parameters:
    -----------
    order_features : dict
        Dictionary of order features
    model_dir : str
        Path to model directory
        
    Returns:
    --------
    Dict : Prediction result with confidence interval
    
    Example:
    --------
    >>> features = {
    ...     'num_items': 3,
    ...     'num_complex_dishes': 1,
    ...     'Total': 450.0,
    ...     'order_hour': 19,
    ...     'is_weekend': 0,
    ...     # ... other features
    ... }
    >>> result = predict_single_order(features)
    >>> print(f"Predicted KPT: {result['prediction_minutes']:.1f} minutes")
    """
    api = PrepTimePredictionAPI(model_dir=model_dir)
    return api.predict(order_features, return_confidence=True)


def predict_from_csv(
    input_csv: str,
    output_csv: str = None,
    model_dir: str = "models/final"
) -> pd.DataFrame:
    """
    Predict prep times for orders in a CSV file.
    
    Parameters:
    -----------
    input_csv : str
        Path to input CSV with features
    output_csv : str, optional
        Path to save predictions (if None, don't save)
    model_dir : str
        Path to model directory
        
    Returns:
    --------
    pd.DataFrame : Dataframe with predictions
    
    Example:
    --------
    >>> df = predict_from_csv('data/new_orders.csv', 'data/predictions.csv')
    >>> print(df[['Order ID', 'predicted_KPT_minutes']].head())
    """
    # Load data
    df = pd.read_csv(input_csv)
    
    # Make predictions
    api = PrepTimePredictionAPI(model_dir=model_dir)
    df_with_predictions = api.predict_batch(df, add_to_dataframe=True)
    
    # Save if output path provided
    if output_csv:
        df_with_predictions.to_csv(output_csv, index=False)
        print(f"Predictions saved to: {output_csv}")
    
    return df_with_predictions


def main():
    """
    Example usage of the prediction API.
    """
    import sys
    
    print("=" * 80)
    print("KITCHEN PREP TIME PREDICTION - INFERENCE API DEMO")
    print("=" * 80)
    
    # Initialize API
    api = PrepTimePredictionAPI(model_dir="models/final")
    
    # Example 1: Single prediction
    print("\n" + "=" * 80)
    print("Example 1: Single Order Prediction")
    print("=" * 80)
    
    # Sample order features (you would need to provide all required features)
    sample_order = {
        'num_items': 3,
        'num_complex_dishes': 1,
        'has_complex_dish': 1,
        'complexity_ratio': 0.33,
        'Total': 450.0,
        'Packaging_charges': 20.0,
        'total_discount_amt': 50.0,
        'has_discount': 1,
        'disc_percent': 1,
        'disc_flat': 0,
        'disc_bogo': 0,
        'disc_bundle': 0,
        'order_hour': 19,
        'order_day': 5,
        'is_weekend': 0,
        'is_lunch_peak': 0,
        'is_dinner_peak': 1,
        'hour_sin': 0.866,
        'hour_cos': -0.5,
        'day_sin': -0.975,
        'day_cos': -0.222,
        'has_event': 0,
        'rest_mean_KPT': 22.5,
        'rest_p75_KPT': 28.0,
        'rest_mean_wait': 5.2,
        'orders_last_30min': 8,
        'avg_item_value': 150.0,
        'is_big_order': 0,
        'is_high_value_order': 1,
        'is_high_load': 0,
        'is_peak_weekend': 0,
        # Add subzone features (one-hot encoded)
        # Add weather features
        # ... (this is just an example)
    }
    
    # Note: This will fail unless all features are provided
    # Just demonstrating the API structure
    print("Sample order features:")
    for k, v in list(sample_order.items())[:10]:
        print(f"  {k}: {v}")
    print("  ... (and more features)")
    
    print("\nPrediction API loaded successfully!")
    print(f"Required features: {len(api.get_feature_names())}")
    print(f"Model: {api.get_model_info()['model_name']}")
    
    # Example 2: Model info
    print("\n" + "=" * 80)
    print("Example 2: Model Information")
    print("=" * 80)
    
    info = api.get_model_info()
    print(f"Model Name: {info['model_name']}")
    print(f"Number of Features: {info['n_features']}")
    print(f"Test MAE: {info['test_mae_minutes']:.3f} minutes")
    print(f"Test R²: {info['test_r2']:.4f}")
    print(f"Best Parameters: {info['best_params']}")
    
    # Example 3: Feature names
    print("\n" + "=" * 80)
    print("Example 3: Required Features (first 20)")
    print("=" * 80)
    
    features = api.get_feature_names()
    for i, feat in enumerate(features[:20], 1):
        print(f"{i:2d}. {feat}")
    print(f"... and {len(features) - 20} more features")
    
    print("\n" + "=" * 80)
    print("INFERENCE API DEMO COMPLETE")
    print("=" * 80)
    print("\nTo use this API:")
    print("1. Prepare your order data with all required features")
    print("2. Call api.predict(order_data) for predictions")
    print("3. Predictions are returned in minutes")
    print("\nFor batch predictions:")
    print("  df_with_predictions = api.predict_batch(df)")


if __name__ == "__main__":
    main()
