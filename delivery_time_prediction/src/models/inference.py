"""
Inference Module for Delivery Time Prediction
=============================================

This module provides inference capabilities for predicting delivery times
using trained models.

Author: Saugat Shakya
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import logging
import joblib

logger = logging.getLogger(__name__)


class DeliveryTimePredictor:
    """
    Production-ready predictor for delivery times.
    
    Usage:
        predictor = DeliveryTimePredictor()
        predictor.load_model('models/final/xgboost_model.pkl')
        prediction = predictor.predict(features_df)
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model file
        """
        self.model = None
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
        
        logger.info("DeliveryTimePredictor initialized")
    
    def load_model(self, model_path: Union[str, Path]):
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to .pkl model file
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        
        # Try to get feature names
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_names = list(self.model.feature_names_in_)
            logger.info(f"Model expects {len(self.feature_names)} features")
        
        logger.info("Model loaded successfully")
    
    def predict(
        self,
        X: pd.DataFrame,
        return_confidence: bool = False
    ) -> Union[np.ndarray, Dict]:
        """
        Predict delivery times.
        
        Args:
            X: DataFrame with features
            return_confidence: If True, also return prediction intervals
            
        Returns:
            Array of predictions, or dict with predictions and confidence
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Validate features
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Reorder columns to match training
            X = X[self.feature_names]
        
        # Make predictions
        predictions = self.model.predict(X)
        
        if not return_confidence:
            return predictions
        
        # Calculate confidence intervals (if possible)
        # For tree-based models, we can use std of trees
        if hasattr(self.model, 'estimators_'):
            # Ensemble model
            all_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            std = np.std(all_predictions, axis=0)
            
            return {
                'predictions': predictions,
                'lower_bound': predictions - 2 * std,
                'upper_bound': predictions + 2 * std,
                'std': std
            }
        else:
            return {'predictions': predictions}
    
    def predict_single(self, features: Dict) -> float:
        """
        Predict delivery time for a single order.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Predicted delivery time in minutes
        """
        # Convert dict to DataFrame
        X = pd.DataFrame([features])
        
        # Make prediction
        prediction = self.predict(X)
        
        return float(prediction[0])
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importances from the model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_")
            return pd.DataFrame()
        
        importances = self.model.feature_importances_
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importances))]
        
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        df_importance = df_importance.sort_values('importance', ascending=False).head(top_n)
        
        return df_importance
    
    def explain_prediction(self, X: pd.DataFrame, index: int = 0):
        """
        Explain a single prediction (requires SHAP).
        
        Args:
            X: DataFrame with features
            index: Index of sample to explain
        """
        try:
            import shap
        except ImportError:
            logger.error("SHAP not installed. Install with: pip install shap")
            return
        
        if self.model is None:
            raise ValueError("No model loaded")
        
        # Create explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X.iloc[[index]])
        
        # Plot
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X.iloc[index],
                feature_names=X.columns.tolist()
            )
        )
