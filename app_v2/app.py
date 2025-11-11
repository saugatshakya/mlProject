"""
ML2025 Multi-Model Web Application
===================================

A unified web interface for three ML models:
1. Dish Prediction - Multi-output regression for dish demand forecasting
2. Demand Prediction - Hourly order volume prediction
3. Dish Recommendation - Association rules for dish recommendations

Author: Saugat Shakya
Date: 2025-11-09
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import pickle
import tempfile
from datetime import datetime, timedelta
import random

# Import model modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models_dish_prediction import DishPredictionModel
from models_demand_prediction import DemandPredictionModel
from models_dish_recommend import DishRecommendationModel

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure directories exist
Path('uploads').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)
Path('static').mkdir(exist_ok=True)

# Global model instances
dish_pred_model = DishPredictionModel()
demand_pred_model = DemandPredictionModel()
dish_rec_model = DishRecommendationModel()


@app.route('/')
def index():
    """Main page with tabs for all three models."""
    return render_template('index.html')


# ============================================================================
# DISH PREDICTION ENDPOINTS
# ============================================================================

@app.route('/api/dish_prediction/train', methods=['POST'])
def train_dish_prediction():
    """Train dish prediction model from uploaded CSV."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filepath = Path('uploads') / f'dish_pred_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        file.save(filepath)
        
        # Train model
        result = dish_pred_model.train(str(filepath))
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dish_prediction/predict', methods=['POST'])
def predict_dish_prediction():
    """Get predictions for N hours ahead."""
    try:
        if not dish_pred_model.is_trained:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        data = request.get_json()
        hours_ahead = data.get('hours_ahead', None)
        
        result = dish_pred_model.predict(hours_ahead)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dish_prediction/status', methods=['GET'])
def status_dish_prediction():
    """Get model status."""
    return jsonify(dish_pred_model.get_status())


# ============================================================================
# DEMAND PREDICTION ENDPOINTS
# ============================================================================

@app.route('/api/demand_prediction/train', methods=['POST'])
def train_demand_prediction():
    """Train demand prediction model from uploaded CSV."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filepath = Path('uploads') / f'demand_pred_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        file.save(filepath)
        
        # Train model
        result = demand_pred_model.train(str(filepath))
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/demand_prediction/predict', methods=['POST'])
def predict_demand_prediction():
    """Get predictions for next N hours."""
    try:
        if not demand_pred_model.is_trained():
            return jsonify({'error': 'Model not trained yet'}), 400
        
        data = request.get_json()
        hours = data.get('hours', 1)
        
        result = demand_pred_model.predict(hours)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/demand_prediction/status', methods=['GET'])
def status_demand_prediction():
    """Get model status."""
    return jsonify(demand_pred_model.get_status())


# ============================================================================
# DISH RECOMMENDATION ENDPOINTS
# ============================================================================

@app.route('/api/dish_recommend/train', methods=['POST'])
def train_dish_recommend():
    """Train dish recommendation model from uploaded CSV."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filepath = Path('uploads') / f'dish_rec_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        file.save(filepath)
        
        # Train model
        result = dish_rec_model.train(str(filepath))
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error training dish recommendation model:\n{error_trace}")
        return jsonify({'error': str(e), 'traceback': error_trace}), 500


@app.route('/api/dish_recommend/recommend', methods=['POST'])
def recommend_dishes():
    """Get recommendations for a dish."""
    try:
        if not dish_rec_model.is_trained:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        data = request.get_json()
        dish_name = data.get('dish_name', '')
        top_n = data.get('top_n', 5)
        
        result = dish_rec_model.recommend(dish_name, top_n)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dish_recommend/available_dishes', methods=['GET'])
def get_available_dishes():
    """Get all available dishes from trained model."""
    try:
        if not dish_rec_model.is_trained:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        # Get all dishes and sort alphabetically
        dishes = sorted(list(dish_rec_model.all_dishes))
        
        return jsonify({
            'dishes': dishes,
            'count': len(dishes)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dish_recommend/search', methods=['POST'])
def search_dishes():
    """Search for dishes."""
    try:
        if not dish_rec_model.is_trained:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        data = request.get_json()
        query = data.get('query', '')
        
        result = dish_rec_model.search(query)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dish_recommend/popular', methods=['GET'])
def popular_dishes():
    """Get popular dishes."""
    try:
        if not dish_rec_model.is_trained:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        top_n = request.args.get('top_n', 20, type=int)
        
        result = dish_rec_model.get_popular(top_n)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dish_recommend/status', methods=['GET'])
def status_dish_recommend():
    """Get model status."""
    return jsonify(dish_rec_model.get_status())


# ============================================================================
# GENERAL ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'models': {
            'dish_prediction': dish_pred_model.is_trained(),
            'demand_prediction': demand_pred_model.is_trained(),
            'dish_recommendation': dish_rec_model.is_trained()
        }
    })


# ============================================================================
# DATA GENERATION ENDPOINTS
# ============================================================================

@app.route('/api/generate/dish_data', methods=['POST'])
def generate_dish_data():
    """Generate synthetic dish prediction data."""
    try:
        data = request.get_json() or {}
        rows = int(data.get('rows', 1008))
        
        # Generate realistic hourly data
        all_dishes = [
            "Burger", "Pizza", "Coke", "Fries", "Salad",
            "Pasta", "IceCream", "Sushi", "Sandwich", "Soup",
            "Wrap", "Taco", "Nuggets", "Donut", "Tea", "Smoothie"
        ]
        
        num_dishes = random.randint(4, 8)
        dishes = random.sample(all_dishes, k=num_dishes)
        base_popularity = list(np.random.randint(1, 6, size=len(dishes)))
        
        end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        timestamps = pd.date_range(end=end_time, periods=rows, freq='H')
        
        data_rows = []
        for ts in timestamps:
            row = {'timestamp': ts}
            hour = ts.hour
            weekday = ts.weekday()
            weekend_factor = 1.2 if weekday >= 5 else 1.0
            
            for idx, dish in enumerate(dishes):
                popularity = base_popularity[idx]
                
                if 11 <= hour <= 14 or 18 <= hour <= 21:
                    popularity_effective = popularity * 1.5
                else:
                    popularity_effective = popularity
                
                if hour < 8 or hour > 22:
                    popularity_effective *= 0.5
                
                popularity_effective *= weekend_factor
                qty = np.random.poisson(lam=max(0.1, popularity_effective))
                
                if random.random() < 0.1:
                    qty += random.randint(1, 3)
                if random.random() < 0.02:
                    qty += random.randint(5, 10)
                
                row[dish] = min(qty, 20)
            
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        
        # Save to uploads
        filepath = Path('uploads') / f'generated_dish_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(filepath, index=False)
        
        return jsonify({
            'status': 'success',
            'filepath': str(filepath),
            'rows': len(df),
            'columns': df.columns.tolist(),
            'dishes': dishes,
            'sample': df.head(5).to_dict(orient='records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/demand_data', methods=['POST'])
def generate_demand_data():
    """Generate synthetic demand prediction data."""
    try:
        data = request.get_json() or {}
        rows = int(data.get('rows', 1008))
        
        end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        timestamps = pd.date_range(end=end_time, periods=rows, freq='H')
        
        data_rows = []
        for ts in timestamps:
            hour = ts.hour
            weekday = ts.weekday()
            
            # Base demand
            base_demand = 15
            
            # Peak hours
            if 11 <= hour <= 14 or 18 <= hour <= 21:
                base_demand *= 2.0
            elif hour < 8 or hour > 22:
                base_demand *= 0.4
            
            # Weekend boost
            if weekday >= 5:
                base_demand *= 1.3
            
            # Add randomness
            total_orders = np.random.poisson(lam=base_demand)
            
            # Random spikes
            if random.random() < 0.1:
                total_orders += random.randint(5, 15)
            
            data_rows.append({
                'timestamp': ts,
                'total_orders': max(0, total_orders)
            })
        
        df = pd.DataFrame(data_rows)
        
        # Save to uploads
        filepath = Path('uploads') / f'generated_demand_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(filepath, index=False)
        
        return jsonify({
            'status': 'success',
            'filepath': str(filepath),
            'rows': len(df),
            'columns': df.columns.tolist(),
            'sample': df.head(5).to_dict(orient='records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/order_data', methods=['POST'])
def generate_order_data():
    """Generate synthetic order data for recommendations."""
    try:
        data = request.get_json() or {}
        num_orders = int(data.get('orders', 1000))
        
        dishes = [
            "Burger", "Pizza", "Coke", "Fries", "Salad",
            "Pasta", "IceCream", "Sushi", "Sandwich", "Soup",
            "Wrap", "Taco", "Nuggets", "Donut", "Tea", "Smoothie"
        ]
        
        # Create association patterns
        combos = [
            ["Burger", "Fries", "Coke"],
            ["Pizza", "Coke"],
            ["Sushi", "Tea"],
            ["Pasta", "Salad"],
            ["Taco", "Wrap"],
            ["IceCream", "Donut"],
        ]
        
        data_rows = []
        for order_id in range(1, num_orders + 1):
            # Randomly choose combo or individual items
            if random.random() < 0.6:  # 60% combos
                items = random.choice(combos).copy()
                # Sometimes add extra item
                if random.random() < 0.3:
                    items.append(random.choice(dishes))
            else:
                # Individual items
                num_items = random.choices([1, 2, 3, 4], weights=[0.3, 0.4, 0.2, 0.1])[0]
                items = random.sample(dishes, k=num_items)
            
            data_rows.append({
                'order_id': order_id,
                'items': ', '.join([f"1 x {item}" for item in items])
            })
        
        df = pd.DataFrame(data_rows)
        
        # Save to uploads
        filepath = Path('uploads') / f'generated_orders_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(filepath, index=False)
        
        return jsonify({
            'status': 'success',
            'filepath': str(filepath),
            'orders': len(df),
            'rows': len(df),
            'columns': df.columns.tolist(),
            'sample': df.head(5).to_dict(orient='records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# USE ORIGINAL DATA ENDPOINTS
# ============================================================================

@app.route('/api/use_original/dish_prediction', methods=['GET'])
def use_original_dish_data():
    """Use original dish prediction data."""
    try:
        source_path = Path('uploads/original_data/dish_prediction.csv')
        
        if not source_path.exists():
            return jsonify({'error': 'Original data not found. Run prepare_original_data.py first.'}), 404
        
        # Read and return info
        df = pd.read_csv(source_path)
        
        # Get dish columns (exclude timestamp)
        dish_columns = [col for col in df.columns if col != 'timestamp']
        
        return jsonify({
            'status': 'success',
            'filepath': str(source_path),
            'rows': len(df),
            'dishes': dish_columns,
            'columns': df.columns.tolist(),
            'sample': df.head(5).to_dict(orient='records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/use_original/demand_prediction', methods=['GET'])
def use_original_demand_data():
    """Use original demand prediction data."""
    try:
        source_path = Path('uploads/original_data/demand_prediction.csv')
        
        if not source_path.exists():
            return jsonify({'error': 'Original data not found. Run prepare_original_data.py first.'}), 404
        
        # Read and return info
        df = pd.read_csv(source_path)
        
        return jsonify({
            'status': 'success',
            'filepath': str(source_path),
            'rows': len(df),
            'columns': df.columns.tolist(),
            'sample': df.head(5).to_dict(orient='records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/use_original/dish_recommendation', methods=['GET'])
def use_original_order_data():
    """Use original dish recommendation data."""
    try:
        source_path = Path('uploads/original_data/dish_recommendation.csv')
        
        if not source_path.exists():
            return jsonify({'error': 'Original data not found. Run prepare_original_data.py first.'}), 404
        
        # Read and return info
        df = pd.read_csv(source_path)
        
        return jsonify({
            'status': 'success',
            'filepath': str(source_path),
            'orders': len(df),
            'rows': len(df),
            'columns': df.columns.tolist(),
            'sample': df.head(5).to_dict(orient='records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview_csv', methods=['POST'])
def preview_csv():
    """Preview uploaded CSV file."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        return jsonify({
            'status': 'success',
            'rows': len(df),
            'columns': df.columns.tolist(),
            'sample': df.head(10).to_dict(orient='records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<path:filename>', methods=['GET'])
def serve_upload(filename):
    """Serve uploaded/generated files."""
    return send_file(Path('uploads') / filename)


if __name__ == '__main__':
    print("="*70)
    print("ML2025 MULTI-MODEL WEB APPLICATION")
    print("="*70)
    print("\nStarting server...")
    print("Available models:")
    print("  1. Dish Prediction (Multi-output regression)")
    print("  2. Demand Prediction (Hourly order volume)")
    print("  3. Dish Recommendation (Association rules)")
    print("\nAccess the app at: http://localhost:5001")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
