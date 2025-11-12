"""
ML2025 Multi-Model Web Application
===================================

A unified web interface for five ML models:
1. Dish Prediction - Multi-output regression for dish demand forecasting
2. Demand Prediction - Hourly order volume prediction (Enhanced XGBoost)
3. Dish Recommendation - Association rules for dish recommendations
4. Prep Time Prediction - Kitchen preparation time forecasting
5. Promotion Effectiveness - Promotion impact on orders and sales

Author: Saugat Shakya
Date: 2025-11-11
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
from models_prep_time_prediction import PrepTimePredictionModel
from models_promotion_effectiveness import PromotionEffectivenessModel

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure directories exist
Path('uploads').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)
Path('static').mkdir(exist_ok=True)
Path('uploads/original_data').mkdir(exist_ok=True)

def prepare_original_data_if_needed():
    """Automatically prepare original data if it doesn't exist."""
    import subprocess
    import sys
    
    required_files = [
        Path('uploads/original_data/dish_prediction.csv'),
        Path('uploads/original_data/demand_prediction.csv'), 
        Path('uploads/original_data/dish_recommendation.csv'),
        Path('uploads/original_data/prep_time_prediction.csv'),
        Path('uploads/original_data/promotion_effectiveness.csv')
    ]
    
    missing_files = [str(f) for f in required_files if not f.exists()]
    
    if missing_files:
        print(f"Missing original data files: {missing_files}")
        print("Preparing original data automatically...")
        
        try:
            # Run the prepare_original_data.py script
            script_path = Path(__file__).parent / 'prepare_original_data.py'
            print(f"Running script: {script_path}")
            result = subprocess.run([sys.executable, str(script_path)], 
                                  capture_output=True, text=True, cwd=str(Path(__file__).parent))
            
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print(f"STDOUT: {result.stdout[-500:]}")  # Last 500 chars
            if result.stderr:
                print(f"STDERR: {result.stderr[-500:]}")  # Last 500 chars
            
            if result.returncode == 0:
                print("✓ Original data prepared successfully")
                return True
            else:
                print(f"✗ Failed to prepare original data (return code: {result.returncode})")
                return False
                
        except Exception as e:
            print(f"✗ Exception during data preparation: {e}")
            return False
    else:
        print("✓ Original data files already exist")
        return True

# Prepare original data if needed
print("Checking original data...")
prepare_original_data_if_needed()

# Route to serve uploaded files
@app.route('/uploads/<path:filename>', methods=['GET'])
def serve_upload(filename):
    """Serve uploaded files."""
    return send_file(Path('uploads') / filename)

# Global model instances
dish_pred_model = DishPredictionModel()
demand_pred_model = DemandPredictionModel()
dish_rec_model = DishRecommendationModel()
prep_time_model = PrepTimePredictionModel()
promo_effect_model = PromotionEffectivenessModel()

# Load saved models on startup
print("Loading saved models...")
try:
    dish_pred_model.load()
except AttributeError:
    pass
try:
    demand_pred_model.load()
except AttributeError:
    pass
try:
    dish_rec_model.load()
except AttributeError:
    pass
try:
    prep_time_model.load()
except AttributeError:
    pass
try:
    promo_effect_model.load()
except AttributeError:
    pass


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
# PREP TIME PREDICTION ENDPOINTS
# ============================================================================

@app.route('/api/prep_time/train', methods=['POST'])
def train_prep_time():
    """Train prep time prediction model from uploaded CSV."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filepath = Path('uploads') / f'prep_time_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        file.save(filepath)
        
        # Train model
        result = prep_time_model.train(str(filepath))
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/prep_time/predict', methods=['POST'])
def predict_prep_time():
    """Predict prep time for an order."""
    try:
        if not prep_time_model.is_trained():
            return jsonify({'error': 'Model not trained yet'}), 400
        
        data = request.get_json()
        
        result = prep_time_model.predict(data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/prep_time/status', methods=['GET'])
def status_prep_time():
    """Get model status."""
    return jsonify(prep_time_model.get_status())


# ============================================================================
# PROMOTION EFFECTIVENESS ENDPOINTS
# ============================================================================

@app.route('/api/promotion/train', methods=['POST'])
def train_promotion():
    """Train promotion effectiveness model from uploaded CSV."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filepath = Path('uploads') / f'promotion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        file.save(filepath)
        
        # Train model
        result = promo_effect_model.train(str(filepath))
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/promotion/predict', methods=['POST'])
def predict_promotion():
    """Predict promotion impact."""
    try:
        if not promo_effect_model.is_trained():
            return jsonify({'error': 'Model not trained yet'}), 400
        
        data = request.get_json()
        
        result = promo_effect_model.predict_promotion_impact(data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/promotion/status', methods=['GET'])
def status_promotion():
    """Get model status."""
    return jsonify(promo_effect_model.get_status())


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
            'dish_recommendation': dish_rec_model.is_trained(),
            'prep_time_prediction': prep_time_model.is_trained(),
            'promotion_effectiveness': promo_effect_model.is_trained()
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
        df.to_csv(filepath, index=False, quoting=1)
        
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
        df.to_csv(filepath, index=False, quoting=1)
        
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
                'items': items
            })
        
        df = pd.DataFrame(data_rows)
        
        # Save to uploads
        filepath = Path('uploads') / f'generated_orders_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(filepath, index=False, quoting=1)
        
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


@app.route('/api/generate/prep_time_data', methods=['POST'])
def generate_prep_time_data():
    """Generate synthetic prep time prediction data."""
    try:
        data = request.get_json() or {}
        rows = int(data.get('rows', 1000))
        
        end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        timestamps = pd.date_range(end=end_time, periods=rows, freq='H')
        
        data_rows = []
        for ts in timestamps:
            hour = ts.hour
            weekday = ts.weekday()
            
            # Base prep time
            base_prep_time = 15
            
            # Peak hour adjustments (kitchen-relevant)
            if 11 <= hour <= 14 or 18 <= hour <= 21:
                base_prep_time *= 1.3
            
            # Order complexity (kitchen-relevant)
            num_items = np.random.poisson(2.5) + 1
            base_prep_time *= (1 + num_items * 0.1)
            
            # Weekend adjustment (kitchen staffing)
            if weekday >= 5:
                base_prep_time *= 1.1
            
            # Add randomness
            prep_time = np.random.normal(base_prep_time, 3.0)
            prep_time = max(5, min(prep_time, 60))  # Clamp between 5-60 minutes
            
            # Generate items string (kitchen-relevant)
            dishes = ["Burger", "Pizza", "Fries", "Salad", "Pasta", "Sandwich", "Wrap"]
            items = []
            for _ in range(num_items):
                dish = random.choice(dishes)
                qty = random.choices([1, 2], weights=[0.8, 0.2])[0]
                items.append(f"{qty} x {dish}")
            items_str = ", ".join(items)
            
            data_rows.append({
                'timestamp': ts,
                'KPT duration (minutes)': prep_time,
                'Items in order': items_str,
                'Order Status': 'Delivered'  # Only kitchen-relevant categorical
            })
        
        df = pd.DataFrame(data_rows)
        
        # Save to uploads
        filepath = Path('uploads') / f'generated_prep_time_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(filepath, index=False, quoting=1)
        
        return jsonify({
            'status': 'success',
            'filepath': str(filepath),
            'rows': len(df),
            'columns': df.columns.tolist(),
            'sample': df.head(5).to_dict(orient='records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/promotion_data', methods=['POST'])
def generate_promotion_data():
    """Generate synthetic promotion effectiveness data."""
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
            base_orders = 12
            base_sales = 4800
            
            # Time factors
            if 11 <= hour <= 14 or 18 <= hour <= 21:
                base_orders *= 1.8
                base_sales *= 1.9
            elif hour < 8 or hour > 22:
                base_orders *= 0.3
                base_sales *= 0.3
            
            # Weekend boost
            if weekday >= 5:
                base_orders *= 1.4
                base_sales *= 1.5
            
            # Promotion effects
            promotion_types = ['no_promo', 'discount_10', 'discount_20', 'free_delivery', 'combo_deal']
            promo_type = random.choice(promotion_types)
            
            if promo_type != 'no_promo':
                # Promotion boosts demand
                promo_boost = {
                    'discount_10': 1.2,
                    'discount_20': 1.4,
                    'free_delivery': 1.3,
                    'combo_deal': 1.5
                }
                boost = promo_boost[promo_type]
                orders = np.random.poisson(base_orders * boost)
                sales = orders * np.random.normal(420, 50)  # Average order value ~420
            else:
                orders = np.random.poisson(base_orders)
                sales = orders * np.random.normal(380, 40)  # Average order value ~380
            
            # Add randomness
            orders = max(0, orders + np.random.randint(-2, 3))
            sales = max(0, sales + np.random.normal(0, 200))
            
            data_rows.append({
                'timestamp': ts,
                'orders_per_hour': orders,
                'sales_per_hour': sales,
                'subtotal': sales * 0.9,  # Subtotal is 90% of total
                'total': sales,
                'promotion_type': promo_type
            })
        
        df = pd.DataFrame(data_rows)
        
        # Save to uploads
        filepath = Path('uploads') / f'generated_promotion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(filepath, index=False, quoting=1)
        
        return jsonify({
            'status': 'success',
            'filepath': str(filepath),
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
            return jsonify({'error': 'Original data file not found. Please try again or contact support.'}), 404
        
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
            return jsonify({'error': 'Original data file not found. Please try again or contact support.'}), 404
        
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
def use_original_recommendation_data():
    """Use original dish recommendation data."""
    try:
        source_path = Path('uploads/original_data/dish_recommendation.csv')
        
        if not source_path.exists():
            return jsonify({'error': 'Original data file not found. Please try again or contact support.'}), 404
        
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


@app.route('/api/use_original/prep_time_prediction', methods=['GET'])
def use_original_prep_time_data():
    """Use original prep time prediction data."""
    try:
        source_path = Path('uploads/original_data/prep_time_prediction.csv')
        
        if not source_path.exists():
            return jsonify({'error': 'Original data file not found. Please try again or contact support.'}), 404
        
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


@app.route('/api/use_original/promotion_effectiveness', methods=['GET'])
def use_original_promotion_data():
    """Use original promotion effectiveness data."""
    try:
        source_path = Path('uploads/original_data/promotion_effectiveness.csv')
        
        if not source_path.exists():
            return jsonify({'error': 'Original data file not found. Please try again or contact support.'}), 404
        
        # Read and return info
        df = pd.read_csv(source_path)
        
        return jsonify({
            'status': 'success',
            'filepath': str(source_path),
            'records': len(df),
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


if __name__ == '__main__':
    print("="*70)
    print("ML2025 MULTI-MODEL WEB APPLICATION")
    print("="*70)
    print("\nStarting server...")
    print("Available models:")
    print("  1. Dish Prediction (Multi-output regression)")
    print("  2. Demand Prediction (Hourly order volume - Enhanced XGBoost)")
    print("  3. Dish Recommendation (Association rules)")
    print("  4. Prep Time Prediction (Kitchen preparation time)")
    print("  5. Promotion Effectiveness (Promotion impact analysis)")
    print("\nAccess the app at: http://localhost:5001")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
