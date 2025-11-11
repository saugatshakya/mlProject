"""
API Test Examples
==================

Simple examples to test the API endpoints using Python requests.
Run these after starting the server (python app.py).
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health check endpoint."""
    print("\n" + "="*70)
    print("TESTING HEALTH ENDPOINT")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_dish_prediction_train():
    """Test dish prediction training."""
    print("\n" + "="*70)
    print("TESTING DISH PREDICTION TRAINING")
    print("="*70)
    
    # You need to have a CSV file ready
    files = {'file': open('../data/dish_hourly_aggregated.csv', 'rb')}
    
    response = requests.post(f"{BASE_URL}/api/dish_prediction/train", files=files)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_dish_prediction_predict():
    """Test dish prediction."""
    print("\n" + "="*70)
    print("TESTING DISH PREDICTION")
    print("="*70)
    
    data = {'hour': 12}
    
    response = requests.post(
        f"{BASE_URL}/api/dish_prediction/predict",
        headers={'Content-Type': 'application/json'},
        data=json.dumps(data)
    )
    print(f"Status: {response.status_code}")
    
    if response.ok:
        result = response.json()
        print(f"Hour: {result['hour']}")
        print(f"Total predicted orders: {result['total_predicted_orders']:.2f}")
        print(f"\nTop 5 dishes:")
        for i, pred in enumerate(result['predictions'][:5], 1):
            print(f"  {i}. {pred['dish']}: {pred['predicted_orders']:.2f}")


def test_demand_prediction_train():
    """Test demand prediction training."""
    print("\n" + "="*70)
    print("TESTING DEMAND PREDICTION TRAINING")
    print("="*70)
    
    # You need to have a CSV file ready
    files = {'file': open('../data/hourly_orders.csv', 'rb')}
    
    response = requests.post(f"{BASE_URL}/api/demand_prediction/train", files=files)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_demand_prediction_predict():
    """Test demand prediction."""
    print("\n" + "="*70)
    print("TESTING DEMAND PREDICTION")
    print("="*70)
    
    data = {'hours': 24}
    
    response = requests.post(
        f"{BASE_URL}/api/demand_prediction/predict",
        headers={'Content-Type': 'application/json'},
        data=json.dumps(data)
    )
    print(f"Status: {response.status_code}")
    
    if response.ok:
        result = response.json()
        print(f"Total predicted orders (24h): {result['total_predicted_orders']:.2f}")
        print(f"\nFirst 5 hours:")
        for pred in result['predictions'][:5]:
            print(f"  {pred['timestamp']}: {pred['predicted_orders']:.2f} orders")


def test_dish_recommend_train():
    """Test dish recommendation training."""
    print("\n" + "="*70)
    print("TESTING DISH RECOMMENDATION TRAINING")
    print("="*70)
    
    # You need to have a CSV file ready
    files = {'file': open('../data/data.csv', 'rb')}
    
    response = requests.post(f"{BASE_URL}/api/dish_recommend/train", files=files)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_dish_recommend_search():
    """Test dish search."""
    print("\n" + "="*70)
    print("TESTING DISH SEARCH")
    print("="*70)
    
    data = {'query': 'pizza'}
    
    response = requests.post(
        f"{BASE_URL}/api/dish_recommend/search",
        headers={'Content-Type': 'application/json'},
        data=json.dumps(data)
    )
    print(f"Status: {response.status_code}")
    
    if response.ok:
        result = response.json()
        print(f"Found {result['num_matches']} dishes")
        print(f"\nTop matches:")
        for dish in result['matches'][:5]:
            print(f"  - {dish}")


def test_dish_recommend_recommend():
    """Test dish recommendation."""
    print("\n" + "="*70)
    print("TESTING DISH RECOMMENDATION")
    print("="*70)
    
    data = {'dish_name': 'Pizza', 'top_n': 5}
    
    response = requests.post(
        f"{BASE_URL}/api/dish_recommend/recommend",
        headers={'Content-Type': 'application/json'},
        data=json.dumps(data)
    )
    print(f"Status: {response.status_code}")
    
    if response.ok:
        result = response.json()
        print(f"Recommendations for: {result['query_dish']}")
        print(f"\nTop {result['num_recommendations']} recommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec['dish']}")
            print(f"     Confidence: {rec['confidence']*100:.1f}%, Lift: {rec['lift']:.2f}x")


def test_dish_recommend_popular():
    """Test popular dishes."""
    print("\n" + "="*70)
    print("TESTING POPULAR DISHES")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/api/dish_recommend/popular?top_n=10")
    print(f"Status: {response.status_code}")
    
    if response.ok:
        result = response.json()
        print(f"\nTop {result['num_dishes']} popular dishes:")
        for i, dish in enumerate(result['dishes'], 1):
            print(f"  {i}. {dish['dish']} - {dish['popularity']}")


if __name__ == "__main__":
    print("="*70)
    print("ML2025 API TESTING SUITE")
    print("="*70)
    print("\nMake sure the server is running at http://localhost:5000")
    print("Press Enter to continue or Ctrl+C to cancel...")
    input()
    
    try:
        # Test health
        test_health()
        
        # Uncomment the tests you want to run:
        
        # DISH PREDICTION
        # test_dish_prediction_train()  # Run this first
        # test_dish_prediction_predict()
        
        # DEMAND PREDICTION
        # test_demand_prediction_train()  # Run this first
        # test_demand_prediction_predict()
        
        # DISH RECOMMENDATION
        # test_dish_recommend_train()  # Run this first
        # test_dish_recommend_search()
        # test_dish_recommend_recommend()
        # test_dish_recommend_popular()
        
        print("\n" + "="*70)
        print("TESTING COMPLETE")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to server.")
        print("Make sure the server is running: python app.py")
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Make sure the data files exist in ../data/")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
