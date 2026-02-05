"""
Test script for the Customer Clustering API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_home():
    """Test home endpoint"""
    print("\n" + "=" * 60)
    print("Testing Home Endpoint")
    print("=" * 60)
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_health():
    """Test health check endpoint"""
    print("\n" + "=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_clusters():
    """Test clusters info endpoint"""
    print("\n" + "=" * 60)
    print("Testing Clusters Info Endpoint")
    print("=" * 60)
    response = requests.get(f"{BASE_URL}/clusters")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_prediction(customer_data, customer_name):
    """Test prediction endpoint"""
    print("\n" + "=" * 60)
    print(f"Testing Prediction for {customer_name}")
    print("=" * 60)
    print(f"Input Data: {json.dumps(customer_data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Prediction Result:")
        print(f"  • Cluster: {result['cluster']}")
        print(f"  • Customer Type: {result['customer_type']}")
        print(f"  • Description: {result['description']}")
        print(f"  • Priority: {result['priority']}")
        print(f"  • Discount Range: {result['discount_range']}")
        print(f"\n  Suggested Offers:")
        for i, offer in enumerate(result['suggested_offers'], 1):
            print(f"    {i}. {offer}")
    else:
        print(f" Error: {response.json()}")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("CUSTOMER CLUSTERING API TESTS")
    print("=" * 60)
    
    try:
        # Test basic endpoints
        test_home()
        test_health()
        test_clusters()
        
        # Test predictions with different customer profiles
        
        # High-value customer
        high_value_customer = {
            "Age": 42,
            "Gender": "M",
            "AnnualIncome": 1200000,
            "TotalSpent": 850000,
            "MonthlyPurchases": 18,
            "AvgOrderValue": 12000,
            "AppTimeMinutes": 120,
            "DiscountUsage": "Low",
            "PreferredShoppingTime": "Night"
        }
        test_prediction(high_value_customer, "High-Value Customer")
        
        # Mid-tier customer
        regular_customer = {
            "Age": 32,
            "Gender": "F",
            "AnnualIncome": 600000,
            "TotalSpent": 250000,
            "MonthlyPurchases": 8,
            "AvgOrderValue": 5000,
            "AppTimeMinutes": 60,
            "DiscountUsage": "Medium",
            "PreferredShoppingTime": "Day"
        }
        test_prediction(regular_customer, "Regular Customer")
        
        # Price-sensitive customer
        price_sensitive_customer = {
            "Age": 24,
            "Gender": "F",
            "AnnualIncome": 250000,
            "TotalSpent": 45000,
            "MonthlyPurchases": 2,
            "AvgOrderValue": 1500,
            "AppTimeMinutes": 25,
            "DiscountUsage": "High",
            "PreferredShoppingTime": "Day"
        }
        test_prediction(price_sensitive_customer, "Price-Sensitive Customer")
        
        print("\n" + "=" * 60)
        print("✓ All Tests Completed Successfully!")
        print("=" * 60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n Error: Cannot connect to API server.")
        print("Please make sure the API is running at http://localhost:8000")
        print("Run: python app.py")
    except Exception as e:
        print(f"\n Error: {str(e)}")


if __name__ == "__main__":
    run_all_tests()
