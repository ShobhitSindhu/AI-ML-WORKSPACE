"""
FastAPI Application for Customer Clustering Prediction
This API accepts customer details and returns their cluster assignment with personalized offers
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import pickle
import numpy as np
from typing import Dict, List, Any
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Customer Clustering API",
    description="Predicts customer cluster and provides personalized offers",
    version="1.0.0"
)

# Load the trained model and preprocessing objects
try:
    with open('customer_clustering_model.pkl', 'rb') as f:
        model_artifacts = pickle.load(f)
    
    kmeans_model = model_artifacts['kmeans_model']
    scaler = model_artifacts['scaler']
    le_gender = model_artifacts['le_gender']
    le_time = model_artifacts['le_time']
    le_discount = model_artifacts['le_discount']
    features = model_artifacts['features']
    cluster_labels = model_artifacts['cluster_labels']
    
    print("✓ Model and artifacts loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Define business offers for each cluster
CLUSTER_OFFERS = {
    "High-Value Loyal Customers": {
        "description": "Premium customers with high income and spending power",
        "offers": [
            "Exclusive early access to new products",
            "Premium membership with free express delivery",
            "VIP customer service hotline",
            "Personalized product recommendations",
            "Complimentary gift wrapping"
        ],
        "discount_range": "5-10%",
        "priority": "Highest"
    },
    "Value-Seeking Regular Customers": {
        "description": "Regular customers seeking good value for money",
        "offers": [
            "Festival discounts (10-15%)",
            "Loyalty reward points on every purchase",
            "Birthday special offers",
            "Referral bonuses",
            "Member-exclusive deals"
        ],
        "discount_range": "10-15%",
        "priority": "Medium"
    },
    "Price-Sensitive Occasional Customers": {
        "description": "Budget-conscious customers who shop occasionally",
        "offers": [
            "Flash sales and coupon-based discounts",
            "Free shipping on minimum order value",
            "Bundle deals and combos",
            "Clearance sale notifications",
            "First-time purchase discount"
        ],
        "discount_range": "15-25%",
        "priority": "Standard"
    }
}


# Define request model
class CustomerData(BaseModel):
    """Customer data input model"""
    Age: int = Field(..., ge=18, le=100, description="Customer age (18-100)")
    Gender: str = Field(..., description="Customer gender: 'M' or 'F'")
    AnnualIncome: float = Field(..., gt=0, description="Annual income in dollars")
    TotalSpent: float = Field(..., ge=0, description="Total amount spent in last year")
    MonthlyPurchases: int = Field(..., ge=0, le=100, description="Number of purchases per month")
    AvgOrderValue: float = Field(..., ge=0, description="Average order value in dollars")
    AppTimeMinutes: float = Field(..., ge=0, le=1440, description="Time spent on app per day (minutes)")
    DiscountUsage: str = Field(..., description="Discount usage: 'Low', 'Medium', or 'High'")
    PreferredShoppingTime: str = Field(..., description="Preferred shopping time: 'Day' or 'Night'")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Age": 35,
                "Gender": "M",
                "AnnualIncome": 850000,
                "TotalSpent": 500000,
                "MonthlyPurchases": 12,
                "AvgOrderValue": 8000,
                "AppTimeMinutes": 90,
                "DiscountUsage": "Low",
                "PreferredShoppingTime": "Night"
            }
        }
    )


class PredictionResponse(BaseModel):
    """Prediction response model"""
    cluster: int
    customer_type: str
    description: str
    suggested_offers: List[str]
    discount_range: str
    priority: str
    customer_profile: Dict[str, Any]


@app.get("/")
def home():
    """Home endpoint"""
    return {
        "message": "Customer Clustering API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict customer cluster",
            "/health": "GET - Check API health",
            "/clusters": "GET - Get cluster information"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "clusters": len(cluster_labels)
    }


@app.get("/clusters")
def get_clusters():
    """Get information about all clusters"""
    cluster_info = {}
    for cluster_id, cluster_name in cluster_labels.items():
        cluster_info[f"Cluster {cluster_id}"] = {
            "name": cluster_name,
            **CLUSTER_OFFERS[cluster_name]
        }
    return cluster_info


@app.post("/predict", response_model=PredictionResponse)
def predict_cluster(customer: CustomerData):
    """
    Predict customer cluster and return personalized offers
    
    Args:
        customer: Customer data
        
    Returns:
        Cluster assignment and personalized offers
    """
    try:
        # Validate and encode categorical features
        gender_encoded = le_gender.transform([customer.Gender])[0]
        time_encoded = le_time.transform([customer.PreferredShoppingTime])[0]
        discount_encoded = le_discount.transform([customer.DiscountUsage])[0]
        
        # Create feature vector in the same order as training
        feature_vector = np.array([[
            customer.Age,
            gender_encoded,
            customer.AnnualIncome,
            customer.TotalSpent,
            customer.MonthlyPurchases,
            customer.AvgOrderValue,
            customer.AppTimeMinutes,
            discount_encoded,
            time_encoded
        ]])
        
        # Scale the features
        feature_scaled = scaler.transform(feature_vector)
        
        # Predict cluster
        cluster_id = int(kmeans_model.predict(feature_scaled)[0])
        customer_type = cluster_labels[cluster_id]
        
        # Get offers for this cluster
        offers_info = CLUSTER_OFFERS[customer_type]
        
        # Create response
        response = PredictionResponse(
            cluster=cluster_id,
            customer_type=customer_type,
            description=offers_info["description"],
            suggested_offers=offers_info["offers"],
            discount_range=offers_info["discount_range"],
            priority=offers_info["priority"],
            customer_profile={
                "age": customer.Age,
                "gender": customer.Gender,
                "annual_income": f"${customer.AnnualIncome:,.2f}",
                "total_spent": f"${customer.TotalSpent:,.2f}",
                "monthly_purchases": customer.MonthlyPurchases,
                "avg_order_value": f"${customer.AvgOrderValue:,.2f}",
                "app_time_minutes": customer.AppTimeMinutes,
                "discount_usage": customer.DiscountUsage,
                "preferred_shopping_time": customer.PreferredShoppingTime
            }
        )
        
        return response
        
    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input value: {str(ve)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict_batch")
def predict_batch(customers: List[CustomerData]):
    """
    Predict clusters for multiple customers at once
    
    Args:
        customers: List of customer data
        
    Returns:
        List of predictions
    """
    predictions = []
    for customer in customers:
        try:
            prediction = predict_cluster(customer)
            predictions.append(prediction)
        except HTTPException as e:
            predictions.append({"error": e.detail})
    
    return {
        "total_customers": len(customers),
        "predictions": predictions
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting Customer Clustering API Server...")
    print("=" * 60)
    print("\nAPI Documentation available at: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
