# Customer Clustering with Business Offers

An unsupervised learning project that segments customers into clusters and provides personalized marketing offers using K-Means clustering.

## 📋 Project Overview

This project implements a complete end-to-end machine learning solution for customer segmentation:
1. **Data Analysis & Preprocessing**
2. **K-Means Clustering**
3. **Visualization & Insights**
4. **Model Deployment with FastAPI**

## 🎯 Business Problem

An online retail company wants to improve its marketing strategy by providing personalized offers to customers based on their behavior, spending patterns, and demographics.

## 📊 Dataset Features

- **Age**: Customer age
- **Gender**: Male/Female
- **City/Region**: Customer location
- **Annual Income**: Yearly income in dollars
- **Total Spent**: Total amount spent in the last year
- **Monthly Purchases**: Number of purchases per month
- **Average Order Value**: Average spending per order
- **App Time Minutes**: Time spent on mobile app (minutes/day)
- **Discount Usage**: Frequency of discount usage (Low/Medium/High)
- **Preferred Shopping Time**: Day/Night

## 🔍 Clustering Results

### Cluster 0: High-Value Loyal Customers (61% of customers)
**Characteristics:**
- High annual income
- Very high total spending (~$289,836)
- High average order value
- Frequent purchases
- Low discount usage
- Moderate app engagement (71.6 min/day)

**Suggested Offers:**
- ✓ Exclusive early access to new products
- ✓ Premium membership with free express delivery
- ✓ VIP customer service hotline
- ✓ Personalized product recommendations

### Cluster 1: Value-Seeking Regular Customers (19% of customers)
**Characteristics:**
- Very high annual income
- Extremely high spending (~$1,015,789)
- Very high average order value
- Regular purchase frequency
- Very low discount usage
- Highest app engagement (147 min/day)

**Suggested Offers:**
- ✓ Festival discounts (10-15%)
- ✓ Loyalty reward points on every purchase
- ✓ Birthday special offers
- ✓ Referral bonuses

### Cluster 2: Price-Sensitive Occasional Customers (20% of customers)
**Characteristics:**
- Low annual income
- Low total spending (~$61,000)
- Low average order value
- Infrequent purchases
- High discount usage
- Low app engagement (21.1 min/day)

**Suggested Offers:**
- ✓ Flash sales and coupon-based discounts
- ✓ Free shipping on minimum order value
- ✓ Bundle deals and combos
- ✓ Clearance sale notifications

## 📈 Visualizations

The project includes 4 key visualizations:

1. **Customer Count per Cluster**: Shows distribution of customers across clusters
2. **Average Spending per Cluster**: Compares spending power
3. **Average App Usage per Cluster**: Measures engagement levels
4. **Income vs Spending Scatter Plot**: Visualizes cluster separation

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the Jupyter notebook:
```bash
jupyter notebook Untitled-1.ipynb
```

3. Execute all cells to:
   - Load and preprocess data
   - Train the K-Means model
   - Generate visualizations
   - Save the model as `customer_clustering_model.pkl`

## 🌐 FastAPI Deployment

### Starting the API Server

```bash
python app.py
```

The API will be available at: `http://localhost:8000`

Interactive API documentation: `http://localhost:8000/docs`

### API Endpoints

#### 1. Home
```
GET /
```
Returns API information and available endpoints.

#### 2. Health Check
```
GET /health
```
Checks if the API and model are loaded correctly.

#### 3. Get Cluster Information
```
GET /clusters
```
Returns details about all customer clusters and their offers.

#### 4. Predict Customer Cluster
```
POST /predict
```

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "cluster": 0,
  "customer_type": "High-Value Loyal Customers",
  "description": "Premium customers with high income and spending power",
  "suggested_offers": [
    "Exclusive early access to new products",
    "Premium membership with free express delivery",
    "VIP customer service hotline",
    "Personalized product recommendations",
    "Complimentary gift wrapping"
  ],
  "discount_range": "5-10%",
  "priority": "Highest",
  "customer_profile": { ... }
}
```

#### 5. Batch Prediction
```
POST /predict_batch
```
Predicts clusters for multiple customers at once.

### Testing the API

Run the test script to verify all endpoints:

```bash
python test_api.py
```

## 📁 Project Structure

```
ai-ml test/
│
├── CustomerData (1).csv              # Dataset
├── Untitled-1.ipynb                  # Main notebook with analysis
├── customer_clustering_model.pkl     # Saved model and artifacts
├── app.py                            # FastAPI application
├── test_api.py                       # API testing script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🛠️ Technologies Used

- **Python 3.12**
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Scikit-learn**: Machine learning (K-Means)
- **Matplotlib & Seaborn**: Visualizations
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

## 📊 Model Performance

- **Silhouette Score**: 0.4540 (K=3)
- **Algorithm**: K-Means Clustering
- **Number of Clusters**: 3
- **Features Used**: 9

The elbow method and silhouette score analysis confirmed that 3 clusters provide optimal separation.

## 🔄 Workflow

1. **Data Loading**: Parse CSV with proper delimiter handling
2. **Data Preprocessing**:
   - Handle missing values (fill with median)
   - Label encoding for binary features
   - One-hot encoding for multi-category features
   - Feature scaling (StandardScaler)
3. **Model Training**:
   - Elbow method to find optimal K
   - Train K-Means with K=3
   - Evaluate using silhouette score
4. **Model Saving**:
   - Save model and all preprocessing objects using pickle
5. **API Deployment**:
   - Load model in FastAPI
   - Create prediction endpoints
   - Map clusters to business offers

## 💡 Business Impact

- **Personalized Marketing**: Target customers with relevant offers
- **Increased Conversion**: Higher engagement through tailored promotions
- **Resource Optimization**: Focus high-touch service on high-value customers
- **Customer Retention**: Appropriate incentives for each segment
- **Revenue Growth**: Maximize value from each customer segment

## 🎓 Questions Answered

### a) Suitable Unsupervised Learning Algorithm
**Answer**: **K-Means Clustering**

K-Means is ideal for this use case because:
- Efficiently handles continuous numerical features
- Scales well with the dataset size
- Produces well-separated, interpretable clusters
- Works well when number of clusters is known or can be determined
- Centroid-based approach aligns with customer segmentation goals

### b) Cluster Characteristics & Offers
All three clusters have been successfully identified with distinct characteristics and personalized business offers as detailed above.

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Created as part of an AI/ML learning project.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

---

**Note**: Make sure to run the Jupyter notebook first to generate the `customer_clustering_model.pkl` file before starting the FastAPI server.
