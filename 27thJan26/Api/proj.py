import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# ===============================
# 1. Train model (ONCE)
# ===============================
df = pd.read_csv("Loan dataset_classification.csv")
df = df.dropna(subset=["Loan_Status"])

X = df.drop(columns=["Loan_Status", "Loan_ID", "Gender", "Dependents"])
y = df["Loan_Status"].map({"Y": 1, "N": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log_cols = ["ApplicantIncome", "CoapplicantIncome"]
num_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History"]
cat_cols = ["Married", "Education", "Self_Employed", "Property_Area"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

log_numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(np.log1p, validate=False)),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("log_num", log_numeric_pipeline, log_cols),
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=3000
    ))
])

model.fit(X_train, y_train)

# ===============================
# 2. Flask App
# ===============================
app = Flask(__name__)

# UI PAGE
@app.route("/")
def index():
    return render_template("index.html")

# API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return jsonify({
        "prediction": "Approved" if prediction == 1 else "Rejected",
        "probability": round(float(probability), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
