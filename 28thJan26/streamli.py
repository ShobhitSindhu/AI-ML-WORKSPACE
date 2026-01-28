import pandas as pd
import numpy as np
import streamlit as st
import sqlite3
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# ======================================
# Streamlit Page Config
# ======================================
st.set_page_config(page_title="Loan Approval Prediction", layout="wide")
st.title("🏦 Loan Approval Prediction System")
st.markdown("Predict loan approval & store predictions in database")

# ======================================
# SQLite Database Setup
# ======================================
def get_db():
    return sqlite3.connect("loan_predictions.db", check_same_thread=False)

def create_table():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS loan_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        applicant_income REAL,
        coapplicant_income REAL,
        loan_amount REAL,
        loan_term INTEGER,
        credit_history INTEGER,
        married TEXT,
        education TEXT,
        self_employed TEXT,
        property_area TEXT,
        prediction TEXT,
        probability REAL
    )
    """)
    conn.commit()
    conn.close()

create_table()

def save_prediction(data, prediction, probability):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO loan_predictions (
        timestamp,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        married,
        education,
        self_employed,
        property_area,
        prediction,
        probability
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data["ApplicantIncome"],
        data["CoapplicantIncome"],
        data["LoanAmount"],
        data["Loan_Amount_Term"],
        data["Credit_History"],
        data["Married"],
        data["Education"],
        data["Self_Employed"],
        data["Property_Area"],
        prediction,
        probability
    ))
    conn.commit()
    conn.close()

# ======================================
# Load Dataset
# ======================================
@st.cache_data
def load_data():
    return pd.read_csv("Loan dataset_classification.csv")

df = load_data()
df = df.dropna(subset=["Loan_Status"])
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# ======================================
# Train Model (Cached)
# ======================================
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["Loan_Status", "Loan_ID", "Gender", "Dependents"])
    y = df["Loan_Status"]

    log_cols = ["ApplicantIncome", "CoapplicantIncome"]
    num_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History"]
    cat_cols = ["Married", "Education", "Self_Employed", "Property_Area"]

    preprocessor = ColumnTransformer([
        ("log", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, validate=False)),
            ("scaler", StandardScaler())
        ]), log_cols),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=3000
        ))
    ])

    model.fit(X, y)
    return model

model = train_model(df)

# ======================================
# Sidebar Navigation
# ======================================
menu = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dataset Overview",
        "📈 Data Analysis",
        "🤖 Loan Prediction",
        "🗄 Prediction History"
    ]
)

# ======================================
# DATASET OVERVIEW
# ======================================
if menu == "📊 Dataset Overview":
    st.subheader("📋 Loan Dataset")
    st.dataframe(df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Loans", len(df))
    c2.metric("Approved Loans", int(df["Loan_Status"].sum()))
    c3.metric("Rejected Loans", int(len(df) - df["Loan_Status"].sum()))

# ======================================
# DATA ANALYSIS
# ======================================
elif menu == "📈 Data Analysis":
    st.subheader("📊 Loan Approval Distribution")
    st.bar_chart(df["Loan_Status"].value_counts())

    st.subheader("💰 Applicant Income")
    st.line_chart(df["ApplicantIncome"])

    st.subheader("🏠 Property Area vs Approval")
    st.bar_chart(df.groupby("Property_Area")["Loan_Status"].mean())

    st.subheader("🎓 Education vs Approval")
    st.bar_chart(df.groupby("Education")["Loan_Status"].mean())

# ======================================
# LOAN PREDICTION
# ======================================
elif menu == "🤖 Loan Prediction":

    st.subheader("🧾 Enter Applicant Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        applicant_income = st.number_input("Applicant Income", min_value=0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0)

    with col2:
        loan_term = st.number_input("Loan Amount Term", min_value=0)
        credit_history = st.selectbox("Credit History", [0, 1])
        married = st.selectbox("Married", ["Yes", "No"])

    with col3:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("🔍 Predict Loan Approval"):

        input_data = {
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Married": married,
            "Education": education,
            "Self_Employed": self_employed,
            "Property_Area": property_area
        }

        input_df = pd.DataFrame([input_data])

        pred = model.predict(input_df)[0]
        prob = float(model.predict_proba(input_df)[0][1])
        result = "Approved" if pred == 1 else "Rejected"

        save_prediction(input_data, result, round(prob, 2))

        if pred == 1:
            st.success(f"✅ Loan Approved (Confidence: {prob:.2f})")
        else:
            st.error(f"❌ Loan Rejected (Confidence: {1 - prob:.2f})")

# ======================================
# PREDICTION HISTORY
# ======================================
elif menu == "🗄 Prediction History":
    st.subheader("🗄 Saved Predictions (SQLite Database)")

    conn = get_db()
    history = pd.read_sql("SELECT * FROM loan_predictions ORDER BY id DESC", conn)
    conn.close()

    st.dataframe(history, use_container_width=True)
