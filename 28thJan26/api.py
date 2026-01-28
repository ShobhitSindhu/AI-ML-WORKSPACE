from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
from datetime import datetime
from typing import List

# ======================================
# FastAPI App
# ======================================
app = FastAPI(title="Loan Prediction API")

# ======================================
# Database
# ======================================
DB_NAME = "loan_predictions.db"

def get_db():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

# ======================================
# Pydantic Models
# ======================================
class LoanPrediction(BaseModel):
    applicant_income: float
    coapplicant_income: float
    loan_amount: float
    loan_term: int
    credit_history: int
    married: str
    education: str
    self_employed: str
    property_area: str
    prediction: str
    probability: float

# ======================================
# ROOT
# ======================================
@app.get("/")
def home():
    return {"message": "Loan Prediction API is running 🚀"}

# ======================================
# PUT / POST → SAVE PREDICTION
# ======================================
@app.post("/prediction")
def save_prediction(data: LoanPrediction):

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
        data.applicant_income,
        data.coapplicant_income,
        data.loan_amount,
        data.loan_term,
        data.credit_history,
        data.married,
        data.education,
        data.self_employed,
        data.property_area,
        data.prediction,
        data.probability
    ))

    conn.commit()
    conn.close()

    return {"status": "success", "message": "Prediction saved"}

# ======================================
# GET → FETCH ALL PREDICTIONS
# ======================================
@app.get("/predictions")
def get_predictions():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM loan_predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "timestamp": row[1],
            "applicant_income": row[2],
            "coapplicant_income": row[3],
            "loan_amount": row[4],
            "loan_term": row[5],
            "credit_history": row[6],
            "married": row[7],
            "education": row[8],
            "self_employed": row[9],
            "property_area": row[10],
            "prediction": row[11],
            "probability": row[12]
        })

    return results
