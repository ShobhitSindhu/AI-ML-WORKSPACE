#FastAPI

import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

model = joblib.load("iris_rf_model.pkl")

@app.post("/predict")
def predict(data):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"predicted_output":int(prediction[0])}

#run command:


#python -m uvicorn fastapi_ .py:app --host 0.0.0.0 --port 5000 --reload