import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "Married": "Yes",
    "Education": "Graduate",
    "Self_Employed": "No",
    "Property_Area": "Urban",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1
}

response = requests.post(url, json=data)
print(response.json())
