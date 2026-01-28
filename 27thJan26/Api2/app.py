from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route("/loans")
def show_loans():
    df = pd.read_csv("Loan dataset_classification.csv")

    # Convert dataframe to list of dicts
    loan_data = df.to_dict(orient="records")

    # Column names (for table header)
    columns = df.columns.tolist()

    return render_template(
        "loans.html",
        columns=columns,
        loans=loan_data
    )

if __name__ == "__main__":
    app.run(debug=True)
