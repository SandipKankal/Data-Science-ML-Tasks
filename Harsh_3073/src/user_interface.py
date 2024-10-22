from flask import Flask, render_template, request
from src.prediction import predict_churn

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract user input from the form
    age = int(request.form['age'])
    tenure = int(request.form['tenure'])
    gender = request.form['gender']
    monthly_charges = float(request.form['monthly_charges'])
    subscription_type = request.form['subscription_type']
    contract_length = request.form['contract_length']

    # Prepare the input list for the prediction function
    input_data = [age, tenure, gender, monthly_charges, subscription_type, contract_length]

    # Get the prediction result
    prediction = predict_churn(input_data)

    # Render result in a new page
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
