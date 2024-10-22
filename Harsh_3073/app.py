from flask import Flask, render_template, request
# Ensure this import matches your prediction.py file
from src.prediction import predict_churn, get_model_accuracy

# Initialize Flask app
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            print("Form Data:", request.form)  # Debug: log form data

            # Collect data from the form without Monthly Charges
            input_data = [
                float(request.form['Age']),
                float(request.form['Tenure']),
                int(request.form['Gender']),
                int(request.form['SubscriptionType']),
                int(request.form['ContractLength'])
            ]

            print("Input Data:", input_data)  # Debug: log input data

            # Choose model type from form (optional: default to 'rf')
            model_type = request.form.get('ModelType', 'rf')

            # Make the prediction
            result = predict_churn(input_data, model_type=model_type)

            print("Prediction Result:", result)  # Debug: log prediction result

            # Get the accuracy of the model
            accuracy = get_model_accuracy(model_type=model_type)

            print("Model Accuracy:", accuracy)  # Debug: log model accuracy

            if result is not None:
                success_message = "Prediction successful! Operation completed."
            else:
                success_message = "Prediction failed. Please try again."

            return render_template('result.html', prediction=result, accuracy=accuracy, success_message=success_message)
        except Exception as e:
            return f"Error occurred: {e}"

    if request.method == 'POST':
        try:
            # Log form data for debugging
            print("Form Data:", request.form)

            # Extract input data from the form
            input_data = [
                float(request.form['Age']),
                float(request.form['Tenure']),
                int(request.form['Gender']),
                float(request.form['MonthlyCharges']),
                int(request.form['SubscriptionType']),
                int(request.form['ContractLength'])
            ]

            # Debugging: log the collected input data
            print("Input Data:", input_data)

            # Select model type from the form (default: 'rf' for Random Forest)
            model_type = request.form.get('ModelType', 'rf')

            # Make the churn prediction using the prediction module
            result = predict_churn(input_data, model_type=model_type)

            # Debug: log the prediction result
            print("Prediction Result:", result)

            # Get model accuracy (optional, assuming the function exists)
            accuracy = get_model_accuracy(model_type=model_type)

            print("Model Accuracy:", accuracy)  # Debug: log model accuracy

            # Check the prediction result and display success message
            if result is not None:
                success_message = "Prediction successful! Operation completed."
            else:
                success_message = "Prediction failed. Please try again."

            # Render the result.html page with prediction and accuracy
            return render_template('result.html', prediction=result, accuracy=accuracy, success_message=success_message)

        except Exception as e:
            # If any error occurs, return an error message
            return f"Error occurred: {e}"


if __name__ == '__main__':
    app.run(debug=True)
