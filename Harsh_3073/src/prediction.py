import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

# Path to your test dataset (adjust the path as necessary)
TEST_DATA_PATH = 'data/customer_churn_dataset-training-master.csv'


def predict_churn(input_data, model_type='rf', model_path=None):
    try:
        # Define model paths based on the selected model type
        if model_type == 'logreg':
            model_path = model_path or 'models/churn_logreg_model.pkl'
        elif model_type == 'rf':
            model_path = model_path or 'models/churn_rf_model.pkl'
        else:
            raise ValueError(
                "Unsupported model type. Choose 'logreg' or 'rf'.")

        print(f"Loading model from {model_path}")  # Debug: log model path

        # Load the trained model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Ensure input data matches the number of features used in training, without 'Monthly Charges'
        input_data_encoded = [
            input_data[0],  # Age
            input_data[1],  # Tenure
            input_data[2],  # Gender (already encoded)
            input_data[3],  # Subscription Type (already encoded)
            input_data[4]   # Contract Length (already encoded)
            # No Monthly Charges
        ]

        # Debug: log encoded input data
        print(f"Encoded input data: {input_data_encoded}")

        # Convert input_data_encoded to a NumPy array and reshape for prediction
        prediction = model.predict(np.array([input_data_encoded]))[0]
        print("Prediction completed successfully!")
        return prediction

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def get_model_accuracy(model_type='rf', model_path=None):
    try:
        # Define model paths based on the selected model type
        if model_type == 'logreg':
            model_path = model_path or 'models/churn_logreg_model.pkl'
        elif model_type == 'rf':
            model_path = model_path or 'models/churn_rf_model.pkl'
        else:
            raise ValueError(
                "Unsupported model type. Choose 'logreg' or 'rf'.")

        # Debug: log model path
        print(f"Loading model from {model_path} to compute accuracy")

        # Load the trained model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Load the test dataset
        test_data = pd.read_csv(TEST_DATA_PATH)

        # Fix any column name discrepancies (ensure they match the training data)
        test_data.rename(
            columns={'Subscription Type': 'SubscriptionType'}, inplace=True)

        # Preprocess the test data to match the training data
        test_data['Gender'] = test_data['Gender'].apply(
            lambda x: 1 if x == 'Male' else 0)
        test_data['SubscriptionType'] = test_data['SubscriptionType'].apply(
            lambda x: 1 if x == 'Premium' else 0)
        test_data['ContractLength'] = test_data['Contract Length'].apply(
            lambda x: 1 if x == 'Long' else 0)

        # Drop unnecessary columns if needed (ensure only the features used in training are kept)
        X_test = test_data[['Age', 'Tenure', 'Gender',
                            'SubscriptionType', 'ContractLength']]
        y_test = test_data['Churn']  # Assuming 'Churn' is the target column

        # Predict on the test data
        y_pred = model.predict(X_test)

        # Compute the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        return accuracy

    except Exception as e:
        print(f"Error during accuracy calculation: {e}")
        return None

    try:
        # Define model paths based on the selected model type
        if model_type == 'logreg':
            model_path = model_path or 'models/churn_logreg_model.pkl'
        elif model_type == 'rf':
            model_path = model_path or 'models/churn_rf_model.pkl'
        else:
            raise ValueError(
                "Unsupported model type. Choose 'logreg' or 'rf'.")

        # Debug: log model path
        print(f"Loading model from {model_path} to compute accuracy")

        # Load the trained model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Load the test dataset
        test_data = pd.read_csv(TEST_DATA_PATH)

        # Fix any column name discrepancies (ensure they match the training data)
        test_data.rename(columns={'Monthly Charges': 'MonthlyCharges',
                         'Subscription Type': 'SubscriptionType'}, inplace=True)

        # Preprocess the test data to match the training data
        test_data['Gender'] = test_data['Gender'].apply(
            lambda x: 1 if x == 'Male' else 0)
        test_data['SubscriptionType'] = test_data['SubscriptionType'].apply(
            lambda x: 1 if x == 'Premium' else 0)
        test_data['ContractLength'] = test_data['Contract Length'].apply(
            lambda x: 1 if x == 'Long' else 0)

        # Drop unnecessary columns if needed (ensure only the features used in training are kept)
        X_test = test_data[['Age', 'Tenure', 'Gender',
                            'MonthlyCharges', 'SubscriptionType', 'ContractLength']]
        y_test = test_data['Churn']  # Assuming 'Churn' is the target column

        # Predict on the test data
        y_pred = model.predict(X_test)

        # Compute the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        return accuracy

    except Exception as e:
        print(f"Error during accuracy calculation: {e}")
        return None
