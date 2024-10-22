from sklearn.ensemble import RandomForestClassifier
import pickle
from src.preprocess import load_and_preprocess_data


def train_model(file_path):
    try:
        # Load and preprocess the data
        X_scaled, y = load_and_preprocess_data(file_path)

        if X_scaled is None or y is None:
            raise ValueError(
                "Preprocessing failed, model training cannot proceed.")

        # Train a Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        # Save the model to a file
        model_path = 'models/churn_rf_model.pkl'
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)

        print("Model training completed successfully!")  # Success message
        print(f"Model saved to {model_path}")

        return model
    except Exception as e:
        print(f"Error during model training: {e}")
        return None


# Example file path
file_path = 'data/customer_churn_dataset-training-master.csv'

# Call the function to train and save the model
train_model(file_path)
