import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Fill missing values
    df.fillna(method='ffill', inplace=True)

    # Encode categorical features
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    le_subscription = LabelEncoder()
    df['Subscription Type'] = le_subscription.fit_transform(
        df['Subscription Type'])

    le_contract = LabelEncoder()
    df['Contract Length'] = le_contract.fit_transform(df['Contract Length'])

    # Drop CustomerID and Churn (target)
    X = df.drop(['CustomerID', 'Churn'], axis=1)
    y = df['Churn']

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Data preprocessing completed successfully!")  # Success message
    return X_scaled, y


# Example file path
file_path = 'data/customer_churn_dataset-training-master.csv'
