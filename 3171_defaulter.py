# -*- coding: utf-8 -*-
"""dscdefaulter.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12GnaxeweGmp37tdpUtxF4J6fMlEDrGHy
"""

#evaluates their performance
#predict employee Churn with decision tree and random forsts.use employee dataset from kaggle
#decision tree-devides the dataset into branches for decision making
#Random forest -its a group of decision tree working togehter
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Load dataset from Google Drive (adjust the path as necessary)
from google.colab import drive
drive.mount('/content/drive')

# Load the dataset
df = pd.read_csv('/content/Employee.csv')

# Preview the dataset
print(df.head())

# Check column names
print("Column names:", df.columns)

# Remove leading/trailing spaces in column names
df.columns = df.columns.str.strip()

# Filter relevant columns
relevant_columns = ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain', 'LeaveOrNot']
df = df[relevant_columns]

# Convert categorical variables to numerical using Label Encoding
label_encoder = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Define features and target variable
X = df.drop('LeaveOrNot', axis=1)  # 'LeaveOrNot' is the target variable
y = df['LeaveOrNot']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)

# Evaluating Decision Tree
print("Decision Tree Classifier Report:")
print(classification_report(y_test, dt_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_predictions))

# Evaluating Random Forest
print("\nRandom Forest Classifier Report:")
print(classification_report(y_test, rf_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

# Visualizing feature importance for Random Forest
importances = rf_classifier.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.show()