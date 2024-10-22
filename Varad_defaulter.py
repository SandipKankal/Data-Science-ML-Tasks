# -*- coding: utf-8 -*-
"""defaulter.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Lw_kXM4pFTql6aLFL4IHFb_Qz39HQtz6
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Load the dataset
data = pd.read_csv('loan_approval_dataset.csv')

# Preprocess the data
# Handling missing values (if any)
data.fillna(method='ffill', inplace=True)

data.head()

# Drop loan_id (not useful for prediction)
data = data.drop('loan_id', axis=1)

# Handle missing values (fill forward or with median/mean depending on the feature)
data.fillna(method='ffill', inplace=True)  # Or handle missing values differently if needed

# Encoding categorical variables
le = LabelEncoder()

print(data.columns)

# Strip leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Now you can proceed with your code
le = LabelEncoder()
data['education'] = le.fit_transform(data['education'])
data['self_employed'] = le.fit_transform(data['self_employed'])

# Encode target variable 'loan_status'
data['loan_status'] = le.fit_transform(data['loan_status'])

# Features (X) and Target (y)
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling (especially for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# Evaluate Logistic Regression
print("Logistic Regression Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_lr))

# 2. Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)  # No scaling needed for Decision Tree
y_pred_dt = dt_model.predict(X_test)

# Evaluate Decision Tree
print("\nDecision Tree Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_dt))

# prompt: visualize the statistics

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data' DataFrame is already created and preprocessed as in the previous code

# Visualize the distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='loan_status', data=data)
plt.title('Distribution of Loan Status')
plt.show()

# Visualize the relationship between 'loan_status' and 'education'
plt.figure(figsize=(8, 6))
sns.countplot(x='education', hue='loan_status', data=data)
plt.title('Loan Status vs. Education')
plt.show()


# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()

# Example: Visualize the distribution of 'applicantincome'
plt.figure(figsize=(8, 6))
sns.histplot(data['applicantincome'], kde=True)
plt.title('Distribution of Applicant Income')
plt.show()

# Box plots to visualize the relationship between numerical features and the target variable.
numerical_features = ['applicantincome', 'coapplicantincome', 'loanamount']
for col in numerical_features:
  plt.figure(figsize=(8,6))
  sns.boxplot(x='loan_status', y=col, data=data)
  plt.title(f'Box Plot of {col} by Loan Status')
  plt.show()

# Confusion Matrix visualization (for Logistic Regression)
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix (Logistic Regression)')
plt.show()


# You can create similar visualizations for the Decision Tree model.

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data' DataFrame is already created and preprocessed

# Visualize the distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='loan_status', data=data)
plt.title('Distribution of Loan Status')
plt.show()

# Visualize the relationship between 'loan_status' and 'education'
plt.figure(figsize=(8, 6))
sns.countplot(x='education', hue='loan_status', data=data)
plt.title('Loan Status vs. Education')
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()

# Example: Visualize the distribution of 'income_annum'
plt.figure(figsize=(8, 6))
sns.histplot(data['income_annum'], kde=True)
plt.title('Distribution of Applicant Income (Annum)')
plt.show()


# Confusion Matrix visualization (for Logistic Regression)
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix (Logistic Regression)')
plt.show()

# Box plots to visualize the relationship between numerical features and the target variable
numerical_features = ['income_annum', 'loan_amount', 'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
for col in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='loan_status', y=col, data=data)
    plt.title(f'Box Plot of {col} by Loan Status')
    plt.show()