# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = 'E:\DS\yield_df.csv'
dataset = pd.read_csv(dataset_path)

# Drop rows with missing values
dataset_clean = dataset.dropna()
#data info
print(dataset_clean.info())
#data description
print(dataset_clean.describe())
#data head
print(dataset_clean.head())
#data tail
print(dataset_clean.tail())
#data shape
print(dataset_clean.shape)
#data columns
print(dataset_clean.columns)
#data null values
print(dataset_clean.isnull().sum())
#data duplicates
print(dataset_clean.duplicated().sum())
#data unique values
print(dataset_clean.nunique())


# Feature selection based on available columns
X = dataset_clean[['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Year']]
y = dataset_clean['hg/ha_yield']  # Target variable (crop yield)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using R-squared and Mean Squared Error (MSE)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Output the evaluation metrics
print(f"R-squared: {r2}")
print(f"Mean Squared Error: {mse}")

# Scatter plot: Actual vs Predicted yields
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.9)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Yield (hg/ha)')
plt.ylabel('Predicted Yield (hg/ha)')
plt.title('Actual vs Predicted Crop Yield')
plt.grid(True)
plt.show()
