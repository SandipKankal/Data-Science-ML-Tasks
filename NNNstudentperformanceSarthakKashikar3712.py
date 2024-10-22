import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv('student_performance_data.csv')

# Prepare features and target
X = df[['Gender', 'Age', 'StudyHoursPerWeek', 'AttendanceRate', 'Major', 'PartTimeJob', 'ExtraCurricularActivities']]
y = df['GPA']  # Target variable

# Preprocessing pipelines for numerical and categorical features
numerical_features = ['Age', 'StudyHoursPerWeek', 'AttendanceRate']
categorical_features = ['Gender', 'Major', 'PartTimeJob', 'ExtraCurricularActivities']

# Define preprocessing steps: one-hot encode categorical variables and scale numerical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Preprocess data
X_preprocessed = preprocessor.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Build the Neural Network Model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer + first hidden layer
model.add(Dense(32, activation='relu'))  # Second hidden layer
model.add(Dense(1, activation='linear'))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1, validation_split=0.2)

# Evaluate the model on test data
y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

# Print actual vs predicted values
print("\nActual GPA values:\n", y_test.values)
print("\nPredicted GPA values:\n", y_pred)

# Plot the training loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# Plot actual vs predicted GPA
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual GPA')
plt.ylabel('Predicted GPA')
plt.title('Actual vs Predicted GPA')
plt.grid(True)
plt.show()


# Function for manual input to predict GPA
def predict_gpa_manually():

    gender = input("Enter Gender (Male/Female): ")
    age = float(input("Enter Age: "))
    study_hours_per_week = float(input("Enter Study Hours Per Week: "))
    attendance_rate = float(input("Enter Attendance Rate (in percentage): "))
    major = input("Enter Major (e.g., CS, Engineering, etc.): ")
    part_time_job = input("Do you have a part-time job? (Yes/No): ")
    extra_curricular = input("Involved in Extra-curricular activities? (Yes/No): ")

    # Create a DataFrame for input
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'StudyHoursPerWeek': [study_hours_per_week],
        'AttendanceRate': [attendance_rate],
        'Major': [major],
        'PartTimeJob': [part_time_job],
        'ExtraCurricularActivities': [extra_curricular]
    })

    # Preprocess the input data
    input_preprocessed = preprocessor.transform(input_data)

    # Predict GPA using the trained model
    predicted_gpa = model.predict(input_preprocessed)
    print(f"Predicted GPA: {predicted_gpa[0][0]}")


# Call the function to input values manually and predict GPA
predict_gpa_manually()
