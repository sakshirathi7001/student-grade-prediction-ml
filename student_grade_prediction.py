
# Student Grade Prediction using Machine Learning
# Author: Student ML Project
# Description: Predicts final student grades using Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("student-mat.csv")

# Display first few rows
print("Dataset Preview:")
print(data.head())

# Select important features
features = ['studytime', 'failures', 'absences', 'G1', 'G2']
target = 'G3'

X = data[features]
y = data[target]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict grades
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Plot Actual vs Predicted Grades
plt.scatter(y_test, predictions)
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted Student Grades")
plt.show()

# Predict grade for a new student
new_student = np.array([[3, 0, 5, 14, 15]])
predicted_grade = model.predict(new_student)

print("\nPredicted Final Grade for New Student:", predicted_grade[0])
