
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load student dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv"

data = pd.read_csv(url, sep=";")

# Select input features
X = data[["studytime", "failures", "absences",
          "G1", "G2", "age"]]

# Target: Final grade
y = data["G3"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train ML model
model = RandomForestRegressor(
    n_estimators=100, random_state=42
)

model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("Model Evaluation")
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Predict new student performance
print("\nEnter Student Details")

studytime = int(input("Study time (1-4): "))
failures = int(input("Previous failures: "))
absences = int(input("Number of absences: "))
G1 = int(input("First period marks (0-20): "))
G2 = int(input("Second period marks (0-20): "))
age = int(input("Student age: "))

new_student = pd.DataFrame([[
    studytime, failures, absences, G1, G2, age
]], columns=X.columns)

result = model.predict(new_student)[0]

print("\nPredicted Final Marks:", round(result, 2))
