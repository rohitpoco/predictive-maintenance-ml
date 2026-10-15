import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Load dataset
data = pd.read_csv("dataset/ai4i2020.csv")

# Select one record for testing
X = data.drop(columns=[
    "UDI", "Product ID", "Type", "Machine failure",
    "TWF", "HDF", "PWF", "OSF", "RNF"
])

sample = X.iloc[[0]]

# Predict machine failure
prediction = model.predict(sample)

if prediction[0] == 1:
    print("Machine Failure Predicted!")
else:
    print("Machine Operating Normally!")

print("Prediction:", prediction[0])