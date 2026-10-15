from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Predictive Maintenance ML Server is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = pd.DataFrame([data])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "failure_probability": round(float(probability) * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)