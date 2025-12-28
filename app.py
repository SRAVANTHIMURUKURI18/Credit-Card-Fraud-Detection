from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Default values for remaining features
default_features = {
    f"V{i}": 0.0 for i in range(1, 29)
}
default_features["Time"] = 0
default_features["Amount"] = 0

@app.route("/")
def home():
    return jsonify({"message": "Credit Card Fraud Detection API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Fill features: take 5 from user, rest use defaults
        features = default_features.copy()
        features.update({k: float(v) for k, v in data.items()})

        # Prepare feature list in correct order
        feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        feature_vector = [features[f] for f in feature_order]

        scaled = scaler.transform([feature_vector])
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        return jsonify({
            "prediction": int(pred),
            "fraud_probability": float(prob)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
