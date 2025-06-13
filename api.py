import os
import boto3
import joblib
from flask import Flask, request, jsonify
import predict
import tempfile

app = Flask(__name__)

BUCKET_NAME = "temperature-prediction-artifacts-2025"
MODEL_FILE = "model_5min.pkl"
SCALER_FILE = "scaler_5min.pkl"
TEMP_DIR = tempfile.gettempdir()  # Gets Windows temp directory (e.g., C:\Temp)
MODEL_PATH = os.path.join(TEMP_DIR, MODEL_FILE)
SCALER_PATH = os.path.join(TEMP_DIR, SCALER_FILE)

s3 = boto3.client("s3")

if not os.path.exists(MODEL_PATH):
    s3.download_file(BUCKET_NAME, MODEL_FILE, MODEL_PATH)
if not os.path.exists(SCALER_PATH):
    s3.download_file(BUCKET_NAME, SCALER_FILE, SCALER_PATH)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route("/predict", methods=["POST"])
def predict_temperature():
    data = request.get_json()
    features = [data[f"Cell Volt {i}"] for i in range(1, 17)] + [data[f"Feature {i}"] for i in range(17, 41)]
    prediction = predict.predict_temperature(model, scaler, features)
    return jsonify({"predictions": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    # app.run(host="0.0.0.0", port=5000)