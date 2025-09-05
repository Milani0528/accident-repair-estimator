import os, random
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import preprocess
from keras.models import load_model   # ✅ Use keras.models with Keras 3.x

app = Flask(__name__)
CORS(app)

# --------------------------
# Load model
# --------------------------
MODEL_PATH = os.path.join("model", "model.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH, compile=False)

# --------------------------
# Class labels (from your .h5 training)
# --------------------------
CLASS_LABELS = [
    "Front_View",
    "Non_Front_View",
    "Non_Rear_Bumper",
    "Non_Sedan_Side_View",
    "Rear_Bumper",
    "Sedan_Side_View",
]

# Cost mapping (rough demo ranges in LKR)
ESTIMATE_RANGES = {
    "Front_View": [60000, 200000],
    "Rear_Bumper": [50000, 180000],
    "Sedan_Side_View": [40000, 150000],
    "Non_Front_View": [0, 0],
    "Non_Rear_Bumper": [0, 0],
    "Non_Sedan_Side_View": [0, 0],
}

# --------------------------
# API routes
# --------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "model_loaded": True,
        "classes": CLASS_LABELS,
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Preprocess & Predict
    x = preprocess(file, target_size=(224, 224))
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    label = CLASS_LABELS[idx]
    confidence = float(preds[idx])

    # Rough estimate
    lo, hi = ESTIMATE_RANGES[label]
    estimate = 0 if lo == hi == 0 else int(random.randint(lo, hi) // 1000) * 1000
    message = "No repair needed" if estimate == 0 else "Rough repair estimate"

    return jsonify({
        "label": label,
        "class_index": idx,
        "confidence": confidence,
        "estimate_lkr": estimate,
        "currency": "LKR",
        "message": message,
        "model_status": "inference"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
