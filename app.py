"""
app.py
------
Flask backend for the Cloud-Based Fraud Detection System.
"""

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load artefacts ────────────────────────────────────────────────────────────
def load_pickle(filename):
    path = os.path.join(BASE_DIR, "model_data", filename)
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model    = load_pickle("fraud_model.pkl")
    scaler   = load_pickle("scaler.pkl")
    features = load_pickle("features.pkl")
    print("[✓] Model artefacts loaded successfully.")
except FileNotFoundError as e:
    print(f"[✗] Could not load model artefacts: {e}")
    print("    Run  python train_model.py  first.")
    model = scaler = features = None

# ── Validation helpers ────────────────────────────────────────────────────────
FIELD_RULES = {
    "amount":                {"type": float, "min": 1.0,    "max": 40_00_000},  # ₹1 to ₹40 lakh
    "hour_of_day":           {"type": int,   "min": 0,     "max": 23},
    "day_of_week":           {"type": int,   "min": 0,     "max": 6},
    "transaction_type":      {"type": int,   "min": 0,     "max": 3},
    "merchant_category":     {"type": int,   "min": 0,     "max": 9},
    "distance_from_home":    {"type": float, "min": 0,     "max": 20_000},
    "num_prev_transactions": {"type": int,   "min": 0,     "max": 500},
    "account_age_days":      {"type": int,   "min": 0,     "max": 36_500},
    "failed_attempts":       {"type": int,   "min": 0,     "max": 10},
    "is_international":      {"type": int,   "min": 0,     "max": 1},
}

def validate_and_parse(data: dict):
    """Returns (parsed_dict, error_message)."""
    parsed = {}
    for field, rules in FIELD_RULES.items():
        raw = data.get(field)
        if raw is None or str(raw).strip() == "":
            return None, f"Missing field: '{field}'"
        try:
            value = rules["type"](raw)
        except (ValueError, TypeError):
            return None, f"'{field}' must be a {rules['type'].__name__}."
        if not (rules["min"] <= value <= rules["max"]):
            return None, (
                f"'{field}' must be between {rules['min']} and {rules['max']}."
            )
        parsed[field] = value
    return parsed, None

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    payload = request.get_json(silent=True) or request.form.to_dict()
    if not payload:
        return jsonify({"error": "No data received."}), 400

    parsed, err = validate_and_parse(payload)
    if err:
        return jsonify({"error": err}), 422

    # Build feature vector in the exact training order
    X_raw = np.array([[parsed[f] for f in features]])
    X_sc  = scaler.transform(X_raw)

    prediction  = int(model.predict(X_sc)[0])
    probability = float(model.predict_proba(X_sc)[0][1])  # P(fraud)

    # Risk tier
    if probability < 0.30:
        risk = "Low"
    elif probability < 0.65:
        risk = "Medium"
    else:
        risk = "High"

    return jsonify({
        "prediction":  prediction,          # 0 = Legitimate, 1 = Fraud
        "label":       "Fraud" if prediction else "Legitimate",
        "probability": round(probability * 100, 2),   # as percentage
        "risk_level":  risk,
        "features_used": features,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)