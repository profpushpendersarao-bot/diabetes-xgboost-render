import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_xgboost_model.joblib")

FEATURES = ["Gender", "AGE", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"]
GENDER_MAP = {"F": 0, "M": 1}
CLASS_NAMES = {0: "Non-Diabetic", 1: "Pre-Diabetic", 2: "Diabetic"}

app = Flask(__name__)
model = joblib.load(MODEL_PATH)  # Load the trained pipeline once at startup.

def get_model_input(form):
    gender = form.get("Gender", "").strip().upper()
    if gender not in GENDER_MAP:
        raise ValueError("Please select Female or Male.")

    values = {"Gender": GENDER_MAP[gender]}
    for name in FEATURES[1:]:
        raw = form.get(name, "").strip()
        if not raw:
            raise ValueError(f"{name} is required.")
        try:
            number = float(raw)
        except ValueError:
            raise ValueError(f"{name} must be a valid number.")
        if not np.isfinite(number) or number < 0:
            raise ValueError(f"{name} must be a finite, non-negative number.")
        values[name] = number

    # Exact feature names and order used to train the saved model.
    return pd.DataFrame([[values[col] for col in FEATURES]], columns=FEATURES)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", values={}, result=None, error=None)

@app.route("/predict", methods=["POST"])
def predict():
    values = request.form.to_dict()
    try:
        model_input = get_model_input(request.form)
        predicted_code = int(model.predict(model_input)[0])
        probabilities = []
        if hasattr(model, "predict_proba"):
            for class_code, probability in zip(model.classes_, model.predict_proba(model_input)[0]):
                code = int(class_code)
                probabilities.append({
                    "name": CLASS_NAMES.get(code, f"Class {code}"),
                    "percentage": float(probability) * 100
                })
        result = {
            "class_code": predicted_code,
            "class_name": CLASS_NAMES.get(predicted_code, f"Class {predicted_code}"),
            "probabilities": probabilities
        }
        return render_template("index.html", values=values, result=result, error=None)
    except Exception as exc:
        return render_template("index.html", values=values, result=None, error=str(exc)), 400

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None,
                    "expected_input_count": len(FEATURES), "features": FEATURES})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
