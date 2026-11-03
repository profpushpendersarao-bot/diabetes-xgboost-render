# Diabetes XGBoost Flask app — Render deployment

The form contains all 11 inputs in the exact saved-model training order:
Gender, AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI.

Gender mapping in the supplied training script: F=0, M=1.
Class mapping: 0=Non-Diabetic, 1=Pre-Diabetic, 2=Diabetic.

## Render
1. Extract this ZIP and upload the contents (not the enclosing folder) to the root of a GitHub repository.
2. Keep app.py, requirements.txt, best_xgboost_model.joblib, and the templates folder together.
3. Create a Render Web Service connected to that repository.
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app`
6. Health check path: `/health`

After deployment, visit `/health`; it should return the expected 11 feature names.

## Local test
Python 3.11 is specified in `.python-version`.
`python -m pip install -r requirements.txt`
`python app.py`
Open http://127.0.0.1:5000.

Do not upload the original patient dataset to a public repository. This is a research prototype, not a diagnostic tool. Model probability estimates are not clinical risk estimates.
