"""Credit-risk scoring tool.

Loads the small LogisticRegression model trained by src/setup_data.py on
synthetic applicant data and scores a single applicant profile. This is a
self-contained example model built for this repo -- it is not imported from
and does not share code or data with any other repository in the portfolio.
"""
from __future__ import annotations

import os

import pandas as pd
from joblib import load

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "credit_risk_model.joblib",
)

_CACHE: dict = {}


def _load_model(model_path: str = MODEL_PATH):
    if model_path not in _CACHE:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Risk model not found at {model_path}. Run `python -m src.setup_data` first."
            )
        _CACHE[model_path] = load(model_path)
    return _CACHE[model_path]


def score_credit_risk(
    age: int,
    monthly_income_clp: float,
    debt_to_income: float,
    months_employed: float,
    n_late_payments: int,
    requested_amount_clp: float,
    model_path: str = MODEL_PATH,
) -> dict:
    """Scores a single credit applicant. Returns a probability of default and a
    risk tier. All inputs describe one applicant's profile."""
    bundle = _load_model(model_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    features = {
        "age": age,
        "monthly_income_clp": monthly_income_clp,
        "debt_to_income": debt_to_income,
        "months_employed": months_employed,
        "n_late_payments": n_late_payments,
        "requested_amount_clp": requested_amount_clp,
    }
    ordered = pd.DataFrame([[features[c] for c in feature_columns]], columns=feature_columns)
    prob_default = float(model.predict_proba(ordered)[0][1])

    if prob_default < 0.15:
        tier = "low"
    elif prob_default < 0.40:
        tier = "medium"
    elif prob_default < 0.65:
        tier = "high"
    else:
        tier = "critical"

    return {
        "probability_default": round(prob_default, 4),
        "risk_tier": tier,
        "input_profile": features,
    }


TOOL_SCHEMAS = [
    {
        "name": "score_credit_risk",
        "description": (
            "Scores an individual credit/loan applicant profile and returns a "
            "probability of default plus a risk tier (low/medium/high/critical)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "description": "Applicant age in years."},
                "monthly_income_clp": {
                    "type": "number",
                    "description": "Applicant's monthly income in Chilean pesos (CLP).",
                },
                "debt_to_income": {
                    "type": "number",
                    "description": "Existing debt-to-income ratio (0-1.5).",
                },
                "months_employed": {
                    "type": "number",
                    "description": "Months at current employment.",
                },
                "n_late_payments": {
                    "type": "integer",
                    "description": "Number of late payments on record.",
                },
                "requested_amount_clp": {
                    "type": "number",
                    "description": "Requested loan/credit amount in CLP.",
                },
            },
            "required": [
                "age",
                "monthly_income_clp",
                "debt_to_income",
                "months_employed",
                "n_late_payments",
                "requested_amount_clp",
            ],
        },
    }
]
