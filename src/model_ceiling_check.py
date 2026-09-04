"""Is the credit-risk model's weak AUC (0.586) a fixable modeling problem, or is it
already close to the best any model could do on this label?

The README calls the low AUC "honest" because the synthetic label is deliberately
noisy, but that claim was never actually checked against a number until this module:
`_generate_credit_applicants` (src/setup_data.py) computes a `true_prob_default` for
every applicant before drawing the Bernoulli `default` label from it. That true
probability is the best possible predictor of `default` that could ever exist for
this label -- no model, however powerful, can score better against it, because the
only randomness left once you know it is the coin flip itself. Scoring that oracle
probability against the real held-out labels gives the AUC ceiling. Comparing the
actually-fitted LogisticRegression (and a more flexible GradientBoostingClassifier,
as a second check that a more expressive model isn't quietly leaving AUC on the
table) against that ceiling answers the question directly instead of asserting it.
"""
from __future__ import annotations

import numpy as np
from joblib import load
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.setup_data import FEATURE_COLUMNS, RANDOM_SEED, RISK_MODEL_PATH, _generate_credit_applicants


def compute_ceiling_comparison(model_path: str = RISK_MODEL_PATH, seed: int = RANDOM_SEED) -> dict:
    rng = np.random.default_rng(seed + 1)
    df = _generate_credit_applicants(rng)
    X = df[FEATURE_COLUMNS]
    y = df["default"]
    true_prob = df["true_prob_default"]

    X_train, X_test, y_train, y_test, _prob_train, prob_test = train_test_split(
        X, y, true_prob, test_size=0.2, random_state=seed
    )

    oracle_auc = float(roc_auc_score(y_test, prob_test))

    bundle = load(model_path)
    logistic_auc = float(roc_auc_score(np.asarray(bundle["y_test"]), np.asarray(bundle["y_proba_test"])))

    gbm = GradientBoostingClassifier(random_state=seed)
    gbm.fit(X_train, y_train)
    gbm_auc = float(roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1]))

    return {
        "oracle_auc": round(oracle_auc, 4),
        "logistic_regression_auc": round(logistic_auc, 4),
        "gradient_boosting_auc": round(gbm_auc, 4),
        "logistic_ceiling_capture_pct": round(100 * logistic_auc / oracle_auc, 1),
        "gbm_ceiling_capture_pct": round(100 * gbm_auc / oracle_auc, 1),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(compute_ceiling_comparison(), indent=2))
