"""Generates the synthetic operational warehouse (DuckDB) and trains the
example credit-risk scoring model used by the tools in src/tools/.

Everything produced here is synthetic data created with a fixed random seed.
It does not come from, and is not a copy of, any other repository in the
portfolio. Run with:

    python -m src.setup_data
"""
from __future__ import annotations

import os

import duckdb
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DUCKDB_PATH = os.path.join(DATA_DIR, "ops.duckdb")
RISK_MODEL_PATH = os.path.join(DATA_DIR, "credit_risk_model.joblib")


def _generate_flotation_batches(rng: np.random.Generator, n_months: int = 12) -> pd.DataFrame:
    """Synthetic monthly copper flotation batches for one plant."""
    rows = []
    batch_id = 1
    for month_offset in range(n_months):
        period = pd.Period("2025-09", freq="M") + month_offset
        n_batches = rng.integers(18, 28)
        for _ in range(n_batches):
            feed_grade = rng.normal(0.85, 0.12)
            recovery = np.clip(rng.normal(88.0, 3.5), 60, 99)
            concentrate_grade = np.clip(rng.normal(28.0, 2.0), 15, 40)
            tonnage = rng.normal(1500, 200)
            rows.append(
                {
                    "batch_id": batch_id,
                    "month": str(period),
                    "feed_grade_pct": round(max(feed_grade, 0.05), 3),
                    "recovery_pct": round(recovery, 2),
                    "concentrate_grade_pct": round(concentrate_grade, 2),
                    "tonnage_processed": round(max(tonnage, 10), 1),
                }
            )
            batch_id += 1
    return pd.DataFrame(rows)


def _generate_maintenance_events(rng: np.random.Generator, n_days: int = 365) -> pd.DataFrame:
    """Synthetic maintenance/alert events across a fleet of equipment."""
    equipment_ids = [f"EQ-{i:03d}" for i in range(1, 26)]
    event_types = ["scheduled", "unscheduled", "inspection", "critical_alert"]
    severities = ["low", "medium", "high", "critical"]
    start_date = pd.Timestamp("2025-09-01")
    rows = []
    event_id = 1
    for day in range(n_days):
        date = start_date + pd.Timedelta(days=day)
        n_events = rng.poisson(1.4)
        for _ in range(n_events):
            equip = rng.choice(equipment_ids)
            etype = rng.choice(event_types, p=[0.45, 0.30, 0.20, 0.05])
            severity = rng.choice(severities, p=[0.5, 0.3, 0.15, 0.05])
            downtime_hours = round(max(rng.exponential(3.0), 0.0), 2)
            rows.append(
                {
                    "event_id": event_id,
                    "date": date.strftime("%Y-%m-%d"),
                    "equipment_id": equip,
                    "event_type": etype,
                    "severity": severity,
                    "downtime_hours": downtime_hours,
                }
            )
            event_id += 1
    return pd.DataFrame(rows)


def _generate_procurement_orders(rng: np.random.Generator, n_orders: int = 400) -> pd.DataFrame:
    """Synthetic procurement orders (spare parts, reagents, services)."""
    categories = ["spare_parts", "reagents", "fuel", "services", "safety_equipment"]
    suppliers = [f"Proveedor-{i}" for i in range(1, 16)]
    statuses = ["delivered", "pending", "delayed", "cancelled"]
    start_date = pd.Timestamp("2025-09-01")
    rows = []
    for order_id in range(1, n_orders + 1):
        date = start_date + pd.Timedelta(days=int(rng.integers(0, 365)))
        category = rng.choice(categories)
        amount_usd = round(max(rng.lognormal(mean=8.5, sigma=1.1), 50), 2)
        status = rng.choice(statuses, p=[0.65, 0.15, 0.15, 0.05])
        rows.append(
            {
                "order_id": order_id,
                "date": date.strftime("%Y-%m-%d"),
                "category": category,
                "supplier": rng.choice(suppliers),
                "amount_usd": amount_usd,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def build_warehouse(db_path: str = DUCKDB_PATH, seed: int = RANDOM_SEED) -> None:
    """Creates data/ops.duckdb with three synthetic operational tables."""
    rng = np.random.default_rng(seed)
    flotation = _generate_flotation_batches(rng)
    maintenance = _generate_maintenance_events(rng)
    procurement = _generate_procurement_orders(rng)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    con = duckdb.connect(db_path)
    con.execute("CREATE TABLE flotation_batches AS SELECT * FROM flotation")
    con.execute("CREATE TABLE maintenance_events AS SELECT * FROM maintenance")
    con.execute("CREATE TABLE procurement_orders AS SELECT * FROM procurement")
    con.close()


def _generate_credit_applicants(rng: np.random.Generator, n: int = 2000) -> pd.DataFrame:
    """Synthetic loan/credit applicant profiles with a latent default label."""
    age = rng.integers(19, 70, size=n)
    monthly_income_clp = np.clip(rng.normal(750000, 300000, size=n), 200000, None)
    debt_to_income = np.clip(rng.normal(0.35, 0.18, size=n), 0.0, 1.5)
    months_employed = np.clip(rng.normal(48, 36, size=n), 0, None)
    n_late_payments = rng.poisson(1.2, size=n)
    requested_amount_clp = np.clip(rng.normal(3_000_000, 1_500_000, size=n), 200000, None)

    # Latent risk score drives a synthetic default probability.
    z = (
        -2.5
        + 2.2 * debt_to_income
        + 0.35 * n_late_payments
        - 0.010 * (months_employed / 12)
        - 0.0000006 * monthly_income_clp
        + 0.00000025 * requested_amount_clp
    )
    prob_default = 1 / (1 + np.exp(-z))
    default = rng.binomial(1, np.clip(prob_default, 0.02, 0.95))

    return pd.DataFrame(
        {
            "age": age,
            "monthly_income_clp": monthly_income_clp.round(0),
            "debt_to_income": debt_to_income.round(3),
            "months_employed": months_employed.round(1),
            "n_late_payments": n_late_payments,
            "requested_amount_clp": requested_amount_clp.round(0),
            "default": default,
        }
    )


FEATURE_COLUMNS = [
    "age",
    "monthly_income_clp",
    "debt_to_income",
    "months_employed",
    "n_late_payments",
    "requested_amount_clp",
]


def train_credit_risk_model(model_path: str = RISK_MODEL_PATH, seed: int = RANDOM_SEED):
    """Trains a small, self-contained LogisticRegression risk-scoring model.

    Also persists the held-out test set and its predicted probabilities inside
    the same joblib bundle, so the evaluation plots (ROC/PR/score distribution)
    are generated from real held-out predictions rather than re-simulated data.
    """
    rng = np.random.default_rng(seed + 1)
    df = _generate_credit_applicants(rng)
    X = df[FEATURE_COLUMNS]
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "test_accuracy": test_accuracy,
            "X_test": X_test.reset_index(drop=True),
            "y_test": y_test.reset_index(drop=True),
            "y_proba_test": y_proba_test,
        },
        model_path,
    )
    return model, test_accuracy


def main() -> None:
    build_warehouse()
    _, acc = train_credit_risk_model()
    print(f"DuckDB warehouse written to {DUCKDB_PATH}")
    print(f"Credit risk model written to {RISK_MODEL_PATH} (holdout accuracy={acc:.3f})")


if __name__ == "__main__":
    main()
