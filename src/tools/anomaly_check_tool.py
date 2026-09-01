"""Anomaly detection tool over the synthetic operations warehouse.

Runs an Isolation Forest over recent maintenance/downtime data to flag
statistically unusual equipment behavior (e.g. abnormal downtime spikes).
"""
from __future__ import annotations

import os

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "ops.duckdb",
)


def _connect(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Warehouse not found at {db_path}. Run `python -m src.setup_data` first."
        )
    return duckdb.connect(db_path, read_only=True)


def check_maintenance_anomalies(
    days: int = 30, contamination: float = 0.1, db_path: str = DB_PATH
) -> dict:
    """Flags equipment with anomalous aggregate downtime/event patterns over
    the last `days` days, using an Isolation Forest on per-equipment features
    (event count, total downtime, share of critical/high severity events).
    """
    if days <= 0:
        raise ValueError("days must be a positive integer")
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5")

    con = _connect(db_path)
    try:
        max_date = con.execute("SELECT MAX(date) FROM maintenance_events").fetchone()[0]
        df = con.execute(
            """
            SELECT equipment_id, event_type, severity, downtime_hours
            FROM maintenance_events
            WHERE CAST(date AS DATE) >= CAST(? AS DATE) - CAST(? AS INTEGER)
            """,
            [max_date, days],
        ).fetchdf()
    finally:
        con.close()

    if df.empty:
        return {"window_days": days, "n_equipment_flagged": 0, "flagged": []}

    grouped = df.groupby("equipment_id").agg(
        n_events=("event_type", "count"),
        total_downtime_hours=("downtime_hours", "sum"),
        n_critical_or_high=(
            "severity",
            lambda s: int(((s == "critical") | (s == "high")).sum()),
        ),
    )
    grouped["critical_share"] = grouped["n_critical_or_high"] / grouped["n_events"]

    if len(grouped) < 3:
        # Not enough equipment to fit a meaningful model.
        return {"window_days": days, "n_equipment_flagged": 0, "flagged": []}

    features = grouped[["n_events", "total_downtime_hours", "critical_share"]]
    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=42
    )
    labels = model.fit_predict(features)
    scores = model.decision_function(features)

    grouped = grouped.assign(anomaly_score=scores, is_anomaly=(labels == -1))
    flagged = (
        grouped[grouped["is_anomaly"]]
        .sort_values("anomaly_score")
        .reset_index()
    )

    result_records = []
    for _, r in flagged.iterrows():
        result_records.append(
            {
                "equipment_id": r["equipment_id"],
                "n_events": int(r["n_events"]),
                "total_downtime_hours": round(float(r["total_downtime_hours"]), 2),
                "critical_share": round(float(r["critical_share"]), 3),
                "anomaly_score": round(float(r["anomaly_score"]), 4),
            }
        )

    return {
        "window_days": days,
        "n_equipment_evaluated": int(len(grouped)),
        "n_equipment_flagged": len(result_records),
        "flagged": result_records,
    }


TOOL_SCHEMAS = [
    {
        "name": "check_maintenance_anomalies",
        "description": (
            "Runs an Isolation Forest over recent equipment maintenance data to "
            "flag equipment with statistically anomalous event/downtime patterns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Lookback window in days (default 30).",
                },
                "contamination": {
                    "type": "number",
                    "description": "Expected fraction of anomalous equipment (0-0.5, default 0.1).",
                },
            },
            "required": [],
        },
    }
]
