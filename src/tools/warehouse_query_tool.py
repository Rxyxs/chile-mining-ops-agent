"""Named, parameterized queries against the synthetic operations warehouse.

By design there is no arbitrary-SQL-from-the-user entry point here: every
query the agent can run is a fixed, named Python function with a small,
validated set of parameters. This keeps the tool safe to expose to an LLM
tool-use loop.
"""
from __future__ import annotations

import os

import duckdb

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


def get_flotation_summary(month: str, db_path: str = DB_PATH) -> dict:
    """Summarizes flotation batch performance for a given month (YYYY-MM)."""
    con = _connect(db_path)
    try:
        row = con.execute(
            """
            SELECT
                COUNT(*) AS n_batches,
                ROUND(AVG(feed_grade_pct), 3) AS avg_feed_grade_pct,
                ROUND(AVG(recovery_pct), 2) AS avg_recovery_pct,
                ROUND(AVG(concentrate_grade_pct), 2) AS avg_concentrate_grade_pct,
                ROUND(SUM(tonnage_processed), 1) AS total_tonnage_processed
            FROM flotation_batches
            WHERE month = ?
            """,
            [month],
        ).fetchone()
    finally:
        con.close()

    columns = [
        "n_batches",
        "avg_feed_grade_pct",
        "avg_recovery_pct",
        "avg_concentrate_grade_pct",
        "total_tonnage_processed",
    ]
    result = dict(zip(columns, row))
    result["month"] = month
    if result["n_batches"] == 0:
        result["note"] = "No flotation batches recorded for this month."
    return result


def get_maintenance_alerts(days: int, db_path: str = DB_PATH) -> dict:
    """Returns maintenance events from the last `days` days, most severe first."""
    if days <= 0:
        raise ValueError("days must be a positive integer")

    con = _connect(db_path)
    try:
        max_date = con.execute("SELECT MAX(date) FROM maintenance_events").fetchone()[0]
        rows = con.execute(
            """
            SELECT event_id, date, equipment_id, event_type, severity, downtime_hours
            FROM maintenance_events
            WHERE CAST(date AS DATE) >= CAST(? AS DATE) - CAST(? AS INTEGER)
            ORDER BY
                CASE severity
                    WHEN 'critical' THEN 0
                    WHEN 'high' THEN 1
                    WHEN 'medium' THEN 2
                    ELSE 3
                END,
                date DESC
            LIMIT 50
            """,
            [max_date, days],
        ).fetchall()
    finally:
        con.close()

    columns = ["event_id", "date", "equipment_id", "event_type", "severity", "downtime_hours"]
    events = [dict(zip(columns, r)) for r in rows]
    return {"window_days": days, "n_events": len(events), "events": events}


def get_procurement_summary(status: str | None = None, db_path: str = DB_PATH) -> dict:
    """Summarizes procurement orders, optionally filtered by status."""
    con = _connect(db_path)
    try:
        if status:
            row = con.execute(
                """
                SELECT COUNT(*), ROUND(SUM(amount_usd), 2), ROUND(AVG(amount_usd), 2)
                FROM procurement_orders WHERE status = ?
                """,
                [status],
            ).fetchone()
        else:
            row = con.execute(
                """
                SELECT COUNT(*), ROUND(SUM(amount_usd), 2), ROUND(AVG(amount_usd), 2)
                FROM procurement_orders
                """
            ).fetchone()
    finally:
        con.close()

    return {
        "status_filter": status,
        "n_orders": row[0],
        "total_amount_usd": row[1],
        "avg_amount_usd": row[2],
    }


WAREHOUSE_QUERY_FUNCTIONS = {
    "get_flotation_summary": get_flotation_summary,
    "get_maintenance_alerts": get_maintenance_alerts,
    "get_procurement_summary": get_procurement_summary,
}

TOOL_SCHEMAS = [
    {
        "name": "get_flotation_summary",
        "description": (
            "Returns aggregate flotation plant performance (feed grade, recovery, "
            "concentrate grade, tonnage) for a given month."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "month": {
                    "type": "string",
                    "description": "Month in YYYY-MM format, e.g. '2025-09'.",
                }
            },
            "required": ["month"],
        },
    },
    {
        "name": "get_maintenance_alerts",
        "description": (
            "Returns maintenance/alert events from the last N days across the "
            "equipment fleet, ordered by severity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Size of the lookback window in days (e.g. 7).",
                }
            },
            "required": ["days"],
        },
    },
    {
        "name": "get_procurement_summary",
        "description": (
            "Summarizes procurement orders (count, total and average amount in USD), "
            "optionally filtered by status (delivered, pending, delayed, cancelled)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Optional status filter.",
                    "enum": ["delivered", "pending", "delayed", "cancelled"],
                }
            },
            "required": [],
        },
    },
]
