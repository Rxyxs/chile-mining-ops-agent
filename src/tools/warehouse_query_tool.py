"""Named, parameterized queries against the operations warehouse (`data/ops.duckdb`,
built from real public data by `src/setup_data.py`).

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

PROCUREMENT_STATUSES = ["delivered", "accepted", "pending", "cancelled"]
PROCUREMENT_CATEGORIES = ["mining_machinery", "mining_services", "industrial_chemicals", "fuels_lubricants"]


def _connect(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Warehouse not found at {db_path}. Run `python -m src.setup_data` first."
        )
    return duckdb.connect(db_path, read_only=True)


def plant_today(con: duckdb.DuckDBPyConnection) -> str:
    """Last day with plant data: the "today" that lookback windows count back from."""
    return con.execute("SELECT MAX(date) FROM flotation_daily").fetchone()[0]


def get_flotation_summary(month: str, db_path: str = DB_PATH) -> dict:
    """Summarizes the flotation plant for a given month (YYYY-MM)."""
    con = _connect(db_path)
    try:
        row = con.execute(
            """
            SELECT
                COUNT(*) AS n_days,
                SUM(hours_with_data) AS hours_with_data,
                ROUND(AVG(iron_feed_pct), 2) AS avg_iron_feed_pct,
                ROUND(AVG(iron_concentrate_pct), 2) AS avg_iron_concentrate_pct,
                ROUND(AVG(silica_concentrate_pct), 2) AS avg_silica_concentrate_pct,
                SUM(hours_silica_high) AS hours_silica_high,
                ROUND(SUM(ore_pulp_tonnes), 0) AS ore_pulp_tonnes
            FROM flotation_daily
            WHERE month = ?
            """,
            [month],
        ).fetchone()
    finally:
        con.close()

    columns = [
        "n_days",
        "hours_with_data",
        "avg_iron_feed_pct",
        "avg_iron_concentrate_pct",
        "avg_silica_concentrate_pct",
        "hours_silica_high",
        "ore_pulp_tonnes",
    ]
    result = dict(zip(columns, row))
    for column in ("hours_with_data", "hours_silica_high"):
        if result[column] is not None:
            result[column] = int(result[column])  # DuckDB SUM returns HUGEINT
    result["month"] = month
    if result["n_days"] == 0:
        result["note"] = "No plant data for this month (data covers 2017-03 to 2017-09)."
    return result


def get_maintenance_alerts(days: int, db_path: str = DB_PATH) -> dict:
    """Returns equipment/instrument events from the last `days` days of plant data,
    most severe first."""
    if days <= 0:
        raise ValueError("days must be a positive integer")

    con = _connect(db_path)
    try:
        today = plant_today(con)
        rows = con.execute(
            """
            SELECT event_id, start, equipment_id, event_type, severity, downtime_hours
            FROM maintenance_events
            WHERE CAST(date AS DATE) > CAST(? AS DATE) - CAST(? AS INTEGER)
            ORDER BY
                CASE severity
                    WHEN 'critical' THEN 0
                    WHEN 'high' THEN 1
                    WHEN 'medium' THEN 2
                    ELSE 3
                END,
                start DESC
            LIMIT 50
            """,
            [today, days],
        ).fetchall()
    finally:
        con.close()

    columns = ["event_id", "start", "equipment_id", "event_type", "severity", "downtime_hours"]
    events = [dict(zip(columns, r)) for r in rows]
    return {"as_of": today, "window_days": days, "n_events": len(events), "events": events}


def get_procurement_summary(
    status: str | None = None, category: str | None = None, month: str | None = None,
    db_path: str = DB_PATH,
) -> dict:
    """Summarizes public purchase orders (ChileCompra) in mining-relevant categories,
    optionally filtered by status, category and month (YYYY-MM)."""
    if status is not None and status not in PROCUREMENT_STATUSES:
        raise ValueError(f"status must be one of {PROCUREMENT_STATUSES}")
    if category is not None and category not in PROCUREMENT_CATEGORIES:
        raise ValueError(f"category must be one of {PROCUREMENT_CATEGORIES}")

    clauses, params = [], []
    for column, value in (("status", status), ("category", category)):
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    if month is not None:
        clauses.append("substr(date, 1, 7) = ?")
        params.append(month)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    con = _connect(db_path)
    try:
        row = con.execute(
            f"""
            SELECT COUNT(*), ROUND(SUM(amount_clp) / 1e6, 2), ROUND(AVG(amount_clp) / 1e6, 2),
                   ROUND(MEDIAN(amount_clp) / 1e6, 2)
            FROM procurement_orders {where}
            """,
            params,
        ).fetchone()
    finally:
        con.close()

    return {
        "status_filter": status,
        "category_filter": category,
        "month_filter": month,
        "n_orders": row[0],
        # Millions of CLP: with raw pesos (965616295) the 3B model miscopied the
        # amount in most answers; see the README evaluation section.
        "total_amount_million_clp": row[1],
        "avg_amount_million_clp": row[2],
        "median_amount_million_clp": row[3],
    }


WAREHOUSE_QUERY_FUNCTIONS = {
    "get_flotation_summary": get_flotation_summary,
    "get_maintenance_alerts": get_maintenance_alerts,
    "get_procurement_summary": get_procurement_summary,
}

TOOL_DEFINITIONS = [
    {
        "name": "get_flotation_summary",
        "description": (
            "Returns monthly performance of the iron-ore flotation plant: days and hours "
            "with data, average iron in feed and concentrate, average silica in concentrate, "
            "hours with high silica (above 4.14%) and ore pulp tonnes. Data covers "
            "2017-03 to 2017-09."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "month": {
                    "type": "string",
                    "description": "Month in YYYY-MM format, e.g. '2017-06'.",
                }
            },
            "required": ["month"],
        },
    },
    {
        "name": "get_maintenance_alerts",
        "description": (
            "Returns equipment and instrument events (stuck sensors, historian freezes, "
            "data gaps) from the last N days of plant data, most severe first."
        ),
        "parameters": {
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
            "Summarizes Chilean public purchase orders (ChileCompra, 2017-03 to 2017-09) in "
            "mining-relevant categories: count, total, average and median amount in millions of CLP. "
            "Optional filters: status, category, month."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Optional status filter.",
                    "enum": PROCUREMENT_STATUSES,
                },
                "category": {
                    "type": "string",
                    "description": "Optional category filter.",
                    "enum": PROCUREMENT_CATEGORIES,
                },
                "month": {
                    "type": "string",
                    "description": "Optional month in YYYY-MM format.",
                },
            },
            "required": [],
        },
    },
]
