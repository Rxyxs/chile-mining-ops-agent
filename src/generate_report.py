"""Generates all result figures and a metrics.json summary from the tools'
real outputs. Run after src/setup_data.py:

    python -m src.setup_data
    python -m src.generate_report
"""
from __future__ import annotations

import json
import os

from src.visualization.plots import (
    plot_anomaly_scores,
    plot_warehouse_overview,
    plot_warehouse_overview_animated,
)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def main() -> dict:
    anomalies = plot_anomaly_scores()
    warehouse = plot_warehouse_overview()
    plot_warehouse_overview_animated()

    summary = {
        "anomaly_check_tool": {
            "window_days": anomalies["window_days"],
            "n_equipment_evaluated": anomalies["n_equipment_evaluated"],
            "n_equipment_flagged": anomalies["n_equipment_flagged"],
            "flagged": [r["equipment_id"] for r in anomalies["flagged"]],
        },
        "warehouse_query_tool": {
            "n_months": warehouse["n_months"],
            "n_procurement_categories": warehouse["n_procurement_categories"],
            "avg_silica_concentrate_pct_by_month": warehouse["avg_silica_concentrate_pct_by_month"],
        },
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nFigures written to reports/figures/, summary written to {metrics_path}")
    return summary


if __name__ == "__main__":
    main()
