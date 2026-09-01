import json
import os

from src.generate_report import main as generate_report


def test_generate_report_writes_metrics_json_with_expected_keys():
    summary = generate_report()

    metrics_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "metrics.json"
    )
    assert os.path.exists(metrics_path)

    with open(metrics_path) as f:
        on_disk = json.load(f)
    assert on_disk == summary

    for key in ("credit_risk_tool", "anomaly_check_tool", "warehouse_query_tool"):
        assert key in summary
