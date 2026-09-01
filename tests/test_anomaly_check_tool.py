import pytest

from src.tools.anomaly_check_tool import check_maintenance_anomalies


def test_check_maintenance_anomalies_runs():
    result = check_maintenance_anomalies(days=365, contamination=0.1)
    assert result["window_days"] == 365
    assert "flagged" in result
    assert isinstance(result["flagged"], list)
    assert result["n_equipment_flagged"] == len(result["flagged"])
    if result["flagged"]:
        first = result["flagged"][0]
        assert set(first.keys()) == {
            "equipment_id",
            "n_events",
            "total_downtime_hours",
            "critical_share",
            "anomaly_score",
        }


def test_check_maintenance_anomalies_rejects_bad_days():
    with pytest.raises(ValueError):
        check_maintenance_anomalies(days=0)


def test_check_maintenance_anomalies_rejects_bad_contamination():
    with pytest.raises(ValueError):
        check_maintenance_anomalies(days=30, contamination=0.9)


def test_check_maintenance_anomalies_short_window_handles_sparse_data():
    result = check_maintenance_anomalies(days=1, contamination=0.1)
    assert result["window_days"] == 1
    assert isinstance(result["flagged"], list)
