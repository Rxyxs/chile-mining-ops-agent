import pytest

from src.tools.warehouse_query_tool import (
    get_flotation_summary,
    get_maintenance_alerts,
    get_procurement_summary,
)


def test_get_flotation_summary_known_month():
    result = get_flotation_summary("2025-09")
    assert result["n_batches"] > 0
    assert 0 < result["avg_recovery_pct"] <= 100
    assert result["month"] == "2025-09"


def test_get_flotation_summary_unknown_month():
    result = get_flotation_summary("1999-01")
    assert result["n_batches"] == 0
    assert "note" in result


def test_get_maintenance_alerts_returns_events():
    result = get_maintenance_alerts(days=30)
    assert result["window_days"] == 30
    assert result["n_events"] >= 0
    assert isinstance(result["events"], list)
    if result["events"]:
        first = result["events"][0]
        assert set(first.keys()) == {
            "event_id",
            "date",
            "equipment_id",
            "event_type",
            "severity",
            "downtime_hours",
        }


def test_get_maintenance_alerts_rejects_bad_days():
    with pytest.raises(ValueError):
        get_maintenance_alerts(days=0)


def test_get_procurement_summary_all():
    result = get_procurement_summary()
    assert result["n_orders"] > 0
    assert result["total_amount_usd"] > 0


def test_get_procurement_summary_filtered():
    result = get_procurement_summary(status="delivered")
    assert result["status_filter"] == "delivered"
    assert result["n_orders"] >= 0
