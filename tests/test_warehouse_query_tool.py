import pytest

from src.tools.warehouse_query_tool import (
    get_flotation_summary,
    get_maintenance_alerts,
    get_procurement_summary,
)


def test_get_flotation_summary_full_month_matches_real_plant():
    result = get_flotation_summary("2017-06")
    assert result["n_days"] == 30
    assert result["hours_with_data"] == 720
    # Iron ore concentrate around 65% Fe, silica around 2%: the real plant's range.
    assert 63 < result["avg_iron_concentrate_pct"] < 67
    assert 1 < result["avg_silica_concentrate_pct"] < 4
    assert 0 <= result["hours_silica_high"] <= result["hours_with_data"]
    assert result["month"] == "2017-06"


def test_get_flotation_summary_unknown_month():
    result = get_flotation_summary("2025-09")
    assert result["n_days"] == 0
    assert "note" in result


def test_get_maintenance_alerts_counts_back_from_last_plant_day():
    result = get_maintenance_alerts(days=60)
    assert result["as_of"] == "2017-09-09"
    assert result["window_days"] == 60
    assert isinstance(result["events"], list)
    if result["events"]:
        first = result["events"][0]
        assert set(first.keys()) == {"event_id", "start", "equipment_id", "event_type", "severity", "downtime_hours"}


def test_get_maintenance_alerts_last_week_is_empty_in_real_data():
    # The last detected event is on 2017-08-17; the plant data ends 2017-09-09.
    assert get_maintenance_alerts(days=7)["n_events"] == 0


def test_get_maintenance_alerts_rejects_bad_days():
    with pytest.raises(ValueError):
        get_maintenance_alerts(days=0)


def test_get_procurement_summary_all():
    result = get_procurement_summary()
    assert result["n_orders"] > 10_000
    assert result["total_amount_million_clp"] > 0


def test_get_procurement_summary_filters_combine():
    everything = get_procurement_summary(category="mining_machinery")
    delivered = get_procurement_summary(status="delivered", category="mining_machinery")
    june = get_procurement_summary(category="mining_machinery", month="2017-06")
    assert 0 < delivered["n_orders"] < everything["n_orders"]
    assert 0 < june["n_orders"] < everything["n_orders"]


def test_get_procurement_summary_rejects_unknown_status():
    with pytest.raises(ValueError):
        get_procurement_summary(status="delayed")
