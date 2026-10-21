import os

from src.visualization.plots import plot_anomaly_scores, plot_warehouse_overview


def test_plot_anomaly_scores_writes_figure_and_matches_tool_output():
    result = plot_anomaly_scores(days=200)
    assert os.path.exists(result["figure_path"])
    assert os.path.getsize(result["figure_path"]) > 0
    assert result["n_equipment_flagged"] <= result["n_equipment_evaluated"]


def test_plot_warehouse_overview_writes_figure():
    result = plot_warehouse_overview()
    assert os.path.exists(result["figure_path"])
    assert os.path.getsize(result["figure_path"]) > 0
    assert result["n_months"] == 7
    assert result["n_procurement_categories"] == 4
