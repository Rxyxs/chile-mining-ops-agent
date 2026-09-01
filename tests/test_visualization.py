import os

from src.visualization.plots import (
    plot_anomaly_scores,
    plot_credit_risk_evaluation,
    plot_warehouse_overview,
)


def test_plot_credit_risk_evaluation_writes_figure_and_real_metrics():
    result = plot_credit_risk_evaluation()
    assert os.path.exists(result["figure_path"])
    assert os.path.getsize(result["figure_path"]) > 0
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert 0.0 <= result["pr_auc"] <= 1.0
    assert result["n_test"] > 0


def test_plot_anomaly_scores_writes_figure_and_matches_tool_output():
    result = plot_anomaly_scores(days=60)
    assert os.path.exists(result["figure_path"])
    assert os.path.getsize(result["figure_path"]) > 0
    assert result["n_equipment_flagged"] <= result["n_equipment_evaluated"]


def test_plot_warehouse_overview_writes_figure():
    result = plot_warehouse_overview()
    assert os.path.exists(result["figure_path"])
    assert os.path.getsize(result["figure_path"]) > 0
    assert result["n_months"] > 0
    assert result["n_procurement_categories"] > 0
