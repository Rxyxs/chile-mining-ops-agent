import os

from src.visualization.plots import (
    plot_anomaly_scores,
    plot_credit_risk_evaluation,
    plot_credit_risk_interactive,
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


def test_plot_credit_risk_interactive_writes_self_contained_html():
    result = plot_credit_risk_interactive()
    assert os.path.exists(result["figure_path"])
    assert result["figure_path"].endswith(".html")
    with open(result["figure_path"], encoding="utf-8") as f:
        content = f.read()
    assert os.path.getsize(result["figure_path"]) > 0
    # plotly.js must be embedded as an inline <script> block, not loaded from
    # a CDN <script src="..."> tag (a literal "cdn.plot.ly" string can still
    # appear inside the bundled library code itself, which is fine).
    assert '<script src="https://cdn' not in content
    assert "Plotly.newPlot" in content
    # a large inline bundle is expected (plotly.js is several MB unminified)
    assert os.path.getsize(result["figure_path"]) > 500_000
    assert result["n_test"] > 0
