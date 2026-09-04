"""Result plots for the tools' own outputs -- credit-risk evaluation, anomaly
scores, and warehouse trends. Every figure here is generated from the real
held-out predictions / query results produced by src/setup_data.py and
src/tools/*, not from re-simulated or hardcoded numbers.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

from src.tools.anomaly_check_tool import check_maintenance_anomalies, DB_PATH as WAREHOUSE_DB_PATH
from src.tools.credit_risk_tool import MODEL_PATH as RISK_MODEL_PATH
from src.tools.warehouse_query_tool import _connect

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
CAT = {"blue": "#2a78d6", "orange": "#eb6834", "red": "#e34948", "green": "#008300", "violet": "#4a3aa7"}

FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "reports",
    "figures",
)

INTERACTIVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "outputs",
    "interactive",
)


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_credit_risk_evaluation(
    model_path: str = RISK_MODEL_PATH, out_dir: str = FIGURES_DIR
) -> dict:
    """ROC curve, PR curve, and predicted-probability distribution over the
    held-out test set saved by src/setup_data.py. Returns the real metrics
    computed (ROC-AUC, PR-AUC, test accuracy) so callers can report them.
    """
    bundle = load(model_path)
    y_test = np.asarray(bundle["y_test"])
    y_proba = np.asarray(bundle["y_proba_test"])

    roc_auc = float(roc_auc_score(y_test, y_proba))
    pr_auc = float(average_precision_score(y_test, y_proba))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=SURFACE)

    ax = axes[0]
    ax.plot(fpr, tpr, color=CAT["blue"], linewidth=2, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=INK_SECONDARY, linewidth=1, linestyle="--")
    _style_axes(ax)
    ax.set_xlabel("False Positive Rate", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("True Positive Rate", color=INK_SECONDARY, fontsize=9)
    ax.set_title("ROC curve", color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    ax.plot(recall, precision, color=CAT["orange"], linewidth=2, label=f"PR-AUC = {pr_auc:.3f}")
    baseline = float(y_test.mean())
    ax.axhline(baseline, color=INK_SECONDARY, linewidth=1, linestyle="--", label=f"base rate = {baseline:.3f}")
    _style_axes(ax)
    ax.set_xlabel("Recall", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Precision", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Precision-Recall curve", color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    bins = np.linspace(0, 1, 30)
    ax.hist(y_proba[y_test == 0], bins=bins, alpha=0.65, color=CAT["blue"], label="no default", density=True)
    ax.hist(y_proba[y_test == 1], bins=bins, alpha=0.65, color=CAT["red"], label="default", density=True)
    _style_axes(ax)
    ax.set_xlabel("Predicted P(default)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Density", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Predicted probability by outcome", color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        "Credit risk tool -- held-out test evaluation", color=INK_PRIMARY, fontsize=13, fontweight="bold", x=0.02, ha="left"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join(out_dir, "credit_risk_evaluation.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "test_accuracy": round(float(bundle.get("test_accuracy", float("nan"))), 4),
        "n_test": int(len(y_test)),
        "base_rate": round(baseline, 4),
        "figure_path": out_path,
    }


def plot_ceiling_comparison(out_dir: str = FIGURES_DIR) -> dict:
    """Bar chart comparing the fitted models' AUC against the oracle ceiling --
    the best AUC any model could ever get against this label, computed from the
    true data-generating probability itself (see src/model_ceiling_check.py)."""
    from src.model_ceiling_check import compute_ceiling_comparison

    result = compute_ceiling_comparison()
    os.makedirs(out_dir, exist_ok=True)

    labels = ["LogisticRegression\n(deployed)", "GradientBoosting\n(comparison only)", "Oracle ceiling\n(true probability)"]
    values = [result["logistic_regression_auc"], result["gradient_boosting_auc"], result["oracle_auc"]]
    colors = [CAT["blue"], CAT["orange"], INK_SECONDARY]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    bars = ax.bar(labels, values, color=colors, zorder=3, width=0.55)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, color=INK_PRIMARY)
    ax.axhline(0.5, color=INK_SECONDARY, linewidth=1, linestyle="--", alpha=0.6)
    ax.text(0.0, 0.51, "random guessing", fontsize=8, color=INK_SECONDARY, ha="left")
    _style_axes(ax)
    ax.set_ylim(0.45, max(values) + 0.08)
    ax.set_ylabel("ROC-AUC (held-out test)", color=INK_SECONDARY, fontsize=9)
    ax.set_title(
        f"Deployed model captures {result['logistic_ceiling_capture_pct']:.0f}% of the AUC ceiling",
        color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left",
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, "ceiling_comparison.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    return {**result, "figure_path": out_path}


def plot_credit_risk_interactive(
    model_path: str = RISK_MODEL_PATH, out_dir: str = INTERACTIVE_DIR
) -> dict:
    """Interactive Plotly scatter of the held-out test set: predicted
    P(default) vs. debt-to-income, colored by the real outcome and with a
    hover tooltip showing the applicant's other features. Same held-out
    predictions as plot_credit_risk_evaluation() -- nothing re-simulated.
    Saved as a single self-contained HTML file (inline plotly.js) so it
    renders standalone, e.g. via htmlpreview.github.io.
    """
    import plotly.graph_objects as go

    bundle = load(model_path)
    X_test = bundle["X_test"]
    y_test = np.asarray(bundle["y_test"])
    y_proba = np.asarray(bundle["y_proba_test"])

    os.makedirs(out_dir, exist_ok=True)

    fig = go.Figure()
    for label, name, color in [(0, "no default (actual)", CAT["blue"]), (1, "default (actual)", CAT["red"])]:
        mask = y_test == label
        rows = X_test[mask]
        hover = [
            f"age={a}<br>income_clp={inc:,.0f}<br>debt_to_income={d:.2f}<br>"
            f"months_employed={m:.0f}<br>late_payments={lp}<br>requested_clp={r:,.0f}<br>"
            f"P(default)={p:.3f}"
            for a, inc, d, m, lp, r, p in zip(
                rows["age"], rows["monthly_income_clp"], rows["debt_to_income"],
                rows["months_employed"], rows["n_late_payments"], rows["requested_amount_clp"],
                y_proba[mask],
            )
        ]
        fig.add_trace(
            go.Scatter(
                x=rows["debt_to_income"],
                y=y_proba[mask],
                mode="markers",
                name=name,
                marker=dict(color=color, size=8, opacity=0.7, line=dict(width=0.5, color="white")),
                text=hover,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=f"Credit risk tool -- held-out test set (n={len(y_test)}): predicted P(default) vs. debt-to-income",
        xaxis_title="Debt-to-income ratio",
        yaxis_title="Predicted P(default)",
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(color=INK_PRIMARY),
        legend=dict(orientation="h", y=-0.15),
        width=950,
        height=560,
    )

    out_path = os.path.join(out_dir, "credit_risk_scores.html")
    fig.write_html(out_path, include_plotlyjs="inline", full_html=True)

    return {"n_test": int(len(y_test)), "figure_path": out_path}


def plot_anomaly_scores(
    days: int = 60, contamination: float = 0.1, db_path: str = WAREHOUSE_DB_PATH, out_dir: str = FIGURES_DIR
) -> dict:
    """Bar chart of per-equipment anomaly scores from the Isolation Forest
    tool, flagged equipment highlighted. Uses check_maintenance_anomalies()
    directly -- the same function the agent calls at runtime -- so the plot
    reflects exactly what the tool would report.
    """
    result = check_maintenance_anomalies(days=days, contamination=contamination, db_path=db_path)
    flagged_ids = {r["equipment_id"] for r in result["flagged"]}

    con = _connect(db_path)
    try:
        max_date = con.execute("SELECT MAX(date) FROM maintenance_events").fetchone()[0]
        df = con.execute(
            """
            SELECT equipment_id, COUNT(*) AS n_events, SUM(downtime_hours) AS total_downtime_hours
            FROM maintenance_events
            WHERE CAST(date AS DATE) >= CAST(? AS DATE) - CAST(? AS INTEGER)
            GROUP BY equipment_id
            ORDER BY equipment_id
            """,
            [max_date, days],
        ).fetchdf()
    finally:
        con.close()

    os.makedirs(out_dir, exist_ok=True)
    colors = [CAT["red"] if eq in flagged_ids else CAT["blue"] for eq in df["equipment_id"]]

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=SURFACE)
    ax.bar(df["equipment_id"], df["total_downtime_hours"], color=colors, zorder=3)
    _style_axes(ax)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["equipment_id"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Total downtime (hours)", color=INK_SECONDARY, fontsize=9)
    ax.set_title(
        f"Maintenance downtime by equipment, last {days}d "
        f"({result['n_equipment_flagged']}/{result['n_equipment_evaluated']} flagged anomalous)",
        color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left",
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CAT["red"], label="flagged anomalous"),
        plt.Rectangle((0, 0), 1, 1, color=CAT["blue"], label="normal"),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "anomaly_scores.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    return {**result, "figure_path": out_path}


def plot_warehouse_overview(db_path: str = WAREHOUSE_DB_PATH, out_dir: str = FIGURES_DIR) -> dict:
    """Two-panel overview of the synthetic warehouse: monthly flotation
    recovery trend, and procurement spend by category."""
    con = _connect(db_path)
    try:
        flotation = con.execute(
            """
            SELECT month, AVG(recovery_pct) AS avg_recovery_pct, AVG(feed_grade_pct) AS avg_feed_grade_pct
            FROM flotation_batches GROUP BY month ORDER BY month
            """
        ).fetchdf()
        procurement = con.execute(
            """
            SELECT category, SUM(amount_usd) AS total_amount_usd
            FROM procurement_orders GROUP BY category ORDER BY total_amount_usd DESC
            """
        ).fetchdf()
    finally:
        con.close()

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)

    ax = axes[0]
    ax.plot(flotation["month"], flotation["avg_recovery_pct"], color=CAT["green"], linewidth=2, marker="o", markersize=4)
    _style_axes(ax)
    ax.set_xticks(range(len(flotation)))
    ax.set_xticklabels(flotation["month"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Avg. recovery (%)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Flotation recovery by month", color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left")

    ax = axes[1]
    ax.barh(procurement["category"], procurement["total_amount_usd"], color=CAT["violet"], zorder=3)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Total spend (USD)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Procurement spend by category", color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left")

    fig.tight_layout()
    out_path = os.path.join(out_dir, "warehouse_overview.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    return {
        "n_months": int(len(flotation)),
        "n_procurement_categories": int(len(procurement)),
        "figure_path": out_path,
    }


def plot_warehouse_overview_animated(db_path: str = WAREHOUSE_DB_PATH, out_dir: str = FIGURES_DIR) -> dict:
    """Animated 'racing line chart' GIF of the monthly flotation recovery
    trend, built from the exact same query as plot_warehouse_overview(). The
    static PNG remains the primary artifact -- this is a companion GIF for
    visual impact, saved via the Pillow writer (no ffmpeg dependency).
    """
    con = _connect(db_path)
    try:
        flotation = con.execute(
            """
            SELECT month, AVG(recovery_pct) AS avg_recovery_pct, AVG(feed_grade_pct) AS avg_feed_grade_pct
            FROM flotation_batches GROUP BY month ORDER BY month
            """
        ).fetchdf()
    finally:
        con.close()

    os.makedirs(out_dir, exist_ok=True)

    months = flotation["month"].tolist()
    values = flotation["avg_recovery_pct"].to_numpy()
    n_points = len(values)

    # Subsample real data points into ~30-60 animation frames (no fabrication).
    n_frames = min(45, n_points) if n_points > 1 else 1
    frame_indices = sorted(set(np.linspace(0, n_points - 1, n_frames).round().astype(int).tolist()))

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 6))
        line, = ax.plot([], [], color="#3ddc84", linewidth=2.5, marker="o", markersize=5)
        label = ax.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            fontsize=11, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1f1f1f", ec="#3ddc84", lw=1.2),
        )

        ax.set_xlim(-0.5, n_points - 0.5)
        pad = (values.max() - values.min()) * 0.15 + 0.5
        ax.set_ylim(values.min() - pad, values.max() + pad)
        ax.set_xticks(range(n_points))
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Avg. recovery (%)", fontsize=10)
        ax.set_title("Flotation recovery by month (animated)", fontsize=13, fontweight="bold", loc="left")
        ax.grid(color="#444444", linewidth=0.6, alpha=0.6)
        fig.tight_layout()

        def update(frame_idx):
            i = frame_indices[frame_idx]
            x = list(range(i + 1))
            y = values[: i + 1]
            line.set_data(x, y)
            label.xy = (i, values[i])
            label.set_text(f"Recovery: {values[i]:.1f}%  ({months[i]})")
            return line, label

        ani = FuncAnimation(fig, update, frames=len(frame_indices), interval=150, blit=False, repeat=True)

        out_path = os.path.join(out_dir, "warehouse_overview_animated.gif")
        ani.save(out_path, writer="pillow", fps=7)
        plt.close(fig)

    return {
        "n_months": int(len(flotation)),
        "n_frames": len(frame_indices),
        "figure_path": out_path,
    }
