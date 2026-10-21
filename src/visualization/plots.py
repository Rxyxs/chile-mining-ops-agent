"""Result plots for the tools' own outputs -- anomaly scores and warehouse trends.
Every figure here is generated from the real query results produced by
src/setup_data.py and src/tools/*, not from re-simulated or hardcoded numbers.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # figures are written to files; no display needed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation
import numpy as np

from src.tools.anomaly_check_tool import check_maintenance_anomalies, DB_PATH as WAREHOUSE_DB_PATH
from src.tools.warehouse_query_tool import _connect, plant_today

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


def plot_anomaly_scores(
    days: int = 200, contamination: float = 0.1, db_path: str = WAREHOUSE_DB_PATH, out_dir: str = FIGURES_DIR
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
        max_date = plant_today(con)
        df = con.execute(
            """
            SELECT equipment_id, COUNT(*) AS n_events, SUM(downtime_hours) AS total_downtime_hours
            FROM maintenance_events
            WHERE CAST(date AS DATE) > CAST(? AS DATE) - CAST(? AS INTEGER)
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
    ax.set_yscale("log")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["equipment_id"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Event hours (log scale)", color=INK_SECONDARY, fontsize=9)
    ax.set_title(
        f"Event hours by equipment, last {days}d of plant data "
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
    """Two-panel overview of the warehouse: monthly silica in concentrate at the
    flotation plant, and public procurement spend by category."""
    con = _connect(db_path)
    try:
        flotation = con.execute(
            """
            SELECT month, AVG(silica_concentrate_pct) AS avg_silica_concentrate_pct,
                   SUM(hours_silica_high) * 100.0 / SUM(hours_with_data) AS pct_hours_silica_high
            FROM flotation_daily GROUP BY month ORDER BY month
            """
        ).fetchdf()
        procurement = con.execute(
            """
            SELECT category, SUM(amount_clp) / 1e9 AS total_amount_bn_clp
            FROM procurement_orders GROUP BY category ORDER BY total_amount_bn_clp DESC
            """
        ).fetchdf()
    finally:
        con.close()

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)

    ax = axes[0]
    ax.plot(flotation["month"], flotation["avg_silica_concentrate_pct"], color=CAT["green"], linewidth=2, marker="o", markersize=4)
    _style_axes(ax)
    ax.set_xticks(range(len(flotation)))
    ax.set_xticklabels(flotation["month"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Avg. silica in concentrate (%)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Silica in concentrate by month (lower is better)", color=INK_PRIMARY, fontsize=11,
                 fontweight="bold", loc="left")

    ax = axes[1]
    ax.barh(procurement["category"], procurement["total_amount_bn_clp"], color=CAT["violet"], zorder=3)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Total net amount (billion CLP)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Public procurement, Mar-Sep 2017", color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left")

    fig.tight_layout()
    out_path = os.path.join(out_dir, "warehouse_overview.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    return {
        "n_months": int(len(flotation)),
        "n_procurement_categories": int(len(procurement)),
        "avg_silica_concentrate_pct_by_month": {
            m: round(float(v), 2) for m, v in zip(flotation["month"], flotation["avg_silica_concentrate_pct"])
        },
        "figure_path": out_path,
    }


def plot_warehouse_overview_animated(db_path: str = WAREHOUSE_DB_PATH, out_dir: str = FIGURES_DIR) -> dict:
    """Animated 'racing line chart' GIF of the monthly silica in concentrate
    trend, built from the exact same query as plot_warehouse_overview(). The
    static PNG remains the primary artifact -- this is a companion GIF for
    visual impact, saved via the Pillow writer (no ffmpeg dependency).
    """
    con = _connect(db_path)
    try:
        flotation = con.execute(
            """
            SELECT month, AVG(silica_concentrate_pct) AS avg_silica_concentrate_pct,
                   SUM(hours_silica_high) * 100.0 / SUM(hours_with_data) AS pct_hours_silica_high
            FROM flotation_daily GROUP BY month ORDER BY month
            """
        ).fetchdf()
    finally:
        con.close()

    os.makedirs(out_dir, exist_ok=True)

    months = flotation["month"].tolist()
    values = flotation["avg_silica_concentrate_pct"].to_numpy()
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
        pad = (values.max() - values.min()) * 0.15 + 0.1
        ax.set_ylim(values.min() - pad, values.max() + pad)
        ax.set_xticks(range(n_points))
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Avg. silica in concentrate (%)", fontsize=10)
        ax.set_title("Silica in concentrate by month (animated)", fontsize=13, fontweight="bold", loc="left")
        ax.grid(color="#444444", linewidth=0.6, alpha=0.6)
        fig.tight_layout()

        def update(frame_idx):
            i = frame_indices[frame_idx]
            x = list(range(i + 1))
            y = values[: i + 1]
            line.set_data(x, y)
            label.xy = (i, values[i])
            label.set_text(f"Silica: {values[i]:.2f}%  ({months[i]})")
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
