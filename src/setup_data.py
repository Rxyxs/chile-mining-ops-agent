"""Builds the operations warehouse (`data/ops.duckdb`) from real public data.

Three tables, all from real sources:

- `flotation_daily`: daily averages of the Kaggle "Quality Prediction in a Mining
  Process" dataset (an iron-ore reverse flotation plant, 20-second readings and hourly
  lab assays, March to September 2017).
- `maintenance_events`: equipment/instrument events detected in the same plant data.
  A `stuck_sensor` event is a process tag that repeats exactly the same 20-second value
  for at least one hour (a frozen transmitter or a tag held by hand); a `data_gap` is
  an hour missing from the plant historian, and a `historian_freeze` is five or more tags
  frozen from the same hour (the feed holding its last values). Severity comes from the
  duration: under 4 h low, 4-12 h medium, 12-24 h high, 24 h or more critical.
- `procurement_orders`: real ChileCompra purchase-order lines (Mercado Público open
  data) in four mining-relevant categories, same months as the plant.

The raw files are downloaded by `src/sources.py` into `data/raw/` (gitignored, about
900 MB). The derived warehouse is small and is committed, so tests and the tools run
from a clean clone without downloading anything. Rebuild with:

    python -m src.sources
    python -m src.setup_data
"""
from __future__ import annotations

import os
import zipfile

import duckdb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
DUCKDB_PATH = os.path.join(DATA_DIR, "ops.duckdb")
FLOTATION_CSV = os.path.join(RAW_DIR, "MiningProcess_Flotation_Plant_Database.csv")
MONTHS = [(2017, m) for m in range(3, 10)]

# Plant's high-silica cut: P90 of the hourly silica in concentrate (4.14%), the same
# threshold the slm-industrial-gateway repo uses for this plant.
SILICA_HIGH_PCT = 4.14
SAMPLES_PER_HOUR = 180  # 20-second readings
STUCK_MIN_SAMPLES = SAMPLES_PER_HOUR  # one hour of identical readings
FREEZE_MIN_TAGS = 5  # this many tags frozen from the same hour = historian freeze

# Process tags (20-second instruments). The four % columns are lab assays, hourly or
# interpolated by design, so a flat run there is not a fault and they are left out.
PROCESS_TAGS = {
    "Starch Flow": "starch_flow",
    "Amina Flow": "amina_flow",
    "Ore Pulp Flow": "ore_pulp_flow",
    "Ore Pulp pH": "ore_pulp_ph",
    "Ore Pulp Density": "ore_pulp_density",
    **{f"Flotation Column 0{i} Air Flow": f"column_0{i}_air_flow" for i in range(1, 8)},
    **{f"Flotation Column 0{i} Level": f"column_0{i}_level" for i in range(1, 8)},
}

# ChileCompra RubroN1 -> category used by the tool.
PROCUREMENT_CATEGORIES = {
    "Maquinaria para minería y perforación": "mining_machinery",
    "Servicios de perforación de minería, petróleo y gas": "mining_services",
    "Productos químicos industriales": "industrial_chemicals",
    "Combustibles, lubricantes y anticorrosivos": "fuels_lubricants",
}
# ChileCompra order status -> status used by the tool.
PROCUREMENT_STATUS = {
    "Recepcion Conforme": "delivered",
    "Aceptada": "accepted",
    "Enviada a proveedor": "pending",
    "En proceso": "pending",
    "Cancelacion solicitada": "cancelled",
}


def _severity(hours: float) -> str:
    if hours >= 24:
        return "critical"
    if hours >= 12:
        return "high"
    if hours >= 4:
        return "medium"
    return "low"


def load_flotation_raw(path: str = FLOTATION_CSV) -> pd.DataFrame:
    raw = pd.read_csv(path, decimal=",")
    raw["date"] = pd.to_datetime(raw["date"])
    return raw


def build_flotation_daily(raw: pd.DataFrame) -> pd.DataFrame:
    hourly = raw.groupby("date").mean(numeric_only=True)
    hourly["silica_high"] = hourly["% Silica Concentrate"] > SILICA_HIGH_PCT
    day = hourly.index.floor("D")
    daily = hourly.groupby(day).agg(
        hours_with_data=("Ore Pulp Flow", "size"),
        iron_feed_pct=("% Iron Feed", "mean"),
        silica_feed_pct=("% Silica Feed", "mean"),
        iron_concentrate_pct=("% Iron Concentrate", "mean"),
        silica_concentrate_pct=("% Silica Concentrate", "mean"),
        hours_silica_high=("silica_high", "sum"),
        amina_flow=("Amina Flow", "mean"),
        starch_flow=("Starch Flow", "mean"),
        ore_pulp_ph=("Ore Pulp pH", "mean"),
        # Ore Pulp Flow is in t/h; one hourly mean x one hour = tonnes that hour.
        ore_pulp_tonnes=("Ore Pulp Flow", "sum"),
    )
    daily.index.name = "date"
    daily = daily.reset_index()
    daily["month"] = daily["date"].dt.strftime("%Y-%m")
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    for col in ["iron_feed_pct", "silica_feed_pct", "iron_concentrate_pct", "silica_concentrate_pct"]:
        daily[col] = daily[col].round(3)
    for col in ["amina_flow", "starch_flow", "ore_pulp_tonnes"]:
        daily[col] = daily[col].round(1)
    daily["ore_pulp_ph"] = daily["ore_pulp_ph"].round(3)
    daily["hours_silica_high"] = daily["hours_silica_high"].astype(int)
    return daily[["date", "month", "hours_with_data", "iron_feed_pct", "silica_feed_pct",
                  "iron_concentrate_pct", "silica_concentrate_pct", "hours_silica_high",
                  "amina_flow", "starch_flow", "ore_pulp_ph", "ore_pulp_tonnes"]]


def detect_events(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Stuck sensors: runs of identical consecutive 20-second values, at least one hour long.
    for column, tag in PROCESS_TAGS.items():
        values = raw[column].to_numpy()
        change = np.r_[True, values[1:] != values[:-1]]
        run_id = np.cumsum(change)
        lengths = np.bincount(run_id)
        starts = np.flatnonzero(change)
        for start in starts:
            n = lengths[run_id[start]]
            if n >= STUCK_MIN_SAMPLES:
                hours = round(n / SAMPLES_PER_HOUR, 2)
                rows.append({"start": raw["date"].iloc[start], "equipment_id": tag,
                             "event_type": "stuck_sensor", "downtime_hours": hours})
    # Historian gaps: hours with no data at all.
    hours = pd.Series(sorted(raw["date"].unique()))
    gap = hours.diff().dt.total_seconds().div(3600) - 1
    for i in np.flatnonzero(gap.fillna(0).to_numpy() > 0):
        rows.append({"start": hours.iloc[i - 1] + pd.Timedelta(hours=1), "equipment_id": "plant_historian",
                     "event_type": "data_gap", "downtime_hours": float(gap.iloc[i])})

    events = pd.DataFrame(rows)
    # Many tags frozen from the same hour is the historian (or the whole data feed)
    # holding its last values, not one transmitter failing: one event, not twenty.
    stuck = events["event_type"].eq("stuck_sensor")
    hour = events["start"].dt.floor("h")
    n_tags = events[stuck].groupby(hour[stuck])["equipment_id"].transform("size")
    frozen = n_tags[n_tags >= FREEZE_MIN_TAGS].index
    if len(frozen):
        freezes = (events.loc[frozen].assign(hour=hour.loc[frozen]).groupby("hour")
                   .agg(start=("start", "min"), downtime_hours=("downtime_hours", "max"))
                   .reset_index(drop=True)
                   .assign(equipment_id="plant_historian", event_type="historian_freeze"))
        events = pd.concat([events.drop(index=frozen), freezes], ignore_index=True)
    events = events.sort_values(["start", "equipment_id"]).reset_index(drop=True)
    events["severity"] = events["downtime_hours"].map(_severity)
    events.insert(0, "event_id", np.arange(1, len(events) + 1))
    events["date"] = events["start"].dt.strftime("%Y-%m-%d")
    events["start"] = events["start"].dt.strftime("%Y-%m-%d %H:%M")
    return events[["event_id", "date", "start", "equipment_id", "event_type", "severity", "downtime_hours"]]


def load_procurement(raw_dir: str = RAW_DIR, months=MONTHS) -> pd.DataFrame:
    usecols = ["Codigo", "FechaCreacion", "Estado", "OrganismoPublico", "NombreProveedor",
               "RubroN1", "monedaItem", "totalLineaNeto"]
    frames = []
    for year, month in months:
        path = os.path.join(raw_dir, f"oc_{year}-{month}.zip")
        with zipfile.ZipFile(path) as z:
            name = z.namelist()[0]
            df = pd.read_csv(z.open(name), sep=";", encoding="latin-1", usecols=usecols,
                             low_memory=False, decimal=",")
        df = df[df["RubroN1"].isin(PROCUREMENT_CATEGORIES) & df["Estado"].isin(PROCUREMENT_STATUS)]
        df = df[df["monedaItem"].eq("CLP")]
        frames.append(df)
    lines = pd.concat(frames, ignore_index=True)
    lines["totalLineaNeto"] = pd.to_numeric(lines["totalLineaNeto"], errors="coerce")
    orders = (lines.groupby(["Codigo", "RubroN1"], as_index=False)
              .agg(date=("FechaCreacion", "first"), status=("Estado", "first"),
                   buyer=("OrganismoPublico", "first"), supplier=("NombreProveedor", "first"),
                   amount_clp=("totalLineaNeto", "sum")))
    orders = orders[orders["amount_clp"] > 0]
    orders["date"] = pd.to_datetime(orders["date"]).dt.strftime("%Y-%m-%d")
    orders["category"] = orders["RubroN1"].map(PROCUREMENT_CATEGORIES)
    orders["status"] = orders["status"].map(PROCUREMENT_STATUS)
    orders["amount_clp"] = orders["amount_clp"].round(0)
    orders = orders.rename(columns={"Codigo": "order_id"}).sort_values(["date", "order_id"])
    return orders[["order_id", "date", "category", "buyer", "supplier", "amount_clp", "status"]].reset_index(drop=True)


def build_warehouse(db_path: str = DUCKDB_PATH, raw_dir: str = RAW_DIR) -> dict:
    """Writes data/ops.duckdb from the raw files and returns the row counts."""
    raw = load_flotation_raw(os.path.join(raw_dir, os.path.basename(FLOTATION_CSV)))
    flotation = build_flotation_daily(raw)
    maintenance = detect_events(raw)
    procurement = load_procurement(raw_dir)

    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    con.execute("CREATE TABLE flotation_daily AS SELECT * FROM flotation")
    con.execute("CREATE TABLE maintenance_events AS SELECT * FROM maintenance")
    con.execute("CREATE TABLE procurement_orders AS SELECT * FROM procurement")
    con.close()
    return {"flotation_daily": len(flotation), "maintenance_events": len(maintenance),
            "procurement_orders": len(procurement)}


def main() -> None:
    counts = build_warehouse()
    print(f"DuckDB warehouse written to {DUCKDB_PATH}: {counts}")


if __name__ == "__main__":
    main()
