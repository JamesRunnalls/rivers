"""Extract discharge for all gauges on a specific date.

For each gauge, uses the observed discharge (discharge_vol in m3/s) where
available, falling back to the simulated value where the observed is missing
or NaN.

Reads:
  - external/camels_ch/timeseries/observation_based/CAMELS_CH_obs_based_*.csv
  - external/camels_ch/timeseries/simulation_based/CAMELS_CH_sim_based_*.csv

Outputs:
  - public/geodata/outputs/discharge_{date}.json — {gauge_id: discharge_m3s}

Run from the project root:
  python scripts/discharge.py 2020-06-15
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

OBS_DIR = Path("external/camels_ch/timeseries/observation_based")
SIM_DIR = Path("external/camels_ch/timeseries/simulation_based")
OUT_DIR = Path("public/geodata/outputs")
OBS_COL = "discharge_vol(m3/s)"
SIM_COL = "discharge_vol_sim(m3/s)"


def _gauge_id(path: Path) -> int:
    match = re.search(r"_(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"Cannot extract gauge ID from {path.name}")
    return int(match.group(1))


def _lookup(path: Path, date: str, col: str) -> float | None:
    df = pd.read_csv(path, usecols=["date", col], parse_dates=["date"])
    row = df[df["date"] == date]
    if row.empty:
        return None
    val = row.iloc[0][col]
    return None if pd.isna(val) else float(val)


def get_discharge(date: str) -> dict[int, float]:
    """Return {gauge_id: discharge_m3s} for all gauges on *date*.

    Observed values are preferred; simulated values are used as a fallback
    when the observed value is absent or NaN.
    """
    obs_files = {_gauge_id(p): p for p in OBS_DIR.glob("CAMELS_CH_obs_based_*.csv")}
    sim_files = {_gauge_id(p): p for p in SIM_DIR.glob("CAMELS_CH_sim_based_*.csv")}
    all_ids = obs_files.keys() | sim_files.keys()

    result: dict[int, float] = {}
    for gauge_id in sorted(all_ids):
        discharge = None

        if gauge_id in obs_files:
            discharge = _lookup(obs_files[gauge_id], date, OBS_COL)

        if discharge is None and gauge_id in sim_files:
            discharge = _lookup(sim_files[gauge_id], date, SIM_COL)

        if discharge is not None:
            result[gauge_id] = discharge

    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <date>  (e.g. 2020-06-15)")
        sys.exit(1)

    date = sys.argv[1]
    discharge = get_discharge(date)

    out_path = OUT_DIR / f"discharge_{date}.json"
    out_path.write_text(json.dumps(discharge, indent=2))
    print(f"Wrote {len(discharge)} gauges to {out_path}")
