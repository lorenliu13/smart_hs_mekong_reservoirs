"""
Shared utilities for lake-based SWOT-GNN datacube builders.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_lake_ids_from_graph(lake_graph_csv: Path) -> np.ndarray:
    """
    Extract unique lake IDs from the GRIT PLD lake graph CSV.
    Excludes the terminal node (-1).

    Returns:
        Sorted int64 array of lake IDs.
    """
    ids = pd.read_csv(lake_graph_csv, usecols=["lake_id"])["lake_id"].to_numpy(dtype=np.int64)
    ids = np.unique(ids)
    return ids[ids != -1]


def derive_climate_vars(raw: dict) -> dict:
    """
    Derive 13 model climate variables from raw ERA5-Land / ECMWF IFS variables.

    Accepts arrays of any shape (2-D or 3-D). Element-wise numpy operations
    broadcast correctly for both ERA5 (n_lakes, n_dates) and ECMWF
    (n_lakes, n_init_dates, forecast_horizon) arrays.

    Returns:
        Dict with keys: LWd, SWd, P, Pres, Temp, Td, Wind, sf, sd,
                        swvl1, swvl2, swvl3, swvl4
    """
    return {
        "LWd":   (raw["strd"]  / 86400.0).astype(np.float32),   # J/m² → W/m²
        "SWd":   (raw["ssrd"]  / 86400.0).astype(np.float32),   # J/m² → W/m²
        "P":     (raw["tp"]    * 1000.0).astype(np.float32),    # m   → mm
        "Pres":  raw["sp"].astype(np.float32),                   # Pa
        "Temp":  raw["t2m"].astype(np.float32),                  # K
        "Td":    raw["d2m"].astype(np.float32),                  # K  (2-m dewpoint)
        "Wind":  np.sqrt(raw["u10"]**2 + raw["v10"]**2).astype(np.float32),  # m/s
        "sf":    (raw["sf"]    * 1000.0).astype(np.float32),    # m   → mm (snowfall)
        "sd":    (raw["sd"]    * 1000.0).astype(np.float32),    # m   → mm (snow depth)
        "swvl1": raw["swvl1"].astype(np.float32),                # m³/m³
        "swvl2": raw["swvl2"].astype(np.float32),                # m³/m³
        "swvl3": raw["swvl3"].astype(np.float32),                # m³/m³
        "swvl4": raw["swvl4"].astype(np.float32),                # m³/m³
    }
