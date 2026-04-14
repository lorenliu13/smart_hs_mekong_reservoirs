"""
Regional spatial cross-validation dataset builder for lake-based SWOT-GNN.

Lakes are assigned to one of five pre-defined geographic regions based on their
HydroSHEDS HYBAS Level-4 sub-basin ID.  One region is held out as the test set;
the remaining four form the train/val pool.

The HYBAS ID for each lake is read from the `hybas_col` column (default:
"hybasin_level_4") of the existing lake graph CSV — no extra file required.

Region → fold index (REGIONAL_FOLD_MAP):
    0: Upper Mekong + Northern Highlands          (~21% of lakes)
    1: Red River Basin + Pearl River Tributaries  (~13% of lakes)
    2: Vietnam Coastal Basins + Mekong Delta      (~20% of lakes)
    3: Khorat Plateau                             (~25% of lakes)
    4: Tonle Sap Basin + 3S Basin                 (~19% of lakes)

Lakes whose HYBAS ID is absent from REGIONAL_FOLD_MAP are excluded from the
loss but remain in the graph for message passing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

from .temporal_graph_dataset_lake import (
    assemble_lake_features_from_datacubes,
    TemporalGraphDatasetLake,
    ECMWF_CLIMATE_VARS,
    SWOT_DIM,
)
from .graph_builder import build_graph_from_lake_graph


# ─── Regional fold definitions ──────────────────────────────────────────────────
# Maps HydroSHEDS HYBAS Level-4 sub-basin ID → fold index (0-indexed).
# Approximate lake-count share per fold:
#   0: Upper Mekong + Northern Highlands          (~21%)
#   1: Red River Basin + Pearl River Tributaries  (~13%)
#   2: Vietnam Coastal Basins + Mekong Delta      (~20%)
#   3: Khorat Plateau                             (~25%)
#   4: Tonle Sap Basin + 3S Basin                 (~19%)
REGIONAL_FOLD_MAP: Dict[int, int] = {
    # Fold 0 — Upper Mekong + Northern Highlands
    4041043580: 0, 4040783120: 0,                   # Upper Mekong (Lancang)
    4041108500: 0, 4041128230: 0, 4041043590: 0,    # Northern Highlands
    # Fold 1 — Red River + Pearl River Tributaries
    4040015000: 1, 4040015010: 1,                   # Red River Basin
    4040013000: 1,                                  # Pearl River (Xi Jiang)
    # Fold 2 — Vietnam Coastal Basins + Mekong Delta
    4040015090: 2,                                  # Vietnam Coastal Basins
    4040017020: 2, 4040017030: 2,                   # Mekong Delta
    # Fold 3 — Khorat Plateau
    4041108580: 3,                                  # Khorat Plateau
    # Fold 4 — Tonle Sap Basin + 3S Basin
    4041144880: 4, 4041144860: 4,                   # Tonle Sap Basin
    4041128330: 4,                                  # 3S Basin
}

REGION_NAMES: Dict[int, str] = {
    0: "Upper Mekong + Northern Highlands",
    1: "Red River + Pearl River",
    2: "Vietnam Coastal + Mekong Delta",
    3: "Khorat Plateau",
    4: "Tonle Sap + 3S Basin",
}

N_REGIONAL_FOLDS: int = 5

# Normalization indices — must match the values used in build_spatial_cv_fold
# in temporal_graph_dataset_lake.py to ensure consistent feature processing.
_LOG1P_DYN  = [5, 10, 15]           # days_since_last_obs, P, sf — zero-bounded, right-skewed
_ZSCORE_DYN = [2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


# ─── Internal helper ────────────────────────────────────────────────────────────

def _assign_lake_folds(
    lake_ids_out: np.ndarray,
    lake_graph_csv: Union[str, Path],
    hybas_col: str,
) -> np.ndarray:
    """
    Map each lake (by position in lake_ids_out) to a fold index.

    Reads `lake_graph_csv` to get each lake's HYBAS Level-4 sub-basin ID from
    `hybas_col`, then maps that ID to a fold via REGIONAL_FOLD_MAP.

    Returns:
        lake_fold_indices : (n_lakes,) int64 array.
            fold index in [0, N_REGIONAL_FOLDS-1] for assigned lakes,
            -1 for lakes with a missing or unrecognised HYBAS ID.
    """
    graph_df = pd.read_csv(lake_graph_csv, usecols=["lake_id", hybas_col])
    graph_df["lake_id"] = pd.to_numeric(graph_df["lake_id"], errors="coerce")
    graph_df = graph_df.dropna(subset=["lake_id", hybas_col])
    graph_df["lake_id"] = graph_df["lake_id"].astype(np.int64)

    hybas_map: Dict[int, int] = dict(
        zip(graph_df["lake_id"].tolist(), graph_df[hybas_col].tolist())
    )

    lake_fold_indices = np.full(len(lake_ids_out), -1, dtype=np.int64)
    for pos, lid in enumerate(lake_ids_out):
        hybas = hybas_map.get(int(lid))
        if hybas is not None:
            fold = REGIONAL_FOLD_MAP.get(int(hybas))
            if fold is not None:
                lake_fold_indices[pos] = fold
    return lake_fold_indices


# ─── Public API ─────────────────────────────────────────────────────────────────

def build_regional_cv_fold(
    wse_datacube_path: Union[str, Path],
    era5_climate_datacube_path: Union[str, Path],
    ecmwf_forecast_datacube_path: Union[str, Path],
    static_datacube_path: Union[str, Path],
    lake_graph_path: Union[str, Path],
    fold_idx: int = 0,
    seq_len: int = 30,
    forecast_horizon: int = 10,
    val_frac: float = 0.15,
    val_method: str = "temporal",
    spatial_val_frac: float = 0.1,
    spatial_val_seed: int = 43,
    hybas_col: str = "hybasin_level_4",
    require_obs_on_any_forecast_day: bool = True,
) -> Tuple[
    "TemporalGraphDatasetLake",
    "TemporalGraphDatasetLake",
    "TemporalGraphDatasetLake",
    Dict,
]:
    """
    Build train / val / test datasets for one fold of regional spatial CV.

    Lake split:
        Each lake is assigned to a region (fold 0–4) via the `hybas_col` column
        (default: "hybasin_level_4") in the lake graph CSV.  The region at
        `fold_idx` becomes the held-out test set; the remaining four regions
        form the train/val pool.  Lakes with no HYBAS match in REGIONAL_FOLD_MAP
        remain as graph nodes for message passing but are excluded from the loss.

    Validation strategy (controlled by `val_method`):

        ``"temporal"`` (default):
            Hold out last `val_frac` fraction of init_dates from train-region
            lakes for validation.  Both train and val share the same lake mask.

        ``"spatial"``:
            Use all init_dates for train and val.  Hold out `spatial_val_frac`
            of train-region lakes (drawn with `spatial_val_seed`) as spatial val;
            the rest are train-train lakes.

    Normalization:
        Z-score statistics are derived from train-train lakes only (no leakage
        from val or test lakes), then applied uniformly to all lakes.

    Args:
        wse_datacube_path:            Path to swot_lake_wse_datacube_*.nc
        era5_climate_datacube_path:   Path to swot_lake_era5_climate_datacube.nc
        ecmwf_forecast_datacube_path: Path to swot_lake_ecmwf_forecast_datacube.nc
        static_datacube_path:         Path to swot_lake_static_datacube.nc
        lake_graph_path:              Path to GRIT PLD lake graph CSV (must contain
                                      ``lake_id`` and `hybas_col` columns)
        fold_idx:                     Which region to hold out as the test set (0–4)
        seq_len:                      ERA5 history window length (days, default 30)
        forecast_horizon:             ECMWF forecast window (days, default 10)
        val_frac:                     Fraction of init_dates held back for temporal val
                                      (only used when val_method="temporal")
        val_method:                   ``"temporal"`` or ``"spatial"``
        spatial_val_frac:             Fraction of train-region lakes held out for
                                      spatial val (only when val_method="spatial")
        spatial_val_seed:             RNG seed for the spatial val lake draw
        hybas_col:                    Column in the lake graph CSV holding the HYBAS
                                      Level-4 sub-basin ID (default: "hybasin_level_4")
        require_obs_on_any_forecast_day: Skip init_dates with no SWOT obs on any
                                         forecast day

    Returns:
        (train_ds, val_ds, test_ds, norm_stats)

        Each dataset carries a ``spatial_mask`` attribute: (n_lakes,) float32
        tensor with 1 for active (loss-contributing) nodes, 0 otherwise.
        Pass ``spatial_mask=ds.spatial_mask`` to ``_run_epoch``.

        norm_stats keys: log1p_dynamic_indices, zscore_dynamic_indices,
                         dynamic_mean, dynamic_std, static_mean, static_std,
                         val_method, fold_idx, region_name,
                         n_train_lakes, n_val_lakes, n_test_lakes, n_unassigned
    """
    if not 0 <= fold_idx < N_REGIONAL_FOLDS:
        raise ValueError(
            f"fold_idx must be in [0, {N_REGIONAL_FOLDS - 1}], got {fold_idx}"
        )

    # ── Load all arrays from datacubes ────────────────────────────────────────
    (
        era5_dynamic,       # (n_lakes, n_era5_dates, SWOT_DIM+CLIMATE_DIM)
        ecmwf_forecast,     # (n_lakes, n_ecmwf_init_dates, forecast_horizon, CLIMATE_DIM)
        static_features,    # (n_lakes, n_static)
        wse_labels,         # (n_lakes, n_time_steps)
        obs_mask,           # (n_lakes, n_time_steps)
        lake_ids_out,       # (n_lakes,)
        era5_dates,         # list of pd.Timestamp
        ecmwf_init_dates,   # list of pd.Timestamp
    ) = assemble_lake_features_from_datacubes(
        wse_datacube_path=wse_datacube_path,
        era5_climate_datacube_path=era5_climate_datacube_path,
        ecmwf_forecast_datacube_path=ecmwf_forecast_datacube_path,
        static_datacube_path=static_datacube_path,
    )

    n_lakes_total = len(lake_ids_out)

    # ── Build lake graph ──────────────────────────────────────────────────────
    edge_index, _, _, _ = build_graph_from_lake_graph(
        lake_graph_csv=lake_graph_path,
        lake_ids=lake_ids_out,
    )

    # ── Regional fold assignment ──────────────────────────────────────────────
    # Read hybasin_level_4 from the lake graph CSV and map each lake to a fold.
    lake_fold_indices = _assign_lake_folds(lake_ids_out, lake_graph_path, hybas_col)

    test_positions  = np.where(lake_fold_indices == fold_idx)[0]
    train_positions = np.where(
        (lake_fold_indices >= 0) & (lake_fold_indices != fold_idx)
    )[0]
    n_unassigned = int((lake_fold_indices == -1).sum())

    if len(test_positions) == 0:
        raise ValueError(
            f"No lakes found for fold_idx={fold_idx} ({REGION_NAMES[fold_idx]}). "
            f"Check that '{hybas_col}' in the lake graph CSV contains HYBAS IDs "
            "present in REGIONAL_FOLD_MAP."
        )
    if len(train_positions) == 0:
        raise ValueError(
            "No training lakes found. Check REGIONAL_FOLD_MAP and the lake graph CSV."
        )

    print(
        f"Regional CV fold {fold_idx + 1}/{N_REGIONAL_FOLDS} "
        f"[{REGION_NAMES[fold_idx]}]: "
        f"{len(train_positions)} train / {len(test_positions)} test lakes "
        f"({n_unassigned} unassigned — graph-only nodes)"
    )

    # ── Valid init_date positions ─────────────────────────────────────────────
    era5_date_to_idx = {d: i for i, d in enumerate(era5_dates)}
    all_valid = []
    for j, init_date in enumerate(ecmwf_init_dates):
        last_hist_day = init_date - pd.Timedelta(days=1)
        era5_idx = era5_date_to_idx.get(last_hist_day)
        if era5_idx is None or era5_idx < seq_len - 1:
            continue
        if require_obs_on_any_forecast_day:
            has_obs = False
            for d in range(forecast_horizon):
                target_date = init_date + pd.Timedelta(days=d)
                t_idx = era5_date_to_idx.get(target_date)
                if t_idx is not None and obs_mask[:, t_idx].sum() > 0:
                    has_obs = True
                    break
            if not has_obs:
                continue
        all_valid.append(j)

    all_valid = np.array(all_valid, dtype=np.int64)
    n_valid   = len(all_valid)
    if n_valid == 0:
        raise ValueError(
            "No valid init_dates found. Check ERA5/ECMWF date ranges and SWOT observations."
        )

    # ── Val strategy ─────────────────────────────────────────────────────────
    if val_method == "temporal":
        val_start  = int(n_valid * (1.0 - val_frac))
        train_idx  = all_valid[:val_start]
        val_idx    = all_valid[val_start:]
        test_idx   = all_valid
        norm_positions         = train_positions
        train_active_positions = train_positions
        val_active_positions   = train_positions
        print(
            f"  Val method: temporal — "
            f"{len(train_idx)} train dates / {len(val_idx)} val dates "
            f"({len(train_positions)} lakes each)"
        )

    elif val_method == "spatial":
        train_idx = all_valid
        val_idx   = all_valid
        test_idx  = all_valid
        rng_val   = np.random.default_rng(spatial_val_seed)
        n_spatial_val          = max(1, int(len(train_positions) * spatial_val_frac))
        perm                   = rng_val.permutation(len(train_positions))
        val_active_positions   = train_positions[perm[:n_spatial_val]]
        train_active_positions = train_positions[perm[n_spatial_val:]]
        norm_positions         = train_active_positions
        print(
            f"  Val method: spatial — "
            f"{len(train_active_positions)} train / "
            f"{len(val_active_positions)} val lakes (seed={spatial_val_seed})"
        )

    else:
        raise ValueError(f"val_method must be 'temporal' or 'spatial', got '{val_method}'")

    # ── Feature normalization (no leakage from val/test lakes) ────────────────
    era5_dynamic   = era5_dynamic.copy()
    ecmwf_forecast = ecmwf_forecast.copy()

    # Step 1: log1p on skewed features (all lakes, all dates)
    for i in _LOG1P_DYN:
        era5_dynamic[:, :, i] = np.log1p(np.clip(era5_dynamic[:, :, i], 0, None))

    # Step 2: z-score statistics from norm_positions lakes only
    norm_era5_slice = era5_dynamic[norm_positions, :, :]
    n_dyn    = era5_dynamic.shape[-1]
    dyn_mean = np.zeros(n_dyn, dtype=np.float32)
    dyn_std  = np.ones(n_dyn,  dtype=np.float32)
    for i in _ZSCORE_DYN:
        vals        = norm_era5_slice[:, :, i].ravel()
        dyn_mean[i] = float(vals.mean())
        dyn_std[i]  = float(vals.std()) + 1e-8

    # Step 3: apply z-score to all lakes and all ERA5 dates
    for i in _ZSCORE_DYN:
        era5_dynamic[:, :, i] = (era5_dynamic[:, :, i] - dyn_mean[i]) / dyn_std[i]

    # Step 4: apply identical normalization to ECMWF climate features
    ecmwf_log1p_indices  = [k for k, _ in enumerate(ECMWF_CLIMATE_VARS) if (SWOT_DIM + k) in _LOG1P_DYN]
    ecmwf_zscore_indices = [k for k, _ in enumerate(ECMWF_CLIMATE_VARS) if (SWOT_DIM + k) in _ZSCORE_DYN]
    for k in ecmwf_log1p_indices:
        ecmwf_forecast[:, :, :, k] = np.log1p(np.clip(ecmwf_forecast[:, :, :, k], 0, None))
    for k in ecmwf_zscore_indices:
        era5_idx = SWOT_DIM + k
        ecmwf_forecast[:, :, :, k] = (
            ecmwf_forecast[:, :, :, k] - dyn_mean[era5_idx]
        ) / dyn_std[era5_idx]

    # Step 5: z-score static features from norm_positions lakes only
    stat_mean = static_features[norm_positions, :].mean(axis=0).astype(np.float32)
    stat_std  = static_features[norm_positions, :].std(axis=0).astype(np.float32) + 1e-8
    static_features = (static_features - stat_mean) / stat_std

    norm_stats: Dict = {
        "log1p_dynamic_indices":  _LOG1P_DYN,
        "zscore_dynamic_indices": _ZSCORE_DYN,
        "dynamic_mean":  dyn_mean,
        "dynamic_std":   dyn_std,
        "static_mean":   stat_mean,
        "static_std":    stat_std,
        "val_method":    val_method,
        "fold_idx":      fold_idx,
        "region_name":   REGION_NAMES[fold_idx],
        "n_train_lakes": len(train_active_positions),
        "n_val_lakes":   len(val_active_positions),
        "n_test_lakes":  len(test_positions),
        "n_unassigned":  n_unassigned,
    }

    # ── Construct three datasets sharing the same normalized arrays ───────────
    # All lakes remain as nodes in the graph; spatial_mask gates the loss.
    shared_kwargs = dict(
        era5_dynamic=era5_dynamic,
        ecmwf_forecast=ecmwf_forecast,
        static_features=static_features,
        edge_index=edge_index,
        era5_dates=era5_dates,
        ecmwf_init_dates=ecmwf_init_dates,
        wse_labels=wse_labels,
        obs_mask=obs_mask,
        lake_ids=lake_ids_out,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
    )
    train_ds = TemporalGraphDatasetLake(**shared_kwargs, indices=train_idx)
    val_ds   = TemporalGraphDatasetLake(**shared_kwargs, indices=val_idx)
    test_ds  = TemporalGraphDatasetLake(**shared_kwargs, indices=test_idx)

    def _make_mask(positions: np.ndarray) -> torch.Tensor:
        m = np.zeros(n_lakes_total, dtype=np.float32)
        m[positions] = 1.0
        return torch.from_numpy(m)

    train_ds.spatial_mask = _make_mask(train_active_positions)
    val_ds.spatial_mask   = _make_mask(val_active_positions)
    test_ds.spatial_mask  = _make_mask(test_positions)

    print(
        f"Regional CV datasets built: "
        f"{len(train_ds)} init dates | "
        f"{len(train_active_positions)} train / "
        f"{len(val_active_positions)} val / "
        f"{len(test_positions)} test lakes"
    )

    return train_ds, val_ds, test_ds, norm_stats
