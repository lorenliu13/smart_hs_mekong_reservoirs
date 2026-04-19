"""
Temporal expanding-window cross-validation dataset builder for lake-based SWOT-GNN.

Three pre-defined folds split the 2023-12 to 2026-02 SWOT record on the time axis.
All lakes are active in every fold (no spatial split); only the init_date window differs.

Fold definitions (TEMPORAL_FOLD_DATES):
    0: Train 2023-12-01 → 2024-11-30  |  Test 2024-12-01 → 2025-04-30  (~12 / 5 months)
    1: Train 2023-12-01 → 2025-04-30  |  Test 2025-05-01 → 2025-09-30  (~17 / 5 months)
    2: Train 2023-12-01 → 2025-09-30  |  Test 2025-10-01 → 2026-02-28  (~22 / 5 months)

Validation:
    Last val_frac fraction of train init_dates (chronological, no shuffling).

Normalization:
    Z-score statistics derived from ERA5 dates within
    [train_start - seq_len days, train_end] (all lakes, train time window only)
    to prevent temporal leakage into val/test.

Pooled evaluation:
    Run all three folds independently, collect test predictions, and compute one
    global metric over all (fold, sample, lake, lead_day) tuples with obs_mask=1.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch

from .temporal_graph_dataset_lake import (
    assemble_lake_features_from_datacubes,
    TemporalGraphDatasetLake,
    SingleLeadDatasetLake,
    ECMWF_CLIMATE_VARS,
    SWOT_DIM,
)
from .graph_builder import build_graph_from_lake_graph


# ─── Fold definitions ────────────────────────────────────────────────────────────

TEMPORAL_FOLD_DATES: List[Dict[str, str]] = [
    {
        "train_start": "2023-12-01",
        "train_end":   "2024-11-30",
        "test_start":  "2024-12-01",
        "test_end":    "2025-04-30",
    },
    {
        "train_start": "2023-12-01",
        "train_end":   "2025-04-30",
        "test_start":  "2025-05-01",
        "test_end":    "2025-09-30",
    },
    {
        "train_start": "2023-12-01",
        "train_end":   "2025-09-30",
        "test_start":  "2025-10-01",
        "test_end":    "2026-02-28",
    },
]

N_TEMPORAL_FOLDS: int = len(TEMPORAL_FOLD_DATES)

_LOG1P_DYN  = [5, 10, 15]
_ZSCORE_DYN = [2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


# ─── Public API ──────────────────────────────────────────────────────────────────

def build_temporal_cv_fold(
    wse_datacube_path: Union[str, Path],
    era5_climate_datacube_path: Union[str, Path],
    ecmwf_forecast_datacube_path: Union[str, Path],
    static_datacube_path: Union[str, Path],
    lake_graph_path: Union[str, Path],
    fold_idx: int = 0,
    seq_len: int = 30,
    forecast_horizon: int = 10,
    val_frac: float = 0.15,
    require_obs_on_any_forecast_day: bool = True,
) -> Tuple[
    "TemporalGraphDatasetLake",
    "TemporalGraphDatasetLake",
    "TemporalGraphDatasetLake",
    Dict,
]:
    """
    Build train / val / test datasets for one fold of temporal expanding-window CV.

    Time split:
        init_dates are partitioned into train and test windows using the pre-defined
        date boundaries in TEMPORAL_FOLD_DATES[fold_idx].  Val is the last val_frac
        fraction of train init_dates (chronological).

    All lakes are active in every fold; the spatial_mask on each returned dataset
    is all-ones so _run_epoch_nd computes loss over every lake.

    Normalization:
        Z-score statistics are derived from ERA5 dates within
        [train_start - seq_len days, train_end] (all lakes) to prevent temporal
        leakage into val/test.

    Args:
        wse_datacube_path:            Path to swot_lake_wse_datacube_*.nc
        era5_climate_datacube_path:   Path to swot_lake_era5_climate_datacube.nc
        ecmwf_forecast_datacube_path: Path to swot_lake_ecmwf_forecast_datacube.nc
        static_datacube_path:         Path to swot_lake_static_datacube.nc
        lake_graph_path:              Path to GRIT PLD lake graph CSV
        fold_idx:                     Which fold to use (0, 1, or 2)
        seq_len:                      ERA5 history window length (days, default 30)
        forecast_horizon:             ECMWF forecast window (days, default 10)
        val_frac:                     Fraction of train init_dates held back for val
        require_obs_on_any_forecast_day: Skip init_dates with no SWOT obs in forecast window

    Returns:
        (train_ds, val_ds, test_ds, norm_stats)

        Each dataset carries a ``spatial_mask`` attribute: (n_lakes,) all-ones float32
        tensor.  Pass ``spatial_mask=ds.spatial_mask`` to ``_run_epoch_nd``.

        norm_stats keys: log1p_dynamic_indices, zscore_dynamic_indices,
                         dynamic_mean, dynamic_std, static_mean, static_std,
                         fold_idx, train_start, train_end, test_start, test_end,
                         n_train_dates, n_val_dates, n_test_dates, n_lakes
    """
    if not 0 <= fold_idx < N_TEMPORAL_FOLDS:
        raise ValueError(
            f"fold_idx must be in [0, {N_TEMPORAL_FOLDS - 1}], got {fold_idx}"
        )

    fold_def    = TEMPORAL_FOLD_DATES[fold_idx]
    train_start = pd.Timestamp(fold_def["train_start"])
    train_end   = pd.Timestamp(fold_def["train_end"])
    test_start  = pd.Timestamp(fold_def["test_start"])
    test_end    = pd.Timestamp(fold_def["test_end"])

    # ── Load all arrays from datacubes ────────────────────────────────────────
    (
        era5_dynamic,
        ecmwf_forecast,
        static_features,
        wse_labels,
        obs_mask,
        lake_ids_out,
        era5_dates,
        ecmwf_init_dates,
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

    # ── Partition valid init_dates into train and test windows ────────────────
    era5_date_to_idx = {d: i for i, d in enumerate(era5_dates)}

    train_valid: List[int] = []
    test_valid:  List[int] = []

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

        if train_start <= init_date <= train_end:
            train_valid.append(j)
        elif test_start <= init_date <= test_end:
            test_valid.append(j)

    train_valid = np.array(train_valid, dtype=np.int64)
    test_valid  = np.array(test_valid,  dtype=np.int64)

    if len(train_valid) == 0:
        raise ValueError(
            f"No valid train init_dates in [{train_start.date()}, {train_end.date()}]. "
            "Check ERA5/ECMWF coverage and SWOT observations."
        )
    if len(test_valid) == 0:
        raise ValueError(
            f"No valid test init_dates in [{test_start.date()}, {test_end.date()}]. "
            "Check ERA5/ECMWF coverage and SWOT observations."
        )

    # Val: last val_frac of train init_dates (chronological)
    val_start_pos = int(len(train_valid) * (1.0 - val_frac))
    train_idx = train_valid[:val_start_pos]
    val_idx   = train_valid[val_start_pos:]
    test_idx  = test_valid

    print(
        f"Temporal CV fold {fold_idx + 1}/{N_TEMPORAL_FOLDS} "
        f"[train {train_start.date()} → {train_end.date()} | "
        f"test {test_start.date()} → {test_end.date()}]: "
        f"{len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test init_dates "
        f"({n_lakes_total} lakes, all active)"
    )

    # ── Feature normalization (train time window, all lakes) ──────────────────
    era5_dynamic   = era5_dynamic.copy()
    ecmwf_forecast = ecmwf_forecast.copy()

    # Step 1: log1p on skewed features (all dates, all lakes)
    for i in _LOG1P_DYN:
        era5_dynamic[:, :, i] = np.log1p(np.clip(era5_dynamic[:, :, i], 0, None))

    # Step 2: z-score stats from ERA5 dates in [train_start - seq_len, train_end] only
    norm_start = train_start - pd.Timedelta(days=seq_len)
    norm_era5_mask = np.array(
        [norm_start <= d <= train_end for d in era5_dates], dtype=bool
    )
    norm_era5_slice = era5_dynamic[:, norm_era5_mask, :]

    n_dyn    = era5_dynamic.shape[-1]
    dyn_mean = np.zeros(n_dyn, dtype=np.float32)
    dyn_std  = np.ones(n_dyn,  dtype=np.float32)

    for i in _ZSCORE_DYN:
        vals        = norm_era5_slice[:, :, i].ravel()
        dyn_mean[i] = float(vals.mean())
        dyn_std[i]  = float(vals.std()) + 1e-8

    # Step 3: apply z-score to all ERA5 dates and all lakes
    for i in _ZSCORE_DYN:
        era5_dynamic[:, :, i] = (era5_dynamic[:, :, i] - dyn_mean[i]) / dyn_std[i]

    # Step 4: apply identical normalization to ECMWF climate features
    ecmwf_log1p_indices  = [k for k, _ in enumerate(ECMWF_CLIMATE_VARS) if (SWOT_DIM + k) in _LOG1P_DYN]
    ecmwf_zscore_indices = [k for k, _ in enumerate(ECMWF_CLIMATE_VARS) if (SWOT_DIM + k) in _ZSCORE_DYN]

    for k in ecmwf_log1p_indices:
        ecmwf_forecast[:, :, :, k] = np.log1p(np.clip(ecmwf_forecast[:, :, :, k], 0, None))
    for k in ecmwf_zscore_indices:
        era5_idx_k = SWOT_DIM + k
        ecmwf_forecast[:, :, :, k] = (
            ecmwf_forecast[:, :, :, k] - dyn_mean[era5_idx_k]
        ) / dyn_std[era5_idx_k]

    # Step 5: z-score static features (all lakes — no spatial split in temporal CV)
    stat_mean = static_features.mean(axis=0).astype(np.float32)
    stat_std  = static_features.std(axis=0).astype(np.float32) + 1e-8
    static_features = (static_features - stat_mean) / stat_std

    norm_stats: Dict = {
        "log1p_dynamic_indices":  _LOG1P_DYN,
        "zscore_dynamic_indices": _ZSCORE_DYN,
        "dynamic_mean":  dyn_mean,
        "dynamic_std":   dyn_std,
        "static_mean":   stat_mean,
        "static_std":    stat_std,
        "fold_idx":      fold_idx,
        "train_start":   str(train_start.date()),
        "train_end":     str(train_end.date()),
        "test_start":    str(test_start.date()),
        "test_end":      str(test_end.date()),
        "n_train_dates": len(train_idx),
        "n_val_dates":   len(val_idx),
        "n_test_dates":  len(test_idx),
        "n_lakes":       n_lakes_total,
    }

    # ── Construct three datasets sharing the same normalized arrays ───────────
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

    # All lakes active — spatial_mask is all-ones (required by _run_epoch_nd)
    all_ones = torch.ones(n_lakes_total, dtype=torch.float32)
    train_ds.spatial_mask = all_ones
    val_ds.spatial_mask   = all_ones
    test_ds.spatial_mask  = all_ones

    print(
        f"Temporal CV datasets built: "
        f"{len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test init_dates | "
        f"{n_lakes_total} lakes (all active)"
    )

    return train_ds, val_ds, test_ds, norm_stats


# ─── Single-lead-day CV fold builder ────────────────────────────────────────────

def build_temporal_cv_fold_single_lead(
    wse_datacube_path: Union[str, Path],
    era5_climate_datacube_path: Union[str, Path],
    ecmwf_forecast_datacube_path: Union[str, Path],
    static_datacube_path: Union[str, Path],
    lake_graph_path: Union[str, Path],
    fold_idx: int = 0,
    seq_len: int = 30,
    forecast_horizon: int = 10,
    val_frac: float = 0.15,
    lead_assign: str = "round_robin",
    seed: int = 42,
) -> Tuple[
    "SingleLeadDatasetLake",
    "SingleLeadDatasetLake",
    "SingleLeadDatasetLake",
    Dict,
]:
    """
    Build train / val / test SingleLeadDatasetLake for one temporal CV fold.

    Each ERA5 date with at least one SWOT observation is used exactly once as a
    training sample.  A lead day d ∈ {0 .. forecast_horizon-1} is assigned per
    obs_date; init_date = obs_date - d days must be a valid ECMWF init_date with
    a full seq_len ERA5 history window.  Obs_dates that do not satisfy these
    constraints are skipped.

    Lead assignment strategies:
        "round_robin": d = chronological_position_of_obs_date % forecast_horizon
        "random":      d drawn uniformly from {0..forecast_horizon-1} with `seed`

    Time split: same convention as build_temporal_cv_fold — init_date is used
    to decide the fold window (train or test).  Val is the last val_frac of
    training samples (in obs_date chronological order).

    Normalization: identical to build_temporal_cv_fold (ERA5 train-window z-score).

    Args:
        wse_datacube_path:            Path to swot_lake_wse_datacube_*.nc
        era5_climate_datacube_path:   Path to swot_lake_era5_climate_datacube.nc
        ecmwf_forecast_datacube_path: Path to swot_lake_ecmwf_forecast_datacube.nc
        static_datacube_path:         Path to swot_lake_static_datacube.nc
        lake_graph_path:              Path to GRIT PLD lake graph CSV
        fold_idx:                     Which fold to use (0, 1, or 2)
        seq_len:                      ERA5 history window length (default 30)
        forecast_horizon:             ECMWF forecast window (default 10)
        val_frac:                     Fraction of train samples held back for val
        lead_assign:                  "round_robin" or "random"
        seed:                         RNG seed for "random" assignment

    Returns:
        (train_ds, val_ds, test_ds, norm_stats)
        Each dataset has a spatial_mask attribute (all-ones, all lakes active).
        norm_stats includes n_train_samples, n_val_samples, n_test_samples.
    """
    if not 0 <= fold_idx < N_TEMPORAL_FOLDS:
        raise ValueError(
            f"fold_idx must be in [0, {N_TEMPORAL_FOLDS - 1}], got {fold_idx}"
        )
    if lead_assign not in ("round_robin", "random"):
        raise ValueError(f"lead_assign must be 'round_robin' or 'random', got '{lead_assign}'")

    fold_def    = TEMPORAL_FOLD_DATES[fold_idx]
    train_start = pd.Timestamp(fold_def["train_start"])
    train_end   = pd.Timestamp(fold_def["train_end"])
    test_start  = pd.Timestamp(fold_def["test_start"])
    test_end    = pd.Timestamp(fold_def["test_end"])

    # ── Load all arrays from datacubes ────────────────────────────────────────
    (
        era5_dynamic,
        ecmwf_forecast,
        static_features,
        wse_labels,
        obs_mask,
        lake_ids_out,
        era5_dates,
        ecmwf_init_dates,
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

    era5_date_to_idx  = {d: i for i, d in enumerate(era5_dates)}
    ecmwf_date_to_idx = {d: j for j, d in enumerate(ecmwf_init_dates)}

    # ── Step 1: find valid multi-lead init_dates (same logic as build_temporal_cv_fold) ──
    # These define the train/val/test windows for standard multi-lead evaluation.
    # Val and test will use TemporalGraphDatasetLake with these indices.
    train_val_valid: List[int] = []
    test_valid:      List[int] = []

    for j, init_date in enumerate(ecmwf_init_dates):
        last_hist_day = init_date - pd.Timedelta(days=1)
        era5_idx = era5_date_to_idx.get(last_hist_day)
        if era5_idx is None or era5_idx < seq_len - 1:
            continue

        has_obs = False
        for d in range(forecast_horizon):
            target_date = init_date + pd.Timedelta(days=d)
            t_idx = era5_date_to_idx.get(target_date)
            if t_idx is not None and obs_mask[:, t_idx].sum() > 0:
                has_obs = True
                break
        if not has_obs:
            continue

        if train_start <= init_date <= train_end:
            train_val_valid.append(j)
        elif test_start <= init_date <= test_end:
            test_valid.append(j)

    if len(train_val_valid) == 0:
        raise ValueError(
            f"No valid train init_dates in [{train_start.date()}, {train_end.date()}]. "
            "Check ERA5/ECMWF coverage and SWOT observations."
        )
    if len(test_valid) == 0:
        raise ValueError(
            f"No valid test init_dates in [{test_start.date()}, {test_end.date()}]. "
            "Check ERA5/ECMWF coverage and SWOT observations."
        )

    # Split train_val_valid chronologically: first (1-val_frac) → train window, last val_frac → val
    val_start_pos      = int(len(train_val_valid) * (1.0 - val_frac))
    train_window_inits = np.array(train_val_valid[:val_start_pos], dtype=np.int64)
    val_init_idx       = np.array(train_val_valid[val_start_pos:],  dtype=np.int64)
    test_init_idx      = np.array(test_valid,                        dtype=np.int64)

    # ── Step 2: build single-lead (ecmwf_j, lead_day) pairs for train only ───
    # Only obs_dates whose derived init_date falls within the train window
    # (train_window_inits) are eligible — this keeps single-lead train and
    # multi-lead val on non-overlapping init_date ranges.
    train_window_set = set(train_window_inits.tolist())

    obs_dates = [
        d for d in era5_dates
        if obs_mask[:, era5_date_to_idx[d]].sum() > 0
    ]

    n_obs = len(obs_dates)
    if lead_assign == "round_robin":
        assigned_leads = [i % forecast_horizon for i in range(n_obs)]
    else:
        rng = np.random.default_rng(seed)
        assigned_leads = rng.integers(0, forecast_horizon, size=n_obs).tolist()

    train_samples: List[List[int]] = []
    for obs_date, lead_day in zip(obs_dates, assigned_leads):
        init_date = obs_date - pd.Timedelta(days=lead_day)
        ecmwf_j   = ecmwf_date_to_idx.get(init_date)
        if ecmwf_j is None or ecmwf_j not in train_window_set:
            continue
        last_hist_day = init_date - pd.Timedelta(days=1)
        era5_idx = era5_date_to_idx.get(last_hist_day)
        if era5_idx is None or era5_idx < seq_len - 1:
            continue
        train_samples.append([ecmwf_j, lead_day])

    train_sample_arr = (
        np.array(train_samples, dtype=np.int64) if train_samples
        else np.empty((0, 2), dtype=np.int64)
    )
    if len(train_sample_arr) == 0:
        raise ValueError(
            f"No valid single-lead train samples in the train window. "
            "Check ERA5/ECMWF coverage and SWOT observations."
        )

    print(
        f"Single-lead temporal CV fold {fold_idx + 1}/{N_TEMPORAL_FOLDS} "
        f"[train {train_start.date()} → {train_end.date()} | "
        f"test {test_start.date()} → {test_end.date()}]: "
        f"{len(train_sample_arr)} train (single-lead) / "
        f"{len(val_init_idx)} val / {len(test_init_idx)} test (multi-lead) "
        f"| lead_assign={lead_assign}, {n_lakes_total} lakes"
    )

    # ── Feature normalization (identical to build_temporal_cv_fold) ───────────
    era5_dynamic   = era5_dynamic.copy()
    ecmwf_forecast = ecmwf_forecast.copy()

    for i in _LOG1P_DYN:
        era5_dynamic[:, :, i] = np.log1p(np.clip(era5_dynamic[:, :, i], 0, None))

    norm_start = train_start - pd.Timedelta(days=seq_len)
    norm_era5_mask = np.array(
        [norm_start <= d <= train_end for d in era5_dates], dtype=bool
    )
    norm_era5_slice = era5_dynamic[:, norm_era5_mask, :]

    n_dyn    = era5_dynamic.shape[-1]
    dyn_mean = np.zeros(n_dyn, dtype=np.float32)
    dyn_std  = np.ones(n_dyn,  dtype=np.float32)

    for i in _ZSCORE_DYN:
        vals        = norm_era5_slice[:, :, i].ravel()
        dyn_mean[i] = float(vals.mean())
        dyn_std[i]  = float(vals.std()) + 1e-8

    for i in _ZSCORE_DYN:
        era5_dynamic[:, :, i] = (era5_dynamic[:, :, i] - dyn_mean[i]) / dyn_std[i]

    ecmwf_log1p_indices  = [k for k, _ in enumerate(ECMWF_CLIMATE_VARS) if (SWOT_DIM + k) in _LOG1P_DYN]
    ecmwf_zscore_indices = [k for k, _ in enumerate(ECMWF_CLIMATE_VARS) if (SWOT_DIM + k) in _ZSCORE_DYN]

    for k in ecmwf_log1p_indices:
        ecmwf_forecast[:, :, :, k] = np.log1p(np.clip(ecmwf_forecast[:, :, :, k], 0, None))
    for k in ecmwf_zscore_indices:
        era5_idx_k = SWOT_DIM + k
        ecmwf_forecast[:, :, :, k] = (
            ecmwf_forecast[:, :, :, k] - dyn_mean[era5_idx_k]
        ) / dyn_std[era5_idx_k]

    stat_mean = static_features.mean(axis=0).astype(np.float32)
    stat_std  = static_features.std(axis=0).astype(np.float32) + 1e-8
    static_features = (static_features - stat_mean) / stat_std

    norm_stats: Dict = {
        "log1p_dynamic_indices":  _LOG1P_DYN,
        "zscore_dynamic_indices": _ZSCORE_DYN,
        "dynamic_mean":  dyn_mean,
        "dynamic_std":   dyn_std,
        "static_mean":   stat_mean,
        "static_std":    stat_std,
        "fold_idx":      fold_idx,
        "lead_assign":   lead_assign,
        "train_start":   str(train_start.date()),
        "train_end":     str(train_end.date()),
        "test_start":    str(test_start.date()),
        "test_end":      str(test_end.date()),
        "n_train_samples": len(train_sample_arr),
        "n_val_dates":     len(val_init_idx),
        "n_test_dates":    len(test_init_idx),
        "n_lakes":         n_lakes_total,
    }

    # ── Construct datasets ────────────────────────────────────────────────────
    # train: SingleLeadDatasetLake — one obs_date used once at one lead day
    # val / test: TemporalGraphDatasetLake — standard multi-lead evaluation
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
    train_ds = SingleLeadDatasetLake(**shared_kwargs, samples=train_sample_arr)
    val_ds   = TemporalGraphDatasetLake(**shared_kwargs, indices=val_init_idx)
    test_ds  = TemporalGraphDatasetLake(**shared_kwargs, indices=test_init_idx)

    all_ones = torch.ones(n_lakes_total, dtype=torch.float32)
    train_ds.spatial_mask = all_ones
    val_ds.spatial_mask   = all_ones
    test_ds.spatial_mask  = all_ones

    print(
        f"Datasets built: "
        f"{len(train_ds)} train (single-lead) / "
        f"{len(val_ds)} val / {len(test_ds)} test (multi-lead) | "
        f"{n_lakes_total} lakes (all active)"
    )

    return train_ds, val_ds, test_ds, norm_stats
