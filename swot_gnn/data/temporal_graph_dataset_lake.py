"""
Temporal graph dataset for lake-based SWOT-GNN with 10-day multi-step forecasting.

Each training sample is indexed by an ECMWF init_date and contains:
  - History window (seq_len=30 days): ERA5-Land climate + SWOT WSE features
  - Forecast window (forecast_horizon=10 days): ECMWF IFS climate + zeroed SWOT features
  - Labels: WSE at init_date + days 0..9 (shape: n_lakes × 10)
  - Mask:   obs_mask at those forecast dates

Usage:
    train_ds, val_ds, test_ds, norm_stats = build_temporal_dataset_from_lake_datacubes(
        era5_dynamic_datacube_path=...,
        ecmwf_forecast_datacube_path=...,
        static_datacube_path=...,
        lake_graph_path=...,
    )
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict

import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from .feature_assembler_lake import (
    assemble_lake_features_from_datacubes,
    ERA5_INPUT_VARS,
    ECMWF_CLIMATE_VARS,
    SWOT_DIM,
    CLIMATE_DIM,
)
from .graph_builder import build_graph_from_lake_graph


class TemporalGraphDatasetLake(Dataset):
    """
    Dataset for 10-day-ahead multi-step WSE forecasting for lakes.

    Each sample:
      Given 30 days of ERA5 history + 10 ECMWF forecast days → predict WSE for those 10 days.

    Input tensor shape: (n_lakes, 40, 14)
      - Timesteps  0–29: ERA5 history (SWOT features + ERA5 climate)
      - Timesteps 30–39: ECMWF forecast (zeroed SWOT features + ECMWF climate + DOY encoding)

    Labels shape: (n_lakes, 10) — normalised WSE at each of the 10 forecast days.
    Mask shape:   (n_lakes, 10) — 1 where SWOT observed, 0 otherwise (loss at observed only).
    """

    def __init__(
        self,
        era5_dynamic: np.ndarray,           # (n_lakes, n_era5_dates, 14)
        ecmwf_forecast: np.ndarray,         # (n_lakes, n_init_dates, forecast_horizon, 9)
        static_features: np.ndarray,        # (n_lakes, n_static)
        edge_index: np.ndarray,             # (2, n_edges)
        era5_dates: pd.DatetimeIndex,
        ecmwf_init_dates: pd.DatetimeIndex,
        wse_labels: np.ndarray,             # (n_lakes, n_era5_dates)
        obs_mask: np.ndarray,               # (n_lakes, n_era5_dates)
        lake_ids: np.ndarray,               # (n_lakes,)
        seq_len: int = 30,
        forecast_horizon: int = 10,
        indices: Optional[np.ndarray] = None,
    ):
        """
        Args:
            era5_dynamic:      (n_lakes, n_era5_dates, 14) — normalized ERA5 input features
            ecmwf_forecast:    (n_lakes, n_init_dates, forecast_horizon, 9) — normalized ECMWF climate
            static_features:   (n_lakes, n_static) — normalized static attributes
            edge_index:        (2, n_edges)
            era5_dates:        All ERA5 dates in chronological order
            ecmwf_init_dates:  All ECMWF init_dates in chronological order
            wse_labels:        (n_lakes, n_era5_dates) — WSE for labels extraction
            obs_mask:          (n_lakes, n_era5_dates) — binary obs mask
            lake_ids:          (n_lakes,) lake IDs in graph order
            seq_len:           ERA5 history window (default 30 days)
            forecast_horizon:  ECMWF forecast window (default 10 days)
            indices:           Optional subset of valid ECMWF init_date positions (train/val/test)
        """
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric is required: pip install torch-geometric")

        self.era5_dynamic    = era5_dynamic
        self.ecmwf_forecast  = ecmwf_forecast
        self.static_features = static_features.astype(np.float32)
        self.edge_index      = torch.from_numpy(edge_index).long()
        self.era5_dates      = era5_dates
        self.ecmwf_init_dates = ecmwf_init_dates
        self.wse_labels      = wse_labels.astype(np.float32)
        self.obs_mask        = obs_mask.astype(np.float32)
        self.lake_ids        = lake_ids
        self.seq_len         = seq_len
        self.forecast_horizon = forecast_horizon
        self.n_lakes         = era5_dynamic.shape[0]

        # O(1) date → index lookup
        self.era5_date_to_idx  = {d: i for i, d in enumerate(era5_dates)}
        self.ecmwf_date_to_idx = {d: i for i, d in enumerate(ecmwf_init_dates)}

        # valid_starts: positions j in ecmwf_init_dates where the ERA5 history window fits
        self.valid_starts = indices if indices is not None else self._find_valid_starts()

    def _find_valid_starts(self) -> np.ndarray:
        """
        Return all ECMWF init_date positions where the 30-day ERA5 history window
        is fully available (init_date - 30 days through init_date - 1 must be in ERA5).
        """
        valid = []
        for j, init_date in enumerate(self.ecmwf_init_dates):
            # Last ERA5 history day = init_date - 1 day
            last_hist_day = init_date - pd.Timedelta(days=1)
            era5_idx = self.era5_date_to_idx.get(last_hist_day)
            if era5_idx is None or era5_idx < self.seq_len - 1:
                continue
            valid.append(j)
        return np.array(valid, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(
        self, idx: int
    ) -> Tuple[List["Data"], torch.Tensor, torch.Tensor, torch.Tensor]:
        j = int(self.valid_starts[idx])
        init_date = self.ecmwf_init_dates[j]

        # ── ERA5 history window (timesteps 0–29) ──────────────────────────────
        # History: 30 ERA5 days ending the day BEFORE init_date.
        last_hist_day = init_date - pd.Timedelta(days=1)
        era5_end_idx  = self.era5_date_to_idx[last_hist_day]  # inclusive end
        era5_start_idx = era5_end_idx - self.seq_len + 1

        history = self.era5_dynamic[:, era5_start_idx : era5_end_idx + 1, :]
        # shape: (n_lakes, 30, 14)

        # ── ECMWF forecast window (timesteps 30–39) ───────────────────────────
        # Build a (n_lakes, 10, 14) block:
        #   slots 0–2: zeroed (obs_mask=0, latest_wse=0, days_since_last_obs=0)
        #   slots 3–4: sin/cos of DOY for each valid forecast date
        #   slots 5–13: ECMWF climate (same 9-variable order as ERA5 slots 5–13)
        ecmwf_slice = self.ecmwf_forecast[:, j, :, :]  # (n_lakes, 10, 9)

        fc_block = np.zeros(
            (self.n_lakes, self.forecast_horizon, 14), dtype=np.float32
        )
        # DOY encoding for each forecast valid_date (init_date + 0, 1, ..., 9)
        for d in range(self.forecast_horizon):
            valid_date = init_date + pd.Timedelta(days=d)
            doy = valid_date.dayofyear
            fc_block[:, d, 3] = float(np.sin(2 * np.pi * doy / 365.25))
            fc_block[:, d, 4] = float(np.cos(2 * np.pi * doy / 365.25))

        # Climate features from ECMWF (slots 5–13)
        fc_block[:, :, SWOT_DIM:] = ecmwf_slice  # SWOT_DIM=5

        # ── Concatenate full 40-step sequence ─────────────────────────────────
        seq_features = np.concatenate([history, fc_block], axis=1)
        # shape: (n_lakes, 40, 14)

        # ── Labels and mask ───────────────────────────────────────────────────
        # Target: WSE and obs_mask at ERA5 indices corresponding to
        #         valid_dates init_date+0, init_date+1, ..., init_date+9
        labels     = np.zeros((self.n_lakes, self.forecast_horizon), dtype=np.float32)
        label_mask = np.zeros((self.n_lakes, self.forecast_horizon), dtype=np.float32)

        for d in range(self.forecast_horizon):
            target_date = init_date + pd.Timedelta(days=d)
            t_idx = self.era5_date_to_idx.get(target_date)
            if t_idx is not None:
                labels[:, d]     = np.nan_to_num(self.wse_labels[:, t_idx], nan=0.0)
                label_mask[:, d] = self.obs_mask[:, t_idx]

        # ── Build PyG Data list (one per timestep, 40 total) ──────────────────
        data_list = [
            Data(
                x=torch.from_numpy(seq_features[:, t, :]).float(),
                edge_index=self.edge_index,
                num_nodes=self.n_lakes,
            )
            for t in range(self.seq_len + self.forecast_horizon)
        ]

        static = torch.from_numpy(self.static_features)  # (n_lakes, n_static)

        return (
            data_list,
            static,
            torch.from_numpy(labels).float(),     # (n_lakes, 10)
            torch.from_numpy(label_mask).float(),  # (n_lakes, 10)
        )


def collate_temporal_graph_batch_lake(
    batch: List[Tuple[List["Data"], torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[List[List["Data"]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a batch of lake forecast samples.

    Returns:
        data_lists:   list of per-sample 40-graph sequences
        static_feats: (batch_size, n_lakes, n_static)
        labels:       (batch_size, n_lakes, 10)
        masks:        (batch_size, n_lakes, 10)
    """
    data_lists   = [b[0] for b in batch]
    static_feats = torch.stack([b[1] for b in batch])   # (B, n_lakes, n_static)
    labels       = torch.stack([b[2] for b in batch])   # (B, n_lakes, 10)
    masks        = torch.stack([b[3] for b in batch])   # (B, n_lakes, 10)
    return data_lists, static_feats, labels, masks


def build_temporal_dataset_from_lake_datacubes(
    era5_dynamic_datacube_path: Union[str, Path],
    ecmwf_forecast_datacube_path: Union[str, Path],
    static_datacube_path: Union[str, Path],
    lake_graph_path: Union[str, Path],
    lake_ids: Optional[np.ndarray] = None,
    seq_len: int = 30,
    forecast_horizon: int = 10,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    require_obs_on_any_forecast_day: bool = True,
    lake_graph_source_col: str = "lake_id",
    lake_graph_downstream_col: str = "downstream_lake_id",
) -> Tuple[
    TemporalGraphDatasetLake,
    TemporalGraphDatasetLake,
    TemporalGraphDatasetLake,
    Dict,
]:
    """
    Build train / val / test TemporalGraphDatasetLake splits from datacubes.

    Feature normalization (training-set statistics only, no leakage):
      - log1p then z-score: days_since_last_obs (idx 2), P (idx 7) — right-skewed
      - z-score only:       LWd(5), SWd(6), P(7), Pres(8), Temp(9),
                            Wind(10), RelHum(11), sd(12), swvl1(13)
      - unchanged:          obs_mask(0, binary), latest_wse(1, pre-normalised),
                            time_doy_sin/cos(3/4, bounded [-1,1])
      The same mean/std are applied to ECMWF features (identical variable slots 5–13).

    Args:
        era5_dynamic_datacube_path:   Path to swot_lake_era5_dynamic_datacube_*.nc
        ecmwf_forecast_datacube_path: Path to swot_lake_ecmwf_forecast_datacube.nc
        static_datacube_path:         Path to swot_lake_static_datacube.nc
        lake_graph_path:              Path to GRIT PLD lake graph CSV
        lake_ids:                     Optional subset of lake IDs. If None, use all.
        seq_len:                      ERA5 history window length (default 30)
        forecast_horizon:             ECMWF forecast window (default 10)
        train_frac, val_frac, test_frac: Chronological split fractions
        require_obs_on_any_forecast_day: Only keep init_dates where at least one
                                         of the 10 forecast days has a SWOT observation.
        lake_graph_source_col:        Source lake_id column in lake graph CSV
        lake_graph_downstream_col:    Downstream lake_id column in lake graph CSV

    Returns:
        (train_ds, val_ds, test_ds, norm_stats)
        norm_stats keys: log1p_dynamic_indices, zscore_dynamic_indices,
                         dynamic_mean, dynamic_std, static_mean, static_std
    """
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
        era5_dynamic_datacube_path=era5_dynamic_datacube_path,
        ecmwf_forecast_datacube_path=ecmwf_forecast_datacube_path,
        static_datacube_path=static_datacube_path,
        lake_ids=lake_ids,
    )

    # ── Build lake graph ──────────────────────────────────────────────────────
    edge_index, _, _, _ = build_graph_from_lake_graph(
        lake_graph_csv=lake_graph_path,
        lake_ids=lake_ids_out,
        source_col=lake_graph_source_col,
        downstream_col=lake_graph_downstream_col,
    )

    # ── Find valid init_date positions and split ───────────────────────────────
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
            "No valid init_dates found. Check that ERA5 and ECMWF date ranges overlap "
            "and that SWOT WSE observations exist within the forecast windows."
        )

    train_end = int(n_valid * train_frac)
    val_end   = int(n_valid * (train_frac + val_frac))
    train_idx = all_valid[:train_end]
    val_idx   = all_valid[train_end:val_end]
    test_idx  = all_valid[val_end:]

    # ── Feature normalization (training-set statistics only) ──────────────────
    # Dynamic feature index reference (ERA5_INPUT_VARS):
    #   0=obs_mask  1=latest_wse  2=days_since_last_obs  3=doy_sin  4=doy_cos
    #   5=LWd  6=SWd  7=P  8=Pres  9=Temp  10=Wind  11=RelHum  12=sd  13=swvl1
    _LOG1P_DYN   = [2, 7]                              # right-skewed, zero-bounded
    _ZSCORE_DYN  = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13] # all continuous features

    era5_dynamic    = era5_dynamic.copy()
    ecmwf_forecast  = ecmwf_forecast.copy()

    # Step 1: log1p on skewed features (all dates)
    for i in _LOG1P_DYN:
        era5_dynamic[:, :, i] = np.log1p(np.clip(era5_dynamic[:, :, i], 0, None))

    # Step 2: compute z-score stats from training ERA5 window only
    # Training ERA5 window: from (first train init_date - 30 days) to (last train init_date - 1 day)
    if len(train_idx) > 0:
        first_train_init = ecmwf_init_dates[train_idx[0]]
        last_train_init  = ecmwf_init_dates[train_idx[-1]]
        era5_train_start = era5_date_to_idx.get(
            first_train_init - pd.Timedelta(days=seq_len)
        )
        era5_train_end   = era5_date_to_idx.get(
            last_train_init - pd.Timedelta(days=1)
        )
        if era5_train_start is None:
            era5_train_start = 0
        if era5_train_end is None:
            era5_train_end = len(era5_dates) - 1
        train_era5_slice = era5_dynamic[:, era5_train_start : era5_train_end + 1, :]
    else:
        train_era5_slice = era5_dynamic  # fallback

    n_dyn    = era5_dynamic.shape[-1]
    dyn_mean = np.zeros(n_dyn, dtype=np.float32)
    dyn_std  = np.ones(n_dyn,  dtype=np.float32)

    for i in _ZSCORE_DYN:
        vals       = train_era5_slice[:, :, i].ravel()
        dyn_mean[i] = float(vals.mean())
        dyn_std[i]  = float(vals.std()) + 1e-8

    # Step 3: apply z-score to all ERA5 dates
    for i in _ZSCORE_DYN:
        era5_dynamic[:, :, i] = (era5_dynamic[:, :, i] - dyn_mean[i]) / dyn_std[i]

    # Step 4: apply the same normalization to ECMWF climate features.
    # ECMWF feature index k (0..8) → ERA5 index (5 + k).
    # ecmwf_forecast shape: (n_lakes, n_init_dates, forecast_horizon, 9)
    ecmwf_log1p_indices = [
        k for k, var in enumerate(ECMWF_CLIMATE_VARS)
        if (SWOT_DIM + k) in _LOG1P_DYN
    ]
    ecmwf_zscore_indices = [
        k for k, var in enumerate(ECMWF_CLIMATE_VARS)
        if (SWOT_DIM + k) in _ZSCORE_DYN
    ]

    for k in ecmwf_log1p_indices:
        ecmwf_forecast[:, :, :, k] = np.log1p(
            np.clip(ecmwf_forecast[:, :, :, k], 0, None)
        )

    for k in ecmwf_zscore_indices:
        era5_idx = SWOT_DIM + k
        ecmwf_forecast[:, :, :, k] = (
            ecmwf_forecast[:, :, :, k] - dyn_mean[era5_idx]
        ) / dyn_std[era5_idx]

    # Step 5: z-score static features
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
    }

    # ── Construct three datasets sharing the same arrays ──────────────────────
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

    print(
        f"Lake dataset built: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test "
        f"samples  ({len(lake_ids_out)} lakes)"
    )

    return train_ds, val_ds, test_ds, norm_stats
