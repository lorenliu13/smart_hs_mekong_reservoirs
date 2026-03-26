"""
Temporal graph dataset for lake-based SWOT-GNN with 10-day multi-step forecasting.

Each training sample is indexed by an ECMWF init_date and contains:
  - History window (seq_len=30 days): ERA5-Land climate + SWOT WSE features
  - Forecast window (forecast_horizon=10 days): ECMWF IFS climate + zeroed SWOT features
  - Labels: WSE at init_date + days 0..9 (shape: n_lakes × 10)
  - Mask:   obs_mask at those forecast dates

Data flow:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                         NetCDF Datacubes                             │
    │                                                                      │
    │  swot_lake_wse_datacube.nc          swot_lake_static_datacube.nc     │
    │  (lake, time) → obs_mask,           (lake, static_feature)           │
    │                 latest_wse, …                    │                   │
    │         │                                        │                   │
    │  swot_lake_era5_climate_datacube.nc              │                   │
    │  (lake, time) → LWd, SWd, P, …                   │                   │
    │         │                                        │                   │
    │  swot_lake_ecmwf_forecast_datacube.nc            │                   │
    │  (lake, init_time, lead_day) → LWd, SWd, P, …    │                   │
    │         │                                        │                   │
    └─────────┼────────────────────────────────────────┼───────────────────┘
              │                                        │
              ▼                                        │
    assemble_lake_features_from_datacubes()            │
      · intersects lake / date axes across all cubes   │
      · stacks WSE (8) + ERA5 (13) → dynamic_features  │
        shape: (n_lakes, n_dates, 21)                  │
      · returns wse_target, obs_mask arrays            │
              │                                        │
              ▼                                        ▼
    build_temporal_dataset_from_lake_datacubes()
      · normalises features (log1p + z-score, train stats only)
      · filters valid init_dates (full ERA5 history available)
      · chronological split → train / val / test index arrays
      · builds edge_index from lake graph CSV
              │
      ┌───────┼───────┐
      ▼       ▼       ▼
    train   val    test    ← TemporalGraphDatasetLake (share same arrays)
              │
              │  __getitem__(idx)  →  one sample per ECMWF init_date
              │
              ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  One training sample                                            │
    │                                                                 │
    │  t=0 ──────────────── t=29 │ t=30 ────────────────── t=39       │
    │  ◄── ERA5 history (30d) ──► │ ◄── ECMWF forecast (10d) ──►      │
    │  SWOT features (8) filled   │ SWOT features (6) zeroed out      │
    │  ERA5 climate  (13) filled  │ DOY sin/cos (2) kept              │
    │                             │ ECMWF climate  (13) filled        │
    │                                                                 │
    │  data_list : list[PyG Data] × 40  — one graph snapshot/step     │
    │  static    : Tensor (n_lakes, n_static)                         │
    │  labels    : Tensor (n_lakes, 10)  — WSE, NaN→0                 │
    │  label_mask: Tensor (n_lakes, 10)  — 1 where SWOT observed      │
    └─────────────────────────────────────────────────────────────────┘

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
import xarray as xr
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict

import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from .graph_builder import build_graph_from_lake_graph

# ─── Feature ordering ──────────────────────────────────────────────────────────

# 8 SWOT-derived input features loaded from the WSE datacube (indices 0–7).
WSE_INPUT_VARS: List[str] = [
    "obs_mask",            # 0:  1 where SWOT observed, 0 otherwise
    "latest_wse",          # 1:  forward-filled normalised WSE
    "latest_wse_u",        # 2:  forward-filled WSE uncertainty
    "latest_wse_std",      # 3:  forward-filled within-pass WSE std
    "latest_area_total",   # 4:  forward-filled total water area
    "days_since_last_obs", # 5:  days since last SWOT pass
    "time_doy_sin",        # 6:  sin(2π × doy / 365.25)
    "time_doy_cos",        # 7:  cos(2π × doy / 365.25)
]

# 13 ERA5-Land climate features (indices 8–20). Order matches ECMWF_CLIMATE_VARS
# so ERA5 reanalysis and ECMWF forecast tensors can be concatenated directly.
ERA5_CLIMATE_VARS: List[str] = [
    "LWd", "SWd", "P", "Pres", "Temp", "Td", "Wind",
    "sf", "sd", "swvl1", "swvl2", "swvl3", "swvl4",
]

ERA5_INPUT_VARS:    List[str] = WSE_INPUT_VARS + ERA5_CLIMATE_VARS
ECMWF_CLIMATE_VARS: List[str] = ERA5_CLIMATE_VARS

SWOT_DIM:    int = 8   # indices 0–7
CLIMATE_DIM: int = 13  # indices 8–20

# Feature indices zeroed in forecast mode (future timesteps have no SWOT data).
# Indices 6–7 (doy_sin/cos) are kept because the calendar date is still known.
WSE_LAKE_DYNAMIC_INDICES: List[int] = [0, 1, 2, 3, 4, 5]

# ───────────────────────────────────────────────────────────────────────────────


def _stack_vars(ds: xr.Dataset, var_names: List[str], **sel) -> np.ndarray:
    """Stack named variables from a dataset into a float32 array with NaN→0."""
    arrays = np.stack([ds[v].sel(**sel).values for v in var_names], axis=-1)
    return np.nan_to_num(arrays.astype(np.float32), nan=0.0)


def assemble_lake_features_from_datacubes(
    wse_datacube_path: Union[str, Path],
    era5_climate_datacube_path: Union[str, Path],
    ecmwf_forecast_datacube_path: Union[str, Path],
    static_datacube_path: Union[str, Path],
    lake_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Load lake features from four NetCDF datacubes into numpy arrays.

    Returns:
        dynamic_features:     (n_lakes, n_dates, 21) — WSE (8) + ERA5 climate (13)
        ecmwf_forecast:       (n_lakes, n_init_dates, n_forecast_days, 13)
        static_features:      (n_lakes, n_static)
        wse_target:           (n_lakes, n_dates)  — NaN where not observed
        obs_mask:             (n_lakes, n_dates)  — 1 where SWOT observed
        lake_ids_out:         (n_lakes,)
        dates_out:            DatetimeIndex
        ecmwf_init_dates_out: DatetimeIndex
    """
    ds_wse    = xr.open_dataset(wse_datacube_path)
    ds_era5   = xr.open_dataset(era5_climate_datacube_path)
    ds_ecmwf  = xr.open_dataset(ecmwf_forecast_datacube_path)
    ds_static = xr.open_dataset(static_datacube_path)

    try:
        # Lake IDs: intersection of all four cubes, then subset if requested
        all_cube_lakes = np.intersect1d(
            np.intersect1d(ds_wse.lake.values, ds_era5.lake.values),
            np.intersect1d(ds_ecmwf.lake.values, ds_static.lake.values),
        ).astype(np.int64)
        if lake_ids is None:
            lake_ids = all_cube_lakes
        else:
            lake_ids = np.array(
                [lid for lid in lake_ids if lid in all_cube_lakes], dtype=np.int64
            )
        if len(lake_ids) == 0:
            raise ValueError("No lakes in common across all four datacubes.")

        # Dates: intersection of WSE and ERA5 time axes
        dates = pd.DatetimeIndex(ds_wse.time.values).intersection(
            pd.DatetimeIndex(ds_era5.time.values)
        )
        if len(dates) == 0:
            raise ValueError("No overlapping dates across WSE and ERA5 datacubes.")

        ecmwf_init_dates = pd.DatetimeIndex(ds_ecmwf.init_time.values)

        # Dynamic features: WSE block (8) + ERA5 block (13) → (n_lakes, n_dates, 21)
        wse_feat  = _stack_vars(ds_wse,  WSE_INPUT_VARS,    lake=lake_ids, time=dates)
        era5_feat = _stack_vars(ds_era5, ERA5_CLIMATE_VARS, lake=lake_ids, time=dates)
        dynamic_features = np.concatenate([wse_feat, era5_feat], axis=-1)

        wse_target = ds_wse["wse"].sel(lake=lake_ids, time=dates).values.astype(np.float32)
        obs_mask   = ds_wse["obs_mask"].sel(lake=lake_ids, time=dates).values.astype(np.float32)

        # ECMWF forecast → (n_lakes, n_init_dates, n_forecast_days, 13)
        ecmwf_forecast = _stack_vars(
            ds_ecmwf, ECMWF_CLIMATE_VARS, lake=lake_ids, init_time=ecmwf_init_dates
        )

        static_arr = np.nan_to_num(
            ds_static["static_feature"].sel(lake=lake_ids).values.astype(np.float32), nan=0.0
        )

        return (
            dynamic_features, ecmwf_forecast, static_arr,
            wse_target, obs_mask, lake_ids, dates, ecmwf_init_dates,
        )

    finally:
        ds_wse.close()
        ds_era5.close()
        ds_ecmwf.close()
        ds_static.close()


class TemporalGraphDatasetLake(Dataset):
    """
    Dataset for 10-day-ahead multi-step WSE forecasting for lakes.

    Each sample:
      Given 30 days of ERA5 history + 10 ECMWF forecast days → predict WSE for those 10 days.

    Input tensor shape: (n_lakes, seq_len+forecast_horizon, SWOT_DIM+CLIMATE_DIM)
      - Timesteps  0-(seq_len-1): ERA5 history (SWOT features + ERA5 climate)
      - Timesteps  seq_len-end:   ECMWF forecast (zeroed SWOT features + ECMWF climate + DOY encoding)

    Labels shape: (n_lakes, forecast_horizon) — WSE at each forecast day (NaN→0, use mask).
    Mask shape:   (n_lakes, forecast_horizon) — 1 where SWOT observed, 0 otherwise.
    """

    def __init__(
        self,
        era5_dynamic: np.ndarray,           # (n_lakes, n_era5_dates, SWOT_DIM+CLIMATE_DIM)
        ecmwf_forecast: np.ndarray,         # (n_lakes, n_init_dates, forecast_horizon, CLIMATE_DIM)
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
            era5_dynamic:      (n_lakes, n_era5_dates, SWOT_DIM+CLIMATE_DIM) — normalized dynamic features
            ecmwf_forecast:    (n_lakes, n_init_dates, forecast_horizon, CLIMATE_DIM) — normalized ECMWF climate
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

        # O(1) date → index lookup — avoids repeated linear scans during __getitem__
        self.era5_date_to_idx  = {d: i for i, d in enumerate(era5_dates)}
        self.ecmwf_date_to_idx = {d: i for i, d in enumerate(ecmwf_init_dates)}

        # valid_starts: integer positions j in ecmwf_init_dates where the full
        # ERA5 history window is available. Passed in as a pre-computed subset
        # (train/val/test split) or computed fresh from scratch.
        self.valid_starts = indices if indices is not None else self._find_valid_starts()

    def _find_valid_starts(self) -> np.ndarray:
        """
        Return all ECMWF init_date positions where the 30-day ERA5 history window
        is fully available (init_date - 30 days through init_date - 1 must be in ERA5).
        """
        valid = []
        for j, init_date in enumerate(self.ecmwf_init_dates):
            # The history window uses ERA5 up to (but not including) the init_date,
            # so the last history day is init_date - 1.
            last_hist_day = init_date - pd.Timedelta(days=1)
            era5_idx = self.era5_date_to_idx.get(last_hist_day)
            # era5_idx < seq_len - 1 means there aren't enough prior ERA5 days
            # to fill the full history window (would run before the array start).
            if era5_idx is None or era5_idx < self.seq_len - 1:
                continue
            valid.append(j)
        return np.array(valid, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(
        self, idx: int
    ) -> Tuple[List["Data"], torch.Tensor, torch.Tensor, torch.Tensor]:
        # Map dataset index → position in ecmwf_init_dates (skips invalid starts)
        j = int(self.valid_starts[idx])
        init_date = self.ecmwf_init_dates[j]

        # ── ERA5 history window (timesteps 0–29) ──────────────────────────────
        # History: 30 ERA5 days ending the day BEFORE init_date.
        # We use ERA5 (reanalysis) for the past because it is a complete,
        # gap-free gridded product — no missing dates unlike raw SWOT observations.
        last_hist_day = init_date - pd.Timedelta(days=1)
        era5_end_idx  = self.era5_date_to_idx[last_hist_day]  # inclusive end
        era5_start_idx = era5_end_idx - self.seq_len + 1       # inclusive start

        history = self.era5_dynamic[:, era5_start_idx : era5_end_idx + 1, :]
        # shape: (n_lakes, seq_len=30, SWOT_DIM+CLIMATE_DIM=14)

        # ── ECMWF forecast window (timesteps 30–39) ───────────────────────────
        # For the forecast horizon we don't have observed SWOT data yet, so the
        # SWOT-derived slots are zeroed out. Only climate and DOY are populated.
        #
        # Feature layout within each timestep (total 14 dims):
        #   [0] obs_mask            → 0 (no observation available in future)
        #   [1] latest_wse          → 0 (unknown)
        #   [2] latest_wse_u        → 0
        #   [3] latest_wse_std      → 0
        #   [4] latest_area_total   → 0
        #   [5] days_since_last_obs → 0 (not meaningful in forecast window)
        #   [6] doy_sin             → computed from valid_date dayofyear
        #   [7] doy_cos             → computed from valid_date dayofyear
        #   [8–13]                  → ECMWF climate variables (9 vars, SWOT_DIM=5 offset)
        ecmwf_slice = self.ecmwf_forecast[:, j, :, :]  # (n_lakes, forecast_horizon, CLIMATE_DIM)

        fc_block = np.zeros(
            (self.n_lakes, self.forecast_horizon, SWOT_DIM + CLIMATE_DIM), dtype=np.float32
        )
        # Fill DOY encoding for each forecast valid_date.
        # Indices 6/7 (doy_sin/cos) match their position in the ERA5 SWOT block,
        # keeping the feature layout identical across history and forecast timesteps.
        for d in range(self.forecast_horizon):
            valid_date = init_date + pd.Timedelta(days=d)
            doy = valid_date.dayofyear
            fc_block[:, d, 6] = float(np.sin(2 * np.pi * doy / 365.25))
            fc_block[:, d, 7] = float(np.cos(2 * np.pi * doy / 365.25))

        # Paste ECMWF climate into the trailing CLIMATE_DIM slots (after SWOT_DIM=5 zeros)
        fc_block[:, :, SWOT_DIM:] = ecmwf_slice
        # shape: (n_lakes, forecast_horizon=10, 14)

        # ── Concatenate full 40-step sequence ─────────────────────────────────
        # Axis 1 is the time dimension; history comes first, then forecast.
        seq_features = np.concatenate([history, fc_block], axis=1)
        # shape: (n_lakes, seq_len+forecast_horizon=40, SWOT_DIM+CLIMATE_DIM=14)

        # ── Labels and mask ───────────────────────────────────────────────────
        # Look up actual SWOT WSE observations for each of the 10 forecast days.
        # NaN WSE (lake not observed on that day) is replaced with 0; label_mask
        # records which lakes were actually observed (1) vs missing (0).
        # The model loss should only be computed where label_mask == 1.
        labels     = np.zeros((self.n_lakes, self.forecast_horizon), dtype=np.float32)
        label_mask = np.zeros((self.n_lakes, self.forecast_horizon), dtype=np.float32)

        for d in range(self.forecast_horizon):
            target_date = init_date + pd.Timedelta(days=d)
            t_idx = self.era5_date_to_idx.get(target_date)
            if t_idx is not None:
                # nan_to_num: lakes not observed on this date have NaN WSE → set to 0
                labels[:, d]     = np.nan_to_num(self.wse_labels[:, t_idx], nan=0.0)
                label_mask[:, d] = self.obs_mask[:, t_idx]
            # If target_date is not in ERA5 (beyond ERA5 coverage), labels/mask stay 0

        # ── Build PyG Data list (one per timestep, 40 total) ──────────────────
        # Each Data object is one temporal "snapshot" of the lake graph.
        # The sequence of 40 Data objects is passed to the GNN encoder which
        # processes them in order (e.g. with an RNN/Transformer over time).
        data_list = [
            Data(
                x=torch.from_numpy(seq_features[:, t, :]).float(),  # (n_lakes, 14)
                edge_index=self.edge_index,                          # (2, n_edges)
                num_nodes=self.n_lakes,
            )
            for t in range(self.seq_len + self.forecast_horizon)
        ]

        static = torch.from_numpy(self.static_features)  # (n_lakes, n_static)

        return (
            data_list,                                     # list of 40 PyG Data objects
            static,                                        # (n_lakes, n_static)
            torch.from_numpy(labels).float(),              # (n_lakes, forecast_horizon=10)
            torch.from_numpy(label_mask).float(),          # (n_lakes, forecast_horizon=10)
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
    wse_datacube_path: Union[str, Path],
    era5_climate_datacube_path: Union[str, Path],
    ecmwf_forecast_datacube_path: Union[str, Path],
    static_datacube_path: Union[str, Path],
    lake_graph_path: Union[str, Path],
    lake_ids: Optional[np.ndarray] = None,
    seq_len: int = 30,
    forecast_horizon: int = 10,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    require_obs_on_any_forecast_day: bool = True
) -> Tuple[
    TemporalGraphDatasetLake,
    TemporalGraphDatasetLake,
    TemporalGraphDatasetLake,
    Dict,
]:
    """
    Build train / val / test TemporalGraphDatasetLake splits from datacubes.

    Feature normalization (training-set statistics only, no leakage):
      Dynamic feature indices (21 total — WSE_INPUT_VARS 0-7 + ERA5_CLIMATE_VARS 8-20):
        0=obs_mask  1=latest_wse  2=latest_wse_u  3=latest_wse_std  4=latest_area_total
        5=days_since_last_obs  6=doy_sin  7=doy_cos
        8=LWd  9=SWd  10=P  11=Pres  12=Temp  13=Td  14=Wind  15=sf  16=sd  17-20=swvl1-4
      - log1p then z-score: days_since_last_obs(5), P(10), sf(15) — right-skewed, zero-bounded
      - z-score only:       latest_wse_u(2), latest_wse_std(3), latest_area_total(4),
                            LWd(8), SWd(9), Pres(11), Temp(12), Td(13), Wind(14), sd(16), swvl1-4(17-20)
      - unchanged:          obs_mask(0, binary), latest_wse(1, pre-normalised per lake),
                            doy_sin/cos(6/7, bounded [-1,1])
      The same mean/std are applied to ECMWF features (identical variable slots 8–20).

    Args:
        wse_datacube_path:            Path to swot_lake_wse_datacube_*.nc
        era5_climate_datacube_path:   Path to swot_lake_era5_climate_datacube.nc
        ecmwf_forecast_datacube_path: Path to swot_lake_ecmwf_forecast_datacube.nc
        static_datacube_path:         Path to swot_lake_static_datacube.nc
        lake_graph_path:              Path to GRIT PLD lake graph CSV
        lake_ids:                     Optional subset of lake IDs. If None, use all.
        seq_len:                      ERA5 history window length (default 30)
        forecast_horizon:             ECMWF forecast window (default 10)
        train_frac, val_frac, test_frac: Chronological split fractions
        require_obs_on_any_forecast_day: Only keep init_dates where at least one
                                         of the 10 forecast days has a SWOT observation.

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
        wse_datacube_path=wse_datacube_path,
        era5_climate_datacube_path=era5_climate_datacube_path,
        ecmwf_forecast_datacube_path=ecmwf_forecast_datacube_path,
        static_datacube_path=static_datacube_path,
        lake_ids=lake_ids,
    )

    # ── Build lake graph ──────────────────────────────────────────────────────
    edge_index, _, _, _ = build_graph_from_lake_graph(
        lake_graph_csv=lake_graph_path,
        lake_ids=lake_ids_out
    )

    # ── Find valid init_date positions and split ───────────────────────────────
    # Pre-build ERA5 date lookup once here (same logic as inside the Dataset class,
    # but needed here before the Dataset objects are constructed).
    era5_date_to_idx = {d: i for i, d in enumerate(era5_dates)}

    all_valid = []
    for j, init_date in enumerate(ecmwf_init_dates):
        # Require a full seq_len ERA5 history before init_date
        last_hist_day = init_date - pd.Timedelta(days=1)
        era5_idx = era5_date_to_idx.get(last_hist_day)
        if era5_idx is None or era5_idx < seq_len - 1:
            continue

        # Optionally skip init_dates where no lake has a SWOT observation in the
        # forecast window — those samples contribute nothing to supervised training.
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

    # Chronological split — no shuffling to prevent future data leaking into training.
    # train_frac / val_frac / test_frac must sum to 1.
    train_end = int(n_valid * train_frac)
    val_end   = int(n_valid * (train_frac + val_frac))
    train_idx = all_valid[:train_end]
    val_idx   = all_valid[train_end:val_end]
    test_idx  = all_valid[val_end:]

    # ── Feature normalization (training-set statistics only) ──────────────────
    # All normalization statistics are computed exclusively from the training
    # portion of the ERA5 time series to prevent data leakage into val/test sets.
    #
    # Dynamic feature index layout (n_features = SWOT_DIM + CLIMATE_DIM):
    #   0=obs_mask  1=latest_wse  2=latest_wse_u  3=latest_wse_std  4=latest_area_total
    #   5=days_since_last_obs  6=doy_sin  7=doy_cos
    #   8=LWd  9=SWd  10=P  11=Pres  12=Temp  13=Td  14=Wind  15=sf  16=sd  17-20=swvl1-4
    #
    # log1p first, then z-score for right-skewed zero-bounded vars (precip, snowfall, age):
    _LOG1P_DYN   = [5, 10, 15]
    # z-score only for continuous vars (uncertainty, std, area, radiation, met vars, soil moisture):
    _ZSCORE_DYN  = [2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # Not normalized: obs_mask (0, binary), latest_wse (1, already per-lake normalised),
    #                 doy_sin/cos (6/7, naturally bounded [-1, 1])

    # Work on copies so the original arrays from the assembler are not mutated
    era5_dynamic    = era5_dynamic.copy()
    ecmwf_forecast  = ecmwf_forecast.copy()

    # Step 1: log1p transform on skewed features across all ERA5 dates.
    # Clip to [0, ∞) first to handle any small negative values from floating-point noise.
    for i in _LOG1P_DYN:
        era5_dynamic[:, :, i] = np.log1p(np.clip(era5_dynamic[:, :, i], 0, None))

    # Step 2: compute z-score statistics from the training ERA5 window only.
    # The training window spans from (first_train_init_date - seq_len days) to
    # (last_train_init_date - 1 day), i.e., all ERA5 dates used as history in training.
    if len(train_idx) > 0:
        first_train_init = ecmwf_init_dates[train_idx[0]]
        last_train_init  = ecmwf_init_dates[train_idx[-1]]
        era5_train_start = era5_date_to_idx.get(
            first_train_init - pd.Timedelta(days=seq_len)
        )
        era5_train_end   = era5_date_to_idx.get(
            last_train_init - pd.Timedelta(days=1)
        )
        # Fall back to array boundaries if the exact date isn't in ERA5
        if era5_train_start is None:
            era5_train_start = 0
        if era5_train_end is None:
            era5_train_end = len(era5_dates) - 1
        train_era5_slice = era5_dynamic[:, era5_train_start : era5_train_end + 1, :]
    else:
        train_era5_slice = era5_dynamic  # edge-case fallback: no train samples

    # Allocate mean/std arrays — non-normalized indices keep mean=0, std=1 (identity)
    n_dyn    = era5_dynamic.shape[-1]
    dyn_mean = np.zeros(n_dyn, dtype=np.float32)
    dyn_std  = np.ones(n_dyn,  dtype=np.float32)

    for i in _ZSCORE_DYN:
        vals        = train_era5_slice[:, :, i].ravel()
        dyn_mean[i] = float(vals.mean())
        dyn_std[i]  = float(vals.std()) + 1e-8  # +epsilon prevents division by zero

    # Step 3: apply training z-score to ALL ERA5 dates (including val/test)
    for i in _ZSCORE_DYN:
        era5_dynamic[:, :, i] = (era5_dynamic[:, :, i] - dyn_mean[i]) / dyn_std[i]

    # Step 4: apply identical normalization to ECMWF climate features.
    # ECMWF variables mirror ERA5 climate but have their own array dimension k (0-based).
    # The mapping is: ECMWF index k → ERA5 feature index (SWOT_DIM + k).
    # We identify which ECMWF indices need log1p / z-score by checking the ERA5 lookup.
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
        # Reuse the ERA5 mean/std for the corresponding variable to keep
        # ERA5-history and ECMWF-forecast features on the same scale.
        era5_idx = SWOT_DIM + k
        ecmwf_forecast[:, :, :, k] = (
            ecmwf_forecast[:, :, :, k] - dyn_mean[era5_idx]
        ) / dyn_std[era5_idx]

    # Step 5: z-score static features (morphological / topographic lake attributes).
    # Statistics are computed over all lakes (no time dimension here).
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
    # All three splits point at the same normalized numpy arrays in memory;
    # only the `indices` argument differs, so there is no data duplication.
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
