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
    # Open all four datacubes. Caller is responsible for closing them after use.
    ds_wse    = xr.open_dataset(wse_datacube_path)
    ds_era5   = xr.open_dataset(era5_climate_datacube_path)
    ds_ecmwf  = xr.open_dataset(ecmwf_forecast_datacube_path)
    ds_static = xr.open_dataset(static_datacube_path)

    try:
        # Get the intersection of lake IDs across all datacubes to ensure alignment. The graph construction
        all_cube_lakes = np.intersect1d(
            np.intersect1d(ds_wse.lake.values, ds_era5.lake.values),
            np.intersect1d(ds_ecmwf.lake.values, ds_static.lake.values),
        ).astype(np.int64)

        # if lake_ids is provided, filter to those that are in the datacubes; otherwise use all common lakes
        if lake_ids is None:
            lake_ids = all_cube_lakes
        else:
            # Filter the provided lake_ids to those that are actually present in all datacubes. This ensures we only attempt to load data for lakes that have complete information across all sources.
            lake_ids = np.array(
                [lid for lid in lake_ids if lid in all_cube_lakes], dtype=np.int64
            )
        if len(lake_ids) == 0:
            raise ValueError("No lakes in common across all four datacubes.")

        # Find the intersection of dates across WSE and ERA5 datacubes to ensure temporal alignment. ECMWF init_dates will be filtered later based on ERA5 history availability.
        dates = pd.DatetimeIndex(ds_wse.time.values).intersection(
            pd.DatetimeIndex(ds_era5.time.values)
        )
        if len(dates) == 0:
            raise ValueError("No overlapping dates across WSE and ERA5 datacubes.")

        # ECMWF init_dates (forecast issue dates) may not align perfectly with ERA5/WSE dates. We'll filter to valid init_dates later, but for now we just need the full list of init_dates available in the ECMWF datacube.
        ecmwf_init_dates = pd.DatetimeIndex(ds_ecmwf.init_time.values)

        # Dynamic features: WSE block (8) + ERA5 block (13) → (n_lakes, n_dates, 21)
        wse_feat  = _stack_vars(ds_wse,  WSE_INPUT_VARS,    lake=lake_ids, time=dates) # dimensions: (n_lakes, n_dates, 8)
        era5_feat = _stack_vars(ds_era5, ERA5_CLIMATE_VARS, lake=lake_ids, time=dates) # dimensions: (n_lakes, n_dates, 13)
        dynamic_features = np.concatenate([wse_feat, era5_feat], axis=-1) # shape: (n_lakes, n_dates, 21)

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
        era5_dates: pd.DatetimeIndex,       # All ERA5 dates in chronological order
        ecmwf_init_dates: pd.DatetimeIndex, # All ECMWF init_dates in chronological order
        wse_labels: np.ndarray,             # (n_lakes, n_era5_dates) # Observed WSE values for label extraction (NaN where not observed)
        obs_mask: np.ndarray,               # (n_lakes, n_era5_dates) # 1 where SWOT observed, 0 otherwise (for label masking)
        lake_ids: np.ndarray,               # (n_lakes,)
        seq_len: int = 30,                  # ERA5 history window length in days
        forecast_horizon: int = 10,         # ECMWF forecast window length in days
        indices: Optional[np.ndarray] = None, # Optional pre-computed array of valid ECMWF init_date positions (train/val/test split
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
        # {date: i} maps ERA5 dates to their position in the era5_dynamic array, which is needed to efficiently extract the history window and label values for each init_date during __getitem__.
        self.era5_date_to_idx  = {d: i for i, d in enumerate(era5_dates)}
        # {init_date: j} maps ECMWF init_dates to their position in the ecmwf_forecast array. We filter valid init_dates later based on ERA5 history availability, but we need this mapping to efficiently look up the forecast climate for each init_date during __getitem__.
        self.ecmwf_date_to_idx = {d: i for i, d in enumerate(ecmwf_init_dates)}

        # valid_starts: indices if provided, otherwise find valid init_dates with full ERA5 history coverage
        # indices: provided when building the train/val/test datasets
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
        """
        
        For the given sample index, construct the input feature sequence, labels, and mask.

        Inputs:
            idx: index into self.valid_starts, which maps to a valid ECMWF init_date with full ERA5 history coverage.

        Returns: 
            data_list: list of PyG Data objects, one per timestep (length seq_len + forecast_horizon)
            static: Tensor of shape (n_lakes, n_static) — static features for all lakes
            labels: Tensor of shape (n_lakes, forecast_horizon) — WSE values for forecast days (NaN→0)
            label_mask: Tensor of shape (n_lakes, forecast_horizon) — 1 where SWOT observed, 0 otherwise

        """

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
        ecmwf_slice = self.ecmwf_forecast[:, j, :self.forecast_horizon, :]  # (n_lakes, forecast_horizon, CLIMATE_DIM)

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
        ] # dim: list of 40 Data objects, each with x shape (n_lakes, 14)

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
    Create a batch of temporal graph samples by collating lists of PyG Data objects and tensors.

    Inputs: 
    batch: List of samples, where each sample is a tuple of:
        - data_list: list of 40 PyG Data objects (one per timestep)
        - static: Tensor (n_lakes, n_static)
        - labels: Tensor (n_lakes, 10)
        - label_mask: Tensor (n_lakes, 10)
    Returns:
        data_lists:   batch size of [40 * PyG Data]
        static_feats: (batch_size, n_lakes, n_static)
        labels:       (batch_size, n_lakes, 10)
        masks:        (batch_size, n_lakes, 10)
    """

    # 
    data_lists   = [b[0] for b in batch] # B * list[40 * Data]
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


def build_spatial_cv_fold(
    wse_datacube_path: Union[str, Path],
    era5_climate_datacube_path: Union[str, Path],
    ecmwf_forecast_datacube_path: Union[str, Path],
    static_datacube_path: Union[str, Path],
    lake_graph_path: Union[str, Path],
    n_folds: int = 5,
    fold_idx: int = 0,
    spatial_split_seed: int = 42,
    seq_len: int = 30,
    forecast_horizon: int = 10,
    val_frac: float = 0.15,
    val_method: str = "temporal",
    spatial_val_frac: float = 0.1,
    require_obs_on_any_forecast_day: bool = True,
) -> Tuple[
    "TemporalGraphDatasetLake",
    "TemporalGraphDatasetLake",
    "TemporalGraphDatasetLake",
    Dict,
]:
    """
    Build train / val / test datasets for one fold of random spatial cross-validation.

    Lake split (spatial axis):
        Lakes are randomly shuffled with `spatial_split_seed`, then divided into
        `n_folds` equal-sized groups.  The group at position `fold_idx` becomes the
        held-out test set; the remaining (n_folds - 1) / n_folds lakes form the
        train / val pool.

    Validation strategy (controlled by `val_method`):

        ``"temporal"`` (default):
            The full time series of train-fold lakes is split chronologically.
            The last ``val_frac`` fraction of valid init_dates forms the val set;
            the rest forms the training set.  Both sets use the same lake mask
            (all train-fold lakes).

        ``"spatial"``:
            All valid init_dates are used for both training and validation.
            A random ``spatial_val_frac`` fraction of the train-fold lakes is
            held out as a spatial validation set.  Training loss is gated to the
            remaining train-train lakes; validation loss is gated to the val lakes.
            Use this mode when you want to train on the complete time series.

    Graph structure:
        ALL lakes remain as nodes in the graph for every dataset so that message
        passing can propagate information across the full network topology.
        The ``spatial_mask`` attribute on each returned dataset tells
        ``_run_epoch`` which nodes contribute to the loss for that dataset.

    Normalization (no leakage):
        Dynamic feature z-score statistics are computed exclusively from the
        lakes that contribute to the training loss (train-train lakes for spatial
        val, all train-fold lakes for temporal val), using all ERA5 dates.
        Static feature statistics follow the same rule.  Both are then applied
        to all lakes uniformly.

    Args:
        wse_datacube_path:            Path to swot_lake_wse_datacube_*.nc
        era5_climate_datacube_path:   Path to swot_lake_era5_climate_datacube.nc
        ecmwf_forecast_datacube_path: Path to swot_lake_ecmwf_forecast_datacube.nc
        static_datacube_path:         Path to swot_lake_static_datacube.nc
        lake_graph_path:              Path to GRIT PLD lake graph CSV
        n_folds:                      Total number of spatial folds (default 5)
        fold_idx:                     Which fold to use as the test set (0-indexed)
        spatial_split_seed:           RNG seed for the lake shuffle
        seq_len:                      ERA5 history window length (default 30)
        forecast_horizon:             ECMWF forecast window (default 10)
        val_frac:                     Fraction of init_dates held back for val
                                      (only used when val_method="temporal")
        val_method:                   ``"temporal"`` or ``"spatial"`` — see above
        spatial_val_frac:             Fraction of train-fold lakes used as the
                                      spatial val set (only when val_method="spatial")
        require_obs_on_any_forecast_day: Skip init_dates with no SWOT observation in
                                         any forecast day

    Returns:
        (train_ds, val_ds, test_ds, norm_stats)

        Each dataset has one extra attribute set after construction:
            spatial_mask : (n_lakes,) float32 tensor — 1 for this dataset's
                           active lakes (loss-active nodes)

        Pass ``spatial_mask=ds.spatial_mask`` to ``_run_epoch``.

        norm_stats keys: log1p_dynamic_indices, zscore_dynamic_indices,
                         dynamic_mean, dynamic_std, static_mean, static_std
    """
    if not 0 <= fold_idx < n_folds:
        raise ValueError(f"fold_idx must be in [0, {n_folds - 1}], got {fold_idx}")

    # ── Load all arrays from datacubes ────────────────────────────────────────
    (
        era5_dynamic, # (n_lakes, n_era5_dates, SWOT_DIM+CLIMATE_DIM)
        ecmwf_forecast, # (n_lakes, n_ecmwf_init_dates, forecast_horizon, CLIMATE_DIM)
        static_features, # (n_lakes, n_static)
        wse_labels, # (n_lakes, n_time_steps) actual SWOT WSE observations (NaN if not observed)
        obs_mask, # (n_lakes, n_time_steps) binary mask of SWOT WSE observations
        lake_ids_out, # (n_lakes,) array of lake IDs corresponding to the first dimension of all arrays
        era5_dates, # list of pd.Timestamp, length n_era5_dates
        ecmwf_init_dates, # list of pd.Timestamp, length n_ecmwf_init_dates
    ) = assemble_lake_features_from_datacubes(
        wse_datacube_path=wse_datacube_path,
        era5_climate_datacube_path=era5_climate_datacube_path,
        ecmwf_forecast_datacube_path=ecmwf_forecast_datacube_path,
        static_datacube_path=static_datacube_path,
    )

    # Tally total lakes for sanity checks and graph construction
    n_lakes_total = len(lake_ids_out)

    # ── Build lake graph ──────────────────────────────────────────────────────
    edge_index, _, _, _ = build_graph_from_lake_graph(
        lake_graph_csv=lake_graph_path,
        lake_ids=lake_ids_out,
    )

    # ── Random spatial fold assignment ────────────────────────────────────────
    # Shuffle lake array positions (not IDs) with the fixed seed, then chunk
    # into n_folds groups.  fold_idx picks the test chunk; all others are train.
    rng = np.random.default_rng(spatial_split_seed) # deterministic shuffling of lake positions
    shuffled_positions = rng.permutation(n_lakes_total)   # positions in lake_ids_out

    # np.array_split produces n_folds chunks of as-equal-as-possible size
    fold_chunks = np.array_split(shuffled_positions, n_folds) # list of n_folds arrays of lake positions
    test_positions  = fold_chunks[fold_idx]               # dim: (n_test_lakes,) positions of test lakes in the original arrays
    train_positions = np.concatenate(
        [fold_chunks[i] for i in range(n_folds) if i != fold_idx]
    ) # remaining positions → train (includes val in both val_method modes)

    print(
        f"Spatial CV fold {fold_idx + 1}/{n_folds}: "
        f"{len(train_positions)} train lakes / {len(test_positions)} test lakes "
        f"(seed={spatial_split_seed})"
    )

    # ── Find all valid init_date positions ────────────────────────────────────
    era5_date_to_idx = {d: i for i, d in enumerate(era5_dates)} 
    # A reverse lookup to find the index of any given dates
    # era5_dates: is a list of all dates in the ERA5 time series
    # {pd.Timestamp('1980-01-01 00:00:00'): 0, pd.Timestamp('1980-01-02 00:00:00'): 1, ...}

    # Loop through each ECMWF initialize date
    # Check if it has a full ERA5 history window before it 
    # Check if it has a SWOT observation on the forecast days
    # all_valid: list of indices of ecmwf_init_dates that meet the criteria to be included in the dataset
    # criteria: 
    # 1) There must be a full seq_len ERA5 history before the init_date (i.e., era5_idx >= seq_len - 1)
    # 2） If require_obs_on_any_forecast_day is True, at least one of the forecast days must have a SWOT observation (obs_mask sum > 0)
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

    # Convert to numpy array for easier indexing later; also get the count of valid init_dates
    all_valid = np.array(all_valid, dtype=np.int64)
    # Get number of valid init_dates that will be used for splitting into train/val/test
    n_valid   = len(all_valid)

    if n_valid == 0:
        raise ValueError(
            "No valid init_dates found. Check that ERA5 and ECMWF date ranges overlap "
            "and that SWOT WSE observations exist within the forecast windows."
        )

    # ── Val strategy: determine init_date indices and per-dataset lake masks ─────
    if val_method == "temporal":
        # All train-fold lakes, last val_frac of dates reserved for val.
        val_start  = int(n_valid * (1.0 - val_frac))
        train_idx  = all_valid[:val_start]
        val_idx    = all_valid[val_start:]
        test_idx   = all_valid

        # Normalization source: all train-fold lakes, all ERA5 dates
        norm_positions = train_positions

        # Spatial masks: train and val share the same lake set
        train_active_positions = train_positions
        val_active_positions   = train_positions

        print(
            f"  Val method: temporal — "
            f"{len(train_idx)} train dates / {len(val_idx)} val dates "
            f"({len(train_positions)} lakes each)"
        )

    elif val_method == "spatial":
        # Train/val/test should use all initial dates 
        train_idx = all_valid
        val_idx   = all_valid
        test_idx  = all_valid

        # Randomly hold out a spatial_val_frac fraction of the train-fold lakes as a spatial val set.
        rng_val = np.random.default_rng(spatial_split_seed + 1)
        # Get the number of lakes to hold out for validation, ensuring at least one lake is in the val set
        n_spatial_val  = max(1, int(len(train_positions) * spatial_val_frac))
        # Permutation of train_positions to randomly select lakes for spatial validation
        perm           = rng_val.permutation(len(train_positions))
        # Get the positions of the lakes for validation and training based on the random permutation
        val_active_positions   = train_positions[perm[:n_spatial_val]] # dim: (n_val_lakes,) positions of val lakes in the original arrays
        train_active_positions = train_positions[perm[n_spatial_val:]] # dim: (n_train_lakes,) positions of train-train lakes in the original arrays

        # Normalization source: train-train lakes only (no val or test lake leakage)
        norm_positions = train_active_positions

        print(
            f"  Val method: spatial"
            f"{len(train_active_positions)} train lakes / "
            f"{len(val_active_positions)} val lakes"
        )

    else:
        raise ValueError(f"val_method must be 'temporal' or 'spatial', got '{val_method}'")

    # ── Feature normalization ─────────────────────────────────────────────────
    # Statistics are derived from `norm_positions` lakes only (no leakage into
    # val or test lakes).  All ERA5 dates are included.
    _LOG1P_DYN  = [5, 10, 15]
    # 5： days_since_last_obs, 10: P, 15: sf — all right-skewed, zero-bounded → log1p + z-score
    _ZSCORE_DYN = [2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    era5_dynamic   = era5_dynamic.copy() # dim： (n_lakes, n_era5_dates, n_features)
    ecmwf_forecast = ecmwf_forecast.copy() # dim: (n_lakes, n_ecmwf_init_dates, forecast_horizon, n_climate_features)

    # Step 1: log1p on skewed features (all lakes, all dates)
    for i in _LOG1P_DYN:
        era5_dynamic[:, :, i] = np.log1p(np.clip(era5_dynamic[:, :, i], 0, None))

    # Step 2: z-score statistics from norm_positions lakes only
    norm_era5_slice = era5_dynamic[norm_positions, :, :]  # (n_norm_lakes, n_dates, 21)

    n_dyn    = era5_dynamic.shape[-1]
    dyn_mean = np.zeros(n_dyn, dtype=np.float32)
    dyn_std  = np.ones(n_dyn,  dtype=np.float32)

    for i in _ZSCORE_DYN:
        vals        = norm_era5_slice[:, :, i].ravel() # dim: (n_norm_lakes * n_dates,) all values for this feature across the norm_positions lakes and all dates
        dyn_mean[i] = float(vals.mean())
        dyn_std[i]  = float(vals.std()) + 1e-8

    # Step 3: apply z-score to all ERA5 dates and all lakes
    for i in _ZSCORE_DYN:
        era5_dynamic[:, :, i] = (era5_dynamic[:, :, i] - dyn_mean[i]) / dyn_std[i]

    # Step 4: apply identical normalization to ECMWF climate features
    ecmwf_log1p_indices  = [k for k, _ in enumerate(ECMWF_CLIMATE_VARS) if (SWOT_DIM + k) in _LOG1P_DYN]
    ecmwf_zscore_indices = [k for k, _ in enumerate(ECMWF_CLIMATE_VARS) if (SWOT_DIM + k) in _ZSCORE_DYN]

    # Apply log1p to ECMWF variables of precipitaiton and snowfall
    for k in ecmwf_log1p_indices:
        ecmwf_forecast[:, :, :, k] = np.log1p(np.clip(ecmwf_forecast[:, :, :, k], 0, None))
    # Apply z-score to ECMWF climate features using the same ERA5-derived mean/std for consistency
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
        "val_method":          val_method,
        "n_folds":             n_folds,
        "fold_idx":            fold_idx,
        "spatial_split_seed":  spatial_split_seed,
        "n_train_lakes":       len(train_active_positions),
        "n_val_lakes":         len(val_active_positions),
        "n_test_lakes":        len(test_positions),
    }

    # ── Construct three datasets sharing the same normalized arrays ───────────
    # All three use the full graph (all lakes as nodes) so message passing is
    # unaffected by the spatial split.
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

    # Build per-dataset spatial masks (1 = active node for loss, 0 = silent node).
    # train_ds and val_ds may differ when val_method="spatial".
    def _make_mask(positions: np.ndarray) -> torch.Tensor:
        # Output: (n_lakes,) binary mask with 1 for positions in the input array, 0 elsewhere
        m = np.zeros(n_lakes_total, dtype=np.float32)
        m[positions] = 1.0
        return torch.from_numpy(m)

    # get the mask for each dataset 
    train_mask_t = _make_mask(train_active_positions)
    val_mask_t   = _make_mask(val_active_positions)
    test_mask_t  = _make_mask(test_positions)

    # Each dataset carries the mask for its own active lakes.
    train_ds.spatial_mask = train_mask_t # dim: (n_lakes,) 1 for train-train lakes, 0 for val/test lakes
    val_ds.spatial_mask   = val_mask_t # dimn: (n_lakes,) 1 for val lakes (either same as train or a subset), 0 for train-train and test lakes
    test_ds.spatial_mask  = test_mask_t

    print(
        f"Spatial CV datasets built: "
        f"{len(train_ds)} init dates (shared across train/val/test) | "
        f"{len(train_active_positions)} train lakes / "
        f"{len(val_active_positions)} val lakes / "
        f"{len(test_positions)} test lakes"
    )

    return train_ds, val_ds, test_ds, norm_stats
