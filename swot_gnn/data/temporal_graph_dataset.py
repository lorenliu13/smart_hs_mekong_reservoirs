"""
Temporal graph dataset for SWOT-GNN.
Produces 21-day sequences of state-of-river graphs. Loss is computed at all observed nodes
(obs_mask=1). Supports train/val/test split by time.
"""
# Numerical arrays and DataFrames
import numpy as np
import pandas as pd
# Path handling (str or Path)
from pathlib import Path
# Type hints for function signatures
from typing import Optional, Union, Tuple, List
# PyTorch tensors and Dataset base class
import torch
from torch.utils.data import Dataset

# PyTorch Geometric: Data class for graph (node features + edges). Optional dependency.
try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

# Build river network graph from GRIT CSV (reach-level or segment-level)
from .graph_builder import (
    build_graph_from_grit,           # Reach-level graph from GRIT reaches
    build_graph_from_segment_darea,  # Segment graph from darea downstream links
)
# Load features from NetCDF datacubes (reach or segment coords)
from .feature_assembler import (
    assemble_node_features_from_datacubes,
    assemble_node_features_from_datacubes_segment_based,
    WSE_DYNAMIC_INDICES,  # indices of WSE features zeroed in forecast step
)


class TemporalGraphDataset(Dataset):
    """
    Dataset for 1-day-ahead WSE forecasting with temporal graph sequences.

    Each sample: given seq_len days of history + 1 forecast-day climate step,
    predict WSE on day seq_len+1 (the day after the history window).

    Returns per sample: (list of seq_len+1 PyG Data objects, static_features, labels, obs_mask).
    - The last Data object in the list (step seq_len+1) has WSE features zeroed;
      only climate and time encoding are filled in (the "perfect forecast" input).
    - static_features: (num_nodes, static_dim) time-invariant attributes, passed
      to the model separately to seed the LSTM initial hidden state.
    - Loss is computed at nodes where obs_mask=1 on the forecast day.
    """

    def __init__(
        self,
        node_features: np.ndarray,
        static_features: np.ndarray,
        edge_index: np.ndarray,
        dates: pd.DatetimeIndex,
        reach_ids: np.ndarray,
        seq_len: int = 30,
        wse_labels: Optional[np.ndarray] = None,
        obs_mask: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
    ):
        """
        Args:
            node_features: (num_nodes, num_dates, n_dyn) -- dynamic features only
            static_features: (num_nodes, n_static) -- time-invariant attributes
            edge_index: (2, num_edges)
            dates: All dates in chronological order
            reach_ids: (num_nodes,) node IDs in graph order
            seq_len: History window length in days (default 30). The forecast step
                     (day seq_len+1) is appended automatically.
            wse_labels: (num_nodes, num_dates) WSE values for target
            obs_mask: (num_nodes, num_dates) 1 where SWOT observed (loss computed here)
            indices: Optional subset of valid start indices for train/val/test split.
        """
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required for TemporalGraphDataset")
        if wse_labels is None or obs_mask is None:
            raise ValueError("wse_labels and obs_mask are required")

        self.node_features = node_features    # (n_nodes, n_date, n_dyn)
        self.static_features = static_features.astype(np.float32)  # (n_nodes, n_static)
        self.edge_index = torch.from_numpy(edge_index).long()
        self.dates = dates
        self.reach_ids = reach_ids
        self.seq_len = seq_len
        self.wse_labels = wse_labels.astype(np.float32)
        self.label_avail = obs_mask.astype(np.float32)

        self.n_reach, self.n_date, self.feat_dim = node_features.shape

        # Need start_idx + seq_len (forecast day) to be a valid index, so
        # start_idx <= n_date - seq_len - 1 → range: arange(n_date - seq_len)
        all_valid = np.arange(self.n_date - seq_len)
        self.valid_starts = indices if indices is not None else all_valid

    def __len__(self) -> int:
        return len(self.valid_starts)

    def _get_labels_and_mask(self, start_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Labels and mask for the forecast day (start_idx + seq_len), the day after
        the history window ends.
        """
        target_idx = start_idx + self.seq_len
        labels = self.wse_labels[:, target_idx].copy()
        labels = np.nan_to_num(labels, nan=0.0)
        label_mask = self.label_avail[:, target_idx].copy().astype(np.float32)
        return labels, label_mask

    def __getitem__(self, idx: int) -> Tuple[List[Data], torch.Tensor, torch.Tensor, torch.Tensor]:
        start_idx = self.valid_starts[idx]

        # History window: days [start_idx, start_idx + seq_len)
        seq_features = self.node_features[:, start_idx:start_idx + self.seq_len, :]  # (n_nodes, seq_len, n_dyn)

        # Forecast step (day start_idx + seq_len): climate + time only, WSE features zeroed.
        # Take the full row from the datacube then zero out WSE-related indices so
        # the model never sees the actual WSE for the target day.
        fc_step = self.node_features[:, start_idx + self.seq_len, :].copy()  # (n_nodes, n_dyn)
        fc_step[:, WSE_DYNAMIC_INDICES] = 0.0  # zero obs_mask, latest_wse, days_since_last_obs

        # Concatenate: (n_nodes, seq_len + 1, n_dyn)
        seq_features = np.concatenate([seq_features, fc_step[:, np.newaxis, :]], axis=1)

        labels, label_mask = self._get_labels_and_mask(start_idx)

        # Build one PyG Data object per timestep (seq_len + 1 total)
        data_list = []
        for t in range(self.seq_len + 1):
            x = torch.from_numpy(seq_features[:, t, :]).float()
            data = Data(x=x, edge_index=self.edge_index, num_nodes=self.n_reach)
            data_list.append(data)

        static = torch.from_numpy(self.static_features)  # (n_nodes, n_static)

        # Return (graph sequence, static features, labels, mask)
        return (
            data_list,
            static,
            torch.from_numpy(labels).float(),
            torch.from_numpy(label_mask).float(),
        )


def collate_temporal_graph_batch(
    batch: List[Tuple[List[Data], torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[List[List[Data]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate batch of temporal graph samples.
    Each sample has same graph structure (same basin), so we stack tensors directly.
    Returns (data_lists, static_feats, labels, masks).
    """
    data_lists = [b[0] for b in batch]
    # static_feats: all samples share the same static (same basin), stack for consistency
    static_feats = torch.stack([b[1] for b in batch])   # (batch_size, n_nodes, n_static)
    labels = torch.stack([b[2] for b in batch])          # (batch_size, n_nodes)
    masks = torch.stack([b[3] for b in batch])           # (batch_size, n_nodes)
    return data_lists, static_feats, labels, masks


def save_dataset_cache(
    cache_path: Union[str, Path],
    train_ds: "TemporalGraphDataset",
    val_ds: "TemporalGraphDataset",
    test_ds: "TemporalGraphDataset",
    norm_stats: dict,
) -> None:
    """
    Save processed dataset arrays to a .npz file for fast reloading.

    Caches the normalized dynamic/static arrays, labels, masks, graph, and
    split indices. On reload, skips NetCDF reading and feature normalization
    entirely. All three datasets share the same underlying arrays.

    Args:
        cache_path: Path to output .npz file (e.g. 'dataset_cache.npz').
        train_ds, val_ds, test_ds: Datasets from build_temporal_dataset_from_datacubes_segment_based.
        norm_stats: Normalization stats dict returned by the same function.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        # Shared arrays (all splits reference the same underlying data)
        dynamic_features=train_ds.node_features,          # (n_seg, n_date, n_dyn) float32 — normalized
        static_features=train_ds.static_features,         # (n_seg, n_static) float32 — normalized
        wse_labels=train_ds.wse_labels,                   # (n_seg, n_date) float32
        obs_mask=train_ds.label_avail,                    # (n_seg, n_date) float32
        edge_index=train_ds.edge_index.numpy(),           # (2, E) int64
        segment_ids=train_ds.reach_ids,                   # (n_seg,)
        dates=train_ds.dates.asi8,                        # int64 nanoseconds; reload as datetime64[ns]
        seq_len=np.array([train_ds.seq_len], dtype=np.int64),
        # Split indices
        train_idx=train_ds.valid_starts,
        val_idx=val_ds.valid_starts,
        test_idx=test_ds.valid_starts,
        # Normalization stats (for inference / diagnostics)
        norm_log1p_dynamic_indices=np.array(norm_stats["log1p_dynamic_indices"], dtype=np.int64),
        norm_zscore_dynamic_indices=np.array(norm_stats["zscore_dynamic_indices"], dtype=np.int64),
        norm_dynamic_mean=norm_stats["dynamic_mean"],
        norm_dynamic_std=norm_stats["dynamic_std"],
        norm_static_mean=norm_stats["static_mean"],
        norm_static_std=norm_stats["static_std"],
    )
    size_mb = cache_path.stat().st_size / 1e6
    print(f"Dataset cache saved → {cache_path} ({size_mb:.1f} MB)")


def load_dataset_cache(
    cache_path: Union[str, Path],
    seq_len: Optional[int] = None,
) -> Tuple["TemporalGraphDataset", "TemporalGraphDataset", "TemporalGraphDataset", dict]:
    """
    Load processed dataset from a .npz cache produced by save_dataset_cache.

    Skips NetCDF loading and normalization; reconstructs the three
    TemporalGraphDataset splits directly from stored arrays.

    Args:
        cache_path: Path to .npz cache file.
        seq_len: Override the stored seq_len (must match what was used to build the cache).

    Returns:
        (train_dataset, val_dataset, test_dataset, norm_stats)
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Dataset cache not found: {cache_path}")

    data = np.load(cache_path, allow_pickle=False)

    if seq_len is None:
        seq_len = int(data["seq_len"][0])

    dates = pd.DatetimeIndex(data["dates"].astype("datetime64[ns]"))

    shared_kwargs = dict(
        node_features=data["dynamic_features"],
        static_features=data["static_features"],
        edge_index=data["edge_index"],
        dates=dates,
        reach_ids=data["segment_ids"],
        seq_len=seq_len,
        wse_labels=data["wse_labels"],
        obs_mask=data["obs_mask"],
    )

    train_ds = TemporalGraphDataset(**shared_kwargs, indices=data["train_idx"])
    val_ds   = TemporalGraphDataset(**shared_kwargs, indices=data["val_idx"])
    test_ds  = TemporalGraphDataset(**shared_kwargs, indices=data["test_idx"])

    norm_stats = {
        "log1p_dynamic_indices":  data["norm_log1p_dynamic_indices"].tolist(),
        "zscore_dynamic_indices": data["norm_zscore_dynamic_indices"].tolist(),
        "dynamic_mean": data["norm_dynamic_mean"],
        "dynamic_std":  data["norm_dynamic_std"],
        "static_mean":  data["norm_static_mean"],
        "static_std":   data["norm_static_std"],
    }

    print(
        f"Dataset loaded from cache: "
        f"{len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test samples"
    )
    return train_ds, val_ds, test_ds, norm_stats


def build_temporal_dataset_from_datacubes(
    dynamic_datacube_path: Union[str, Path],
    static_datacube_path: Union[str, Path],
    grit_reach_path: Union[str, Path],
    reach_ids: Optional[np.ndarray] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    seq_len: int = 21,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    require_obs_on_last_day: bool = True,
) -> Tuple[TemporalGraphDataset, TemporalGraphDataset, TemporalGraphDataset]:
    """
    Build TemporalGraphDataset from dynamic and static NetCDF datacubes.
    Returns train, val, test splits (chronological by time).

    Args:
        dynamic_datacube_path: Path to ba_river_swot_dynamic_datacube_*.nc
        static_datacube_path: Path to ba_river_swot_static_datacube_*.nc
        grit_reach_path: Path to GRIT reach CSV for graph construction
        reach_ids: Optional subset of reach IDs. If None, use all from datacube.
        start_date: Optional start date (YYYY-MM-DD). If None, use full datacube range.
        end_date: Optional end date. If None, use full datacube range.
        seq_len: Temporal sequence length (default 21)
        train_frac: Fraction for train split (default 0.7)
        val_frac: Fraction for validation split (default 0.15)
        test_frac: Fraction for test split (default 0.15)
        require_obs_on_last_day: If True, keep only sequences whose last day has at least
            one valid WSE observation (obs_mask=1). Shrinks dataset to SWOT-available dates.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Ensure path object for file ops
    grit_reach_path = Path(grit_reach_path)
    # Load GRIT reach table (fid = reach ID)
    grit_df = pd.read_csv(grit_reach_path)
    # All unique reach IDs, sorted
    grit_reach_ids = np.sort(grit_df["fid"].unique())

    # Optionally restrict date range; None = use full datacube range
    dates = None
    if start_date and end_date:
        dates = pd.date_range(start_date, end_date, freq="D")

    # Load features from NetCDF: node_features, wse, obs_mask; reach_ids/dates actually used
    node_features, wse_labels, obs_mask, reach_ids_out, dates_out = (
        assemble_node_features_from_datacubes(
            dynamic_datacube_path=dynamic_datacube_path,
            static_datacube_path=static_datacube_path,
            reach_ids=reach_ids if reach_ids is not None else grit_reach_ids,
            dates=dates,
        )
    )

    # Build directed graph: edge_index (2, E), node_ids, node_to_idx, metadata
    edge_index, _, _, _ = build_graph_from_grit(grit_reach_path, reach_ids=reach_ids_out)

    # Valid sequence start indices: need seq_len consecutive days
    all_valid = np.arange(len(dates_out) - seq_len + 1)
    if require_obs_on_last_day:
        # Keep only sequences whose last day has at least one WSE observation
        all_valid = np.array([
            i for i in all_valid
            if obs_mask[:, i + seq_len - 1].sum() > 0
        ])
    n_valid = len(all_valid)

    # Chronological split: earlier dates train, mid val, later test (avoids data leakage)
    train_end = int(n_valid * train_frac)   # Exclusive end of train indices
    val_end = int(n_valid * (train_frac + val_frac))  # Exclusive end of val indices
    train_idx = all_valid[:train_end]
    val_idx = all_valid[train_end:val_end]
    test_idx = all_valid[val_end:]

    # Create three datasets sharing same arrays, different index subsets
    train_ds = TemporalGraphDataset(
        node_features=node_features,
        edge_index=edge_index,
        dates=dates_out,
        reach_ids=reach_ids_out,
        seq_len=seq_len,
        wse_labels=wse_labels,
        obs_mask=obs_mask,
        indices=train_idx,
    )
    val_ds = TemporalGraphDataset(
        node_features=node_features,
        edge_index=edge_index,
        dates=dates_out,
        reach_ids=reach_ids_out,
        seq_len=seq_len,
        wse_labels=wse_labels,
        obs_mask=obs_mask,
        indices=val_idx,
    )
    test_ds = TemporalGraphDataset(
        node_features=node_features,
        edge_index=edge_index,
        dates=dates_out,
        reach_ids=reach_ids_out,
        seq_len=seq_len,
        wse_labels=wse_labels,
        obs_mask=obs_mask,
        indices=test_idx,
    )
    # Return train, validation, and test datasets
    return train_ds, val_ds, test_ds


def build_temporal_dataset_from_datacubes_segment_based(
    dynamic_datacube_path: Union[str, Path],
    static_datacube_path: Union[str, Path],
    segment_darea_path: Union[str, Path],
    segment_ids: Optional[np.ndarray] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    seq_len: int = 30,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    require_obs_on_forecast_day: bool = True,
) -> Tuple[TemporalGraphDataset, TemporalGraphDataset, TemporalGraphDataset, dict]:
    """
    Build TemporalGraphDataset for 1-day-ahead WSE forecasting from segment-based datacubes.
    Returns train, val, test splits (chronological by time) plus normalization stats.

    Each sample uses seq_len days of history plus 1 forecast-step (climate only,
    WSE zeroed), and predicts WSE on the day after the history window ends.

    Feature normalization applied here (training-set statistics only, no leakage):
      - log1p then z-score: days_since_last_obs (idx 2), P (idx 6) — right-skewed
      - z-score only: LWd (5), Pres (7), RelHum (8), SWd (9), Temp (10), Wind (11)
      - unchanged: obs_mask (0, binary), latest_wse (1, already per-reach z-scored),
                   time_doy_sin/cos (3/4, already in [-1, 1])
      - static features: z-score (all 33 features)

    Args:
        dynamic_datacube_path: Path to segment-based dynamic datacube
        static_datacube_path: Path to segment-based static datacube
        segment_darea_path: Path to segment darea CSV/shapefile with downstream links
        segment_ids: Optional subset of segment IDs. If None, use all from datacube.
        start_date, end_date: Optional date range strings (YYYY-MM-DD)
        seq_len: History window length in days (default 30). Forecast step appended automatically.
        train_frac, val_frac, test_frac: Chronological split fractions
        require_obs_on_forecast_day: If True, only keep sequences where the forecast
            day (start_idx + seq_len) has at least one SWOT observation (obs_mask=1).

    Returns:
        (train_dataset, val_dataset, test_dataset, norm_stats)
        norm_stats keys: log1p_dynamic_indices, zscore_dynamic_indices,
                         dynamic_mean, dynamic_std, static_mean, static_std
    """
    dates = None
    if start_date and end_date:
        dates = pd.date_range(start_date, end_date, freq="D")

    # Load segment-based datacubes; static returned separately (not concatenated)
    # dynamic_features: (num_segments, num_dates, n_dyn)
    # static_features:  (num_segments, n_static)
    dynamic_features, static_features, wse_labels, obs_mask, segment_ids_out, dates_out = (
        assemble_node_features_from_datacubes_segment_based(
            dynamic_datacube_path=dynamic_datacube_path,
            static_datacube_path=static_datacube_path,
            segment_ids=segment_ids,
            dates=dates,
        )
    )

    edge_index, _, _ = build_graph_from_segment_darea(
        segment_darea_path, segment_ids=segment_ids_out
    )

    # Valid starts: need start_idx + seq_len (forecast day) to exist in dates_out
    # so start_idx <= n_date - seq_len - 1 → range: arange(n_date - seq_len)
    n_date = len(dates_out)
    all_valid = np.arange(n_date - seq_len)
    if require_obs_on_forecast_day:
        # Keep only sequences where the forecast target day has SWOT observations
        all_valid = np.array([
            i for i in all_valid
            if obs_mask[:, i + seq_len].sum() > 0
        ])
    n_valid = len(all_valid)

    train_end = int(n_valid * train_frac)
    val_end = int(n_valid * (train_frac + val_frac))
    train_idx = all_valid[:train_end]
    val_idx = all_valid[train_end:val_end]
    test_idx = all_valid[val_end:]

    # ── Feature normalization (training-set statistics only) ──────────────────
    # Dynamic feature indices (matches DYNAMIC_FEATURE_VARS in feature_assembler.py):
    #   0=obs_mask  1=latest_wse  2=days_since_last_obs  3=doy_sin  4=doy_cos
    #   5=LWd  6=P  7=Pres  8=RelHum  9=SWd  10=Temp  11=Wind
    _LOG1P_DYN = [2, 6]              # log1p first (right-skewed, zero-bounded)
    _ZSCORE_DYN = [2, 5, 6, 7, 8, 9, 10, 11]  # then z-score (includes log1p'd ones)

    # Copy to avoid mutating the assembler's output arrays in-place
    dynamic_features = dynamic_features.copy()

    # Step 1: log1p on skewed features (applied to all dates before computing stats)
    for i in _LOG1P_DYN:
        dynamic_features[:, :, i] = np.log1p(dynamic_features[:, :, i])

    # Step 2: compute mean/std from training time window only.
    # Training sequences span dates [train_idx[0], train_idx[-1] + seq_len] inclusive.
    train_dyn = dynamic_features[:, train_idx[0] : train_idx[-1] + seq_len + 1, :]

    n_dyn = dynamic_features.shape[-1]
    dyn_mean = np.zeros(n_dyn, dtype=np.float32)
    dyn_std  = np.ones(n_dyn,  dtype=np.float32)
    for i in _ZSCORE_DYN:
        vals = train_dyn[:, :, i].ravel()
        dyn_mean[i] = float(vals.mean())
        dyn_std[i]  = float(vals.std()) + 1e-8

    # Step 3: apply z-score to all dates using training stats (no leakage)
    for i in _ZSCORE_DYN:
        dynamic_features[:, :, i] = (dynamic_features[:, :, i] - dyn_mean[i]) / dyn_std[i]

    # Step 4: z-score static features (time-invariant; stats from all nodes)
    stat_mean = static_features.mean(axis=0).astype(np.float32)  # (n_static,)
    stat_std  = static_features.std(axis=0).astype(np.float32) + 1e-8
    static_features = (static_features - stat_mean) / stat_std

    norm_stats = {
        "log1p_dynamic_indices":  _LOG1P_DYN,
        "zscore_dynamic_indices": _ZSCORE_DYN,
        "dynamic_mean": dyn_mean,
        "dynamic_std":  dyn_std,
        "static_mean":  stat_mean,
        "static_std":   stat_std,
    }
    # ─────────────────────────────────────────────────────────────────────────

    shared_kwargs = dict(
        node_features=dynamic_features,
        static_features=static_features,
        edge_index=edge_index,
        dates=dates_out,
        reach_ids=segment_ids_out,
        seq_len=seq_len,
        wse_labels=wse_labels,
        obs_mask=obs_mask,
    )
    train_ds = TemporalGraphDataset(**shared_kwargs, indices=train_idx)
    val_ds = TemporalGraphDataset(**shared_kwargs, indices=val_idx)
    test_ds = TemporalGraphDataset(**shared_kwargs, indices=test_idx)
    return train_ds, val_ds, test_ds, norm_stats
