"""
Build river/lake network graphs from GRIT reach data.
Converts to PyTorch Geometric Data format for SWOT-GNN.

Supported graph sources
-----------------------
- build_graph_from_grit          : reach-level graph from raw GRIT CSV
- build_graph_from_lake_graph    : lake-level graph from the output of
                                   build_lake_graph_from_reaches.py
                                   (one row per lake; upstream/downstream IDs
                                   are comma-separated in upstream_lake_ids /
                                   downstream_lake_ids columns)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Union

# Optional PyTorch Geometric dependency for graph neural network support
try:
    import torch
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def build_graph_from_grit(
    grit_reach_path: Union[str, Path],
    downstream_col: str = "downstre_1",
    node_id_col: str = "fid",
    reach_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build directed graph from GRIT reach CSV.

    Args:
        grit_reach_path: Path to GRIT reach CSV (e.g. ba_river_watershed_reaches_gritv06_with_centroid_lake.csv)
        downstream_col: Column with downstream IDs (comma-separated for multiple)
        node_id_col: Column for reach/node ID
        reach_ids: Optional subset of reach IDs to include (e.g. reaches with SWOT data).
                   If None, include all reaches in the CSV.

    Returns:
        edge_index: (2, num_edges) array [source, target] for downstream edges
        node_ids: (num_nodes,) array of reach IDs in graph order
        node_to_idx: dict mapping reach_id -> index
        metadata: dict with num_nodes, num_edges
    """
    # --- Load and prepare reach data ---
    # Load reach metadata from GRIT CSV
    reach_df = pd.read_csv(grit_reach_path)
    # Normalize column names so we always use "fid" internally
    reach_df = reach_df.rename(columns={node_id_col: "fid"}) if node_id_col != "fid" else reach_df

    # Filter to subset of reaches if specified (e.g. only those with SWOT observations)
    if reach_ids is not None:
        reach_df = reach_df[reach_df["fid"].isin(reach_ids)].copy()
        reach_df = reach_df.reset_index(drop=True)

    reach = reach_df.copy()
    # --- Parse downstream connectivity ---
    # GRIT stores downstream IDs as comma-separated strings (e.g. "101,102,103").
    # Split into separate columns (down1, down2, ...) for multiple downstream reaches.
    newcol = reach[downstream_col].astype(str).str.split(",", expand=True)
    col_names = [f"down{i+1}" for i in range(newcol.shape[1])]
    reach[col_names] = newcol
    reach[col_names] = reach[col_names].replace("", np.nan).replace("nan", np.nan)

    # --- Build node index ---
    # Each reach gets a sequential integer index (0, 1, 2, ...) for graph construction.
    # node_to_idx maps reach_id -> index; node_ids holds reach IDs in sorted order.
    valid_node_ids = set(reach["fid"].values)
    node_ids = np.sort(reach["fid"].unique())
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # --- Extract directed edges ---
    # For each reach and each downstream column, create edge (source_idx, target_idx).
    # Only include edges whose target is in our node set (avoids dangling refs to
    # reaches outside the filtered subset). Skip self-loops (src != tgt).
    edges = []
    for col in col_names:
        if col not in reach.columns:
            continue
        pairs = reach[["fid", col]].dropna()
        pairs = pairs.astype(np.int64)
        pairs = pairs[pairs[col].isin(valid_node_ids)]
        for _, row in pairs.iterrows():
            src, tgt = int(row["fid"]), int(row[col])
            if src != tgt:
                edges.append((node_to_idx[src], node_to_idx[tgt]))

    # --- Build edge_index for PyG ---
    # PyG expects edge_index shape (2, num_edges) with row0=sources, row1=targets.
    # Add reverse edges so the graph is undirected: GNN can pass info both upstream
    # and downstream. np.unique removes duplicate edges (e.g. if A->B and B->A).
    if not edges:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        edge_index = np.array(edges, dtype=np.int64).T  # (2, num_edges): [sources; targets]
        rev = np.flip(edge_index, axis=0)  # Swap source<->target for reverse edges
        edge_index = np.concatenate([edge_index, rev], axis=1)
        edge_index = np.unique(edge_index, axis=1)

    metadata = {"num_nodes": len(node_ids), "num_edges": edge_index.shape[1]}
    return edge_index, node_ids, node_to_idx, metadata


def build_graph_from_lake_graph(
    lake_graph_csv: Union[str, Path],
    lake_ids: Optional[np.ndarray] = None,
    lake_id_col: str = "lake_id",
    downstream_col: str = "downstream_lake_ids",
) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Build directed lake connectivity graph from the output of
    build_lake_graph_from_reaches.py.

    Expected CSV columns (one row per lake):
        lake_id              - unique lake ID
        lon, lat             - centroid coordinates
        most_downstream_fid  - fid of the most-downstream reach
        downstream_river_fid - first non-lake reach(es) downstream
        upstream_lake_ids    - comma-separated upstream lake IDs (empty if none)
        downstream_lake_ids  - comma-separated downstream lake IDs (-1 = basin outlet)

    Terminal nodes (lake_id == -1 or appearing as a downstream target with value -1)
    are excluded. Both forward and reverse edges are added so the GNN can propagate
    information in both directions.

    Args:
        lake_graph_csv:  Path to gritv06_*_pld_lake_graph_*.csv
        lake_ids:        Optional subset of lake IDs to include. If None, use all.
        lake_id_col:     Column name for the lake node ID (default: "lake_id").
        downstream_col:  Column with comma-separated downstream lake IDs
                         (default: "downstream_lake_ids").

    Returns:
        edge_index:   (2, num_edges) — directed + reversed edges, deduplicated
        node_ids:     (num_nodes,)   — lake IDs in graph order
        node_to_idx:  dict mapping lake_id → node index
        metadata:     dict with num_nodes, num_edges
    """
    df = pd.read_csv(lake_graph_csv)

    # Validate required columns
    for col in (lake_id_col, downstream_col):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in lake graph CSV. "
                f"Available columns: {list(df.columns)}"
            )

    # Parse lake_id, drop terminal node rows (-1)
    df[lake_id_col] = pd.to_numeric(df[lake_id_col], errors="coerce")
    df = df.dropna(subset=[lake_id_col]).copy()
    df[lake_id_col] = df[lake_id_col].astype(np.int64)
    df = df[df[lake_id_col] != -1].reset_index(drop=True)

    # Apply optional lake_ids filter
    if lake_ids is not None:
        lake_id_set = set(lake_ids.tolist())
        df = df[df[lake_id_col].isin(lake_id_set)].reset_index(drop=True)

    # Build node index from lake rows
    node_ids = np.sort(df[lake_id_col].unique()).astype(np.int64)
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    valid_set = set(node_ids.tolist())

    def _parse_ids(value) -> List[int]:
        """Parse a comma-separated cell of lake IDs into a list of ints."""
        if pd.isna(value) or str(value).strip() == "":
            return []
        return [int(x.strip()) for x in str(value).split(",") if x.strip()]

    # Extract directed downstream edges from each lake row
    edges = []
    for _, row in df.iterrows():
        src = int(row[lake_id_col])
        if src not in node_to_idx:
            continue
        src_idx = node_to_idx[src]
        for tgt in _parse_ids(row[downstream_col]):
            if tgt == -1 or tgt not in valid_set or tgt == src:
                continue
            edges.append((src_idx, node_to_idx[tgt]))

    # Build edge_index with reverse edges for undirected GNN
    if not edges:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        edge_index = np.array(edges, dtype=np.int64).T  # (2, num_directed_edges)
        rev = np.flip(edge_index, axis=0)
        edge_index = np.concatenate([edge_index, rev], axis=1)
        edge_index = np.unique(edge_index, axis=1)

    metadata = {"num_nodes": len(node_ids), "num_edges": edge_index.shape[1]}
    return edge_index, node_ids, node_to_idx, metadata


def grit_to_pyg_data(
    grit_reach_path: Union[str, Path],
    reach_ids: Optional[np.ndarray] = None,
    node_features: Optional[np.ndarray] = None,
    pos_embedding: Optional[np.ndarray] = None,
) -> "Data":
    """
    Convert GRIT river network to PyTorch Geometric Data object.

    Args:
        grit_reach_path: Path to GRIT reach CSV
        reach_ids: Optional subset of reach IDs
        node_features: Optional (num_nodes, feat_dim) tensor for initial node features
        pos_embedding: Optional (num_nodes, pos_dim) positional encoding (e.g. RWSE, Laplacian)

    Returns:
        PyG Data with edge_index, x (if provided), pos (if provided), num_nodes
    """
    if not HAS_PYG:
        raise ImportError("PyTorch Geometric is required. Install with: pip install torch-geometric")

    # Build the river network graph structure (reach-level connectivity)
    edge_index, node_ids, node_to_idx, _ = build_graph_from_grit(grit_reach_path, reach_ids=reach_ids)

    # --- Assemble PyG Data object ---
    # edge_index: COO format (2, num_edges), required for message passing.
    # num_nodes: number of nodes (reaches).
    # node_ids: original reach IDs for mapping predictions back to reaches.
    data_dict = {
        "edge_index": torch.from_numpy(edge_index).long(),
        "num_nodes": len(node_ids),
        "node_ids": torch.from_numpy(node_ids).long(),
    }

    # Optional node features: e.g. reach attributes, hydrological stats, or precomputed
    # embeddings. Must have shape (num_nodes, feat_dim) and align with node_ids order.
    if node_features is not None:
        data_dict["x"] = torch.as_tensor(node_features, dtype=torch.float32)
    # Optional positional embedding: e.g. random walk structural encoding (RWSE) or
    # Laplacian eigenvectors for capturing graph structure. Shape (num_nodes, pos_dim).
    if pos_embedding is not None:
        data_dict["pos"] = torch.as_tensor(pos_embedding, dtype=torch.float32)

    return Data(**data_dict)
