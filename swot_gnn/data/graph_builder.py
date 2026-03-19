"""
Build river network graph from GRIT reach data.
Converts to PyTorch Geometric Data format for SWOT-GNN.
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


def build_graph_from_segment_darea(
    segment_darea_path: Union[str, Path],
    segment_ids: Optional[np.ndarray] = None,
    downstream_col: str = "downstre_1",
    node_id_col: str = "fid",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build directed graph from GRIT segment darea (shapefile or CSV with segment connectivity).

    Args:
        segment_darea_path: Path to segment darea CSV/shapefile with fid (segment_id), downstre_1
        segment_ids: Optional subset of segment IDs. If None, use all.
        downstream_col: Column with downstream segment IDs
        node_id_col: Column for segment ID

    Returns:
        edge_index, node_ids, node_to_idx
    """
    # --- Load segment darea file ---
    # Supports CSV or vector formats (.shp, .gpkg). Uses pyogrio if available,
    # otherwise geopandas for shapefiles.
    path = Path(segment_darea_path)
    if path.suffix.lower() in (".shp", ".gpkg"):
        try:
            from pyogrio import read_dataframe
            df = read_dataframe(path)
        except ImportError:
            import geopandas as gpd
            df = gpd.read_file(path)
    else:
        df = pd.read_csv(path)

    df = df.rename(columns={node_id_col: "fid"}) if node_id_col != "fid" else df
    if segment_ids is not None:
        df = df[df["fid"].isin(segment_ids)].copy()
    df = df.reset_index(drop=True)

    # --- Parse downstream segment IDs ---
    # Same comma-separated format as GRIT reach downstream column
    newcol = df[downstream_col].astype(str).str.split(",", expand=True)
    col_names = [f"down{i+1}" for i in range(newcol.shape[1])]
    df[col_names] = newcol
    df[col_names] = df[col_names].replace("", np.nan).replace("nan", np.nan)

    # --- Build node index (segments) ---
    node_ids = np.sort(df["fid"].unique())
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    valid = set(node_ids)

    # --- Extract edges from segment-level connectivity ---
    # This file has direct segment-to-segment downstream links (fid = segment_id,
    # downstre_1 = downstream segment IDs). No reach-level mapping needed.
    edges = []
    for _, row in df.iterrows():
        src = row["fid"]
        if src not in node_to_idx:
            continue
        src_idx = node_to_idx[src]
        for col in col_names:
            if col not in row.index:
                continue
            ds_val = row[col]
            if pd.isna(ds_val):
                continue
            try:
                ds_seg = int(float(ds_val))
            except (ValueError, TypeError):
                continue
            if ds_seg in valid and ds_seg != src:
                tgt_idx = node_to_idx[ds_seg]
                edges.append((src_idx, tgt_idx))

    # Add reverse edges for undirected GNN
    if not edges:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        edge_index = np.array(edges, dtype=np.int64).T
        rev = np.flip(edge_index, axis=0)
        edge_index = np.concatenate([edge_index, rev], axis=1)
        edge_index = np.unique(edge_index, axis=1)

    return edge_index, node_ids, node_to_idx


def build_graph_from_lake_graph(
    lake_graph_csv: Union[str, Path],
    lake_ids: Optional[np.ndarray] = None,
    source_col: str = "lake_id",
    downstream_col: str = "downstream_lake_id",
) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Build directed lake connectivity graph from the GRIT PLD lake graph CSV.

    The lake graph CSV has one row per directed edge (source_lake → downstream_lake).
    Terminal nodes (-1) are excluded. Both reverse edges are added so the GNN
    can propagate information in both directions.

    Args:
        lake_graph_csv:   Path to gritv06_pld_lake_graph_0sqkm.csv
        lake_ids:         Optional subset of lake IDs. If None, use all non-(-1) lakes.
        source_col:       Column name for the source lake ID.
        downstream_col:   Column name for the downstream lake ID.

    Returns:
        edge_index:   (2, num_edges) — directed + reversed edges, deduplicated
        node_ids:     (num_nodes,)   — lake IDs in graph order
        node_to_idx:  dict mapping lake_id → node index
        metadata:     dict with num_nodes, num_edges
    """
    lake_graph = pd.read_csv(lake_graph_csv)

    # Validate column names — give a helpful error listing available columns
    available_cols = list(lake_graph.columns)
    if source_col not in lake_graph.columns:
        raise KeyError(
            f"Source column '{source_col}' not found in lake graph CSV. "
            f"Available columns: {available_cols}"
        )
    if downstream_col not in lake_graph.columns:
        raise KeyError(
            f"Downstream column '{downstream_col}' not found in lake graph CSV. "
            f"Available columns: {available_cols}\n"
            f"Hint: pass the correct column name via the downstream_col parameter."
        )

    # Parse both columns as int64, drop rows with NaN or terminal node -1
    lake_graph[source_col]     = pd.to_numeric(lake_graph[source_col], errors="coerce")
    lake_graph[downstream_col] = pd.to_numeric(lake_graph[downstream_col], errors="coerce")
    lake_graph = lake_graph.dropna(subset=[source_col, downstream_col])
    lake_graph[source_col]     = lake_graph[source_col].astype(np.int64)
    lake_graph[downstream_col] = lake_graph[downstream_col].astype(np.int64)

    # Remove rows involving the terminal node (-1)
    lake_graph = lake_graph[
        (lake_graph[source_col] != -1) & (lake_graph[downstream_col] != -1)
    ]

    # Determine node set
    all_lake_ids = np.sort(
        pd.concat([lake_graph[source_col], lake_graph[downstream_col]]).unique()
    ).astype(np.int64)

    if lake_ids is not None:
        lake_id_set = set(lake_ids.tolist())
        all_lake_ids = np.sort(np.array([lid for lid in all_lake_ids if lid in lake_id_set]))

    node_ids    = all_lake_ids
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    valid_set   = set(node_ids.tolist())

    # Extract edges — both endpoints must be in the node set
    edges = []
    for _, row in lake_graph.iterrows():
        src = int(row[source_col])
        tgt = int(row[downstream_col])
        if src in valid_set and tgt in valid_set and src != tgt:
            edges.append((node_to_idx[src], node_to_idx[tgt]))

    # Build edge_index with reverse edges for undirected GNN
    if not edges:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        edge_index = np.array(edges, dtype=np.int64).T     # (2, num_directed_edges)
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
