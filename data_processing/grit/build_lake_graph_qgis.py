"""
Build a QGIS-compatible GeoPackage from the lake-level connectivity graph.

Reads
-----
1. gritv06_great_mekong_pld_lake_graph_{threshold}sqkm.csv
     lake_id, most_downstream_fid, downstream_river_fid,
     upstream_lake_ids, downstream_lake_ids
     (negative lake_ids are terminal river nodes — -1, -2, … one per basin outlet)
2. swot_prior_lake_database_great_mekong_overlap_with_grit.csv
     lake_id, lon, lat, names, poly_area, ref_area
3. gritv06_reaches_great_mekong_with_lake_id.csv
     reach_id — used to verify terminal reach IDs (no geometry; terminal nodes get null geometry)

Output
------
gritv06_great_mekong_pld_lake_graph_{threshold}sqkm.gpkg  (GeoPackage, EPSG:4326)
  Layer: lake_nodes  – one point per lake (lon/lat from PLD) + all graph attrs.
                       Terminal nodes (lake_id < 0, name = "BASIN_OUTLET_-1" …) have
                       null geometry (reaches CSV has no coordinates).
  Layer: lake_edges  – one directed line per edge (upstream → downstream lake),
                       including edges from lakes that drain to the terminal node.
                       attrs: from_lake_id, to_lake_id, from_name, to_name
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
AREA_THRESHOLD_SQKM = 0.1  # Must match the value used in build_lake_graph_from_reaches.py
OBS_COUNT_THRESHOLD = 30   # Must match the value used in build_lake_graph_from_reaches.py
# Terminal node IDs are all negative integers (-1, -2, …); real lake IDs are positive.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SUFFIX = rf"area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}"
_RESERVOIRS_DIR = (
    rf"E:\Project_2025_2026\Smart_hs\raw_data\grit"
    rf"\GRIT_mekong_mega_reservoirs\reservoirs\lake_graph_{_SUFFIX}"
)
GRAPH_CSV = rf"{_RESERVOIRS_DIR}\gritv06_great_mekong_pld_lake_graph_{_SUFFIX}.csv"
PLD_SHP = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit"
    r"\GRIT_mekong_mega_reservoirs\prior_lake_database"
    r"\swot_prior_lake_database_great_mekong_overlap_with_grit.gpkg"
)
REACHES_SHP = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit"
    r"\GRIT_mekong_mega_reservoirs\reaches"
    r"\gritv06_reaches_great_mekong_with_lake_id.gpkg"
)
OUTPUT_GPKG = rf"{_RESERVOIRS_DIR}\gritv06_great_mekong_pld_lake_graph_{_SUFFIX}.gpkg"

# ---------------------------------------------------------------------------
# Load graph CSV
# ---------------------------------------------------------------------------
graph = pd.read_csv(GRAPH_CSV)

# Normalise lake_id to int64 (may be stored as float string in the CSV)
graph["lake_id"] = pd.to_numeric(graph["lake_id"], errors="coerce").astype("int64")

# Fill NaN connectivity cells with empty string so string ops are safe
for col in ("upstream_lake_ids", "downstream_lake_ids", "downstream_river_fid"):
    graph[col] = graph[col].fillna("").astype(str)

print(f"Graph rows: {len(graph)}  (includes terminal node if present)")

# ---------------------------------------------------------------------------
# Resolve terminal nodes: all graph rows with lake_id < 0.
# Each has a single most_downstream_fid (one outlet reach per terminal node).
# ---------------------------------------------------------------------------
terminal_rows = graph[graph["lake_id"] < 0]

# node_id → outlet reach_id
terminal_node_to_reach: dict[int, int] = {
    int(row["lake_id"]): int(float(row["most_downstream_fid"]))
    for _, row in terminal_rows.iterrows()
    if str(row["most_downstream_fid"]).strip()
}
terminal_node_ids: set[int] = set(terminal_node_to_reach.keys())
print(f"Terminal nodes: {sorted(terminal_node_ids)}  "
      f"(outlet reach_ids: {terminal_node_to_reach})")

# ---------------------------------------------------------------------------
# Verify terminal node reach IDs from GRIT reaches CSV.
# The CSV has no geometry/coordinates; terminal nodes will have null geometry.
# ---------------------------------------------------------------------------
terminal_node_to_centroid: dict[int, Point] = {}

if terminal_node_to_reach:
    print("Loading GRIT reaches CSV to verify terminal node reach IDs …")
    reaches_df = pd.read_csv(REACHES_SHP)
    reaches_df["reach_id"] = pd.to_numeric(reaches_df["reach_id"], errors="coerce").astype("int64")

    for node_id, reach_id in terminal_node_to_reach.items():
        match = reaches_df[reaches_df["reach_id"] == reach_id]
        if match.empty:
            print(f"  WARNING: reach_id={reach_id} not found for terminal node {node_id}.")
        else:
            print(f"  Terminal node {node_id}: reach_id={reach_id} found (no geometry in CSV — null geometry).")

# ---------------------------------------------------------------------------
# Load PLD shapefile (polygons)
# ---------------------------------------------------------------------------
pld = gpd.read_file(PLD_SHP)
pld["lake_id"] = pd.to_numeric(pld["lake_id"], errors="coerce").astype("int64")

# Each physical lake may appear as multiple sub-polygons in PLD.
# Dissolve them into a single multi-polygon per lake_id so we get one centroid.
pld_dissolved = (
    pld[["lake_id", "names", "poly_area", "ref_area", "lon", "lat", "geometry"]]
    .dissolve(by="lake_id", aggfunc={"names": "first", "poly_area": "sum", "ref_area": "first", "lon": "first", "lat": "first"})
    .reset_index()
)
# Project to metric CRS for accurate centroid computation, then convert back
pld_dissolved["centroid"] = (
    pld_dissolved.geometry.to_crs(epsg=3857).centroid.to_crs(epsg=4326)
)

print(f"PLD unique lake_ids after dissolve: {len(pld_dissolved)}")

# ---------------------------------------------------------------------------
# Build nodes GeoDataFrame
# ---------------------------------------------------------------------------
# Merge graph data with dissolved PLD geometries.
# Terminal nodes (lake_id < 0) have no PLD entry → centroid will be null
# until we inject the reach-derived geometry below.
nodes_df = graph.merge(
    pld_dissolved[["lake_id", "names", "poly_area", "ref_area", "lon", "lat", "centroid"]],
    on="lake_id",
    how="left",
)

# Assign "BASIN_OUTLET_<id>" as the name for each terminal node
for node_id in terminal_node_ids:
    nodes_df.loc[nodes_df["lake_id"] == node_id, "names"] = f"BASIN_OUTLET_{node_id}"

nodes_gdf = gpd.GeoDataFrame(
    nodes_df.drop(columns=["centroid"]),
    geometry=nodes_df["centroid"],
    crs="EPSG:4326",
)

# Inject geometry and lon/lat for each terminal node from its outlet reach
for node_id, centroid_pt in terminal_node_to_centroid.items():
    mask = nodes_gdf["lake_id"] == node_id
    nodes_gdf.loc[mask, "geometry"] = centroid_pt
    nodes_gdf.loc[mask, "lon"] = centroid_pt.x
    nodes_gdf.loc[mask, "lat"] = centroid_pt.y

missing_geom = nodes_gdf.geometry.isna().sum()
if missing_geom:
    print(f"WARNING: {missing_geom} node(s) still have null geometry after all lookups.")

print(f"Nodes layer: {len(nodes_gdf)} features  ({nodes_gdf.geometry.notna().sum()} with geometry)")

# ---------------------------------------------------------------------------
# Build a quick centroid lookup: lake_id → (x, y)
# ---------------------------------------------------------------------------
centroid_lookup: dict[int, tuple[float, float]] = {}

# From PLD dissolved centroids
for _, row in pld_dissolved.iterrows():
    c = row["centroid"]
    if c is not None and not c.is_empty:
        centroid_lookup[int(row["lake_id"])] = (c.x, c.y)

# Fall back to lon/lat columns from original PLD if centroid is missing
for _, row in pld.iterrows():
    lid = int(row["lake_id"])
    if lid not in centroid_lookup:
        try:
            centroid_lookup[lid] = (float(row["lon"]), float(row["lat"]))
        except (TypeError, ValueError):
            pass

# Terminal nodes — use reach-derived geometry
for node_id, centroid_pt in terminal_node_to_centroid.items():
    centroid_lookup[node_id] = (centroid_pt.x, centroid_pt.y)

# ---------------------------------------------------------------------------
# Build edges GeoDataFrame
# ---------------------------------------------------------------------------
# Expand comma-separated downstream_lake_ids into individual directed edges.
# Edges to/from the terminal node now have geometry thanks to centroid_lookup.
name_lookup = dict(zip(pld_dissolved["lake_id"], pld_dissolved["names"]))
for node_id in terminal_node_ids:
    name_lookup[node_id] = f"BASIN_OUTLET_{node_id}"

edge_records = []

for _, row in graph.iterrows():
    src_id = int(row["lake_id"])
    src_xy = centroid_lookup.get(src_id)

    for tgt_str in str(row["downstream_lake_ids"]).split(","):
        tgt_str = tgt_str.strip()
        if not tgt_str:
            continue
        tgt_id = int(float(tgt_str))
        tgt_xy = centroid_lookup.get(tgt_id)
        geom = LineString([src_xy, tgt_xy]) if (src_xy and tgt_xy) else None
        edge_records.append(
            {
                "from_lake_id": src_id,
                "to_lake_id": tgt_id,
                "geometry": geom,
            }
        )

edges_gdf = gpd.GeoDataFrame(edge_records, geometry="geometry", crs="EPSG:4326")

# Drop edges where geometry could not be built (endpoint missing from all lookups)
valid_mask = edges_gdf["geometry"].notna()
dropped = (~valid_mask).sum()
if dropped:
    print(f"WARNING: Dropped {dropped} edge(s) with missing endpoint geometry.")
edges_gdf = edges_gdf[valid_mask].reset_index(drop=True)

edges_gdf["from_name"] = edges_gdf["from_lake_id"].map(name_lookup)
edges_gdf["to_name"] = edges_gdf["to_lake_id"].map(name_lookup)

print(f"Edges layer: {len(edges_gdf)} features")
terminal_edges = edges_gdf[
    (edges_gdf["from_lake_id"] < 0) |
    (edges_gdf["to_lake_id"] < 0)
]
print(f"  Edges connected to terminal node(s): {len(terminal_edges)}")

# ---------------------------------------------------------------------------
# Write GeoPackage
# ---------------------------------------------------------------------------
nodes_gdf.to_file(OUTPUT_GPKG, layer="lake_nodes", driver="GPKG")
edges_gdf.to_file(OUTPUT_GPKG, layer="lake_edges", driver="GPKG")

print(f"\nSaved GeoPackage: {OUTPUT_GPKG}")
print("  Layers: lake_nodes, lake_edges")
print("\nTo load in QGIS:")
print("  Layer > Add Layer > Add Vector Layer, select the .gpkg file.")
print("  Both layers will appear in the layer browser.")
