"""
Build a QGIS-compatible GeoPackage from the lake-level connectivity graph.

Reads
-----
1. gritv06_pld_lake_graph_{threshold}sqkm.csv
     lake_id, most_downstream_fid, downstream_river_fid,
     upstream_lake_ids, downstream_lake_ids
     (lake_id = -1 is the terminal river node — most-downstream reach, basin outlet)
2. swot_prior_lake_database_mekong_overlap_with_grit.shp
     lake_id, lon, lat, names, poly_area, ref_area, geometry (polygon)
3. gritv06_reaches_mekong_basin_with_pld_lakes.shp
     fid, geometry (LineString) — used to place the terminal node on the map

Output
------
gritv06_pld_lake_graph_{threshold}sqkm.gpkg  (GeoPackage, EPSG:4326)
  Layer: lake_nodes  – one point per lake (polygon centroid) + all graph attrs.
                       Terminal node (lake_id = -1, name = "BASIN_OUTLET") uses
                       the centroid of the terminal river reach geometry.
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
LAKE_AREA_THRESHOLD_SQKM = 1   # Must match the value used in build_lake_graph_from_reaches.py
TERMINAL_NODE_ID = -1

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GRAPH_CSV = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit"
    rf"\GRIT_mekong_mega_reservoirs\reservoirs\gritv06_pld_lake_graph_{LAKE_AREA_THRESHOLD_SQKM}sqkm.csv"
)
PLD_SHP = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit"
    r"\GRIT_mekong_mega_reservoirs\prior_lake_database"
    r"\swot_prior_lake_database_mekong_overlap_with_grit.shp"
)
REACHES_SHP = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit"
    r"\GRIT_mekong_mega_reservoirs\reaches"
    r"\gritv06_reaches_mekong_basin_with_pld_lakes.shp"
)
OUTPUT_GPKG = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit"
    rf"\GRIT_mekong_mega_reservoirs\reservoirs\gritv06_pld_lake_graph_{LAKE_AREA_THRESHOLD_SQKM}sqkm.gpkg"
)

# ---------------------------------------------------------------------------
# Load graph CSV
# ---------------------------------------------------------------------------
graph = pd.read_csv(GRAPH_CSV)

# Normalise lake_id to int64 (may be stored as float in the CSV)
graph["lake_id"] = graph["lake_id"].astype("int64")

# Fill NaN connectivity cells with empty string so string ops are safe
for col in ("upstream_lake_ids", "downstream_lake_ids", "downstream_river_fid"):
    graph[col] = graph[col].fillna("").astype(str)

print(f"Graph rows: {len(graph)}  (includes terminal node if present)")

# ---------------------------------------------------------------------------
# Resolve terminal reach fids from the terminal node row
# ---------------------------------------------------------------------------
terminal_rows = graph[graph["lake_id"] == TERMINAL_NODE_ID]
terminal_reach_fids: list[int] = []
if not terminal_rows.empty:
    for fid_str in str(terminal_rows.iloc[0]["most_downstream_fid"]).split(","):
        fid_str = fid_str.strip()
        if fid_str:
            terminal_reach_fids.append(int(float(fid_str)))

print(f"Terminal node: lake_id={TERMINAL_NODE_ID}, reach fid(s): {terminal_reach_fids}")

# ---------------------------------------------------------------------------
# Get terminal node geometry from GRIT reaches shapefile
# The terminal reach is the most-downstream river reach with no downstream
# neighbour — this is the river mouth / basin outlet.
# We use the centroid of the reach line as the node location.
# ---------------------------------------------------------------------------
terminal_centroid: Point | None = None

if terminal_reach_fids:
    print("Loading GRIT reaches shapefile to resolve terminal node geometry …")
    reaches_gdf = gpd.read_file(REACHES_SHP)
    reaches_gdf["fid"] = reaches_gdf["fid"].astype("int64")
    terminal_reaches = reaches_gdf[reaches_gdf["fid"].isin(terminal_reach_fids)].copy()

    if not terminal_reaches.empty:
        # Compute centroid in projected CRS for accuracy, convert back to WGS84
        merged_geom = terminal_reaches.to_crs(epsg=3857).geometry.union_all()
        centroid_3857 = merged_geom.centroid
        terminal_centroid = (
            gpd.GeoSeries([centroid_3857], crs=3857).to_crs(4326).iloc[0]
        )
        print(f"Terminal node geometry resolved: lon={terminal_centroid.x:.4f}, lat={terminal_centroid.y:.4f}")
    else:
        print("WARNING: Terminal reach fid(s) not found in reaches shapefile — terminal node will have null geometry.")

# ---------------------------------------------------------------------------
# Load PLD shapefile (polygons)
# ---------------------------------------------------------------------------
pld = gpd.read_file(PLD_SHP)
pld["lake_id"] = pld["lake_id"].astype("int64")

# Each physical lake may appear as multiple sub-polygons in PLD.
# Dissolve them into a single multi-polygon per lake_id so we get one centroid.
pld_dissolved = (
    pld[["lake_id", "names", "poly_area", "ref_area", "geometry"]]
    .dissolve(by="lake_id", aggfunc={"names": "first", "poly_area": "sum", "ref_area": "first"})
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
# The terminal node (lake_id = -1) has no PLD entry → centroid will be null
# until we inject the reach-derived geometry below.
nodes_df = graph.merge(
    pld_dissolved[["lake_id", "names", "poly_area", "ref_area", "centroid"]],
    on="lake_id",
    how="left",
)

# Assign "BASIN_OUTLET" as the name for the terminal node
nodes_df.loc[nodes_df["lake_id"] == TERMINAL_NODE_ID, "names"] = "BASIN_OUTLET"

nodes_gdf = gpd.GeoDataFrame(
    nodes_df.drop(columns=["centroid"]),
    geometry=nodes_df["centroid"],
    crs="EPSG:4326",
)

# Inject terminal node geometry derived from the reaches shapefile
if terminal_centroid is not None:
    mask = nodes_gdf["lake_id"] == TERMINAL_NODE_ID
    nodes_gdf.loc[mask, "geometry"] = terminal_centroid

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

# Terminal node — use reach-derived geometry
if terminal_centroid is not None:
    centroid_lookup[TERMINAL_NODE_ID] = (terminal_centroid.x, terminal_centroid.y)

# ---------------------------------------------------------------------------
# Build edges GeoDataFrame
# ---------------------------------------------------------------------------
# Expand comma-separated downstream_lake_ids into individual directed edges.
# Edges to/from the terminal node now have geometry thanks to centroid_lookup.
name_lookup = dict(zip(pld_dissolved["lake_id"], pld_dissolved["names"]))
name_lookup[TERMINAL_NODE_ID] = "BASIN_OUTLET"

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
    (edges_gdf["from_lake_id"] == TERMINAL_NODE_ID) |
    (edges_gdf["to_lake_id"] == TERMINAL_NODE_ID)
]
print(f"  Edges connected to terminal node: {len(terminal_edges)}")

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
