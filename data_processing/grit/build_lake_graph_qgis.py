"""
Build a QGIS-compatible GeoPackage from the lake-level connectivity graph.

Reads
-----
1. gritv06_pld_lake_graph.csv
     lake_id, most_downstream_fid, upstream_lake_ids, downstream_lake_ids
2. swot_prior_lake_database_mekong_overlap_with_grit.shp
     lake_id, lon, lat, names, poly_area, geometry (polygon)

Output
------
gritv06_pld_lake_graph.gpkg  (GeoPackage, EPSG:4326)
  Layer: lake_nodes  – one point per lake (polygon centroid), all graph attrs
  Layer: lake_edges  – one line per directed edge (upstream → downstream lake)
                       attrs: from_lake_id, to_lake_id, from_name, to_name
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GRAPH_CSV = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit"
    r"\GRIT_mekong_mega_reservoirs\reservoirs\gritv06_pld_lake_graph.csv"
)
PLD_SHP = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit"
    r"\GRIT_mekong_mega_reservoirs\prior_lake_database"
    r"\swot_prior_lake_database_mekong_overlap_with_grit.shp"
)
OUTPUT_GPKG = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit"
    r"\GRIT_mekong_mega_reservoirs\reservoirs\gritv06_pld_lake_graph.gpkg"
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
graph = pd.read_csv(GRAPH_CSV)

# Normalise lake_id to int64 (it may be stored as float in the CSV)
graph["lake_id"] = graph["lake_id"].astype("int64")

# Fill NaN connectivity cells with empty string so string ops are safe
graph["upstream_lake_ids"] = graph["upstream_lake_ids"].fillna("").astype(str)
graph["downstream_lake_ids"] = graph["downstream_lake_ids"].fillna("").astype(str)

print(f"Graph rows: {len(graph)}")

# Load PLD shapefile (polygons)
pld = gpd.read_file(PLD_SHP)

# lake_id is float64 in the shapefile — cast to int64
pld["lake_id"] = pld["lake_id"].astype("int64")

# Each physical lake may appear as multiple sub-polygons in PLD.
# Dissolve them into a single multi-polygon per lake_id so we get one centroid.
pld_dissolved = (
    pld[["lake_id", "names", "poly_area", "geometry"]]
    .dissolve(by="lake_id", aggfunc={"names": "first", "poly_area": "sum"})
    .reset_index()
)
pld_dissolved["centroid"] = pld_dissolved.geometry.centroid

print(f"PLD unique lake_ids after dissolve: {len(pld_dissolved)}")

# ---------------------------------------------------------------------------
# Build nodes GeoDataFrame
# ---------------------------------------------------------------------------
# Merge graph data with dissolved PLD geometries
nodes_df = graph.merge(
    pld_dissolved[["lake_id", "names", "poly_area", "centroid"]],
    on="lake_id",
    how="left",
)

missing_geom = nodes_df["centroid"].isna().sum()
if missing_geom:
    print(f"WARNING: {missing_geom} lake(s) from the graph have no PLD geometry.")

nodes_gdf = gpd.GeoDataFrame(
    nodes_df.drop(columns=["centroid"]),
    geometry=nodes_df["centroid"],
    crs="EPSG:4326",
)

print(f"Nodes layer: {len(nodes_gdf)} features")

# ---------------------------------------------------------------------------
# Build a quick centroid lookup: lake_id → (x, y)
# ---------------------------------------------------------------------------
centroid_lookup: dict[int, tuple[float, float]] = {}
for _, row in pld_dissolved.iterrows():
    c = row["centroid"]
    if c is not None and not c.is_empty:
        centroid_lookup[int(row["lake_id"])] = (c.x, c.y)

# Also fall back to lon/lat columns from original PLD if centroid is missing
for _, row in pld.iterrows():
    lid = int(row["lake_id"])
    if lid not in centroid_lookup:
        try:
            centroid_lookup[lid] = (float(row["lon"]), float(row["lat"]))
        except (TypeError, ValueError):
            pass

# ---------------------------------------------------------------------------
# Build edges GeoDataFrame
# ---------------------------------------------------------------------------
# Each row in the graph CSV can encode multiple upstream and downstream
# relationships. Expand them into individual directed edges.
edge_records = []

for _, row in graph.iterrows():
    src_id = int(row["lake_id"])
    src_xy = centroid_lookup.get(src_id)

    # downstream edges: src_id → each downstream lake
    for tgt_str in str(row["downstream_lake_ids"]).split(","):
        tgt_str = tgt_str.strip()
        if not tgt_str:
            continue
        tgt_id = int(tgt_str)
        tgt_xy = centroid_lookup.get(tgt_id)
        if src_xy and tgt_xy:
            geom = LineString([src_xy, tgt_xy])
        else:
            geom = None
        edge_records.append(
            {
                "from_lake_id": src_id,
                "to_lake_id": tgt_id,
                "geometry": geom,
            }
        )

edges_gdf = gpd.GeoDataFrame(edge_records, geometry="geometry", crs="EPSG:4326")

# Drop edges where geometry could not be built (both endpoints missing from PLD)
valid_mask = edges_gdf["geometry"].notna()
dropped = (~valid_mask).sum()
if dropped:
    print(f"WARNING: Dropped {dropped} edge(s) with missing endpoint geometry.")
edges_gdf = edges_gdf[valid_mask].reset_index(drop=True)

# Add lake names for convenience
name_lookup = dict(zip(pld_dissolved["lake_id"], pld_dissolved["names"]))
edges_gdf["from_name"] = edges_gdf["from_lake_id"].map(name_lookup)
edges_gdf["to_name"] = edges_gdf["to_lake_id"].map(name_lookup)

print(f"Edges layer: {len(edges_gdf)} features")

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
