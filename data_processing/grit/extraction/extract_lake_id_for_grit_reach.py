"""
Assign SWOT PLD lake_id to GRIT reaches that intersect lake polygons.

Workflow:
  1. Load and merge two regional SWOT Prior Lake Database (PLD) shapefiles.
  2. Spatial-join GRIT reaches against PLD polygons; each reach gets the
     lake_id of the lake it intersects (longest overlap wins when a reach
     crosses multiple lakes).
  3. From the PLD subset that overlaps any GRIT reach, derive the HydroBasins
     sub-basin ID at levels 1–8 for each lake centroid.
  4. Propagate the HydroBasins IDs from the matched lake back onto each reach
     and save both the reach-level result and the lake-level result (GPKG + CSV).
"""

import geopandas as gpd
from pathlib import Path

# --- Input paths ---
# Two regional PLD tiles covering the Great Mekong: Mekong main stem and Yangtze.
PLD_SHP_1   = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\prior_lake_database\swot_prior_lake_database_great_mekong_mekong.shp"
PLD_SHP_2 = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\prior_lake_database\swot_prior_lake_database_great_mekong_yang.shp"
REACH_SHP = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\reaches\gritv06_reaches_great_mekong_basin.shp"

# --- Output paths ---
# Reaches with an assigned lake_id column (NaN for reaches outside any lake).
OUTPUT_GPKG    = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\reaches\gritv06_reaches_great_mekong_with_lake_id.gpkg"
OUTPUT_CSV     = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\reaches\gritv06_reaches_great_mekong_with_lake_id.csv"
# PLD lakes that overlap at least one GRIT reach, enriched with HydroBasins IDs.
OUTPUT_PLD_GPKG = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\prior_lake_database\swot_prior_lake_database_great_mekong_overlap_with_grit.gpkg"
OUTPUT_PLD_CSV  = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\prior_lake_database\swot_prior_lake_database_great_mekong_overlap_with_grit.csv"

# --- Column name in PLD that holds the lake identifier ---
PLD_LAKE_ID_COL = "lake_id"   # <-- adjust to actual column name in your PLD shapefile

# --- HydroBasins config ---
# Directory that contains one shapefile per hierarchical level (01–08).
# Expected filename pattern: hybas_as_lev{NN}_v1c_great_mekong.shp
HYDROBASINS_DIR = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs"
    r"\basin_shapefile\hydrobasins"
)
HYDROBASINS_LEVELS = range(1, 9)  # levels 1–8 (coarsest to finest subdivision)


def main():
    # ------------------------------------------------------------------ #
    # 1. Load and merge the two PLD regional tiles                        #
    # ------------------------------------------------------------------ #
    print("Loading PLD polygons...")
    pld1 = gpd.read_file(PLD_SHP_1)
    print(f"  PLD_1 CRS: {pld1.crs}, features: {len(pld1)}")
    pld2 = gpd.read_file(PLD_SHP_2)
    print(f"  PLD_2 CRS: {pld2.crs}, features: {len(pld2)}")

    # Reproject the second tile to match the first before concatenating
    # so that the merged GeoDataFrame has a single, consistent CRS.
    if pld2.crs != pld1.crs:
        print(f"  Reprojecting PLD_2 to PLD_1 CRS ({pld1.crs})...")
        pld2 = pld2.to_crs(pld1.crs)

    # Concatenate and remove any lakes that appear in both regional tiles.
    pld = gpd.GeoDataFrame(
        gpd.pd.concat([pld1, pld2], ignore_index=True),
        crs=pld1.crs,
    ).drop_duplicates(subset=[PLD_LAKE_ID_COL])
    print(f"  Merged PLD features: {len(pld)}")
    print(f"  PLD columns: {pld.columns.tolist()}")

    # ------------------------------------------------------------------ #
    # 2. Load GRIT reaches and align CRS with PLD                         #
    # ------------------------------------------------------------------ #
    print("Loading GRIT reaches...")
    reaches = gpd.read_file(REACH_SHP)
    print(f"  Reaches CRS: {reaches.crs}, features: {len(reaches)}")

    # Reproject PLD to the reaches CRS so the spatial join uses a consistent
    # coordinate system (reaches CRS is used as the reference).
    if pld.crs != reaches.crs:
        print(f"  Reprojecting PLD to reaches CRS ({reaches.crs})...")
        pld = pld.to_crs(reaches.crs)

    # ------------------------------------------------------------------ #
    # 3. Spatial join: assign lake_id to each reach                       #
    # ------------------------------------------------------------------ #
    print("Running spatial join (this may take a moment)...")
    # Left join keeps all reaches; those outside every lake polygon get NaN.
    joined = gpd.sjoin(
        reaches,
        pld[[PLD_LAKE_ID_COL, "geometry"]],
        how="left",          # keep all reaches; non-intersecting get NaN
        predicate="intersects",
    )

    # A single reach may intersect more than one lake polygon.
    # Build a geometry lookup keyed by lake_id so we can measure overlap length.
    pld_geom = pld.set_index(PLD_LAKE_ID_COL)["geometry"]

    def pick_best_lake(group):
        """Return the lake_id whose polygon overlaps the reach the most."""
        # Drop rows where the left join produced no match (NaN lake_id).
        valid = group.dropna(subset=[PLD_LAKE_ID_COL])
        if valid.empty:
            return None
        if len(valid) == 1:
            return valid[PLD_LAKE_ID_COL].iloc[0]
        # Compute intersection length with each candidate lake and keep the max.
        reach_geom = valid.geometry.iloc[0]
        lengths = valid[PLD_LAKE_ID_COL].apply(
            lambda lid: reach_geom.intersection(pld_geom.loc[lid]).length
        )
        return valid.iloc[lengths.values.argmax()][PLD_LAKE_ID_COL]

    print("  Resolving reaches that intersect multiple lakes (keeping largest overlap)...")
    # Group all sjoin rows back to their original reach index, pick one lake_id.
    lake_id_series = (
        joined.groupby(joined.index, group_keys=False)
        .apply(pick_best_lake)
        .rename("lake_id")
    )

    # Deduplicate to one row per reach, drop the sjoin helper columns,
    # then attach the resolved lake_id.
    joined = joined[~joined.index.duplicated(keep="first")].drop(
        columns=[PLD_LAKE_ID_COL, "index_right"], errors="ignore"
    )
    joined = joined.join(lake_id_series)

    print(f"  Reaches with a lake_id: {joined['lake_id'].notna().sum()} / {len(joined)}")

    # GeoPackage reserves 'fid' as its internal row ID — rename if present
    # to avoid a conflict that would silently overwrite the column on write.
    if "fid" in joined.columns:
        joined = joined.rename(columns={"fid": "reach_id"})

    # ------------------------------------------------------------------ #
    # 4. Build the PLD subset that overlaps any GRIT reach                #
    # ------------------------------------------------------------------ #
    # Only keep lakes that were matched to at least one reach.
    matched_lake_ids = set(joined["lake_id"].dropna())
    pld_overlap = pld[pld[PLD_LAKE_ID_COL].isin(matched_lake_ids)].copy()

    # ------------------------------------------------------------------ #
    # 5. Assign HydroBasins sub-basin ID for each lake centroid           #
    # ------------------------------------------------------------------ #
    # Use each lake's centroid (in EPSG:4326) as the point for the spatial
    # join against the HydroBasins polygon layer for each level.
    # Result columns: "hybasin_level_1" … "hybasin_level_8".
    print("\nAssigning HydroBasins sub-basin IDs to PLD lakes...")
    lakes_gdf = gpd.GeoDataFrame(
        pld_overlap[[PLD_LAKE_ID_COL]].copy(),
        geometry=pld_overlap.geometry.centroid,
        crs=pld_overlap.crs,
    ).to_crs("EPSG:4326")  # HydroBasins shapefiles are distributed in EPSG:4326

    for level in HYDROBASINS_LEVELS:
        level_str = f"{level:02d}"
        shp_path = Path(HYDROBASINS_DIR) / f"hybas_as_lev{level_str}_v1c_great_mekong.shp"
        col_name = f"hybasin_level_{level}"

        if not shp_path.exists():
            print(f"  [SKIP] Level {level}: file not found at {shp_path}")
            # Placeholder column so the output schema is consistent across runs.
            pld_overlap[col_name] = float("nan")
            continue

        # Reproject basins to match lakes_gdf CRS to avoid sjoin CRS mismatch
        # errors that occur when CRS strings differ even if the datum is the same.
        basins = gpd.read_file(shp_path)[["HYBAS_ID", "geometry"]].to_crs(lakes_gdf.crs)

        # Point-in-polygon join: each lake centroid gets the HYBAS_ID of the
        # basin polygon it falls within. "within" is exact for interior points;
        # boundary-touching centroids would need "intersects" as a fallback.
        joined_basins = lakes_gdf.sjoin(basins, how="left", predicate="within")

        # Drop duplicate rows that arise when a centroid touches two basin
        # polygons exactly on their shared boundary.
        joined_basins = joined_basins.drop_duplicates(subset=PLD_LAKE_ID_COL)

        # Build a lake_id → HYBAS_ID mapping and broadcast it onto pld_overlap.
        hybas_map = joined_basins.set_index(PLD_LAKE_ID_COL)["HYBAS_ID"]
        pld_overlap[col_name] = pld_overlap[PLD_LAKE_ID_COL].map(hybas_map)

        assigned = pld_overlap[col_name].notna().sum()
        print(f"  Level {level}: {assigned}/{len(pld_overlap)} lakes assigned a HYBAS_ID.")

    # ------------------------------------------------------------------ #
    # 6. Propagate HydroBasins IDs onto reaches and save reach output     #
    # ------------------------------------------------------------------ #
    # Build a lake_id → hybasin_level_X lookup from pld_overlap, then join
    # it onto the reaches table using each reach's assigned lake_id.
    # Reaches with no lake_id (NaN) receive NaN for all HydroBasins columns.
    hybasin_cols = [f"hybasin_level_{level}" for level in HYDROBASINS_LEVELS]
    hybasin_lookup = pld_overlap.set_index(PLD_LAKE_ID_COL)[hybasin_cols]
    joined = joined.join(hybasin_lookup, on="lake_id")

    print(f"Saving GPKG to {OUTPUT_GPKG} ...")
    joined.to_file(OUTPUT_GPKG, driver="GPKG")

    print(f"Saving CSV to {OUTPUT_CSV} ...")
    joined.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    # ------------------------------------------------------------------ #
    # 7. Save PLD-overlap output                                          #
    # ------------------------------------------------------------------ #
    print(f"Saving {len(pld_overlap)} overlapping PLD lakes to {OUTPUT_PLD_GPKG} ...")
    pld_overlap.to_file(OUTPUT_PLD_GPKG, driver="GPKG")
    pld_overlap.drop(columns="geometry").to_csv(OUTPUT_PLD_CSV, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
