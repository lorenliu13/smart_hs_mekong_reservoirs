"""
Assign SWOT PLD lake_id to GRIT reaches that intersect lake polygons.
"""

import geopandas as gpd

# --- Input paths ---
PLD_SHP   = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\prior_lake_database\swot_prior_lake_database_great_mekong.shp"
REACH_SHP = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\reaches\gritv06_reaches_great_mekong_basin.shp"

# --- Output paths ---
OUTPUT_GPKG    = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\reaches\gritv06_reaches_great_mekong_with_lake_id.gpkg"
OUTPUT_CSV     = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\reaches\gritv06_reaches_great_mekong_with_lake_id.csv"
OUTPUT_PLD_GPKG = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\prior_lake_database\swot_prior_lake_database_great_mekong_overlap_with_grit.gpkg"
OUTPUT_PLD_CSV  = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\prior_lake_database\swot_prior_lake_database_great_mekong_overlap_with_grit.csv"

# --- Column name in PLD that holds the lake identifier ---
PLD_LAKE_ID_COL = "lake_id"   # <-- adjust to actual column name in your PLD shapefile


def main():
    print("Loading PLD polygons...")
    pld = gpd.read_file(PLD_SHP)
    print(f"  PLD CRS: {pld.crs}, features: {len(pld)}")
    print(f"  PLD columns: {pld.columns.tolist()}")

    print("Loading GRIT reaches...")
    reaches = gpd.read_file(REACH_SHP)
    print(f"  Reaches CRS: {reaches.crs}, features: {len(reaches)}")

    # --- Align CRS ---
    if pld.crs != reaches.crs:
        print(f"  Reprojecting PLD to reaches CRS ({reaches.crs})...")
        pld = pld.to_crs(reaches.crs)

    # --- Spatial join: find reaches that intersect any PLD polygon ---
    print("Running spatial join (this may take a moment)...")
    joined = gpd.sjoin(
        reaches,
        pld[[PLD_LAKE_ID_COL, "geometry"]],
        how="left",          # keep all reaches; non-intersecting get NaN
        predicate="intersects",
    )

    # For reaches that intersect multiple lakes, keep the lake with the longest overlap.
    # Build a lookup: pld_lake_id -> lake geometry
    pld_geom = pld.set_index(PLD_LAKE_ID_COL)["geometry"]

    def pick_best_lake(group):
        """Return the lake_id whose polygon overlaps the reach the most."""
        # Rows with no matched lake (NaN lake_id from left join)
        valid = group.dropna(subset=[PLD_LAKE_ID_COL])
        if valid.empty:
            return None
        if len(valid) == 1:
            return valid[PLD_LAKE_ID_COL].iloc[0]
        reach_geom = valid.geometry.iloc[0]
        lengths = valid[PLD_LAKE_ID_COL].apply(
            lambda lid: reach_geom.intersection(pld_geom.loc[lid]).length
        )
        return valid.iloc[lengths.values.argmax()][PLD_LAKE_ID_COL]

    print("  Resolving reaches that intersect multiple lakes (keeping largest overlap)...")
    lake_id_series = (
        joined.groupby(joined.index, group_keys=False)
        .apply(pick_best_lake)
        .rename("lake_id")
    )

    # Collapse to one row per reach and attach the winning lake_id
    joined = joined[~joined.index.duplicated(keep="first")].drop(
        columns=[PLD_LAKE_ID_COL, "index_right"], errors="ignore"
    )
    joined = joined.join(lake_id_series)

    print(f"  Reaches with a lake_id: {joined['lake_id'].notna().sum()} / {len(joined)}")

    # GeoPackage reserves 'fid' as its internal row ID — rename if present
    if "fid" in joined.columns:
        joined = joined.rename(columns={"fid": "reach_id"})

    print(f"Saving GPKG to {OUTPUT_GPKG} ...")
    joined.to_file(OUTPUT_GPKG, driver="GPKG")

    print(f"Saving CSV to {OUTPUT_CSV} ...")
    joined.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    # Save PLD lakes that overlap with at least one GRIT reach
    matched_lake_ids = set(joined["lake_id"].dropna())
    pld_overlap = pld[pld[PLD_LAKE_ID_COL].isin(matched_lake_ids)]
    print(f"Saving {len(pld_overlap)} overlapping PLD lakes to {OUTPUT_PLD_GPKG} ...")
    pld_overlap.to_file(OUTPUT_PLD_GPKG, driver="GPKG")
    pld_overlap.drop(columns="geometry").to_csv(OUTPUT_PLD_CSV, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
