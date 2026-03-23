"""
Extract GRIT catchment polygons within the Mekong River Basin boundary
and save as a shapefile.
"""

import geopandas as gpd

# --- Input paths ---
BASIN_SHP = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\basin_shapefile\GM_boundary_geometry_fixed.shp"
CATCHMENTS_GPKG = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\Grit_ARC\output_v06\catchments\GRITv06_segment_catchments_AS_EPSG4326.gpkg"

# --- Output path ---
OUTPUT_SHP = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\catchments\GRITv06_catchments_great_mekong.shp"


def main():
    print("Loading basin boundary...")
    basin = gpd.read_file(BASIN_SHP)
    print(f"  Basin CRS: {basin.crs}, features: {len(basin)}")

    print("Loading catchment polygons...")
    catchments = gpd.read_file(CATCHMENTS_GPKG)
    print(f"  Catchments CRS: {catchments.crs}, features: {len(catchments)}")

    # Reproject catchments to basin CRS if needed
    if catchments.crs != basin.crs:
        print(f"  Reprojecting catchments from {catchments.crs} to {basin.crs}...")
        catchments = catchments.to_crs(basin.crs)

    # Dissolve basin to a single geometry for the spatial filter
    basin_union = basin.geometry.union_all()

    print("Filtering catchments within basin boundary...")
    # Keep catchments whose geometry intersects the basin
    mask = catchments.geometry.intersects(basin_union)
    catchments_in_basin = catchments[mask].copy()
    print(f"  Catchments within basin: {len(catchments_in_basin)}")

    # Clip to exact basin boundary (removes parts outside)
    print("Clipping to exact basin boundary...")
    catchments_clipped = gpd.clip(catchments_in_basin, basin_union)
    print(f"  Features after clip: {len(catchments_clipped)}")

    print(f"Saving to {OUTPUT_SHP} ...")
    catchments_clipped.to_file(OUTPUT_SHP)
    print("Done.")


if __name__ == "__main__":
    main()
