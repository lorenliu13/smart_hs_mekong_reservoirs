"""
Extract HydroBasins shapefiles (levels 1-12) for the Great Mekong region.

Basins that intersect the Great Mekong boundary are retained.
Outputs are saved as shapefiles in the specified output directory.
"""

import geopandas as gpd
from pathlib import Path

# --- Paths ---
HYDROBASINS_DIR = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\hydrobasins\hybas_as_lev01-12_v1c")
MEKONG_BOUNDARY = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\basin_shapefile\GM_boundary_geometry_fixed.shp")
OUTPUT_DIR = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\basin_shapefile\hydrobasins")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load Great Mekong boundary ---
print("Loading Great Mekong boundary...")
mekong = gpd.read_file(MEKONG_BOUNDARY)

# --- Process each level ---
for level in range(1, 9):
    level_str = f"{level:02d}"
    input_shp = HYDROBASINS_DIR / f"hybas_as_lev{level_str}_v1c.shp"

    if not input_shp.exists():
        print(f"[SKIP] Level {level_str}: file not found at {input_shp}")
        continue

    print(f"[Level {level_str}] Loading {input_shp.name}...")
    basins = gpd.read_file(input_shp)

    # Reproject to match CRS if needed
    if basins.crs != mekong.crs:
        mekong_reproj = mekong.to_crs(basins.crs)
    else:
        mekong_reproj = mekong

    # Dissolve boundary to a single geometry for spatial filter
    mekong_union = mekong_reproj.geometry.union_all()

    # Keep basins that intersect the Mekong boundary
    mask = basins.geometry.intersects(mekong_union)
    basins_mekong = basins[mask].copy()

    print(f"[Level {level_str}] {mask.sum()} / {len(basins)} basins intersect the Mekong region.")

    # Save output shapefile
    out_shp = OUTPUT_DIR / f"hybas_as_lev{level_str}_v1c_great_mekong.shp"
    basins_mekong.to_file(out_shp)
    print(f"[Level {level_str}] Saved -> {out_shp}")

print("\nDone. All levels processed.")
