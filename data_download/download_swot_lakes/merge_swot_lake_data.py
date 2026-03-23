# import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

SWOT_FILL_VALUE = -999_999_999_999.0



def process_month(start_date, end_date, year_str, save_folder, valid_lake_ids):
    """Process all SWOT granules for a single month and write to a per-month CSV.

    Args:
        start_date (str): Month start date string, e.g. '2024-03-01'.
        end_date (str): Month end date string, e.g. '2024-04-01'.
        year_str (str): Four-digit year string used to locate shapefiles on disk.
        save_folder (str): Directory where the per-month CSV is written.
        valid_lake_ids (set): Set of lake_id values from the GRIT reaches CSV;
            only rows whose lake_id appears in this set are retained.

    Returns:
        tuple: (month_save_path, total_rows, logs)
            - month_save_path: path of the CSV written
            - total_rows: number of rows written
            - logs: list of log message strings for printing
    """
    month_label = start_date[:7]  # e.g. '2024-03'
    month_save_path = save_folder + f"/swot_lake_df_{month_label}.csv"

    file_list_path = (
        f"/data/ouce-grit/cenv1160/smart_hs/raw_data/swot/mekong_river_basin/swot_lakes/file_list/"
        f"{start_date}_{end_date}_swot_lake_file_df.csv"
    )
    swot_file_df = pd.read_csv(file_list_path)

    logs = [f"  [{month_label}] {start_date} to {end_date}: {swot_file_df.shape[0]} files found"]
    granule_dfs = []

    for index in range(swot_file_df.shape[0]):
        curr_url = swot_file_df['url'].values[index]
        # Derive the local filename by stripping the URL path and .zip extension
        filename = os.path.basename(curr_url)[:-4]

        # Only process "Prior" lake product files (PLD-based observations);
        # skip other product types (e.g. observed-only granules)
        if 'Prior' not in filename:
            continue

        # Construct the expected path to the shapefile on disk
        file_path = f"/data/ouce-grit/cenv1160/smart_hs/raw_data/swot/mekong_river_basin/swot_lakes/{year_str}/{filename}"
        shp_path = file_path + "/" + f"{filename}.shp"

        # Skip granules that haven't been downloaded yet
        if not os.path.exists(shp_path):
            logs.append(f"    [{index+1}/{swot_file_df.shape[0]}] SKIPPED (missing): {filename}")
            continue

        # Read the shapefile — all lakes in the granule are kept
        swot_lake_df = gpd.read_file(shp_path)

        # Drop geometry column — only tabular attributes are needed downstream
        swot_lake_df = swot_lake_df.drop(columns=['geometry'])

        # Keep only lakes whose lake_id appears in the GRIT reaches CSV
        swot_lake_df['lake_id'] = swot_lake_df['lake_id'].astype(int)
        swot_lake_df = swot_lake_df[swot_lake_df['lake_id'].isin(valid_lake_ids)]

        granule_dfs.append(swot_lake_df)
        logs.append(f"    [{index+1}/{swot_file_df.shape[0]}] {filename}: {swot_lake_df.shape[0]} lakes")

    # Concatenate all granules and write once
    if granule_dfs:
        month_df = pd.concat(granule_dfs, ignore_index=True)
        month_df.to_csv(month_save_path, index=False)
        total_rows = len(month_df)
    else:
        total_rows = 0

    logs.append(f"  [{month_label}] Done — {total_rows} rows written to {month_save_path}")
    return month_save_path, total_rows, logs


# --- Main: merge SWOT lake data from Dec 2023 to Dec 2025 into a single CSV ---

# Output directory for per-month and combined CSV files
save_folder = r"/data/ouce-grit/cenv1160/smart_hs/processed_data/swot/great_mekong_river_basin/lakes"

# Load the set of valid lake IDs from the GRIT reaches CSV; only observations
# for these lakes will be retained from each SWOT granule
grit_reaches_path = "/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reaches/gritv06_reaches_great_mekong_with_lake_id.csv"
grit_reaches_df = pd.read_csv(grit_reaches_path, usecols=['lake_id'])
# load the lake ids
lake_ids = pd.to_numeric(grit_reaches_df['lake_id'], errors='coerce')
# drop the nan
lake_ids = lake_ids[~np.isnan(lake_ids)]
valid_lake_ids = set(grit_reaches_df['lake_id'].dropna().unique())
print(f"Loaded {len(valid_lake_ids)} unique lake IDs from GRIT reaches CSV")

# Build the full list of months in the study period: Dec 2023 – Dec 2025
study_start = datetime(2025, 1, 1)
study_end = datetime(2025, 5, 1)
months = []
current = study_start
while current < study_end:
    months.append(current)
    current += relativedelta(months=1)

print(f"\nProcessing {len(months)} months in parallel (max 12 workers)")

# Build the argument list for each month
month_args = [
    (month_dt.strftime('%Y-%m-%d'),
     (month_dt + relativedelta(months=1)).strftime('%Y-%m-%d'),
     month_dt.strftime('%Y'),
     save_folder,
     valid_lake_ids)
    for month_dt in months
]

# Process months in parallel — each writes to its own file so there are no
# race conditions. Results arrive out of order; collect and sort afterwards.
month_save_paths = []
with ProcessPoolExecutor(max_workers=12) as executor:
    futures = {executor.submit(process_month, *args): args[0] for args in month_args}
    for future in as_completed(futures):
        month_save_path, total_rows, logs = future.result()
        print("\n".join(logs), flush=True)
        month_save_paths.append(month_save_path)

# Sort paths chronologically before merging
month_save_paths.sort()

save_path = os.path.join(save_folder, "full_swot_lake_df_2023_2025.csv")
monthly_dfs = []
for month_save_path in month_save_paths:
    try:
        month_df = pd.read_csv(month_save_path)
    except FileNotFoundError:
        print(f"Skipping missing file: {month_save_path}", flush=True)
        continue

    # Remove rows where WSE (water surface elevation) is the SWOT fill/no-data value
    month_df = month_df[month_df['wse'] != SWOT_FILL_VALUE]

    # Parse the time string column and extract a plain date column for easier grouping
    month_df['time_str'] = pd.to_datetime(month_df['time_str'])
    month_df['date'] = month_df['time_str'].dt.date

    monthly_dfs.append(month_df)

full_df = pd.concat(monthly_dfs, ignore_index=True)
full_df.to_csv(save_path, index=False)
print(f"\nSaved {len(full_df)} rows to {save_path}")
