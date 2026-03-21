# import fiona
import geopandas as gpd
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os




def process_month(start_date, end_date, year_str, save_folder):
    """Process all SWOT granules for a single month and write to a per-month CSV.

    Args:
        start_date (str): Month start date string, e.g. '2024-03-01'.
        end_date (str): Month end date string, e.g. '2024-04-01'.
        year_str (str): Four-digit year string used to locate shapefiles on disk.
        save_folder (str): Directory where the per-month CSV is written.

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
    total_rows = 0
    header_written = False

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

        # Write directly to CSV (append after the first granule) to avoid
        # accumulating a large in-memory DataFrame across all granules
        swot_lake_df.to_csv(month_save_path, mode='a', header=not header_written, index=False)
        header_written = True
        total_rows += len(swot_lake_df)
        logs.append(f"    [{index+1}/{swot_file_df.shape[0]}] {filename}: {swot_lake_df.shape[0]} lakes")

    logs.append(f"  [{month_label}] Done — {total_rows} rows written to {month_save_path}")
    return month_save_path, total_rows, logs


# --- Main: merge SWOT lake data from Dec 2023 to Dec 2025 into a single CSV ---

# Output directory for per-month and combined CSV files
save_folder = r"/data/ouce-grit/cenv1160/smart_hs/processed_data/swot/mekong_river_basin/swot/lakes"

# Build the full list of months in the study period: Dec 2023 – Dec 2025
study_start = datetime(2023, 12, 1)
study_end = datetime(2025, 12, 1)
months = []
current = study_start
while current < study_end:
    months.append(current)
    current += relativedelta(months=1)

print(f"\nProcessing {len(months)} months sequentially")

# Process each month independently, writing a per-month CSV
month_save_paths = []
for month_dt in months:
    month_end_dt = month_dt + relativedelta(months=1)
    start_date = month_dt.strftime('%Y-%m-%d')
    end_date = month_end_dt.strftime('%Y-%m-%d')
    year_str = month_dt.strftime('%Y')

    month_save_path, total_rows, logs = process_month(start_date, end_date, year_str, save_folder)
    print("\n".join(logs), flush=True)
    month_save_paths.append(month_save_path)

# Build the final merged CSV by reading one month at a time, filtering, and
# appending — so only one month is held in memory at any point.
save_path = save_folder + "/" + "full_swot_lake_df_2023_2025.csv"
header_written = False
total_rows = 0
for month_save_path in month_save_paths:
    if not os.path.exists(month_save_path):
        print(f"Skipping missing file: {month_save_path}", flush=True)
        continue
    month_df = pd.read_csv(month_save_path)

    # Remove rows where WSE (water surface elevation) is the SWOT fill/no-data value
    month_df = month_df[month_df['wse'] != -999999999999.0]

    # Parse the time string column and extract a plain date column for easier grouping
    month_df['time_str'] = pd.to_datetime(month_df['time_str'])
    month_df['date'] = month_df['time_str'].dt.date

    month_df.to_csv(save_path, mode='a', header=not header_written, index=False)
    header_written = True
    total_rows += len(month_df)
    del month_df  # free memory before loading the next month

print(f"\nSaved {total_rows} rows to {save_path}")
