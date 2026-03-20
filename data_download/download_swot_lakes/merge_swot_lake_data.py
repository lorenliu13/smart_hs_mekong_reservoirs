# import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
from concurrent.futures import ProcessPoolExecutor, as_completed




def process_year(year_str):
    """Process all months for a given year and return a concatenated DataFrame and log messages.

    For each month in the year, reads a pre-built file list CSV that enumerates
    available SWOT lake shapefiles. All lakes in each shapefile are loaded and
    geometry is dropped to produce a lightweight tabular record. Results are
    accumulated across all months and returned alongside diagnostic log strings.

    Args:
        year_str (str): Four-digit year string, e.g. '2024'.

    Returns:
        tuple: (year_str, year_df, logs)
            - year_str: the input year string (used as a dict key by the caller)
            - year_df: DataFrame of all SWOT lake observations for the year
            - logs: list of log message strings for printing
    """
    # Build the start/end bounds for monthly iteration over this year
    start_month = datetime(int(year_str), 1, 1)
    end_month = datetime(int(year_str) + 1, 1, 1)

    # Clamp to the overall study period: Dec 2023 – Dec 2025
    start_month = max(start_month, datetime(2023, 12, 1))
    end_month = min(end_month, datetime(2025, 12, 1))

    year_df = pd.DataFrame()  # accumulator for all monthly data within this year
    logs = []
    current_month = start_month

    # Iterate month-by-month through the clamped date range
    while current_month < end_month:
        month_end = current_month + relativedelta(months=1)
        start_date = current_month.strftime('%Y-%m-%d')
        end_date = month_end.strftime('%Y-%m-%d')

        # Read the pre-built file list CSV for this month, which contains URLs
        # to each SWOT granule (shapefile directory) covering the Mekong basin
        file_list_path = f"/data/ouce-grit/cenv1160/smart_hs/raw_data/swot/mekong_river_basin/swot_lakes/file_list/{start_date}_{end_date}_swot_lake_file_df.csv"
        swot_file_df = pd.read_csv(file_list_path)
        logs.append(f"  [Year {year_str}] {start_date} to {end_date}: {swot_file_df.shape[0]} files found")

        # Process each SWOT granule file listed for this month
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

            # Append this granule's filtered rows to the year accumulator
            year_df = pd.concat([year_df, swot_lake_df], axis=0, ignore_index=True)
            logs.append(f"    [{index+1}/{swot_file_df.shape[0]}] {filename}: {swot_lake_df.shape[0]} lakes")

        current_month += relativedelta(months=1)

    logs.append(f"  [Year {year_str}] Done — {year_df.shape[0]} rows collected")
    return year_str, year_df, logs


# --- Main: merge SWOT lake data from Dec 2023 to Dec 2025 into a single CSV ---

# Output directory for per-year and combined CSV files
save_folder = r"/data/ouce-grit/cenv1160/smart_hs/processed_data/swot/mekong_river_basin/swot/lakes"
years = ['2023', '2024', '2025']

print(f"\nProcessing {len(years)} years in parallel: {years}")

# Process each year in parallel using one worker process per year.
# Each worker reads, filters, and concatenates SWOT shapefiles independently.
year_dfs = {}
with ProcessPoolExecutor(max_workers=len(years)) as executor:
    futures = {executor.submit(process_year, y): y for y in years}
    for future in as_completed(futures):
        year_str, year_df, logs = future.result()
        print("\n".join(logs), flush=True)
        year_dfs[year_str] = year_df

        # Save an intermediate per-year CSV so progress is not lost if the
        # final concatenation step fails
        year_save_path = save_folder + f"/swot_lake_df_{year_str}.csv"
        year_df.to_csv(year_save_path, index=False)
        print(f"Saved year {year_str}: {year_df.shape[0]} rows -> {year_save_path}", flush=True)

# Concatenate all years in chronological order into a single DataFrame
full_swot_lake_df = pd.concat([year_dfs[y] for y in sorted(year_dfs)], axis=0, ignore_index=True)

# Remove rows where WSE (water surface elevation) is the SWOT fill/no-data value
print(f"\nTotal rows before filtering: {full_swot_lake_df.shape[0]}")
full_swot_lake_df = full_swot_lake_df[full_swot_lake_df['wse'] != -999999999999.0]
print(f"Total rows after removing invalid WSE: {full_swot_lake_df.shape[0]}")

# Parse the time string column and extract a plain date column for easier grouping
full_swot_lake_df['time_str'] = pd.to_datetime(full_swot_lake_df['time_str'])
full_swot_lake_df['date'] = full_swot_lake_df['time_str'].dt.date

# Save the full merged dataset
save_path = save_folder + "/" + "full_swot_lake_df_2023_2025.csv"
full_swot_lake_df.to_csv(save_path, index=False)
print(f"\nSaved {full_swot_lake_df.shape[0]} rows to {save_path}")
