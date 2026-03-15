# import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


# load grit reach with lakes
grit_reach_df = pd.read_csv(r'/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reaches/gritv06_reaches_mekong_basin_with_pld_lakes.csv')
lake_ids = np.unique(grit_reach_df['lake_id'])
lake_ids = lake_ids[~np.isnan(lake_ids)] # get the lake ids along the grit reaches
print(f"Loaded {len(lake_ids)} lake IDs from GRIT reaches")


def process_year(year_str, lake_ids):
    """Process all months for a given year and return a concatenated DataFrame."""
    start_month = datetime(int(year_str), 1, 1)
    end_month = datetime(int(year_str) + 1, 1, 1)

    # clamp to overall date range
    start_month = max(start_month, datetime(2023, 12, 1))
    end_month = min(end_month, datetime(2025, 12, 1))

    year_df = pd.DataFrame()
    current_month = start_month

    while current_month < end_month:
        month_end = current_month + relativedelta(months=1)
        start_date = current_month.strftime('%Y-%m-%d')
        end_date = month_end.strftime('%Y-%m-%d')

        file_list_path = f"/data/ouce-grit/cenv1160/smart_hs/raw_data/swot/mekong_river_basin/swot_lakes/file_list/{start_date}_{end_date}_swot_lake_file_df.csv"
        swot_file_df = pd.read_csv(file_list_path)
        print(f"  [Year {year_str}] {start_date} to {end_date}: {swot_file_df.shape[0]} files found")

        for index in range(swot_file_df.shape[0]):
            curr_url = swot_file_df['url'].values[index]
            filename = os.path.basename(curr_url)[:-4]
            if 'Prior' not in filename:
                continue

            file_path = f"/data/ouce-grit/cenv1160/smart_hs/raw_data/swot/mekong_river_basin/swot_lakes/{year_str}/{filename}"
            swot_lake_df = gpd.read_file(file_path + "/" + f"{filename}.shp")
            swot_lake_ids = swot_lake_df['lake_id'].values.astype(int)
            swot_lake_df = swot_lake_df[np.isin(swot_lake_ids, lake_ids)]
            swot_lake_df = swot_lake_df.drop(columns=['geometry'])

            year_df = pd.concat([year_df, swot_lake_df], axis=0, ignore_index=True)
            print(f"    [{index+1}/{swot_file_df.shape[0]}] {filename}: {swot_lake_df.shape[0]} matching lakes")

        current_month += relativedelta(months=1)

    print(f"  [Year {year_str}] Done — {year_df.shape[0]} rows collected")
    return year_str, year_df


# Merge the swot lake data into a single file
# Start from December 2023 to December 2025
save_folder = r"/data/ouce-grit/cenv1160/smart_hs/processed_data/swot/mekong_river_basin/swot/lakes"
years = ['2023', '2024', '2025']

print(f"\nProcessing {len(years)} years in parallel: {years}")

year_dfs = {}
with ProcessPoolExecutor(max_workers=len(years)) as executor:
    futures = {executor.submit(process_year, y, lake_ids): y for y in years}
    for future in as_completed(futures):
        year_str, year_df = future.result()
        year_dfs[year_str] = year_df

        # save per-year CSV
        year_save_path = save_folder + f"/swot_lake_df_{year_str}.csv"
        year_df.to_csv(year_save_path, index=False)
        print(f"Saved year {year_str}: {year_df.shape[0]} rows -> {year_save_path}")

# combine all years
full_swot_lake_df = pd.concat([year_dfs[y] for y in sorted(year_dfs)], axis=0, ignore_index=True)

# remove invalid data
print(f"\nTotal rows before filtering: {full_swot_lake_df.shape[0]}")
full_swot_lake_df = full_swot_lake_df[full_swot_lake_df['wse'] != -999999999999.0]
print(f"Total rows after removing invalid WSE: {full_swot_lake_df.shape[0]}")
full_swot_lake_df['time_str'] = pd.to_datetime(full_swot_lake_df['time_str'])
full_swot_lake_df['date'] = full_swot_lake_df['time_str'].dt.date

# save combined CSV
save_path = save_folder + "/" + "full_swot_lake_df_2023_2025.csv"
full_swot_lake_df.to_csv(save_path, index=False)
print(f"\nSaved {full_swot_lake_df.shape[0]} rows to {save_path}")
