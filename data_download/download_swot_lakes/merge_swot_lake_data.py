# import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os


# load grit reach with lakes
grit_reach_df = pd.read_csv(r'E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_vietnam\ba_river_watershed\reaches\ba_river_watershed_reaches_gritv06_with_centroid_lake.csv')
lake_ids = np.unique(grit_reach_df['lake_id'])
lake_ids = lake_ids[~np.isnan(lake_ids)] # get the lake ids along the grit reaches


# Merge the swot lake data into a single file
# Start from December 2023 to September 2025
start_month = datetime(2023, 12, 1)
end_month = datetime(2025, 12, 1)

# Generate list of start_date and end_date pairs
date_pairs = []
current_month = start_month

full_swot_lake_df = pd.DataFrame()

while current_month < end_month:
    # Calculate the start and end of the current month
    month_start = current_month
    month_end = current_month + relativedelta(months=1) # - timedelta(days=1)
    
    # Format dates as strings
    start_date = month_start.strftime('%Y-%m-%d')
    end_date = month_end.strftime('%Y-%m-%d')

    # get the year
    year = start_date[:4] # get the year from the start date

    swot_file_df = pd.read_csv(fr"E:\Project_2025_2026\Smart_hs\raw_data\swot\ba_river_watershed\file_list\{start_date}_{end_date}_swot_lake_file_df.csv")

    for index in range(swot_file_df.shape[0]):
        # extract the filename from the url
        curr_url = swot_file_df['url'].values[index]
        filename = os.path.basename(curr_url)[:-4]
        # check if this is the prior file 
        if 'Prior' in filename:
            pass
        else:
            # if not, skip the file
            continue
        
        # get the zip file path
        file_path = rf"E:\Project_2025_2026\Smart_hs\raw_data\swot\ba_river_watershed\swot_lake\{year}" + "/" + filename
        # load the shapefile
        swot_lake_df = gpd.read_file(file_path + "/" + f"{filename}.shp")
        # get the lake ids
        swot_lake_ids = swot_lake_df['lake_id'].values.astype(int)
        # get swot lake measurement
        swot_lake_df = swot_lake_df[np.isin(swot_lake_ids, lake_ids)]
        # drop the geoemtry column
        swot_lake_df = swot_lake_df.drop(columns=['geometry'])

        full_swot_lake_df = pd.concat([full_swot_lake_df, swot_lake_df], axis = 0, ignore_index = True)
        
    # Move to next month
    current_month += relativedelta(months=1)

# remove invalid data
full_swot_lake_df = full_swot_lake_df[full_swot_lake_df['wse'] != -999999999999.0]
full_swot_lake_df['time_str'] = pd.to_datetime(full_swot_lake_df['time_str'])
# get the date
full_swot_lake_df['date'] = full_swot_lake_df['time_str'].dt.date

# save the lake data
save_folder = r"E:\Project_2025_2026\Smart_hs\processed_data\swot\ba_river_watershed"
full_swot_lake_df.to_csv(save_folder + "/" + "full_swot_lake_df_2023_2025.csv", index=False)
