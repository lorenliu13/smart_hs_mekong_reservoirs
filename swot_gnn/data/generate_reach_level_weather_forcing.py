# assign catchment average atmospheric variables to reach level
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pyogrio import read_dataframe


variable_name_list = ['LWd', 'P', 'Pres', 'RelHum', 'SWd', 'Temp', 'Wind']

for variable_name in variable_name_list:
        
    print(f'Processing {variable_name}')
    # load catchment average
    source_folder = r"E:\Project_2025_2026\Smart_hs\processed_data\swot_gnn\training_data\mswx_forcing"
    r = pd.read_csv(source_folder + "/" + f"{variable_name}_Past_Daily_combined_catchment_avg.csv")
    
    # load the segment shapefile
    shapefile_folder = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_vietnam\ba_river_watershed\catchments"
    catch0 = read_dataframe(shapefile_folder + "/" + "gritv06_segment_catchment_ba_river_watershed_epsg4326.shp")

    # assign segment ids for catchment average dataframe
    r['segment_id'] = catch0['fid']

    # load the xarray
    regional_var_xarray = xr.load_dataset(rf"E:\Project_2025_2026\Smart_hs\MSWX_V100_Ba_River\Past\{variable_name}\Daily\{variable_name}_Past_Daily_combined.nc")
    time_steps = regional_var_xarray['time'].data

    # load csv file
    grit_reach_df = pd.read_csv(r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_vietnam\ba_river_watershed\reaches\ba_river_watershed_reaches_gritv06_with_centroid_lake.csv")

    grit_reach_ids = grit_reach_df['fid'].values
    full_reach_var_df = pd.DataFrame()
    for grit_reach_id in grit_reach_ids:
        curr_grit_reach_record = grit_reach_df[grit_reach_df['fid'] == grit_reach_id]
        # current segment id
        curr_segment_id = curr_grit_reach_record['segment_id'].values[0]

        # find the corresponding variables
        curr_var_record = r[r['segment_id'] == curr_segment_id]
        
        # drop the segment column
        curr_var_record = curr_var_record.drop(columns=['segment_id'])

        # get the variable time series 
        curr_var_series = curr_var_record.values.flatten()

        # get the variable time series
        # curr_var_series = curr_var_record.values.flatten()[1:]    

        # build a dataframe
        curr_reach_var_df = pd.DataFrame()
        curr_reach_var_df['time'] = time_steps
        curr_reach_var_df['fid'] = grit_reach_id
        curr_reach_var_df['var'] = curr_var_series

        # concat
        full_reach_var_df = pd.concat([full_reach_var_df, curr_reach_var_df], axis = 0, ignore_index = True)

    # save_folder = r"E:\Project_2025_2026\Smart_hs\processed_data\lstm\ba_river_basin\training_data_mswx_catchment_avg\mswx_forcing\catchment_avg_atmospheric_variables"
    full_reach_var_df.to_csv(source_folder + "/" + f"{variable_name}_Past_Daily_combined_catchment_avg_reach_level.csv", index=False)