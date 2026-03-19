# Calculate the average of the atmospheric variables per catchment
import geopandas as gpd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from exactextract import exact_extract
import rasterio
import rasterio.plot
from pyogrio import read_dataframe
import rioxarray
import os


def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass


# load all the variable xarrays
variable_name_list = ['LWd', 'P', 'Pres', 'RelHum', 'SWd', 'Temp', 'Wind']

for variable_name in variable_name_list:
    print(f'Processing {variable_name}')
    # variable_name = 'P'
    # filename = rf"E:\Project_2025_2026\Smart_hs\MSWX_V100_Sacramento\Past\{variable_name}\Daily\{variable_name}_Past_Daily_combined.nc"
    filename = rf"E:\Project_2025_2026\Smart_hs\MSWX_V100_Ba_River\Past\{variable_name}\Daily\{variable_name}_Past_Daily_combined.nc"

    shapefile_folder = r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_vietnam\ba_river_watershed\catchments"
    catch0 = read_dataframe(shapefile_folder + "/" + "gritv06_segment_catchment_ba_river_watershed_epsg4326.shp")

    res = exact_extract(
            rast=filename,
            vec=catch0, 
            ops='mean', progress = True, output = 'pandas',
            # max_cells_in_memory = 100000,
            strategy = "raster-sequential"
        )

    save_folder = r"E:\Project_2025_2026\Smart_hs\processed_data\swot_gnn\training_data\mswx_forcing"
    create_folder(save_folder)
    
    res.to_csv(save_folder + "/" + f"{variable_name}_Past_Daily_combined_catchment_avg.csv", index=False)


