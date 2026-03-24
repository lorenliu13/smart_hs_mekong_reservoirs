# Collecting SWOT data from the USGS website
import os
import earthaccess
import pandas as pd
import multiprocessing as mp
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta


def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass


def swot_file_collection(start_date, end_date):
    """
    Collecting SWOT data from the USGS website
    Args:
        start_date: the start date of the swot data
        end_date: the end date of the swot data
    Returns:
        swot_file_df: a dataframe containing the swot data
    """
    print(f"Collecting SWOT data from {start_date} to {end_date}")
    
    # define the area of interest for swot data
    # min_lon = -125
    # max_lon = -114
    # min_lat = 32
    # max_lat = 42
    # bbox = (min_lon, min_lat, max_lon, max_lat)
    # swot_name = 'SWOT_L2_HR_RiverSP_reach_2.0'

    swot_file_df = pd.DataFrame()

    # load swot passes 
    swot_passes_df = pd.read_csv("/data/ouce-grit/cenv1160/smart_hs/raw_data/swot/mekong_river_basin/swot_passes.csv")
    pass_names = swot_passes_df['pass_number'].values
    # search for swot data with version C
    # pass_names = ['062', '243', '271', '340', '368', '521', '549']
    # pass_names = ['062', '243', '549']

    n_passes = len(pass_names)
    for pass_idx, pass_name in enumerate(pass_names):
        print(f"  [{start_date}] Pass {pass_name} ({pass_idx+1}/{n_passes}) — searching version C...")
        swot_results_version_c = earthaccess.search_data(
            # short_name='SWOT_L2_HR_RiverSP_reach_2.0',
            short_name='SWOT_L2_HR_LakeSP_2.0',
            # short_name='SWOT_L2_HR_RiverSP_node_2.0',
            # bounding_box=bbox,
            temporal=(start_date, end_date),
            granule_name=f'*Lake*_{pass_name}_AS*',
            count=-1
        )
        print(f"  [{start_date}] Pass {pass_name} — version C: {len(swot_results_version_c)} granules. Searching version D...")

        # search for swot data with version D
        swot_results_version_d = earthaccess.search_data(
            # short_name='SWOT_L2_HR_RiverSP_reach_D',
            short_name='SWOT_L2_HR_LakeSP_D',
            # short_name='SWOT_L2_HR_RiverSP_node_D',
            # bounding_box=bbox,
            temporal=(start_date, end_date),
            granule_name=f'*Lake*_{pass_name}_AS*', # Change the region: NA, AS
            count=-1
        )
        print(f"  [{start_date}] Pass {pass_name} — version D: {len(swot_results_version_d)} granules.")

        for swot_result in swot_results_version_c:
            
            # access the 'umm' (Unified Metadata Model) field, which holds
            # the detailed metadata for the granule
            umm_metadata = swot_result.get('umm', {})

            # extract the url from the granuleUR field
            file_name = umm_metadata.get('GranuleUR')
            file_name = file_name[:-5] + ".zip" # get the reach file filename
            # get the url
            url = swot_result.data_links()[0]
            # url = "https://archive.swot.podaac.earthdata.nasa.gov/podaac-swot-ops-cumulus-protected/SWOT_L2_HR_RiverSP_D/" + file_name

            # extract the time
            beginning_datetime_str = umm_metadata.get('TemporalExtent').get('RangeDateTime')['BeginningDateTime']
            beginning_datetime_str = pd.to_datetime(beginning_datetime_str)
            year = beginning_datetime_str.year
            month = beginning_datetime_str.month
            day = beginning_datetime_str.day
            
            ending_datetime_str = umm_metadata.get('TemporalExtent').get('RangeDateTime')['EndingDateTime']
            ending_datetime_str = pd.to_datetime(ending_datetime_str)

            # get the current swot record
            curr_swot_record = pd.DataFrame()
            curr_swot_record['year'] = [year]
            curr_swot_record['month'] = [month]
            curr_swot_record['day'] = [day]
            curr_swot_record['start_date'] = [beginning_datetime_str] 
            curr_swot_record['end_date'] = [ending_datetime_str]
            curr_swot_record['url'] = [url]

            # append it
            swot_file_df = pd.concat([swot_file_df, curr_swot_record], axis = 0, ignore_index = True)

        for swot_result in swot_results_version_d:
            
            # access the 'umm' (Unified Metadata Model) field, which holds
            # the detailed metadata for the granule
            umm_metadata = swot_result.get('umm', {})

            # extract the url from the granuleUR field
            file_name = umm_metadata.get('GranuleUR')
            file_name = file_name[:-5] + ".zip" # get the reach file filename
            # get the url
            url = swot_result.data_links()[0]
            # url = "https://archive.swot.podaac.earthdata.nasa.gov/podaac-swot-ops-cumulus-protected/SWOT_L2_HR_RiverSP_D/" + file_name

            # extract the time
            beginning_datetime_str = umm_metadata.get('TemporalExtent').get('RangeDateTime')['BeginningDateTime']
            beginning_datetime_str = pd.to_datetime(beginning_datetime_str)
            year = beginning_datetime_str.year
            month = beginning_datetime_str.month
            day = beginning_datetime_str.day
            
            ending_datetime_str = umm_metadata.get('TemporalExtent').get('RangeDateTime')['EndingDateTime']
            ending_datetime_str = pd.to_datetime(ending_datetime_str)

            # get the current swot record
            curr_swot_record = pd.DataFrame()
            curr_swot_record['year'] = [year]
            curr_swot_record['month'] = [month]
            curr_swot_record['day'] = [day]
            curr_swot_record['start_date'] = [beginning_datetime_str] 
            curr_swot_record['end_date'] = [ending_datetime_str]
            curr_swot_record['url'] = [url]

            # append it
            swot_file_df = pd.concat([swot_file_df, curr_swot_record], axis = 0, ignore_index = True)

    # save it as a csv file
    # save_folder = r"E:\Project_2025_2026\Smart_hs\raw_data\swot\swot_reach\file_list\california"
    save_folder = r"/data/ouce-grit/cenv1160/smart_hs/raw_data/swot/mekong_river_basin/swot_lakes/file_list"
    out_path = os.path.join(save_folder, f'{start_date}_{end_date}_swot_lake_file_df.csv')
    swot_file_df.to_csv(out_path, index=False)
    print(f"[DONE] {start_date} → {end_date}: {len(swot_file_df)} total records saved to {out_path}")


def run_task(task):
    swot_file_collection(task['start_date'], task['end_date'])
    return


if __name__ == '__main__':

    auth = earthaccess.login(persist=True)
    print(f"Authenticated: {auth.authenticated}")

    # Start from December 2023 to September 2025
    # start_month = datetime(2023, 1, 1)
    # start_month = datetime(2025, 4, 1)
    # start_month = datetime(2025, 11, 1)
    # start_month = datetime(2023, 12, 1)
    start_month = datetime(2025, 12, 1)
    end_month = datetime(2026, 2, 1)  # inclusive: last month processed is end_month

    # Generate list of start_date and end_date pairs
    date_pairs = []
    current_month = start_month

    while current_month <= end_month:
        # Calculate the start and end of the current month
        month_start = current_month
        month_end = current_month + relativedelta(months=1) # - timedelta(days=1)
        
        # Format dates as strings
        start_date = month_start.strftime('%Y-%m-%d')
        end_date = month_end.strftime('%Y-%m-%d')
        
        # Add to the list
        date_pairs.append((start_date, end_date))
        
        # Move to next month
        current_month += relativedelta(months=1)
    
    print(f"\nTotal months to process: {len(date_pairs)}")
    

    process_num = 10 # number of processes
    # Loop through each month
    task_list = []
    for i, (start_date, end_date) in enumerate(date_pairs):
        task = {'start_date': start_date, 'end_date': end_date}
        task_list.append(task)
    
    # use the process
    print(f"Starting pool with {process_num} processes...")
    pool = mp.Pool(processes=process_num)
    pool.map(run_task, task_list)
    pool.close()
    pool.join()
    print("All tasks completed.")


