# Unzip downloaded swot files
# Collecting SWOT data from the USGS website
import os
import pandas as pd
import multiprocessing as mp
import zipfile
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass

def run_task(task):

    start_date = task['start_date']
    end_date = task['end_date']

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing {start_date} to {end_date}")

    # get the year
    year = start_date[:4] # get the year from the start date

    # download_folder = rf"/home/yliu2232/smart_hs/raw_data/swot/swot_reach/california/{year}"
    # download_folder = rf"/data/ouce-grit/cenv1160/smart_hs/raw_data/swot/swot_node/sacramento/{year}"
    download_folder = rf"/data/ouce-grit/cenv1160/smart_hs/raw_data/swot/mekong_river_basin/swot_lakes/{year}"
    create_folder(download_folder)

    # swot_file_df = pd.read_csv(fr"/home/yliu2232/smart_hs/raw_data/swot/swot_reach/california/file_list/{start_date}_{end_date}_swot_file_df.csv")
    # swot_file_df = pd.read_csv(fr"/data/ouce-grit/cenv1160/smart_hs/raw_data/swot/swot_node/file_list/{start_date}_{end_date}_swot_node_file_df.csv")
    swot_file_df = pd.read_csv(fr"/data/ouce-grit/cenv1160/smart_hs/raw_data/swot/mekong_river_basin/swot_lakes/file_list/{start_date}_{end_date}_swot_lake_file_df.csv")

    total_files = swot_file_df.shape[0]
    print(f"  Found {total_files} files to unzip for {start_date} ~ {end_date}")

    for index in range(total_files):
        # extract the filename from the url
        curr_url = swot_file_df['url'].values[index]
        filename = os.path.basename(curr_url)
        # get the zip file path
        zip_file_path = download_folder + "/" + filename
        # unzip the file into a folder with the same name as the zip file
        unzip_folder_path = download_folder + "/" + filename[:-4]
        # unzip the file
        if not os.path.exists(zip_file_path):
            print(f"  [{index+1}/{total_files}] SKIPPED (missing): {filename}")
            continue
        print(f"  [{index+1}/{total_files}] Unzipping {filename}")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_folder_path)

    print(f"  Done: {start_date} ~ {end_date}")


if __name__ == "__main__":

    # Start from December 2023 to September 2025
    start_month = datetime(2023, 12, 1)
    end_month = datetime(2025, 12, 1)

    # Generate list of start_date and end_date pairs
    date_pairs = []
    current_month = start_month
    
    while current_month < end_month:
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
    pool = mp.Pool(processes=process_num)
    pool.map(run_task, task_list)