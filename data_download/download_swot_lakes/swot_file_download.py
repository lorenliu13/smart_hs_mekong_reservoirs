# Collecting SWOT data from the NASA website
import os
import earthaccess
import pandas as pd
import multiprocessing as mp
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import requests

def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass

auth = earthaccess.login(persist=True) # authenticates with the Earthdata system
print(auth.authenticated)


def download_url(save_file_path, curr_url, filename):    
    try:
        # print(f'Starting download of {filename}')
    
        # send an http get request to the url
        response = requests.get(curr_url, stream=True)
    
        # raise an exception if the request was unsuccessful
        response.raise_for_status()
    
        # open the file in bindary write mode
        with open(save_file_path, 'wb') as f: # opens a local file with the filename
            # iterate over the response content in chunks
            # reads the file in chunks of 8192 bytes and writes each chunk to the local file
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
        print(f"Successfully downloaded '{filename}'")
    
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print("This might be a protected link requiring authentication. You may need to use credentials.")
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")


def run_task(task):

    start_date = task['start_date']
    end_date = task['end_date']

    print(f"\n[{start_date} → {end_date}] Starting task (PID {os.getpid()})")

    # get the year
    year = start_date[:4] # get the year from the start date

    download_folder = rf"/data/ouce-grit/cenv1160/smart_hs/raw_data/swot/mekong_river_basin/swot_lakes/{year}"
    create_folder(download_folder)

    csv_path = fr"/data/ouce-grit/cenv1160/smart_hs/raw_data/swot/mekong_river_basin/swot_lakes/file_list/{start_date}_{end_date}_swot_lake_file_df.csv"
    swot_file_df = pd.read_csv(csv_path)
    total = swot_file_df.shape[0]
    print(f"[{start_date} → {end_date}] {total} files to download → {download_folder}")

    for index in range(total):
        # extract the filename from the url
        curr_url = swot_file_df['url'].values[index]
        filename = os.path.basename(curr_url)
        save_file_path = download_folder + "/" + filename

        if os.path.exists(save_file_path):
            print(f"[{start_date} → {end_date}] ({index + 1}/{total}) SKIPPED (already exists): {filename}")
            continue
        print(f"[{start_date} → {end_date}] ({index + 1}/{total}) Downloading: {filename}")
        download_url(save_file_path, curr_url, filename)

    print(f"[{start_date} → {end_date}] Task complete.")


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
    for s, e in date_pairs:
        print(f"  {s} → {e}")

    process_num = 5 # number of processes
    print(f"\nStarting download pool with {process_num} parallel processes...")

    # Loop through each month
    task_list = []
    for i, (start_date, end_date) in enumerate(date_pairs):
        task = {'start_date': start_date, 'end_date': end_date}
        task_list.append(task)

    # use the process
    pool = mp.Pool(processes=process_num)
    pool.map(run_task, task_list)
    pool.close()
    pool.join()

    print("\nAll tasks complete.")