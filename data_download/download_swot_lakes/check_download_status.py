# Check if all SWOT lake files have been downloaded
import os
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

base_dir = "/data/ouce-grit/cenv1160/smart_hs/raw_data/swot/mekong_river_basin/swot_lakes"
file_list_dir = os.path.join(base_dir, "file_list")

start_month = datetime(2025, 12, 1)
end_month = datetime(2026, 2, 1)

total_expected = 0
total_downloaded = 0
total_missing = 0
missing_files = []

current_month = start_month
while current_month <= end_month:
    month_end = current_month + relativedelta(months=1)
    start_date = current_month.strftime('%Y-%m-%d')
    end_date = month_end.strftime('%Y-%m-%d')
    year = start_date[:4]

    csv_path = os.path.join(file_list_dir, f"{start_date}_{end_date}_swot_lake_file_df.csv")
    download_folder = os.path.join(base_dir, year)

    if not os.path.exists(csv_path):
        print(f"[{start_date} → {end_date}] MISSING file list CSV: {csv_path}")
        current_month += relativedelta(months=1)
        continue

    df = pd.read_csv(csv_path)
    expected = df.shape[0]
    downloaded = 0
    month_missing = []

    for url in df['url'].values:
        filename = os.path.basename(url)
        file_path = os.path.join(download_folder, filename)
        if os.path.exists(file_path):
            downloaded += 1
        else:
            month_missing.append(filename)

    total_expected += expected
    total_downloaded += downloaded
    total_missing += len(month_missing)
    missing_files.extend(month_missing)

    status = "OK" if len(month_missing) == 0 else f"MISSING {len(month_missing)}"
    print(f"[{start_date} → {end_date}] {downloaded}/{expected} downloaded  [{status}]")

    current_month += relativedelta(months=1)

print(f"\n{'='*50}")
print(f"Total expected : {total_expected}")
print(f"Total downloaded: {total_downloaded}")
print(f"Total missing  : {total_missing}")

if missing_files:
    print(f"\nMissing files:")
    for f in missing_files:
        print(f"  {f}")
else:
    print("\nAll files downloaded successfully.")
