import pandas as pd
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

SWOT_FILL_VALUE = -999_999_999_999.0

# --- Config ---
start_month = datetime(2023, 12, 1)
end_month   = datetime(2026, 2, 1)  # inclusive

monthly_csv_folder = "/data/ouce-grit/cenv1160/smart_hs/processed_data/swot/great_mekong_river_basin/lakes"
save_path = os.path.join(
    monthly_csv_folder,
    f"full_swot_lake_df_{start_month.strftime('%Y_%m')}_{end_month.strftime('%Y_%m')}.csv"
)

# --- Build list of expected per-month CSV paths ---
month_save_paths = []
current = start_month
while current <= end_month:
    month_label = current.strftime('%Y-%m')
    month_save_paths.append(os.path.join(monthly_csv_folder, f"swot_lake_df_{month_label}.csv"))
    current += relativedelta(months=1)

print(f"Merging {len(month_save_paths)} monthly CSVs into {save_path}")

# --- Concatenate ---
monthly_dfs = []
for month_save_path in month_save_paths:
    try:
        month_df = pd.read_csv(month_save_path)
    except FileNotFoundError:
        print(f"  SKIPPED (missing): {month_save_path}")
        continue

    # Remove rows where WSE is the SWOT fill/no-data value
    month_df = month_df[month_df['wse'] != SWOT_FILL_VALUE]

    # Parse time string and add plain date column
    month_df['time_str'] = pd.to_datetime(month_df['time_str'])
    month_df['date'] = month_df['time_str'].dt.date

    print(f"  {os.path.basename(month_save_path)}: {len(month_df)} rows")
    monthly_dfs.append(month_df)

full_df = pd.concat(monthly_dfs, ignore_index=True)
full_df.to_csv(save_path, index=False)
print(f"\nSaved {len(full_df)} rows to {save_path}")
