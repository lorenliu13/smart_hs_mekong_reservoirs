"""
Download DAHITI water surface elevation time series for the Great Mekong Area.

Dataset   : DAHITI v2 (Database for Hydrological Time Series of Inland Waters)
            https://dahiti.dgfi.tum.de/api/v2/
Region    : Great Mekong Area  (N=34, W=89, S=7, E=112)
Data type : Water surface elevation (WSE) from satellite altimetry
Format    : CSV, one file per DAHITI target
Output    : OUTDIR/dahiti_mekong_targets.csv          — target metadata index
              Columns: dahiti_id, target_name, latitude, longitude, country, data_points, ...
            OUTDIR/water_level/dahiti_{dahiti_id}.csv — WSE time series per target
              Columns: date, water_level_m, uncertainty_m, satellite

Notes
-----
Authentication requires a DAHITI API key (v2). Register for free at:
    https://dahiti.dgfi.tum.de
The key can be set via the DAHITI_API_KEY constant at the top of this script
or via the environment variable DAHITI_API_KEY.

The list-targets endpoint is called once to discover all targets within the
bounding box. Time series for each target are then downloaded in parallel
using a thread pool (HTTP I/O bound — threads are appropriate here).

The script is idempotent: re-running skips files that already exist.

Requires:
    pip install requests pandas
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "https://dahiti.dgfi.tum.de/api/v2/"

# Bounding box: Great Mekong Area (same coordinates as ERA5-Land and ECMWF downloads)
BBOX = {
    "min_lat":  7.0,   # S
    "max_lat": 34.0,   # N
    "min_lon": 89.0,   # W
    "max_lon": 112.0,  # E
}

# Parallel download workers — keep low to avoid 429 rate-limit errors
MAX_WORKERS = 2

# Retry settings for 429 Too Many Requests responses
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 10  # seconds; delay doubles each attempt

# Request timeout in seconds
REQUEST_TIMEOUT = 60

# Output root directory
# OUTDIR = Path("/data/ouce-grit/cenv1160/smart_hs/raw_data/dahiti")
OUTDIR = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\dahiti")

# Metadata index filename (written to OUTDIR root)
TARGETS_CSV = "dahiti_mekong_targets.csv"

# Sub-directory for per-target time series files
WATER_LEVEL_SUBDIR = "water_level"

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


DAHITI_API_KEY = "D3252F74BCAAE0B9BCC3570033E675DE4899B7FE81519D8AF290C8BBE98AA536"


def get_api_key() -> str:
    """
    Return the DAHITI API key. Falls back to the DAHITI_API_KEY environment
    variable if the module-level constant is not set.
    """
    key = DAHITI_API_KEY or os.environ.get("DAHITI_API_KEY")
    if not key:
        print(
            "[ERROR] No DAHITI API key found.\n"
            "        Set DAHITI_API_KEY at the top of this script or as an\n"
            "        environment variable:  export DAHITI_API_KEY=your_key_here"
        )
        sys.exit(1)
    return key


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def list_targets(api_key: str, bbox: dict) -> list[dict]:
    """
    Call the DAHITI v2 list-targets endpoint and return all targets within
    the given bounding box.

    Parameters
    ----------
    api_key : str
        DAHITI API key.
    bbox : dict
        Keys: min_lat, max_lat, min_lon, max_lon.

    Returns
    -------
    list[dict]
        One dict per target, containing at minimum:
        dahiti_id, target_name, latitude, longitude, country, data_points.
    """
    url = API_BASE_URL + "list-targets/"
    payload = {"api_key": api_key, **bbox}
    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()["data"]


def save_targets_csv(targets: list[dict], outpath: Path) -> None:
    """Write target metadata to a CSV file. Skips if the file already exists."""
    if outpath.exists():
        print(f"[SKIP]  {outpath.name} already exists.")
        return
    pd.DataFrame(targets).to_csv(outpath, index=False)
    print(f"[DONE]  Saved target index → {outpath}")


def build_wl_path(outdir: Path, dahiti_id) -> Path:
    """Return the output path for a single target's water level CSV."""
    return outdir / WATER_LEVEL_SUBDIR / f"dahiti_{dahiti_id}.csv"


def download_water_level(
    api_key: str,
    dahiti_id,
    target_name: str,
    outpath: Path,
) -> str:
    """
    Download water level time series for one DAHITI target and save to CSV.

    Parameters
    ----------
    api_key     : DAHITI API key.
    dahiti_id   : Numeric DAHITI target ID.
    target_name : Human-readable name (used only for log messages).
    outpath     : Destination CSV file path.

    Returns
    -------
    str
        One of "skipped", "ok", or "failed: <reason>".
    """
    if outpath.exists():
        print(f"[SKIP]  {outpath.name}")
        return "skipped"

    try:
        url = API_BASE_URL + "download-water-level/"
        payload = {"api_key": api_key, "dahiti_id": dahiti_id}

        # Retry on 429 with exponential backoff
        for attempt in range(MAX_RETRIES):
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            if response.status_code == 429 and attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                print(f"[WAIT]  ID {dahiti_id} rate-limited; retrying in {wait}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            response.raise_for_status()
            break
        resp_json = response.json()

        # The API wraps records in a "data" key; fall back to the raw response
        # if it is already a list (defensive).
        records = resp_json["data"] if isinstance(resp_json, dict) else resp_json
        df = pd.DataFrame(records)

        # Normalise column names: the API may return slightly different field names.
        # The v2 API uses "data" for the satellite/mission source column.
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ("water_level", "wse", "height"):
                rename_map[col] = "water_level_m"
            elif col_lower in ("uncertainty", "sigma", "error", "wse_u", "std"):
                rename_map[col] = "uncertainty_m"
            elif col_lower in ("date", "time", "datetime"):
                rename_map[col] = "date"
            elif col_lower in ("satellite", "sensor", "mission", "data"):
                rename_map[col] = "satellite"
        df = df.rename(columns=rename_map)

        # Ensure required columns exist; fill missing ones with NaN
        for col in ("date", "water_level_m", "uncertainty_m", "satellite"):
            if col not in df.columns:
                df[col] = float("nan")

        # Normalise date format
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        df[["date", "water_level_m", "uncertainty_m", "satellite"]].to_csv(
            outpath, index=False
        )
        print(f"[DONE]  {outpath.name}  ({len(df)} records)")
        return "ok"

    except Exception as e:
        print(f"[FAIL]  ID {dahiti_id} ({target_name}): {e}")
        return f"failed: {e}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def download_all_water_levels(
    api_key: str,
    targets: list[dict],
    outdir: Path,
    max_workers: int,
) -> list[str]:
    """
    Download water level time series for all targets in parallel.

    Returns
    -------
    list[str]
        Failure strings for each target that could not be downloaded (empty if
        all succeeded).
    """
    wl_dir = outdir / WATER_LEVEL_SUBDIR
    wl_dir.mkdir(parents=True, exist_ok=True)

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for target in targets:
            dahiti_id = target.get("dahiti_id") or target.get("id")
            target_name = target.get("target_name") or target.get("name", str(dahiti_id))
            outpath = build_wl_path(outdir, dahiti_id)
            future = pool.submit(
                download_water_level, api_key, dahiti_id, target_name, outpath
            )
            futures[future] = (dahiti_id, target_name)

        failed = []
        for future in as_completed(futures):
            dahiti_id, target_name = futures[future]
            result = future.result()
            if result.startswith("failed"):
                failed.append(f"ID {dahiti_id} ({target_name}): {result}")

    return failed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DAHITI WSE time series for the Great Mekong Area."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=OUTDIR,
        help="Output root directory (default: cluster path in script)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Parallel download threads (default: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only fetch and save the target list; skip water level downloads",
    )
    args = parser.parse_args()

    api_key = get_api_key()  # fail fast before touching the filesystem

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("-" * 60)
    print("DAHITI download — Great Mekong Area")
    print(f"  Output dir : {outdir}")
    print(
        f"  Bounding box: N={BBOX['max_lat']}, S={BBOX['min_lat']}, "
        f"W={BBOX['min_lon']}, E={BBOX['max_lon']}"
    )
    if not args.skip_download:
        print(f"  Workers    : {args.max_workers}")
    print("-" * 60)

    print("[INFO]  Fetching target list from DAHITI API...")
    targets = list_targets(api_key, BBOX)

    if not targets:
        print(
            "[WARN]  No targets returned for the given bounding box.\n"
            "        Check that your API key is valid and the bbox coordinates are correct."
        )
        sys.exit(0)

    print(f"[INFO]  Found {len(targets)} target(s) in bounding box.")

    save_targets_csv(targets, outdir / TARGETS_CSV)

    if args.skip_download:
        print(f"\nFinished (target list only). Index saved → {outdir / TARGETS_CSV}")
        return

    print(
        f"[INFO]  Downloading water level time series "
        f"({len(targets)} targets, {args.max_workers} threads)..."
    )
    all_failed = download_all_water_levels(api_key, targets, outdir, args.max_workers)

    print(f"\nFinished. {len(targets)} target(s) processed → {outdir}")
    if all_failed:
        print(f"Failed ({len(all_failed)}):")
        for entry in all_failed:
            print(f"  {entry}")
    else:
        print("All targets downloaded successfully.")


if __name__ == "__main__":
    main()
