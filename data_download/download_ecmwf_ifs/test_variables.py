"""
Test MARS retrievals for each variable in VARIABLES.

Submits a minimal request per variable (1 day, 2 steps) to verify
that the param code is accepted and data is returned.

Usage:
    python test_variables.py

Output files are written to TEST_OUTDIR and kept for inspection.
Failed variables are reported at the end.
"""

from pathlib import Path

from ecmwfapi import ECMWFService

# from download_hres_mekong import AREA, GRID, VARIABLES

# Mapping: variable name → GRIB param code
VARIABLES = {
    "tp"   : "228.128",  # total precipitation
    "2t"   : "167.128",  # 2m temperature
    "2d"   : "168.128",  # 2m dewpoint
    "sp"   : "134.128",  # surface pressure
    "10u"  : "165.128",  # 10m U wind
    "10v"  : "166.128",  # 10m V wind
    "ssrd" : "169.128",  # surface solar radiation downwards
    "strd" : "175.128",  # surface thermal radiation downwards
}

AREA   = "34/89/7/112"                    # N/W/S/E — full Mekong River Basin
GRID   = "0.1/0.1"                        # ~11 km regular lat/lon

# Local laptop output directory
TEST_OUTDIR = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\ecmwf_ifs\hres\test_output")

# Minimal request — 1 forecast date, 2 steps only
TEST_DATE  = "2024-01-01"
TEST_STEPS = "0/6/12/18/24"


def test_variable(server: ECMWFService, var_name: str, param_code: str) -> tuple[bool, str]:
    outfile = TEST_OUTDIR / f"test_{var_name}.grib2"
    try:
        server.execute({
            "class"   : "od",
            "expver"  : "1",
            "stream"  : "oper",
            "type"    : "fc",
            "date"    : TEST_DATE,
            "time"    : "00",
            "step"    : TEST_STEPS,
            "levtype" : "sfc",
            "param"   : param_code,
            "area"    : AREA,
            "grid"    : GRID,
        }, str(outfile))

        if outfile.exists() and outfile.stat().st_size > 0:
            size_kb = outfile.stat().st_size / 1024
            return True, f"OK  ({size_kb:.1f} KB → {outfile.name})"
        else:
            return False, "Empty or missing output file"

    except Exception as e:
        return False, str(e)


def main() -> None:
    TEST_OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {TEST_OUTDIR}\n")

    server = ECMWFService("mars")

    results: dict[str, tuple[bool, str]] = {}

    for var_name, param_code in VARIABLES.items():
        print(f"[TEST] {var_name:5s} ({param_code}) ...", flush=True)
        ok, msg = test_variable(server, var_name, param_code)
        results[var_name] = (ok, msg)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {msg}")

    print("\n--- Summary ---")
    passed = [v for v, (ok, _) in results.items() if ok]
    failed = [(v, msg) for v, (ok, msg) in results.items() if not ok]

    print(f"Passed ({len(passed)}): {', '.join(passed) if passed else 'none'}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for var_name, msg in failed:
            print(f"  {var_name}: {msg}")
    else:
        print("All variables OK.")


if __name__ == "__main__":
    main()
