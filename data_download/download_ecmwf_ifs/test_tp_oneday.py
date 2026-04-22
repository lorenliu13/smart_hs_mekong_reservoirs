"""
Minimal test: download total precipitation (tp) for a single day (2023-01-01)
to validate the MARS request before running the full download.
"""

from pathlib import Path
from ecmwfapi import ECMWFService

OUTDIR = Path("/data/ouce-grit/cenv1160/smart_hs/raw_data/ecmwf_ifs/hres")
OUTDIR.mkdir(parents=True, exist_ok=True)

outfile = OUTDIR / "test_tp_2023-01-01.grib2"

server = ECMWFService("mars")

print(f"Submitting test request → {outfile}")

server.execute({
    "class"   : "od",
    "expver"  : "1",
    "stream"  : "oper",
    "type"    : "fc",
    "date"    : "2023-01-01",
    "time"    : "00",
    "step"    : "0/1/2/3/6/12/24",    # small subset of steps (explicit values)
    "levtype" : "sfc",
    "param"   : "228.128",            # tp only
    # "area"    : "34/89/7/112",
    # "grid"    : "0.1/0.1",
}, str(outfile))

print(f"Done → {outfile}  ({outfile.stat().st_size / 1e6:.1f} MB)")
