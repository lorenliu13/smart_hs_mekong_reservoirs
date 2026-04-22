"""
Minimal test: download a single ENS variable on a single date, global extent.

Edit the four constants below, then run:
    python test_download_single.py
"""

from pathlib import Path
from ecmwfapi import ECMWFService

# ---------------------------------------------------------------------------
# Edit these
# ---------------------------------------------------------------------------
DATE   = "2026-01-01"       # single forecast date (YYYY-MM-DD)
PARAM  = "228.128"          # GRIB param code: 228.128 = tp (total precipitation)
MEMBER = "pf"               # "cf" for control, or "pf" for perturbed member 1
NUMBER = "1"                # only used when MEMBER == "pf"
OUTFILE = Path("test_ens_single.grib2")
# ---------------------------------------------------------------------------

server = ECMWFService("mars")

request = {
    "class"  : "od",
    "expver" : "1",
    "stream" : "enfo",
    "type"   : MEMBER,
    "date"   : DATE,
    "time"   : "00",
    "step"   : "0/6/12/24",
    "levtype": "sfc",
    "param"  : PARAM,
}

if MEMBER == "pf":
    request["number"] = NUMBER

print(f"[INFO] Submitting MARS request: {request}")
server.execute(request, str(OUTFILE))
print(f"[DONE] Saved → {OUTFILE.resolve()}")
