#!/usr/bin/env python3
"""Download the public NHANES XPT components required for the analysis."""

from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

CYCLES = [
    (1999, ""), (2001, "_B"), (2003, "_C"), (2005, "_D"),
    (2007, "_E"), (2009, "_F"), (2011, "_G"), (2013, "_H"),
    (2015, "_I"), (2017, "_J"),
]
COMPONENTS = ("DEMO", "RXQ_RX", "BMX", "SMQ")
PROSTATE_FILES = {
    1999: "KIQ.xpt",
    2001: "KIQ_P_B.xpt",
    2003: "KIQ_P_C.xpt",
    2005: "KIQ_P_D.xpt",
    2007: "KIQ_P_E.xpt",
}


def download(year: int, filename: str) -> None:
    destination = OUT / filename
    if destination.exists() and destination.stat().st_size:
        print(f"Present: {destination.name}")
        return
    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{filename}"
    print(f"Downloading {url}")
    try:
        urlretrieve(url, destination)
    except HTTPError:
        destination.unlink(missing_ok=True)
        raise


for year, suffix in CYCLES:
    for component in COMPONENTS:
        download(year, f"{component}{suffix}.xpt")
    if year in PROSTATE_FILES:
        download(year, PROSTATE_FILES[year])
