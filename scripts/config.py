"""Shared paths and constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"

# Large rasters and vectors (NLCD, NFHL, WBD) are kept outside the repository.
BIG = Path(os.environ.get("FLOOD_BIGDATA", Path.home() / "fema_flood_bigdata"))

# Full OpenFEMA NFIP Redacted Policies v2 CSV (about 29 GB), downloaded separately.
NFIP_POLICIES_CSV = Path(os.environ.get("NFIP_POLICIES_CSV", RAW / "openfema" / "FimaNfipPoliciesV2.csv"))

for p in (RAW, INTERIM, PROCESSED, FIG, TAB, BIG):
    p.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "floodplain-mitigation-research/1.0"}

# CONUS = 48 states + DC (state FIPS)
NON_CONUS = {"02", "15", "60", "66", "69", "72", "78"}

PRICE_YEAR = 2024  # constant-dollar base year
STUDY_START, STUDY_END = 1996, 2024  # Storm Events damage reporting is consistent from 1996
