"""Aggregate the OpenFEMA NFIP Policies v2 CSV (read in place, ~28.6 GB) to
county x ZIP code x policy-effective year x construction period x flood-zone class (sum of policyCount),
for the within-county (ZIP-level) test of where HMA mitigation goes.

Same source and zone rules as 08_nfip_policies.py. Policy years 2016-2024 only.
Output: data/processed/nfip_policies_zip.parquet
"""
import importlib
import time

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

from config import PROCESSED

p08 = importlib.import_module("08_nfip_policies")
COLS = ["countyCode", "reportedZipCode", "policyEffectiveDate", "originalConstructionDate", "floodZoneCurrent",
        "ratedFloodZone", "policyCount"]
KEYS = ["GEOID", "zip", "pol_year", "built_2001", "zone"]


def main():
    t0 = time.time()
    opts = pv.ConvertOptions(include_columns=COLS, column_types={c: pa.string() for c in COLS})
    reader = pv.open_csv(p08.SRC, read_options=pv.ReadOptions(block_size=1 << 27), convert_options=opts)
    parts, n = [], 0
    for batch in reader:
        d = batch.to_pandas()
        n += len(d)
        pol_year = pd.to_numeric(d.policyEffectiveDate.str[:4], errors="coerce")
        d = d[pol_year.between(2016, 2024)]
        zc = d.floodZoneCurrent.where(d.floodZoneCurrent.fillna("") != "", d.ratedFloodZone)
        con = pd.to_numeric(d.originalConstructionDate.str[:4], errors="coerce")
        g = pd.DataFrame({
            "GEOID": d.countyCode.str.zfill(5),
            "zip": d.reportedZipCode.str[:5],
            "pol_year": pol_year[d.index],
            "built_2001": (con >= 2001).astype("int8").where(con.notna()),
            "zone": p08.zone_class(zc),
            "policies": pd.to_numeric(d.policyCount, errors="coerce").fillna(1),
        })
        parts.append(g.groupby(KEYS, dropna=False).policies.sum())
        if len(parts) >= 40:  # keep memory bounded
            parts = [pd.concat(parts).groupby(level=list(range(len(KEYS))), dropna=False).sum()]
        if n % 5_000_000 < len(batch):
            print(f"{n:,} rows, {time.time()-t0:,.0f}s", flush=True)
    out = pd.concat(parts).groupby(level=list(range(len(KEYS))), dropna=False).sum().reset_index()
    out.to_parquet(PROCESSED / "nfip_policies_zip.parquet", index=False)
    print(f"done {n:,} rows -> {len(out):,} groups in {time.time()-t0:,.0f}s", flush=True)


if __name__ == "__main__":
    main()
