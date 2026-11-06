"""Aggregate the OpenFEMA NFIP Policies v2 CSV (read in place, ~28.6 GB) to
county x policy-effective year x construction year x flood-zone class (sum of policyCount).

The CSV is read in place from the path in config.NFIP_POLICIES_CSV (set the NFIP_POLICIES_CSV environment
variable to point elsewhere).
Output: data/processed/nfip_policies_agg.parquet
"""
import time

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

from config import NFIP_POLICIES_CSV, PROCESSED

SRC = NFIP_POLICIES_CSV
COLS = ["countyCode", "policyEffectiveDate", "originalConstructionDate", "floodZoneCurrent", "ratedFloodZone",
        "occupancyType", "policyCount"]


def zone_class(z):
    z = z.fillna("").str.upper().str.strip()
    out = pd.Series("other", index=z.index)
    out[z.str.match(r"^(X|B|C)")] = "outside"
    out[z.str.match(r"^V")] = "coastal_v"
    out[z.str.match(r"^A")] = "sfha_a"
    return out


def main():
    t0 = time.time()
    opts = pv.ConvertOptions(include_columns=COLS, column_types={c: pa.string() for c in COLS})
    reader = pv.open_csv(SRC, read_options=pv.ReadOptions(block_size=1 << 27), convert_options=opts)
    parts, n = [], 0
    for batch in reader:
        d = batch.to_pandas()
        n += len(d)
        zc = d.floodZoneCurrent.where(d.floodZoneCurrent.fillna("") != "", d.ratedFloodZone)
        g = pd.DataFrame({
            "GEOID": d.countyCode.str.zfill(5),
            "pol_year": pd.to_numeric(d.policyEffectiveDate.str[:4], errors="coerce"),
            "con_year": pd.to_numeric(d.originalConstructionDate.str[:4], errors="coerce"),
            "zone": zone_class(zc),
            "occ": pd.to_numeric(d.occupancyType, errors="coerce"),
            "policies": pd.to_numeric(d.policyCount, errors="coerce").fillna(1),
        })
        parts.append(g.groupby(["GEOID", "pol_year", "con_year", "zone", "occ"], dropna=False).policies.sum())
        if len(parts) >= 40:  # keep memory bounded
            parts = [pd.concat(parts).groupby(level=[0, 1, 2, 3, 4], dropna=False).sum()]
        if n % 5_000_000 < len(d):
            print(f"{n:,} rows, {time.time()-t0:,.0f}s", flush=True)
    out = pd.concat(parts).groupby(level=[0, 1, 2, 3, 4], dropna=False).sum().reset_index()
    out.to_parquet(PROCESSED / "nfip_policies_agg.parquet", index=False)
    print(f"done {n:,} rows -> {len(out):,} groups in {time.time()-t0:,.0f}s", flush=True)


if __name__ == "__main__":
    main()
