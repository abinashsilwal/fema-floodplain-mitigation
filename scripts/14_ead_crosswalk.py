"""Area-weighted HUC10-to-county crosswalk for EPA modelled riverine expected annual damage (EAD).

The main analysis assigned each HUC10 watershed's EAD to the county containing its centroid (03_analysis.py).
Here each watershed's EAD is split among counties in proportion to the share of the watershed's area inside each
county (USGS Watershed Boundary Dataset, WBDHU10). The +5 C change is recomputed for counties with baseline EAD
of at least US$10,000, as before, and the comparison of high-growth unfunded and funded counties is repeated.

Input: ~/fema_flood_bigdata/wbd/WBD_National_GDB.zip (USGS WBD national geodatabase)
Outputs: data/processed/epa_ead_county_areaweighted.parquet, outputs/tables/exp_ead_crosswalk_check.csv
"""
import importlib

import geopandas as gpd
import numpy as np
import pandas as pd

from config import BIG, PROCESSED, RAW, TAB

E = importlib.import_module("10_exposure_analysis")


def main():
    e = pd.read_excel(RAW / "epa" / "huc10_EAD_outputs.xlsx", sheet_name="output")
    e["huc10"] = e.HUC10.astype("int64").astype(str).str.zfill(10)
    w = gpd.read_file(f"zip://{BIG / 'wbd' / 'WBD_National_GDB.zip'}!WBD_National_GDB.gdb", layer="WBDHU10",
                      columns=["huc10"])
    w = w[w.huc10.isin(e.huc10)].to_crs("EPSG:5070")
    w["huc_area"] = w.geometry.area
    counties = gpd.read_file(PROCESSED / "counties.gpkg")[["GEOID", "geometry"]].to_crs("EPSG:5070")
    x = gpd.overlay(w[["huc10", "huc_area", "geometry"]], counties, how="intersection", keep_geom_type=True)
    x["share"] = x.geometry.area / x.huc_area
    x = x.merge(e[["huc10", "EAD(Base)", "EAD(dT03)", "EAD(dT05)"]], on="huc10")
    for k in ("EAD(Base)", "EAD(dT03)", "EAD(dT05)"):
        x[k] = x[k] * x.share
    cty = x.groupby("GEOID")[["EAD(Base)", "EAD(dT03)", "EAD(dT05)"]].sum(min_count=1)
    cty["ead_chg5_aw"] = 100 * (cty["EAD(dT05)"] - cty["EAD(Base)"]) / cty["EAD(Base)"].where(cty["EAD(Base)"] >= 10_000)
    cty.reset_index().to_parquet(PROCESSED / "epa_ead_county_areaweighted.parquet", index=False)

    c, _ = E.load()
    c["ead_chg5_aw"] = c.index.map(cty.ead_chg5_aw)
    c["ead_base_aw"] = c.index.map(cty["EAD(Base)"])
    t, d = E.typology(c)
    hg = d[d.exp_class == "High growth"]
    out = {"huc10_matched": int(w.huc10.nunique()), "huc10_epa": int(e.huc10.nunique()),
           "counties_with_ead_aw": int(cty["EAD(Base)"].notna().sum()),
           "spearman_base_centroid_vs_aw": c["EAD(Base)"].corr(c.ead_base_aw, method="spearman")
           if "EAD(Base)" in c else np.nan,
           "spearman_chg5_centroid_vs_aw": c.ead_chg5.corr(c.ead_chg5_aw, method="spearman")}
    ua, fa = hg.loc[~hg.funded, "ead_chg5_aw"].dropna(), hg.loc[hg.funded, "ead_chg5_aw"].dropna()
    rng, pool = np.random.default_rng(1), np.r_[ua.values, fa.values]
    perm = np.array([np.median(z_[:len(ua)]) - np.median(z_[len(ua):]) for z_ in (rng.permutation(pool) for _ in range(5000))])
    obs = ua.median() - fa.median()
    out.update({"aw_n_unf": len(ua), "aw_n_fun": len(fa), "aw_unf_median": ua.median(), "aw_fun_median": fa.median(),
                "aw_unf_q25": ua.quantile(.25), "aw_unf_q75": ua.quantile(.75), "aw_fun_q25": fa.quantile(.25),
                "aw_fun_q75": fa.quantile(.75), "aw_perm_p": np.mean(np.abs(perm) >= abs(obs))})
    pd.Series(out).to_csv(TAB / "exp_ead_crosswalk_check.csv")
    print(pd.Series(out).round(3).to_string())


if __name__ == "__main__":
    main()
