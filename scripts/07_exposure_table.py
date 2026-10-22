"""Combine pass A (county NLCD totals) and pass B (NFHL zone x NLCD counts) into county exposure tables.

Outputs
  data/processed/exposure_county_year.parquet  county x year (2000..2025, 5-yr): km2 by zone and class
  data/processed/exposure_county.parquet       one row per county: levels, changes, shares, coverage flags
"""
import numpy as np
import pandas as pd

from config import BIG, PROCESSED

PX_KM2 = 900 / 1e6
DEV = ["dev_open", "dev_low", "dev_med", "dev_high"]
DEV_MH = ["dev_med", "dev_high"]
MIN_BASE_KM2 = 0.5  # relative change reported only above this baseline


def load(sub):
    files = sorted((BIG / "exposure" / sub).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def main():
    tot = load("totals")
    nf = load("nfhl")
    t = tot.pivot_table(index=["GEOID", "year"], columns="nlcd", values="pixels", aggfunc="sum", fill_value=0)
    cy = pd.DataFrame(index=t.index)
    cy["county_km2"] = t.sum(axis=1) * PX_KM2
    cy["water_km2"] = t.get("water", 0) * PX_KM2
    cy["dev_km2"] = t[DEV].sum(axis=1) * PX_KM2
    cy["devmh_km2"] = t[DEV_MH].sum(axis=1) * PX_KM2

    if len(nf):
        nf = nf[nf.zone != "unmapped"]  # deliveries already merged per county in 06 --counts
        z = nf.pivot_table(index=["GEOID", "year"], columns=["zone", "nlcd"], values="pixels", aggfunc="sum",
                           fill_value=0)

        def zsum(zone, classes=None):
            if zone not in z.columns.get_level_values(0):
                return pd.Series(0.0, index=z.index)
            s = z[zone]
            cols = [c for c in (classes or s.columns) if c in s.columns]
            return s[cols].sum(axis=1) * PX_KM2

        e = pd.DataFrame(index=z.index)
        e["mapped_km2"] = sum(zsum(k) for k in z.columns.get_level_values(0).unique())
        e["dev_mapped_km2"] = sum(zsum(k, DEV) for k in z.columns.get_level_values(0).unique())
        e["sfha_km2"] = zsum("sfha_a")
        e["sfha_land_km2"] = e.sfha_km2 - zsum("sfha_a", ["water"])
        e["dev_sfha_km2"] = zsum("sfha_a", DEV)
        e["devmh_sfha_km2"] = zsum("sfha_a", DEV_MH)
        e["dev_v_km2"] = zsum("coastal_v", DEV)
        e["dev_x500_km2"] = zsum("x_500", DEV)
        e["dev_levee_km2"] = zsum("x_levee", DEV)
        cy = cy.join(e, how="left")
        meta = nf.groupby("GEOID").agg(eff_median=("eff_median", "first"), n_deliveries=("n_deliveries", "first"),
                                       dfirm=("dfirm", "first"))
    cy = cy.reset_index()
    cy["mapped_share"] = (cy.mapped_km2 / cy.county_km2).clip(upper=1)
    # coverage judged on developed land: unmapped forest, desert or water does not bias exposure counts
    cy["dev_mapped_share"] = (cy.dev_mapped_km2 / cy.dev_km2).clip(upper=1)
    cy.to_parquet(PROCESSED / "exposure_county_year.parquet", index=False)

    w = cy.pivot(index="GEOID", columns="year")
    c = pd.DataFrame(index=w.index)
    y0, y1 = w.columns.get_level_values(1).min(), w.columns.get_level_values(1).max()
    for v in ("dev_sfha_km2", "devmh_sfha_km2", "dev_km2", "dev_x500_km2", "dev_v_km2"):
        c[f"{v}_{y0}"] = w[(v, y0)]
        c[f"{v}_{y1}"] = w[(v, y1)]
        c[f"d_{v}"] = w[(v, y1)] - w[(v, y0)]
    c["sfha_land_km2"] = w[("sfha_land_km2", y0)]
    c["county_km2"] = w[("county_km2", y0)]
    c["mapped_share"] = w[("mapped_share", y0)]
    c["dev_mapped_share"] = w[("dev_mapped_share", y1)]
    c["rel_growth_sfha"] = np.where(c[f"dev_sfha_km2_{y0}"] >= MIN_BASE_KM2,
                                    100 * c.d_dev_sfha_km2 / c[f"dev_sfha_km2_{y0}"], np.nan)
    c["base_zero_flag"] = c[f"dev_sfha_km2_{y0}"] < MIN_BASE_KM2
    # share of new development landing in the SFHA vs the SFHA share of land available for development
    c["new_dev_share_sfha"] = np.where(c.d_dev_km2 > 0, c.d_dev_sfha_km2 / c.d_dev_km2, np.nan)
    avail_sfha = (c.sfha_land_km2 - c[f"dev_sfha_km2_{y0}"]).clip(lower=0)
    avail_all = (c.county_km2 - w[("water_km2", y0)] - c[f"dev_km2_{y0}"]).clip(lower=1e-9)
    c["avail_share_sfha"] = avail_sfha / avail_all
    c["fp_dev_index"] = c.new_dev_share_sfha / c.avail_share_sfha.where(c.avail_share_sfha > 0)
    c["nfhl_covered"] = c.dev_mapped_share >= 0.9
    if len(nf):
        c = c.join(meta)
    gf = load("gfplain")  # GFPLAIN250m geomorphic floodplain: complete coverage, all counties
    if len(gf):
        gd = gf[gf.nlcd.isin(DEV) & (gf.zone == "gfplain")].groupby(["GEOID", "year"]).pixels.sum().unstack() * PX_KM2
        c = c.reindex(c.index.union(gd.index))
        c[f"dev_gfp_km2_{y0}"] = gd.get(y0)
        c[f"dev_gfp_km2_{y1}"] = gd.get(y1)
        c["d_dev_gfp_km2"] = c[f"dev_gfp_km2_{y1}"] - c[f"dev_gfp_km2_{y0}"]
    c.index.name = "GEOID"
    c.reset_index().to_parquet(PROCESSED / "exposure_county.parquet", index=False)
    print(c.describe().T.round(3).to_string())
    print("counties:", len(c), " NFHL covered (>=90% mapped):", int(c.nfhl_covered.sum()))


if __name__ == "__main__":
    main()
