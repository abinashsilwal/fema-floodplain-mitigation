"""Who and where HMA flood mitigation reaches below the county level.

1. Construction year of HMA-mitigated buildings (Project Site Inventories) compared with the construction year
   of insured A-zone units (NFIP policies in effect in 2024) and of A-zone NFIP claims (2016-2024).
2. Within-county ZIP-code test: among ZIP codes in the same county, are flood-mitigation sites located where
   newer insured floodplain buildings are, or where older buildings and claims are? Linear probability model
   with county fixed effects (within transformation), standard errors clustered by county.

Site inventories are complete only for subapplications in FEMA GO (fiscal year 2020 onward) and legacy records
from the 1990s, so both parts use subapplications submitted in fiscal years 2020-2025.
Outputs: outputs/tables/wc_year_built.csv, wc_zip_model.csv, wc_zip_summary.csv
"""
import importlib

import numpy as np
import pandas as pd

from config import PROCESSED, ROOT

RAW = ROOT / "data" / "raw"
TAB = ROOT / "outputs" / "tables"
ACT = ["Acquisition", "Elevation", "Relocation", "Mitigation reconstruction", "Floodproofing"]
BUILDING_ACT = ["Acquisition", "Elevation", "Relocation", "Mitigation reconstruction"]
FY0 = 2020
M09 = importlib.import_module("09_mitigation_loss_tables")  # claims loader, A-zone rule


def sites():
    s = pd.read_parquet(RAW / "openfema" / "HmaSubapplicationsProjectSiteInventories_v1_paged.parquet")
    h = pd.read_parquet(RAW / "openfema" / "HmaSubapplications_v2.parquet",
                        columns=["subapplicationIdentifier", "program", "fiscalYear", "status", "totalObligatedAmount"])
    s = s[s.primaryActivity.isin(ACT) & (s.projectSiteInventoryType == "Building") & (s.structureType == "Residential")]  # residential buildings only
    s = s.merge(h, left_on="subapplicationId", right_on="subapplicationIdentifier", how="inner")
    s = s[s.fiscalYear >= FY0].copy()
    s["funded"] = s.totalObligatedAmount.fillna(0) > 0
    s["zip"] = s.zipcode.str[:5]
    s["yb"] = s.yearBuilt.where(s.yearBuilt.between(1800, 2026))
    return s


def year_built(s):
    pol = pd.read_parquet(PROCESSED / "nfip_policies_agg.parquet")
    pol = pol[(pol.zone == "sfha_a") & (pol.pol_year == 2024) & pol.con_year.between(1800, 2024)]
    cl = pd.read_parquet(RAW / "openfema" / "NfipClaims_v3_paged.parquet",
                         columns=["yearOfLoss", "ratedFloodZone", "floodZoneCurrent", "originalConstructionDate"])
    cl = cl[cl.yearOfLoss.between(2016, 2024)]
    z = cl.ratedFloodZone.fillna(cl.floodZoneCurrent).fillna("").str.upper()
    cl = cl[z.str.match(M09.SFHA)]
    cyr = pd.to_datetime(cl.originalConstructionDate, errors="coerce").dt.year.dropna()
    b = s[s.primaryActivity.isin(BUILDING_ACT) & s.funded & s.yb.notna()]
    rows = []
    for cut in [1975, 1990, 2001, 2010]:
        rows.append({"built_from": cut,
                     "insured_units_pct": 100 * pol.loc[pol.con_year >= cut, "policies"].sum() / pol.policies.sum(),
                     "claims_pct": 100 * (cyr >= cut).mean(),
                     "mitigated_pct": 100 * (b.yb >= cut).mean()})
    out = pd.DataFrame(rows)
    out["n_mitigated"], out["n_claims"], out["insured_units"] = len(b), len(cyr), pol.policies.sum()
    out["mitigated_known_share"] = s[s.primaryActivity.isin(BUILDING_ACT) & s.funded].yb.notna().mean()
    out["mitigated_median_year"] = b.yb.median()
    out.to_csv(TAB / "wc_year_built.csv", index=False)
    print(out.round(1).to_string())


def zip_county():
    """Each ZIP code is assigned to the county holding most of its A-zone policies (2024)."""
    p = pd.read_parquet(PROCESSED / "nfip_policies_zip.parquet")
    p = p[(p.pol_year == 2024) & (p.zone == "sfha_a") & p.zip.str.match(r"^\d{5}$", na=False)]
    cz = p.groupby(["zip", "GEOID"]).policies.sum().reset_index().sort_values("policies")
    return cz.drop_duplicates("zip", keep="last").set_index("zip").GEOID


def year_built_matched(s):
    """Same comparison within the counties where the mitigated buildings are: each county's share of insured
    A-zone units and of A-zone claims built in or after the cut-off, weighted by its number of mitigated buildings.
    Removes differences in housing age between counties with and without mitigation."""
    b = s[s.primaryActivity.isin(BUILDING_ACT) & s.funded & s.yb.notna()].copy()
    b["GEOID"] = b.zip.map(zip_county())
    b = b.dropna(subset=["GEOID"])
    w = b.GEOID.value_counts()
    pol = pd.read_parquet(PROCESSED / "nfip_policies_agg.parquet")
    pol = pol[(pol.zone == "sfha_a") & (pol.pol_year == 2024) & pol.con_year.between(1800, 2024) & pol.GEOID.isin(w.index)]
    cl = pd.read_parquet(RAW / "openfema" / "NfipClaims_v3_paged.parquet",
                         columns=["yearOfLoss", "ratedFloodZone", "floodZoneCurrent", "originalConstructionDate",
                                  "countyCode"])
    cl = cl[cl.yearOfLoss.between(2016, 2024)]
    z = cl.ratedFloodZone.fillna(cl.floodZoneCurrent).fillna("").str.upper()
    cl = cl[z.str.match(M09.SFHA) & cl.countyCode.isin(w.index)].copy()
    cl["yr"] = pd.to_datetime(cl.originalConstructionDate, errors="coerce").dt.year
    cl = cl.dropna(subset=["yr"])
    rows = []
    for cut in [1990, 2001]:
        u = pol.assign(n=pol.policies.where(pol.con_year >= cut, 0)).groupby("GEOID")[["n", "policies"]].sum()
        c = cl.assign(n=(cl.yr >= cut).astype(float)).groupby("countyCode").n.mean()
        us, cs = (u.n / u.policies).reindex(w.index), c.reindex(w.index)
        rows.append({"built_from": cut, "counties": len(w), "mitigated": int(w.sum()),
                     "insured_units_pct_matched": 100 * np.average(us.fillna(0), weights=w),
                     "claims_pct_matched": 100 * np.average(cs.fillna(0), weights=w),
                     "mitigated_pct": 100 * (b.yb >= cut).mean(),
                     "mitigated_pct_with_claims_data": 100 * (b[b.GEOID.isin(c.index)].yb >= cut).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "wc_year_built_matched.csv", index=False)
    print(out.round(1).to_string())


def zip_claims():
    """A-zone NFIP claims 2016-2024 by ZCTA, from block-group GEOIDs via land-area shares of tracts."""
    cl = pd.read_parquet(RAW / "openfema" / "NfipClaims_v3_paged.parquet",
                         columns=["yearOfLoss", "ratedFloodZone", "floodZoneCurrent", "censusGeoid"])
    cl = cl[cl.yearOfLoss.between(2016, 2024)]
    z = cl.ratedFloodZone.fillna(cl.floodZoneCurrent).fillna("").str.upper()
    cl = cl[z.str.match(M09.SFHA) & cl.censusGeoid.notna()]
    tr = cl.censusGeoid.str[:11].value_counts().rename("claims")
    rel = pd.read_csv(RAW / "census_rel" / "tab20_zcta520_tract20_natl.txt", sep="|", dtype=str, encoding="utf-8-sig",
                      usecols=["GEOID_ZCTA5_20", "GEOID_TRACT_20", "AREALAND_PART", "AREALAND_TRACT_20"])
    rel = rel.dropna(subset=["GEOID_ZCTA5_20"])
    w = rel.AREALAND_PART.astype(float) / rel.AREALAND_TRACT_20.astype(float).replace(0, np.nan)
    rel = rel.assign(w=w.fillna(0)).merge(tr, left_on="GEOID_TRACT_20", right_index=True)
    return (rel.w * rel.claims).groupby(rel.GEOID_ZCTA5_20).sum().rename("claims")


def within(df, y, xs, g):
    """LPM with fixed effects for g (within transformation); SEs clustered by g."""
    d = df[[y, g] + xs].dropna()
    d = d[d.groupby(g)[y].transform("std") > 0]  # groups with variation in y only
    yy = (d[y] - d.groupby(g)[y].transform("mean")).to_numpy()
    X = np.column_stack([(d[x] - d.groupby(g)[x].transform("mean")).to_numpy() for x in xs])
    b = np.linalg.lstsq(X, yy, rcond=None)[0]
    e = yy - X @ b
    XtXi = np.linalg.pinv(X.T @ X)
    meat = np.zeros((len(xs), len(xs)))
    for _, idx in d.groupby(g).indices.items():
        sg = X[idx].T @ e[idx]
        meat += np.outer(sg, sg)
    G, n, k = d[g].nunique(), len(d), len(xs) + d[g].nunique()
    V = XtXi @ meat @ XtXi * G / (G - 1) * (n - 1) / (n - k)
    se = np.sqrt(np.diag(V))
    return pd.DataFrame({"term": xs, "coef": b, "se": se, "lo": b - 1.96 * se, "hi": b + 1.96 * se,
                         "n_zip": n, "n_county": G, "mean_y": d[y].mean()})


def zip_model(s):
    p = pd.read_parquet(PROCESSED / "nfip_policies_zip.parquet")
    p = p[(p.pol_year == 2024) & (p.zone == "sfha_a") & p.zip.str.match(r"^\d{5}$", na=False)]
    zc = zip_county()
    u = p.pivot_table(index="zip", columns="built_2001", values="policies", aggfunc="sum").fillna(0)
    u = u.rename(columns={0: "units_old", 1: "units_new"})[["units_old", "units_new"]]
    u["GEOID"] = zc
    u = u.join(zip_claims(), how="left").fillna({"claims": 0})
    b = s[s.primaryActivity.isin(BUILDING_ACT)]
    u["site_any"] = u.index.isin(b.zip).astype(float)
    u["site_funded"] = u.index.isin(b.loc[b.funded, "zip"]).astype(float)
    u["units_all"] = u.units_old + u.units_new
    for c in ["units_old", "units_new", "claims", "units_all"]:
        v = np.log1p(u[c])
        u["z_" + c] = (v - v.mean()) / v.std()
    sh = u.units_new / u.units_all.replace(0, np.nan)  # share of insured A-zone units built 2001+
    u["z_share_new"] = (sh - sh.mean()) / sh.std()
    # sample: ZIP codes with insured A-zone units in counties with at least one flood-mitigation site
    u = u[u.GEOID.isin(u.loc[u.site_any == 1, "GEOID"])]
    xs = ["z_units_new", "z_units_old", "z_claims"]
    res = []
    for y in ["site_any", "site_funded"]:
        r = within(u, y, xs, "GEOID")
        r["outcome"] = y
        r[["coef", "se", "lo", "hi"]] *= 100  # percentage points per SD
        res.append(r)
    # alternative: total insured units and the share built 2001+ (avoids the correlated new/old pair)
    for y in ["site_any", "site_funded"]:
        r = within(u, y, ["z_share_new", "z_units_all", "z_claims"], "GEOID")
        r["outcome"] = y + "_share"
        r[["coef", "se", "lo", "hi"]] *= 100
        res.append(r)
    res = pd.concat(res)
    res.to_csv(TAB / "wc_zip_model.csv", index=False)
    summ = pd.Series({"zips": len(u), "counties": u.GEOID.nunique(), "zips_with_site": int(u.site_any.sum()),
                      "zips_with_funded_site": int(u.site_funded.sum()), "sites": len(b),
                      "sites_matched_pct": 100 * b.zip.isin(u.index).mean(),
                      "rho_new_old": np.corrcoef(u.z_units_new, u.z_units_old)[0, 1]})
    summ.to_csv(TAB / "wc_zip_summary.csv", header=["value"])
    print(res.round(2).to_string()); print(summ.round(2))


def claim_rates_within():
    """New (2001+) vs older A-zone buildings, claims and payments per policy-year 2016-2024, compared within
    counties: Mantel-Haenszel rate ratio stratified by county (each county compares its own newer and older
    buildings), against the crude national ratio. Same definitions as 09_mitigation_loss_tables.py."""
    c = pd.read_parquet(RAW / "openfema" / "NfipClaims_v3_paged.parquet")  # as in M09.claims()
    c["GEOID"] = c.countyCode.astype(str).str.zfill(5)
    c["sfha"] = c.ratedFloodZone.fillna(c.floodZoneCurrent).fillna("").str.upper().str.match(M09.SFHA)
    c["con_year"] = pd.to_datetime(c.originalConstructionDate, errors="coerce", utc=True).dt.year
    c["paid"] = c[["amountPaidOnBuildingClaim", "amountPaidOnContentsClaim"]].astype(float).fillna(0).sum(axis=1)
    c["paid_real"] = c.paid * c.yearOfLoss.map(M09.defl()).fillna(1.0)
    w = c[c.yearOfLoss.between(2016, 2024) & c.sfha & c.con_year.notna()].assign(new=lambda d: d.con_year >= 2001)
    pol = pd.read_parquet(PROCESSED / "nfip_policies_agg.parquet")
    pp = pol[pol.pol_year.between(2016, 2024) & (pol.zone == "sfha_a") & pol.con_year.notna()]
    T = pp.assign(new=pp.con_year >= 2001).pivot_table(index="GEOID", columns="new", values="policies", aggfunc="sum")
    A = w.pivot_table(index="GEOID", columns="new", values="paid", aggfunc="size")
    P = w.pivot_table(index="GEOID", columns="new", values="paid_real", aggfunc="sum")
    d = pd.concat({"t0": T[False], "t1": T[True], "a0": A[False], "a1": A[True], "p0": P[False], "p1": P[True]},
                  axis=1).fillna(0)
    d = d[(d.t0 > 0) & (d.t1 > 0)]  # counties with both newer and older insured buildings
    tt = d.t0 + d.t1
    rows = []
    for lab, x1, x0 in (("claims", d.a1, d.a0), ("payments", d.p1, d.p0)):
        crude = (x1.sum() / d.t1.sum()) / (x0.sum() / d.t0.sum())
        mh = (x1 * d.t0 / tt).sum() / (x0 * d.t1 / tt).sum()
        # bootstrap over counties for a 95% interval on the stratified ratio
        rng = np.random.default_rng(7)
        idx = np.arange(len(d))
        bs = []
        for _ in range(1000):
            k = rng.choice(idx, len(idx))
            bs.append((x1.iloc[k] * d.t0.iloc[k] / tt.iloc[k]).sum() / (x0.iloc[k] * d.t1.iloc[k] / tt.iloc[k]).sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"measure": lab, "crude_ratio": crude, "mh_ratio_within_county": mh, "mh_lo": lo, "mh_hi": hi,
                     "counties": len(d), "share_policy_years_in_sample": tt.sum() / pp.policies.sum()})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "wc_claim_rates_within_county.csv", index=False)
    print(out.round(3).to_string())


def main():
    s = sites()
    year_built(s)
    year_built_matched(s)
    claim_rates_within()
    zip_model(s)


if __name__ == "__main__":
    main()
