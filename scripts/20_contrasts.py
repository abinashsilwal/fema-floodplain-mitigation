"""Direct contrasts between coefficients, and checks on the site-inventory records.

1. County model (primary logit with state fixed effects, state-clustered errors): differences between average
   marginal effects (baseline floodplain development - new development; NFIP losses - new development), with
   delta-method standard errors from the joint covariance of the marginal effects. The same contrasts are computed
   in the linear probability version with Conley spatial errors (19_spatial_se.py design).
2. ZIP-code model (17_within_county.py): older-unit minus newer-unit coefficient with county-clustered covariance.
3. Site inventories, fiscal years 2020-2025: construction year of building sites by funding status, payment status,
   program and decision status (FEMA describes inventory sites as proposed for mitigation; not all are mitigated).
Outputs: outputs/tables/exp_contrasts.csv, wc_site_status.csv
"""
import importlib

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import PROCESSED, RAW, TAB

S = importlib.import_module("19_spatial_se")
W = importlib.import_module("17_within_county")
X = S.X
PAIRS = [("x_base_exp", "x_new_exp"), ("x_loss", "x_new_exp"), ("x_dr", "x_new_exp")]


def county_contrasts():
    d = S.design()
    fe = pd.get_dummies(d.STATEFP, prefix="st", drop_first=True, dtype=float)
    y = d.y.astype(int)
    keep = fe.columns[(fe.T @ y > 0).values & (fe.T @ (1 - y) > 0).values]
    Xs = sm.add_constant(pd.concat([d[X], fe[keep]], axis=1))
    m = sm.Logit(y, Xs).fit(disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": d.STATEFP.astype(int)})
    me = m.get_margeff(at="overall")
    names = list(m.params.index[1:])
    idx = {k: names.index(k) for k in X}
    C = me.margeff_cov
    rows = []
    for a, b in PAIRS:
        i, j = idx[a], idx[b]
        diff = me.margeff[i] - me.margeff[j]
        se = np.sqrt(C[i, i] + C[j, j] - 2 * C[i, j])
        rows.append({"model": "Logit AME, state-clustered", "contrast": f"{a} - {b}", "diff_pp": 100 * diff,
                     "se_pp": 100 * se, "lo": 100 * (diff - 1.96 * se), "hi": 100 * (diff + 1.96 * se)})
    # linear probability model with Conley errors (largest variance across cut-offs reported)
    st = d.STATEFP.to_numpy()
    Wd = d[["y"] + X] - d.groupby("STATEFP")[["y"] + X].transform("mean")
    yy, Xm = Wd.y.to_numpy(), Wd[X].to_numpy()
    XtXi = np.linalg.inv(Xm.T @ Xm)
    bvec = XtXi @ Xm.T @ yy
    e = yy - Xm @ bvec
    u = Xm * e[:, None]
    import geopandas as gpd
    g = gpd.read_file(PROCESSED / "counties.gpkg", columns=["GEOID"]).to_crs(5070).set_index("GEOID")
    cen = g.geometry.centroid.reindex(d.index)
    xy = np.column_stack([cen.x, cen.y]) / 1000
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    n, k, G = len(yy), len(X), d.STATEFP.nunique()
    dof = (n - 1) / (n - k - G)
    for a, b in PAIRS:
        i, j = X.index(a), X.index(b)
        w = np.zeros(k)
        w[i], w[j] = 1, -1
        diff = w @ bvec
        ses = []
        for cut in S.CUTOFFS_KM:
            K = np.clip(1 - D / cut, 0, None)
            V = XtXi @ (u.T @ K @ u) @ XtXi * dof
            ses.append(np.sqrt(w @ V @ w))
        se = max(ses)
        rows.append({"model": "LPM, Conley (widest of 50-800 km)", "contrast": f"{a} - {b}", "diff_pp": 100 * diff,
                     "se_pp": 100 * se, "lo": 100 * (diff - 1.96 * se), "hi": 100 * (diff + 1.96 * se)})
    return rows


def zip_contrast():
    """Older minus newer insured units in the ZIP model, county fixed effects, county-clustered covariance."""
    import types
    rows = []
    src = W.zip_model.__code__  # rebuild the ZIP sample exactly as in 17_within_county.zip_model
    p = pd.read_parquet(PROCESSED / "nfip_policies_zip.parquet")
    p = p[(p.pol_year == 2024) & (p.zone == "sfha_a") & p.zip.str.match(r"^\d{5}$", na=False)]
    u = p.pivot_table(index="zip", columns="built_2001", values="policies", aggfunc="sum").fillna(0)
    u = u.rename(columns={0: "units_old", 1: "units_new"})[["units_old", "units_new"]]
    u["GEOID"] = W.zip_county()
    u = u.join(W.zip_claims(), how="left").fillna({"claims": 0})
    s = W.sites()
    b = s[s.primaryActivity.isin(W.BUILDING_ACT)]
    u["site_any"] = u.index.isin(b.zip).astype(float)
    u["site_funded"] = u.index.isin(b.loc[b.funded, "zip"]).astype(float)
    for c in ["units_old", "units_new", "claims"]:
        v = np.log1p(u[c])
        u["z_" + c] = (v - v.mean()) / v.std()
    u = u[u.GEOID.isin(u.loc[u.site_any == 1, "GEOID"])]
    xs = ["z_units_new", "z_units_old", "z_claims"]
    for yv in ["site_any", "site_funded"]:
        dd = u[[yv, "GEOID"] + xs].dropna()
        dd = dd[dd.groupby("GEOID")[yv].transform("std") > 0]
        Y = (dd[yv] - dd.groupby("GEOID")[yv].transform("mean")).to_numpy()
        Xm = np.column_stack([(dd[x] - dd.groupby("GEOID")[x].transform("mean")).to_numpy() for x in xs])
        bb = np.linalg.lstsq(Xm, Y, rcond=None)[0]
        e = Y - Xm @ bb
        XtXi = np.linalg.pinv(Xm.T @ Xm)
        meat = np.zeros((3, 3))
        for _, ix in dd.groupby("GEOID").indices.items():
            sg = Xm[ix].T @ e[ix]
            meat += np.outer(sg, sg)
        G, n = dd.GEOID.nunique(), len(dd)
        V = XtXi @ meat @ XtXi * G / (G - 1) * (n - 1) / (n - 3 - G)
        w = np.array([-1.0, 1.0, 0.0])  # older - newer
        diff, se = w @ bb, np.sqrt(w @ V @ w)
        rows.append({"model": f"ZIP LPM, county FE ({yv})", "contrast": "z_units_old - z_units_new",
                     "diff_pp": 100 * diff, "se_pp": 100 * se, "lo": 100 * (diff - 1.96 * se),
                     "hi": 100 * (diff + 1.96 * se)})
    return rows


def site_status():
    s = pd.read_parquet(RAW / "openfema" / "HmaSubapplicationsProjectSiteInventories_v1_paged.parquet")
    h = pd.read_parquet(RAW / "openfema" / "HmaSubapplications_v2.parquet",
                        columns=["subapplicationIdentifier", "program", "fiscalYear", "status",
                                 "totalObligatedAmount", "totalPaymentAmount", "subapplicationTitle"])
    f = s.merge(h, left_on="subapplicationId", right_on="subapplicationIdentifier", how="inner")
    b = f[(f.fiscalYear >= W.FY0) & f.primaryActivity.isin(W.BUILDING_ACT) & (f.projectSiteInventoryType == "Building") & (f.structureType == "Residential")].copy()
    n_all = len(b)
    b["yb"] = b.yearBuilt.where(b.yearBuilt.between(1800, 2026))
    known = b.yb.notna().mean()
    b = b[b.yb.notna()].assign(new=lambda x: x.yb >= 2001, funded=lambda x: x.totalObligatedAmount.fillna(0) > 0,
                               paid=lambda x: x.totalPaymentAmount.fillna(0) > 0)
    # Proposals are often resubmitted in later years, so the same physical site can appear in several
    # subapplications. Sites are matched across subapplications by a fingerprint of their recorded attributes.
    # Records inside one subapplication are always treated as different buildings (identical attributes there
    # describe similar buildings, not repeats), and only fingerprints with at least five known attributes are
    # matched. A matched site is counted once and classed as funded if any of its subapplications was funded.
    fp_main = ["zipcode", "yearBuilt", "buildingSize", "lotSize", "foundationType", "structureType",
               "firstFloorElevation"]
    fp_loose = ["zipcode", "yearBuilt", "structureType"]

    def unique_sites(df, fp, min_known):
        d = df.copy()
        d["fp"] = d[fp].astype(str).agg("|".join, axis=1)
        d["occ"] = d.groupby(["fp", "subapplicationId"]).cumcount()  # n-th similar building in a subapplication
        strong = d[fp].notna().sum(axis=1) >= min_known
        d["uid"] = (d.fp + "#" + d.occ.astype(str)).where(strong, "row" + d.index.astype(str))
        d["uid_funded"] = d.groupby("uid").funded.transform("max")
        u = d.sort_values("funded", ascending=False).drop_duplicates("uid")
        return d, u

    b, uniq = unique_sites(b, fp_main, 5)
    fma = b.program.str.startswith("Flood Mitigation Assistance")
    rows = [("All building sites, FY2020-2025", b), ("Funded (obligation > 0)", b[b.funded]),
            ("Funded with payments > 0", b[b.funded & b.paid]), ("Funded, Flood Mitigation Assistance", b[b.funded & fma]),
            ("Funded, other programs", b[b.funded & ~fma]),
            ("Not funded", b[~b.funded]),
            ("Unique sites, funded in any subapplication", uniq[uniq.uid_funded]),
            ("Unique sites, never funded", uniq[~uniq.uid_funded])]
    out = pd.DataFrame([{"group": lab, "sites": len(x), "subapplications": x.subapplicationId.nunique(),
                         "built_2001_pct": 100 * x.new.mean(), "built_1990_pct": 100 * (x.yb >= 1990).mean(),
                         "subapp_mean_built_2001_pct": 100 * x.groupby("subapplicationId").new.mean().mean()}
                        for lab, x in rows])
    out["building_sites_total"], out["year_known_share"] = n_all, known
    rep = b.groupby("uid").subapplicationId.nunique()
    out["repeated_site_keys"] = int((rep > 1).sum())
    out["rows_with_repeated_key"] = int(b.uid.isin(rep[rep > 1].index).sum())
    out["funded_fma_share"] = b[b.funded].program.str.startswith("Flood Mitigation Assistance").mean()
    out.to_csv(TAB / "wc_site_status.csv", index=False)
    print(out.round(2).to_string())
    # record flow from site rows to unique sites
    strong = b[fp_main].notna().sum(axis=1) >= 5
    flow = pd.Series({
        "building-site records with construction year (FY2020-2025)": len(b),
        "records with fewer than five known attributes (never matched)": int((~strong).sum()),
        "records with a strong fingerprint": int(strong.sum()),
        "sites listed in more than one subapplication": int((rep > 1).sum()),
        "records belonging to those sites": int(b.uid.isin(rep[rep > 1].index).sum()),
        "records removed as repeat listings": len(b) - len(uniq),
        "unique sites": len(uniq),
        "unique sites funded in any subapplication": int(uniq.uid_funded.sum()),
        "unique sites never funded": int((~uniq.uid_funded).sum()),
    })
    flow.to_csv(TAB / "wc_site_flow.csv", header=["count"])
    print(flow.to_string())
    # alternative fingerprints: none (no matching), main, loose (ZIP code, year built, structure type)
    alt = []
    for lab, fp, mk in (("No matching (all records)", None, None), ("Main fingerprint, >=5 known", fp_main, 5),
                        ("Main fingerprint, any known", fp_main, 0), ("Loose: ZIP, year built, structure", fp_loose, 3)):
        if fp is None:
            uu = b.assign(uid_funded=b.funded)
        else:
            _, uu = unique_sites(b.drop(columns=["fp", "occ", "uid", "uid_funded"], errors="ignore"), fp, mk)
        alt.append({"fingerprint": lab, "unique_sites": len(uu),
                    "funded_sites": int(uu.uid_funded.sum()), "funded_built_2001_pct": 100 * uu[uu.uid_funded].new.mean(),
                    "never_funded_sites": int((~uu.uid_funded).sum()),
                    "never_funded_built_2001_pct": 100 * uu[~uu.uid_funded].new.mean()})
    alt = pd.DataFrame(alt)
    alt.to_csv(TAB / "wc_site_fingerprints.csv", index=False)
    print(alt.round(1).to_string())


def main():
    rows = county_contrasts() + zip_contrast()
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "exp_contrasts.csv", index=False)
    print(t.round(2).to_string())
    site_status()


if __name__ == "__main__":
    main()
