"""Applications versus awards: do counties with more new floodplain development apply less for HMA flood
mitigation, or apply and get funded less often?

HMA Subapplications v2 also records applications that were denied, found ineligible, not selected or withdrawn.
Flood-mitigation subapplications (same activity and project-type rules as 02) are matched to benefiting counties
with the same name lookup as 02. Window: fiscal years 2001-2023 (most 2024-25 records are still pending).

Outcome classes: funded (obligation > 0); not funded (denied, not approved, ineligible, not selected, withdrawn);
undecided (pending, submitted, revision requested, or approved without an obligation yet); void (excluded).

Models (covered counties, same covariates and estimator as the primary funding model in 10):
  A. county applied for any flood-mitigation project (logit, AMEs per SD)
  B. among counties that applied, county had any funded application (logit, AMEs per SD)
  C. decided applications: funded vs not funded, county covariates + program and fiscal-year fixed effects
     (linear probability, SEs clustered by state)
Outputs: outputs/tables/exp_applications_models.csv, exp_applications_summary.csv
"""
import importlib
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import PROCESSED, RAW, TAB

P = importlib.import_module("02_build_county_panel")
E = importlib.import_module("10_exposure_analysis")
NEG = {"Not Approved / Denied", "Not Approved", "Ineligible", "Not Selected", "Withdrawn"}
VOID = {"Void", "Not Provided"}
X = ["x_new_exp", "x_base_exp", "x_loss", "x_dr", "x_pop", "x_svi", "x_pen", "x_rural", "x_coastal"]


def applications():
    d = pd.read_parquet(RAW / "openfema" / "HmaSubapplications_v2.parquet")
    d = d[d.program != "Fire Management Assistance Grant - Hazard Mitigation Grant Program"]
    d = d[~d.subapplicationType.isin(P.NON_PROJECT_TYPES) & ~d.primaryActivity.isin(P.NON_PROJECT_ACTIVITIES)]
    d = d[d.primaryActivity.isin(P.FLOOD_ACTIVITIES) & d.fiscalYear.between(2001, 2023) & ~d.status.isin(VOID)]
    d["prog"] = d.program.map(P.PROGRAM_GROUP)
    obl = d.totalObligatedAmount.fillna(0) > 0
    d["cls"] = np.where(obl, "funded", np.where(d.status.isin(NEG), "not_funded", "undecided"))
    # sensitivity: "Closed" with no obligation and no payment treated as closed without funding
    closed0 = (d.status == "Closed") & ~obl & (d.totalPaymentAmount.fillna(0) <= 0)
    d["cls2"] = np.where(closed0, "not_funded", d.cls)
    import geopandas as gpd
    counties = gpd.read_file(PROCESSED / "counties.gpkg")
    st = counties.drop_duplicates("STATE_NAME").set_index("STATE_NAME").STATEFP
    d["STATEFP"] = d.subapplicantState.map(st)
    d = d[d.STATEFP.notna()]
    look = {}
    for r in counties.sort_values("LSAD").itertuples():  # same lookup as 02
        for k in (P.norm(r.NAMELSAD), P.norm(r.NAME)):
            look.setdefault((r.STATEFP, k), r.GEOID)
        if r.LSAD == "25":
            look.setdefault((r.STATEFP, P.norm(r.NAME) + "CITY"), r.GEOID)
    rows = []
    for r in d.itertuples():
        names = [x.strip() for x in str(r.benefitingCounties).split(";") if x.strip()] \
            if r.benefitingCounties is not None else []
        if not names or any(re.search(r"STATE\s*WIDE|ALL COUNTIES", n, re.I) for n in names):
            continue
        for g in dict.fromkeys(x for x in (look.get((r.STATEFP, P.norm(n))) for n in names) if x):
            rows.append((r.subapplicationIdentifier, g))
    x = pd.DataFrame(rows, columns=["subapplicationIdentifier", "GEOID"])
    return x.merge(d[["subapplicationIdentifier", "prog", "fiscalYear", "status", "cls", "cls2"]],
                   on="subapplicationIdentifier")


def covariates(c):
    d = c[c.nfhl_covered.fillna(False)].copy()
    d["x_new_exp"] = np.log1p(d.d_dev_sfha_km2.clip(lower=0))
    d["x_base_exp"] = np.log1p(d.dev_sfha_km2_2000)
    d["x_loss"] = np.log1p(d.claims_paid_1996_2025)
    d["x_dr"], d["x_pop"], d["x_svi"], d["x_pen"] = d.dr_years, np.log(d.pop2020), d.svi_2020, d.resPenetrationRateSfha
    d["x_rural"], d["x_coastal"] = (~d.metro.fillna(True)).astype(float), d.coastal.astype(float)
    d = d.dropna(subset=X + ["STATEFP"])
    Z = d[X].copy()
    for k in X[:7]:
        Z[k] = E.z(Z[k])
    return d, Z


def logit_ame(y, Z, st, label):
    fe = pd.get_dummies(st, prefix="st", drop_first=True, dtype=float)
    keep = fe.columns[(fe.T @ y > 0).values & (fe.T @ (1 - y) > 0).values]
    m = sm.Logit(y, sm.add_constant(pd.concat([Z, fe[keep]], axis=1))).fit(
        disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": st.astype(int)})
    me = m.get_margeff(at="overall")
    idx = list(m.params.index[1:])
    return [(label, k, 100 * me.margeff[idx.index(k)], 100 * me.margeff_se[idx.index(k)], me.pvalues[idx.index(k)],
             len(y), int(y.sum())) for k in X]


def never_applied(ty, a):
    """Among high-growth covered counties: which characteristics are associated with never submitting a flood-
    mitigation application (FY2001-2023)? Logit, AMEs (continuous terms per SD; binary terms per unit), SEs
    clustered by state. Plan and CRS status are current snapshots, so this is descriptive."""
    h = ty[ty.exp_class == "High growth"].copy()
    h["never"] = (~h.index.isin(a.GEOID)).astype(int)
    h["x_plan"], h["x_crs"] = h.plan_ok.astype(float), h.crs_any.astype(float)
    h["x_pop"], h["x_rural"] = np.log(h.pop2020), (~h.metro.fillna(True)).astype(float)
    h["x_loss"], h["x_dr"], h["x_svi"] = np.log1p(h.claims_paid_1996_2025), h.dr_years, h.svi_2020
    h["x_new"], h["x_coastal"] = np.log1p(h.d_dev_sfha_km2.clip(lower=0)), h.coastal.astype(float)
    Xn = ["x_plan", "x_crs", "x_pop", "x_rural", "x_loss", "x_dr", "x_svi", "x_new", "x_coastal"]
    d = h.dropna(subset=Xn + ["STATEFP"])
    Zn = d[Xn].copy()
    for k in ("x_pop", "x_loss", "x_dr", "x_svi", "x_new"):
        Zn[k] = E.z(Zn[k])
    m = sm.Logit(d.never, sm.add_constant(Zn)).fit(disp=0, cov_type="cluster",
                                                     cov_kwds={"groups": d.STATEFP.astype(int)})
    me = m.get_margeff(at="overall")
    out = pd.DataFrame({"ame": 100 * me.margeff, "se": 100 * me.margeff_se, "p": me.pvalues}, index=Xn)
    out["n"], out["n_never"], out["n_all_high"] = len(d), int(d.never.sum()), len(h)
    out.to_csv(TAB / "exp_never_applied_model.csv")
    means = d.groupby("never")[["x_plan", "x_crs", "x_rural", "pop2020", "claims_paid_1996_2025"]].mean()
    means.to_csv(TAB / "exp_never_applied_means.csv")
    return out


def main():
    c, _ = E.load()
    a = applications()
    d, Z = covariates(c)
    by = a.groupby("GEOID")
    d["applied"] = d.index.isin(a.GEOID).astype(int)
    d["n_app"] = by.size().reindex(d.index).fillna(0)
    d["any_funded"] = d.index.isin(a.loc[a.cls == "funded", "GEOID"]).astype(int)
    rows = logit_ame(d.applied, Z, d.STATEFP, "A. Applied (all covered counties)")
    # A2: applied to a pre-disaster program (FMA or PDM/BRIC), which does not require a declaration
    pre = a[a.prog.isin(["FMA", "PDM/BRIC"])]
    d["applied_pre"] = d.index.isin(pre.GEOID).astype(int)
    rows += logit_ame(d.applied_pre, Z, d.STATEFP, "A2. Applied to FMA or PDM/BRIC (all covered counties)")
    ap = d[d.applied == 1]
    rows += logit_ame(ap.any_funded, Z.loc[ap.index], ap.STATEFP, "B. Any application funded (applicant counties)")
    # C: decided applications, county covariates + program and fiscal-year FE (LPM);
    # D: same, counting applications closed without obligation or payment as not funded
    for col, label in (("cls", "C. Application funded (decided applications)"),
                       ("cls2", "D. Application funded, closed without funding counted as not funded")):
        dec = a[a[col] != "undecided"].merge(Z, left_on="GEOID", right_index=True) \
            .merge(d[["STATEFP"]], left_on="GEOID", right_index=True)
        yC = (dec[col] == "funded").astype(float) * 100
        fe = pd.concat([pd.get_dummies(dec.prog, prefix="p", drop_first=True, dtype=float),
                        pd.get_dummies(dec.fiscalYear, prefix="fy", drop_first=True, dtype=float)], axis=1)
        m = sm.OLS(yC, sm.add_constant(pd.concat([dec[X], fe], axis=1))).fit(
            cov_type="cluster", cov_kwds={"groups": dec.STATEFP.astype(int)})
        rows += [(label, k, m.params[k], m.bse[k], m.pvalues[k], len(dec), int((dec[col] == "funded").sum()))
                 for k in X]
    # C3: decided applications FY2001-2019 only (few still undecided; checks right-censoring of recent years)
    dec = a[(a.cls != "undecided") & a.fiscalYear.between(2001, 2019)].merge(Z, left_on="GEOID", right_index=True) \
        .merge(d[["STATEFP"]], left_on="GEOID", right_index=True)
    yC = (dec.cls == "funded").astype(float) * 100
    fe = pd.concat([pd.get_dummies(dec.prog, prefix="p", drop_first=True, dtype=float),
                    pd.get_dummies(dec.fiscalYear, prefix="fy", drop_first=True, dtype=float)], axis=1)
    m = sm.OLS(yC, sm.add_constant(pd.concat([dec[X], fe], axis=1))).fit(
        cov_type="cluster", cov_kwds={"groups": dec.STATEFP.astype(int)})
    rows += [("E. Application funded, FY2001-2019 only", k, m.params[k], m.bse[k], m.pvalues[k], len(dec),
              int((dec.cls == "funded").sum())) for k in X]
    # C4: decided applications FY2003-2023 (drops the earliest years, where unsuccessful records appear missing)
    dec = a[(a.cls != "undecided") & a.fiscalYear.between(2003, 2023)].merge(Z, left_on="GEOID", right_index=True) \
        .merge(d[["STATEFP"]], left_on="GEOID", right_index=True)
    yC = (dec.cls == "funded").astype(float) * 100
    fe = pd.concat([pd.get_dummies(dec.prog, prefix="p", drop_first=True, dtype=float),
                    pd.get_dummies(dec.fiscalYear, prefix="fy", drop_first=True, dtype=float)], axis=1)
    m = sm.OLS(yC, sm.add_constant(pd.concat([dec[X], fe], axis=1))).fit(
        cov_type="cluster", cov_kwds={"groups": dec.STATEFP.astype(int)})
    rows += [("F. Application funded, FY2003-2023 only", k, m.params[k], m.bse[k], m.pvalues[k], len(dec),
              int((dec.cls == "funded").sum())) for k in X]
    # flow from unique submissions to the modelling samples
    u = a.drop_duplicates("subapplicationIdentifier")
    links_cov = a[a.GEOID.isin(d.index)]
    flow = [("Flood-mitigation subapplications submitted, FY2001-2023 (unique)", u.subapplicationIdentifier.nunique()),
            ("  funded (obligation > 0)", int((u.cls == "funded").sum())),
            ("  not funded (denied, ineligible, not selected, withdrawn)", int((u.cls == "not_funded").sum())),
            ("  closed with no obligation and no payment", int(((u.cls == "undecided") & (u.cls2 == "not_funded")).sum())),
            ("  still pending, submitted or otherwise undecided", int(((u.cls == "undecided") & (u.cls2 != "not_funded")).sum())),
            ("Application-county pairs (multi-county applications counted once per county)", len(a)),
            ("  pairs in covered counties with complete covariates", len(links_cov)),
            ("  of which decided (model column 4)", int((links_cov.cls != "undecided").sum())),
            ("  of which decided, funded", int((links_cov.cls == "funded").sum()))]
    pd.DataFrame(flow, columns=["step", "count"]).to_csv(TAB / "exp_applications_flow.csv", index=False)
    by_year = u[u.cls != "undecided"].groupby("fiscalYear").cls.apply(lambda s: 100 * (s == "funded").mean())
    pd.DataFrame({"approval_pct": by_year, "undecided_pct": u.groupby("fiscalYear").cls.apply(
        lambda s: 100 * (s == "undecided").mean())}).to_csv(TAB / "exp_applications_by_year.csv")
    t = pd.DataFrame(rows, columns=["model", "term", "coef", "se", "p", "n", "n_pos"])
    t.to_csv(TAB / "exp_applications_models.csv", index=False)
    # descriptive summary by growth class (top tercile of added A-zone land, as in the typology)
    hi = d.d_dev_sfha_km2 >= d.d_dev_sfha_km2.quantile(2 / 3)
    a2 = a.merge(pd.DataFrame({"hi": hi}), left_on="GEOID", right_index=True)
    s = {"apps_total": a.subapplicationIdentifier.nunique(), "app_county_links": len(a),
         "apps_funded": a.loc[a.cls == "funded", "subapplicationIdentifier"].nunique(),
         "apps_not_funded": a.loc[a.cls == "not_funded", "subapplicationIdentifier"].nunique(),
         "apps_undecided": a.loc[a.cls == "undecided", "subapplicationIdentifier"].nunique(),
         "covered_model_n": len(d), "covered_applied_pct": 100 * d.applied.mean(),
         "hi_applied_pct": 100 * d.loc[hi, "applied"].mean(), "lo_applied_pct": 100 * d.loc[~hi, "applied"].mean()}
    for lab, g in (("hi", a2[a2.hi]), ("lo", a2[~a2.hi])):
        dd = g[g.cls != "undecided"]
        s[f"{lab}_success_pct"] = 100 * (dd.cls == "funded").mean()
    # county matching rate by outcome (to check that unfunded applications are not under-matched)
    raw = pd.read_parquet(RAW / "openfema" / "HmaSubapplications_v2.parquet")
    raw = raw[(raw.program != "Fire Management Assistance Grant - Hazard Mitigation Grant Program")
              & raw.primaryActivity.isin(P.FLOOD_ACTIVITIES) & ~raw.subapplicationType.isin(P.NON_PROJECT_TYPES)
              & ~raw.primaryActivity.isin(P.NON_PROJECT_ACTIVITIES) & raw.fiscalYear.between(2001, 2023)
              & ~raw.status.isin(VOID)]
    raw_obl = raw.totalObligatedAmount.fillna(0) > 0
    raw_cls = np.where(raw_obl, "funded", np.where(raw.status.isin(NEG), "not_funded", "undecided"))
    hit = raw.subapplicationIdentifier.isin(a.subapplicationIdentifier)
    s["match_funded_pct"] = 100 * hit[raw_cls == "funded"].mean()
    s["match_not_funded_pct"] = 100 * hit[raw_cls == "not_funded"].mean()
    hu = d[hi & (d.funded == False)]
    s["hi_unfunded_n"] = len(hu)
    s["hi_unfunded_applied"] = int(hu.applied.sum())
    # same count for all high-growth unfunded covered counties (the 214 in Table 5 and Figure 5)
    _, ty = E.typology(c)
    hu_all = ty[(ty.exp_class == "High growth") & ~ty.funded]
    s["hi_unfunded_all_n"] = len(hu_all)
    s["hi_unfunded_all_never"] = int((~hu_all.index.isin(a.GEOID)).sum())
    pd.Series(s).to_csv(TAB / "exp_applications_summary.csv")
    never_applied(ty, a)
    print(pd.Series(s).round(2).to_string())
    print(t[t.term.isin(["x_new_exp", "x_base_exp", "x_loss", "x_dr", "x_pop"])].round(3).to_string())


if __name__ == "__main__":
    main()
