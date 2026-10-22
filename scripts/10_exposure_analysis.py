"""Emerging floodplain exposure vs FEMA flood-mitigation funding (main analysis of the paper).

Inputs: data/processed/{exposure_county, exposure_county_year, county_mitigation_loss,
        county_period_mitigation_loss, county_year_panel, county_cross_section}.parquet, counties.gpkg
Outputs: outputs/tables/exp_*.csv, outputs/figures/Fig*_*.png

Exposure = developed land (NLCD 21-24) inside the present-day FEMA 1% annual-chance A-zones (fixed mask).
All county analyses use counties whose developed land is >= 90% inside mapped NFHL zones (nfhl_covered),
unless stated. Funding/exposure comparisons over 2001-2025 are retrospective descriptive alignment; the
period panel uses lagged exposure growth to respect temporal order.
"""
import importlib
import os
import sys
import warnings

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import FIG, PROCESSED, TAB

os.environ.setdefault("GDAL_DATA", os.path.join(sys.prefix, "Library", "share", "gdal"))
warnings.filterwarnings("ignore")
A = importlib.import_module("03_analysis")  # twfe estimator, palette, rc settings, numpy rank patch
BLUE, SEQ4, INK, save, twfe = A.BLUE, A.SEQ4, A.INK, A.save, A.twfe
PERIODS = ["2001-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2025"]
EXP_YEARS = [2000, 2005, 2010, 2015, 2020, 2025]


def z(s):
    return (s - s.mean()) / s.std()


def load():
    e = pd.read_parquet(PROCESSED / "exposure_county.parquet")
    ey = pd.read_parquet(PROCESSED / "exposure_county_year.parquet")
    m = pd.read_parquet(PROCESSED / "county_mitigation_loss.parquet")
    p = pd.read_parquet(PROCESSED / "county_year_panel.parquet")
    xs = pd.read_parquet(PROCESSED / "county_cross_section.parquet").reset_index()
    base = p.drop_duplicates("GEOID").set_index("GEOID")[["NAME", "STUSPS", "STATEFP", "region", "coastal", "area_km2"]]
    pop = p.pivot_table(index="GEOID", columns="year", values="pop")
    p["dr_flood_bin"] = (p.dr_flood > 0).astype(int)
    p["dr_narrow_bin"] = (p.dr_flood_narrow > 0).astype(int)
    w = p[(p.year >= 2001) & (p.year <= 2024)].groupby("GEOID")
    dr = pd.concat([w.dr_flood_bin.sum().rename("dr_years"), w.dr_narrow_bin.sum().rename("dr_years_narrow")], axis=1)
    c = base.join(e.set_index("GEOID")).join(m.set_index("GEOID")).join(dr)
    c = c.join(xs.set_index("GEOID")[["EAD(Base)", "ead_chg5"]])
    c["pop2000"], c["pop2020"] = pop[2000], pop[2020]
    fill0 = [k for k in c.columns if k.startswith(("mit_", "claims_", "pol_"))]
    c[fill0] = c[fill0].fillna(0)
    c["resPenetrationRateSfha"] = c.resPenetrationRateSfha.clip(upper=1)
    c["funded"] = c.mit_obl_2001_2025 > 0
    # broader FEMA flood-mitigation measure: HMA + Public Assistance Section 406 + NFIP ICC (2001-2025)
    for k in ("pa406_2001_2025", "icc_2001_2025", "icc_n_2001_2025", "hud_buyout_any"):
        if k not in c:
            c[k] = np.nan
    c["mit_fema_all_2001_2025"] = c[["mit_obl_2001_2025", "pa406_2001_2025", "icc_2001_2025"]].fillna(0).sum(axis=1)
    c["first_firm_year"] = c.first_firm_year.where(c.first_firm_year.between(1960, 2026))
    return c, ey


# ------------------------------------------------------------------ 1 national trends
def national_trends(c, ey):
    cov = c.index[c.nfhl_covered.fillna(False)]
    y = ey[ey.GEOID.isin(cov)].groupby("year")[["dev_km2", "dev_sfha_km2", "devmh_sfha_km2", "dev_x500_km2",
                                                  "dev_v_km2", "sfha_land_km2"]].sum()
    y["dev_out_km2"] = y.dev_km2 - y.dev_sfha_km2 - y.dev_v_km2
    y["share_dev_in_sfha"] = 100 * y.dev_sfha_km2 / y.dev_km2
    d = y.diff()
    y["new_dev_share_sfha"] = 100 * d.dev_sfha_km2 / d.dev_km2
    y["growth_sfha_pct"] = 100 * d.dev_sfha_km2 / y.dev_sfha_km2.shift()
    y["growth_out_pct"] = 100 * d.dev_out_km2 / y.dev_out_km2.shift()
    y["dev_out_km2"] = y.dev_km2 - y.dev_sfha_km2 - y.dev_v_km2 - y.dev_x500_km2  # outside 1% and 0.2% zones
    y["growth_out_pct"] = 100 * y.dev_out_km2.diff() / y.dev_out_km2.shift()
    y["growth_x500_pct"] = 100 * d.dev_x500_km2 / y.dev_x500_km2.shift()
    y["new_dev_share_x500"] = 100 * d.dev_x500_km2 / d.dev_km2
    y.round(3).to_csv(TAB / "exp_national_trends.csv")
    reg = ey[ey.GEOID.isin(cov)].merge(c[["region"]], left_on="GEOID", right_index=True)
    r = reg[reg.year.isin([2000, 2025])].pivot_table(index="region", columns="year",
                                                    values=["dev_sfha_km2", "dev_km2"], aggfunc="sum")
    rt = pd.DataFrame({
        "dev_sfha_2000": r[("dev_sfha_km2", 2000)], "dev_sfha_2025": r[("dev_sfha_km2", 2025)],
        "growth_sfha_pct": 100 * (r[("dev_sfha_km2", 2025)] / r[("dev_sfha_km2", 2000)] - 1),
        "growth_all_pct": 100 * (r[("dev_km2", 2025)] / r[("dev_km2", 2000)] - 1),
        "new_dev_share_sfha": 100 * (r[("dev_sfha_km2", 2025)] - r[("dev_sfha_km2", 2000)])
                              / (r[("dev_km2", 2025)] - r[("dev_km2", 2000)]),
    })
    rt.loc["CONUS"] = [y.dev_sfha_km2[2000], y.dev_sfha_km2[2025], 100 * (y.dev_sfha_km2[2025] / y.dev_sfha_km2[2000] - 1),
                       100 * (y.dev_km2[2025] / y.dev_km2[2000] - 1),
                       100 * (y.dev_sfha_km2[2025] - y.dev_sfha_km2[2000]) / (y.dev_km2[2025] - y.dev_km2[2000])]
    rt.round(2).to_csv(TAB / "exp_region_growth.csv")
    meta = {"n_counties_total": len(c), "n_counties_nfhl": int(c.dev_sfha_km2_2000.notna().sum()),
            "n_counties_covered": len(cov),
            "pop2020_share_covered": c.loc[cov, "pop2020"].sum() / c.pop2020.sum(),
            "dev2025_share_covered": c.loc[cov, "dev_km2_2025"].sum() / c.dev_km2_2025.sum()}
    pd.Series(meta).to_csv(TAB / "exp_coverage.csv")
    return y, rt, meta


# ------------------------------------------------------------------ 2 buildings ratio
def building_ratio(c):
    cov = c[c.nfhl_covered.fillna(False)].copy()
    # FEMA-structure-scaled estimate of residential structures added in the SFHA since 2000
    cov["est_new_struct"] = cov.totalResStructuresSfha * (cov.d_dev_sfha_km2 / cov.dev_sfha_km2_2025).clip(0, 1)
    rows = []
    for lab, d in [("CONUS (covered counties)", cov)] + [(r, g) for r, g in cov.groupby("region")]:
        # matched window: buildings constructed 2001-2024 (insured in 2024) vs buildings acquired/elevated FY2001-2024
        mit = d.mit_acq_2001_2024.sum() + d.mit_elev_2001_2024.sum()
        rows.append((lab, len(d), d.pol_sfha_built2001_2024.sum(), d.est_new_struct.sum(), d.mit_acq_2001_2024.sum(),
                     d.mit_elev_2001_2024.sum(), d.pol_sfha_built2001_2024.sum() / mit, d.est_new_struct.sum() / mit))
    allc = c.copy()
    mit_all = allc.mit_acq_2001_2024.sum() + allc.mit_elev_2001_2024.sum()
    rows.append(("CONUS (all counties)", len(allc), allc.pol_sfha_built2001_2024.sum(), np.nan, allc.mit_acq_2001_2024.sum(),
                 allc.mit_elev_2001_2024.sum(), allc.pol_sfha_built2001_2024.sum() / mit_all, np.nan))
    t = pd.DataFrame(rows, columns=["area", "counties", "insured_new_sfha_bldg", "est_new_sfha_res_struct",
                                    "acquired", "elevated", "ratio_insured_per_mitigated", "ratio_est_per_mitigated"])
    t.round(2).to_csv(TAB / "exp_building_ratio.csv", index=False)
    return t, cov


# ------------------------------------------------------------------ 3 alignment (cross-section)
def alignment(c):
    d = c[c.nfhl_covered.fillna(False)].copy()
    d["x_new_exp"] = np.log1p(d.d_dev_sfha_km2.clip(lower=0))
    d["x_base_exp"] = np.log1p(d.dev_sfha_km2_2000)
    d["x_new_bldg"] = np.log1p(d.pol_sfha_built2001_2024)
    d["x_loss_pre"] = np.log1p(d.claims_paid_1996_2000)
    d["x_loss"] = np.log1p(d.claims_paid_1996_2025)
    d["x_dr"] = d.dr_years
    d["x_pop"] = np.log(d.pop2020)
    d["x_svi"] = d.svi_2020
    d["x_rural"] = (~d.metro.fillna(True)).astype(float)
    d["x_coastal"] = d.coastal.astype(float)
    d["x_pen"] = d.resPenetrationRateSfha
    X = ["x_new_exp", "x_base_exp", "x_loss", "x_dr", "x_pop", "x_svi", "x_pen", "x_rural", "x_coastal"]
    d = d.dropna(subset=X + ["STATEFP"])
    Z = d[X].copy()
    for k in X:
        if k not in ("x_rural", "x_coastal"):
            Z[k] = z(Z[k])
    fe = pd.get_dummies(d.STATEFP, prefix="st", drop_first=True, dtype=float)
    out = []
    for lab, col in [("All programs", "mit_obl_2001_2025"), ("HMGP", "mit_obl_hmgp_2001_2025"),
                     ("FMA", "mit_obl_fma_2001_2025"), ("PDM/BRIC", "mit_obl_pdm_2001_2025"),
                     ("Buildings mitigated", "mit_bldg_2001_2025")]:
        y_any = (d[col] > 0).astype(int)
        keep = fe.columns[(fe.T @ y_any > 0).values & (fe.T @ (1 - y_any) > 0).values]  # drop perfectly predicted states
        Xs = sm.add_constant(pd.concat([Z, fe[keep]], axis=1))
        try:
            m1 = sm.Logit(y_any, Xs).fit(disp=0, maxiter=300, cov_type="cluster",
                                         cov_kwds={"groups": d.STATEFP.astype(int)})
            me = m1.get_margeff(at="overall")
            idx = list(m1.params.index[1:])
            for k in X:
                i = idx.index(k)
                out.append((lab, "part1_any", k, 100 * me.margeff[i], 100 * me.margeff_se[i], me.pvalues[i],
                            len(d), int(y_any.sum())))
            if lab == "All programs":  # fit, discrimination and collinearity of the primary model
                pr = pd.Series(m1.predict(Xs), index=d.index)
                rk = pr.rank()
                n1, n0 = int(y_any.sum()), int((1 - y_any).sum())
                auc = (rk[y_any == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
                C = Z.drop(columns=["x_rural", "x_coastal"]).corr().values
                vif = pd.Series(np.diag(np.linalg.inv(C)), index=[k for k in X if k not in ("x_rural", "x_coastal")])
                diag = {"n": len(d), "n_funded": n1, "states_in_model": len(keep) + 1, "mcfadden_r2": m1.prsquared,
                        "auc": auc, "auc_without_state_fe": np.nan, "max_vif": vif.max(), "max_vif_term": vif.idxmax(),
                        "corr_new_base": Z.x_new_exp.corr(Z.x_base_exp)}
                m0 = sm.Logit(y_any, sm.add_constant(Z)).fit(disp=0, maxiter=300)
                r0 = pd.Series(m0.predict(sm.add_constant(Z)), index=d.index).rank()
                diag["auc_without_state_fe"] = (r0[y_any == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
                pd.Series(diag).to_csv(TAB / "exp_alignment_diagnostics.csv")
        except Exception as e:
            print("logit failed", lab, e)
        pos = d[d[col] > 0]
        if len(pos) > 50:
            Xp = sm.add_constant(pd.concat([Z.loc[pos.index], fe.loc[pos.index]], axis=1))
            Xp = Xp.loc[:, Xp.std() > 0].assign(const=1.0)
            m2 = sm.OLS(np.log(pos[col]), Xp).fit(cov_type="cluster", cov_kwds={"groups": pos.STATEFP.astype(int)})
            for k in X:
                out.append((lab, "part2_log", k, m2.params[k], m2.bse[k], m2.pvalues[k], len(pos), len(pos)))
    t = pd.DataFrame(out, columns=["outcome", "part", "term", "coef", "se", "p", "n", "n_pos"])
    t.to_csv(TAB / "exp_alignment_hurdle.csv", index=False)
    return t, d


def robustness(c):
    """Re-estimate part 1 (any funding, all programs) for the key terms under alternative samples/measures."""
    rows = []
    base = c[c.nfhl_covered.fillna(False)]
    # analytic sample of the primary model: which covariates remove covered counties
    miss = base.resPenetrationRateSfha.isna()
    full = base[["pop2020", "svi_2020", "resPenetrationRateSfha", "claims_paid_1996_2025", "dr_years",
                 "dev_sfha_km2_2000", "STATEFP"]].notna().all(axis=1)
    pd.Series({"covered": len(base), "model": int(full.sum()), "no_takeup": int(miss.sum()),
               "no_takeup_zero_struct": int((miss & (base.totalResStructuresSfha.fillna(0) == 0)).sum()),
               "no_svi": int(base.svi_2020.isna().sum()),
               "dropped_newdev_pct": 100 * base.loc[~full, "d_dev_sfha_km2"].sum() / base.d_dev_sfha_km2.sum(),
               "dropped_pop_median": base.loc[~full, "pop2020"].median(),
               "kept_pop_median": base.loc[full, "pop2020"].median()}).to_csv(TAB / "exp_sample_flow.csv")
    # (group, label, sample, exposure variable, outcome, covariate to drop)
    Y, E0 = "mit_obl_2001_2025", "d_dev_sfha_km2"
    cases = [
        ("Primary", "Primary (covered counties)", base, E0, Y, None),
        ("Sample", "Without insurance take-up (adds counties lacking it)", base, E0, Y, "x_pen"),
        ("Sample", "Coverage threshold 80% of developed land mapped", c[c.dev_mapped_share >= 0.8], E0, Y, None),
        ("Sample", "Coverage threshold 95% of developed land mapped", c[c.dev_mapped_share >= 0.95], E0, Y, None),
        ("Sample", "Maps effective by 2010 (median panel date)", base[base.eff_median <= 2010], E0, Y, None),
        ("Sample", "First FIRM before 1990 (Community Status Book)", base[base.first_firm_year.between(1960, 1989)],
         E0, Y, None),
        ("Sample", "Non-coastal counties", base[~base.coastal.astype(bool)], E0, Y, None),
        ("Sample", "All counties with any NFHL map", c[c.dev_sfha_km2_2000.notna()], E0, Y, None),
        ("Exposure measure", "Medium/high-intensity development", base, "d_devmh_sfha_km2", Y, None),
        ("Exposure measure", "Insured units in new A-zone buildings (NFIP)", base, "pol_sfha_built2001_2024", Y, None),
        ("Exposure measure", "1% + 0.2% annual-chance zones",
         base.assign(d_dev_fp500=base.d_dev_sfha_km2 + base.d_dev_x500_km2), "d_dev_fp500", Y, None),
        ("Exposure measure", "All counties, geomorphic floodplain (GFPLAIN250m)",
         c.assign(dev_sfha_km2_2000=c.dev_gfp_km2_2000) if "dev_gfp_km2_2000" in c else c.iloc[:0],
         "d_dev_gfp_km2", Y, None),
        ("Covariates", "Losses before the funding window only (NFIP 1996–2000)",
         base.assign(claims_paid_1996_2025=base.claims_paid_1996_2000), E0, Y, None),
        ("Covariates", "Losses: riverine NFIP claims only",
         base.assign(claims_paid_1996_2025=base.claims_paid_riverine), E0, Y, None),
        ("Covariates", "Declarations: flood type or flood title only",
         base.assign(dr_years=base.dr_years_narrow), E0, Y, None),
        ("Outcome", "Funding FY2001–2021 only", base, E0, "mit_obl_2001_2021", None),
        ("Outcome", "Closed/completed projects only", base, E0, "mit_obl_closed_2001_2025", None),
        ("Outcome", "Single-county projects only (no equal split)", base, E0, "mit_obl_single_2001_2025", None),
        ("Outcome", "HMA + Public Assistance 406 + NFIP ICC", base, E0, "mit_fema_all_2001_2025", None),
        ("Outcome", "Public Assistance 406 only", base, E0, "pa406_2001_2025", None),
        ("Outcome", "NFIP ICC only", base, E0, "icc_2001_2025", None),
    ]
    for grp, lab, d0, expvar, ycol, drop in cases:
        d = d0.copy()
        d["x_new_exp"] = np.log1p(d[expvar].clip(lower=0))
        d["x_base_exp"] = np.log1p(d.dev_sfha_km2_2000)
        d["x_loss"] = np.log1p(d.claims_paid_1996_2025)
        d["x_dr"], d["x_pop"], d["x_svi"], d["x_pen"] = d.dr_years, np.log(d.pop2020), d.svi_2020, d.resPenetrationRateSfha
        d["x_rural"], d["x_coastal"] = (~d.metro.fillna(True)).astype(float), d.coastal.astype(float)
        X = [k for k in ["x_new_exp", "x_base_exp", "x_loss", "x_dr", "x_pop", "x_svi", "x_pen", "x_rural", "x_coastal"]
             if k != drop]
        d = d.dropna(subset=X + ["STATEFP"])
        X = [k for k in X if d[k].nunique() > 1]  # e.g. coastal is constant among non-coastal counties
        Z = d[X].copy()
        for k in X:
            if k not in ("x_rural", "x_coastal"):  # binary indicators left unstandardized, as in the primary model
                Z[k] = z(Z[k])
        y = (d[ycol].fillna(0) > 0).astype(int)
        fe = pd.get_dummies(d.STATEFP, prefix="st", drop_first=True, dtype=float)
        keep = fe.columns[(fe.T @ y > 0).values & (fe.T @ (1 - y) > 0).values]
        try:
            m = sm.Logit(y, sm.add_constant(pd.concat([Z, fe[keep]], axis=1))).fit(
                disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": d.STATEFP.astype(int)})
            me = m.get_margeff(at="overall")
            idx = list(m.params.index[1:])
            g = lambda k: (100 * me.margeff[idx.index(k)], 100 * me.margeff_se[idx.index(k)])
            rows.append((grp, lab, len(d), *g("x_new_exp"), *g("x_base_exp"), *g("x_loss"), *g("x_dr")))
        except Exception as e:
            print("robustness failed", lab, e)
    t = pd.DataFrame(rows, columns=["group", "case", "n", "new_exp", "new_exp_se", "base_exp", "base_exp_se", "loss",
                                    "loss_se", "dr", "dr_se"])
    t.to_csv(TAB / "exp_robustness.csv", index=False)
    return t


def early_late(c, ey):
    """Strictly time-ordered version of the primary model: every predictor is measured before the funding window.
    Exposure growth 2000-2010, NFIP losses 1996-2010 and flood-declaration years 2001-2010 predict any HMA
    flood-mitigation funding in FY2011-2025. Same covariates, estimator and sample rule as the primary model."""
    d = c[c.nfhl_covered.fillna(False)].copy()
    e = ey.pivot(index="GEOID", columns="year", values="dev_sfha_km2")
    d["d_dev_0010"] = (e[2010] - e[2000]).reindex(d.index)
    mp = pd.read_parquet(PROCESSED / "county_period_mitigation_loss.parquet")
    early = mp[mp.period.isin(["2001-2005", "2006-2010"])].groupby("GEOID").claims_paid.sum()
    late = mp[mp.period.isin(["2011-2015", "2016-2020", "2021-2025"])].groupby("GEOID").obl.sum()
    d["loss_9610"] = d.claims_paid_1996_2000.fillna(0) + early.reindex(d.index).fillna(0)
    d["obl_1125"] = late.reindex(d.index).fillna(0)
    p = pd.read_parquet(PROCESSED / "county_year_panel.parquet", columns=["GEOID", "year", "dr_flood"])
    d["dr_0110"] = (p[p.year.between(2001, 2010)].assign(b=lambda x: x.dr_flood > 0)
                    .groupby("GEOID").b.sum().reindex(d.index))
    d["x_new_exp"] = np.log1p(d.d_dev_0010.clip(lower=0))
    d["x_base_exp"] = np.log1p(d.dev_sfha_km2_2000)
    d["x_loss"] = np.log1p(d.loss_9610)
    d["x_dr"], d["x_pop"], d["x_svi"], d["x_pen"] = d.dr_0110, np.log(d.pop2020), d.svi_2020, d.resPenetrationRateSfha
    d["x_rural"], d["x_coastal"] = (~d.metro.fillna(True)).astype(float), d.coastal.astype(float)
    X = ["x_new_exp", "x_base_exp", "x_loss", "x_dr", "x_pop", "x_svi", "x_pen", "x_rural", "x_coastal"]
    d = d.dropna(subset=X + ["STATEFP"])
    Z = d[X].copy()
    for k in X[:7]:
        Z[k] = z(Z[k])
    y = (d.obl_1125 > 0).astype(int)
    fe = pd.get_dummies(d.STATEFP, prefix="st", drop_first=True, dtype=float)
    keep = fe.columns[(fe.T @ y > 0).values & (fe.T @ (1 - y) > 0).values]
    m = sm.Logit(y, sm.add_constant(pd.concat([Z, fe[keep]], axis=1))).fit(
        disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": d.STATEFP.astype(int)})
    me = m.get_margeff(at="overall")
    idx = list(m.params.index[1:])
    t = pd.DataFrame({"ame": [100 * me.margeff[idx.index(k)] for k in X],
                      "se": [100 * me.margeff_se[idx.index(k)] for k in X],
                      "p": [me.pvalues[idx.index(k)] for k in X]}, index=X)
    t["n"], t["n_funded"] = len(d), int(y.sum())
    t.to_csv(TAB / "exp_early_late.csv")
    return t


def funding_sources(c):
    """Funding per person by source (HMA; HMA + Public Assistance 406 + NFIP ICC; PA 406; ICC), covered counties,
    zeros kept: OLS of log(1 + 2024 USD per person) on the part-1 terms, state FE, state-clustered SEs.
    Effects reported as % change per SD."""
    d = c[c.nfhl_covered.fillna(False)].copy()
    d["x_new_exp"] = np.log1p(d.d_dev_sfha_km2.clip(lower=0))
    d["x_base_exp"] = np.log1p(d.dev_sfha_km2_2000)
    d["x_loss"] = np.log1p(d.claims_paid_1996_2025)
    d["x_dr"], d["x_pop"], d["x_svi"], d["x_pen"] = d.dr_years, np.log(d.pop2020), d.svi_2020, d.resPenetrationRateSfha
    d["x_rural"], d["x_coastal"] = (~d.metro.fillna(True)).astype(float), d.coastal.astype(float)
    X = ["x_new_exp", "x_base_exp", "x_loss", "x_dr", "x_pop", "x_svi", "x_pen", "x_rural", "x_coastal"]
    d = d.dropna(subset=X + ["STATEFP"])
    Z = d[X].copy()
    for k in X[:7]:
        Z[k] = z(Z[k])
    fe = pd.get_dummies(d.STATEFP, prefix="st", drop_first=True, dtype=float)
    rows = []
    for lab, col in [("HMA", "mit_obl_2001_2025"), ("HMA + PA 406 + ICC", "mit_fema_all_2001_2025"),
                     ("PA 406", "pa406_2001_2025"), ("ICC", "icc_2001_2025")]:
        y = np.log1p(d[col].fillna(0) / d.pop2020)
        m = sm.OLS(y, sm.add_constant(pd.concat([Z, fe], axis=1))).fit(
            cov_type="cluster", cov_kwds={"groups": d.STATEFP.astype(int)})
        r = [lab, len(d), 100 * (d[col].fillna(0) > 0).mean()]
        for k in ("x_new_exp", "x_base_exp", "x_loss", "x_dr"):
            r += [100 * (np.exp(m.params[k]) - 1), m.pvalues[k]]
        rows.append(r)
    t = pd.DataFrame(rows, columns=["outcome", "n", "pct_any", "new_exp_pct", "p_new", "base_pct", "p_base",
                                    "loss_pct", "p_loss", "dr_pct", "p_dr"])
    t.round(3).to_csv(TAB / "exp_funding_sources_intensive.csv", index=False)
    return t


# ------------------------------------------------------------------ 4 period panel (lagged exposure growth)
def period_panel(c, ey, tf=None, out_name="exp_period_panel.csv"):
    """tf: transform of net development change; default log(1 + x) with net decreases set to 0."""
    tf = tf or (lambda x: np.log1p(x.clip(lower=0)))
    cov = c.index[c.nfhl_covered.fillna(False)]
    e = ey[ey.GEOID.isin(cov)].pivot(index="GEOID", columns="year", values="dev_sfha_km2")
    g = pd.DataFrame({f"{a+1}-{b}": e[b] - e[a] for a, b in zip(EXP_YEARS[:-1], EXP_YEARS[1:])})
    g = g.stack().rename("d_exp").reset_index().rename(columns={"level_1": "period"})
    mp = pd.read_parquet(PROCESSED / "county_period_mitigation_loss.parquet")
    p = pd.read_parquet(PROCESSED / "county_year_panel.parquet")
    p["dr_flood_bin"] = (p.dr_flood > 0).astype(int)
    p["period"] = pd.cut(p.year, [2000, 2005, 2010, 2015, 2020, 2025], labels=PERIODS).astype(str)
    drp = p[p.year >= 2001].groupby(["GEOID", "period"]).agg(dr=("dr_flood_bin", "sum"), pop=("pop", "mean")).reset_index()
    full = pd.MultiIndex.from_product([cov, PERIODS], names=["GEOID", "period"]).to_frame(index=False)
    f = full.merge(g, on=["GEOID", "period"], how="left").merge(mp, on=["GEOID", "period"], how="left") \
        .merge(drp, on=["GEOID", "period"], how="left")
    for k in ("obl", "bldg", "claims_paid", "claims_n"):
        f[k] = f[k].fillna(0)
    f["pi"] = f.period.map({p_: i for i, p_ in enumerate(PERIODS)})
    f = f.sort_values(["GEOID", "pi"])
    gg = f.groupby("GEOID")
    f["any_obl"] = 100 * (f.obl > 0)
    f["log_obl"] = np.log1p(f.obl)
    f["l_exp"] = tf(gg.d_exp.shift(1))
    f["f_exp"] = tf(gg.d_exp.shift(-1))  # lead: development in the following period
    f["l_loss"] = np.log1p(gg.claims_paid.shift(1))
    f["l_dr"] = gg.dr.shift(1)
    f["log_pop"] = np.log(f["pop"])
    f = f.merge(c[["STATEFP"]], left_on="GEOID", right_index=True)
    res = []
    for y in ("any_obl", "log_obl"):
        for spec, X in [("lagged", ["l_exp", "l_loss", "l_dr", "log_pop"]),
                        ("lagged+lead", ["l_exp", "f_exp", "l_loss", "l_dr", "log_pop"])]:
            d = f.dropna(subset=[y] + X)
            m = twfe(d[y], d[X], d.GEOID, d.period, d.STATEFP.astype(int))
            for k in X:
                res.append((y, spec, k, m.params[k], m.bse[k], m.pvalues[k], int(m.nobs), d.GEOID.nunique()))
    t = pd.DataFrame(res, columns=["y", "spec", "term", "coef", "se", "p", "nobs", "counties"])
    t.to_csv(TAB / out_name, index=False)
    return t


# ------------------------------------------------------------------ 5 typology
def typology(c, q=(1 / 3, 2 / 3)):
    d = c[c.nfhl_covered.fillna(False)].copy()
    hi = d.d_dev_sfha_km2.quantile(q[1])
    d["exp_class"] = np.where(d.d_dev_sfha_km2 >= hi, "High growth", "Lower growth")
    d["type"] = d.exp_class + ", " + np.where(d.funded, "funded", "no funding")
    t = d.groupby("type").agg(counties=("NAME", "size"), pop2020=("pop2020", "sum"),
                              new_dev_sfha_km2=("d_dev_sfha_km2", "sum"),
                              insured_new_bldg=("pol_sfha_built2001_2024", "sum"),
                              claims_paid=("claims_paid_sfha_2001_2025", "sum"),
                              ead_chg5_median=("ead_chg5", "median"))
    t["share_new_dev"] = 100 * t.new_dev_sfha_km2 / t.new_dev_sfha_km2.sum()
    t.round(2).to_csv(TAB / "exp_typology.csv")
    # threshold sensitivity
    sens = []
    for lab, thr in [("tercile", d.d_dev_sfha_km2.quantile(2 / 3)), ("quartile", d.d_dev_sfha_km2.quantile(.75)),
                     ("median", d.d_dev_sfha_km2.median()), ("decile", d.d_dev_sfha_km2.quantile(.9))]:
        h = d[d.d_dev_sfha_km2 >= thr]
        sens.append((lab, thr, len(h), (~h.funded).mean() * 100,
                     h.loc[~h.funded, "d_dev_sfha_km2"].sum() / h.d_dev_sfha_km2.sum() * 100))
    rel = d[~d.base_zero_flag.fillna(True)]
    h = rel[rel.rel_growth_sfha >= rel.rel_growth_sfha.quantile(2 / 3)]
    sens.append(("tercile of % growth", np.nan, len(h), (~h.funded).mean() * 100,
                 h.loc[~h.funded, "d_dev_sfha_km2"].sum() / h.d_dev_sfha_km2.sum() * 100))
    pd.DataFrame(sens, columns=["threshold", "km2", "n_high", "pct_unfunded", "pct_new_dev_unfunded"]).round(2) \
        .to_csv(TAB / "exp_typology_sensitivity.csv", index=False)
    # projected riverine EAD change (+5 C) in high-growth counties, unfunded vs funded: quartiles and a
    # permutation test of the difference in medians (5,000 draws)
    hg = d[d.exp_class == "High growth"]
    ua, fa = hg.loc[~hg.funded, "ead_chg5"].dropna(), hg.loc[hg.funded, "ead_chg5"].dropna()
    rng, pool = np.random.default_rng(1), np.r_[ua.values, fa.values]
    perm = np.array([np.median(x[:len(ua)]) - np.median(x[len(ua):]) for x in (rng.permutation(pool) for _ in range(5000))])
    obs = ua.median() - fa.median()
    pd.Series({"n_unf": len(ua), "n_fun": len(fa), "unf_q25": ua.quantile(.25), "unf_q75": ua.quantile(.75),
               "fun_q25": fa.quantile(.25), "fun_q75": fa.quantile(.75), "diff_median": obs,
               "perm_p": np.mean(np.abs(perm) >= abs(obs))}).to_csv(TAB / "exp_ead_unfunded_test.csv")
    # high-growth counties without HMA: did they receive flood mitigation from other sources?
    u = d[(d.exp_class == "High growth") & ~d.funded]
    pa, icc, hud = u.pa406_2001_2025.fillna(0) > 0, u.icc_2001_2025.fillna(0) > 0, u.hud_buyout_any.fillna(False).astype(bool)
    pd.Series({"n_hi_unfunded_hma": len(u), "with_pa406": int(pa.sum()), "with_icc": int(icc.sum()),
               "with_hud_buyouts": int(hud.sum()), "with_any_other": int((pa | icc | hud).sum()),
               "with_none": int((~(pa | icc | hud)).sum()),
               "none_pop_m": u.loc[~(pa | icc | hud), "pop2020"].sum() / 1e6,
               "pa406_total_musd": u.pa406_2001_2025.fillna(0).sum() / 1e6,
               "icc_total_musd": u.icc_2001_2025.fillna(0).sum() / 1e6}).to_csv(TAB / "exp_unfunded_other_sources.csv")
    # CONUS totals by source, 2001-2025 (2024 USD), all counties
    tot = {k: c[k].fillna(0).sum() / 1e9 for k in ("mit_obl_2001_2025", "pa406_2001_2025", "icc_2001_2025")}
    tot.update(icc_claims=c.icc_n_2001_2025.fillna(0).sum(),
               funded_hma=int((c.mit_obl_2001_2025.fillna(0) > 0).sum()),
               funded_any_fema=int((c.mit_fema_all_2001_2025 > 0).sum()), counties=len(c))
    pd.Series(tot).to_csv(TAB / "exp_mitigation_sources_national.csv")
    return t, d


# ------------------------------------------------------------------ 6 extensions (analysis plan v2, Q5–Q7)
def rl_coverage(c, d_ty):
    """Share of repetitive-loss properties mitigated, by region and by exposure-growth class."""
    rows = []
    for lab, d in [("CONUS (all counties)", c)] + [(r, g) for r, g in c.groupby("region")]:
        rows.append((lab, d.rl_n.sum(), d.rl_mitigated.sum(), 100 * d.rl_mitigated.sum() / d.rl_n.sum(),
                     d.srl_n.sum(), 100 * d.srl_mitigated.sum() / max(d.srl_n.sum(), 1)))
    for lab, d in d_ty.groupby("type"):
        rows.append((lab, d.rl_n.sum(), d.rl_mitigated.sum(), 100 * d.rl_mitigated.sum() / max(d.rl_n.sum(), 1),
                     d.srl_n.sum(), 100 * d.srl_mitigated.sum() / max(d.srl_n.sum(), 1)))
    t = pd.DataFrame(rows, columns=["group", "rl_props", "rl_mitigated", "rl_pct_mitigated", "srl_props", "srl_pct_mitigated"])
    t.round(2).to_csv(TAB / "exp_rl_coverage.csv", index=False)
    return t


def throughput(c):
    """Annual flows, CONUS counties: insured A-zone buildings (in force 2024) by construction year 2001-2024 vs
    buildings acquired or elevated with HMA funds by fiscal year 2001-2024 (county-allocated, as in Table 3)."""
    conus = set(c.index)
    pol = pd.read_parquet(PROCESSED / "nfip_policies_agg.parquet")
    p = pol[(pol.pol_year == 2024) & (pol.zone == "sfha_a") & pol.con_year.between(2001, 2024) & pol.GEOID.isin(conus)]
    new_by_year = p.groupby("con_year").policies.sum()  # policyCount = insured units
    sf_by_year = p[p.occ == 11].groupby("con_year").policies.sum()  # single-family residences: one policy, one building
    h = pd.read_parquet(PROCESSED / "hma_projects_county.parquet")
    v = pd.read_parquet(PROCESSED.parent / "raw" / "openfema" / "HmaSubapplications_v2.parquet",
                        columns=["subapplicationIdentifier", "numberBuildingLocations"])
    h = h[h.flood_activity & h.primaryActivity.isin(["Acquisition", "Relocation", "Elevation", "Mitigation reconstruction"])
          & h.GEOID.isin(conus)].merge(v, on="subapplicationIdentifier", how="left")
    h["bldg"] = h.numberBuildingLocations.fillna(0) * h.share
    mit_by_year = h[h.fiscalYear.between(2001, 2024)].groupby("fiscalYear").bldg.sum()
    # upper bound on mitigated buildings: add every NFIP ICC claim (elevate/floodproof/relocate/demolish), although
    # ICC often pays the local share of HMA projects, so some buildings are counted twice
    cl = pd.read_parquet(PROCESSED.parent / "raw" / "openfema" / "NfipClaims_v3_paged.parquet",
                         columns=["yearOfLoss", "countyCode", "amountPaidOnIncreasedCostOfComplianceClaim"])
    cl = cl[cl.countyCode.astype(str).str.zfill(5).isin(conus)]
    icc_by_year = cl[cl.yearOfLoss.between(2001, 2024) &
                     (pd.to_numeric(cl.amountPaidOnIncreasedCostOfComplianceClaim, errors="coerce") > 0)]         .groupby("yearOfLoss").size()
    unit = pd.read_csv(TAB / "exp_unit_cost.csv", index_col=0).iloc[:, 0]
    pen = c.resPenetrationRateSfha.where(c.resPenetrationRateSfha > 0.05)  # guard against tiny denominators
    adj = (c.pol_sfha_built2001_2024 / pen).clip(upper=c.pol_sfha_built2001_2024 * 10).sum()
    out = {
        "new_insured_per_year": new_by_year.mean(), "mitigated_per_year": mit_by_year.mean(),
        "flow_ratio": new_by_year.mean() / mit_by_year.mean(),
        "new_sf_per_year": sf_by_year.mean(), "flow_ratio_sf": sf_by_year.mean() / mit_by_year.mean(),
        "icc_per_year": icc_by_year.reindex(range(2001, 2025), fill_value=0).mean(),
        "flow_ratio_with_icc": new_by_year.mean() / (mit_by_year.mean() +
                                                     icc_by_year.reindex(range(2001, 2025), fill_value=0).mean()),
        "new_insured_2001_2024": new_by_year.sum(), "mitigated_2001_2024": mit_by_year.sum(),
        "penetration_adjusted_new_bldg": adj,
        "scale_usd_bn_at_acq_cost": new_by_year.sum() * unit.get("Acquisition", np.nan) / 1e9,
        "scale_usd_bn_at_elev_cost": new_by_year.sum() * unit.get("Elevation", np.nan) / 1e9,
        "actual_flood_obl_2001_2025_bn": c.mit_obl_2001_2025.sum() / 1e9,
    }
    hy = h[h.fiscalYear.between(2001, 2024)]
    acq = hy[hy.primaryActivity.isin(["Acquisition", "Relocation"])].groupby("fiscalYear").bldg.sum()
    yrs = range(2001, 2025)
    pd.DataFrame({"new_insured": new_by_year, "new_sf": sf_by_year, "mitigated": mit_by_year,
                  "acquired": acq.reindex(yrs, fill_value=0), "elevated": (mit_by_year - acq).reindex(yrs, fill_value=0),
                  "icc": icc_by_year.reindex(yrs, fill_value=0)}).reindex(yrs).to_csv(TAB / "exp_flows_by_year.csv")
    pd.Series(out).to_csv(TAB / "exp_throughput.csv")
    return out


def left_behind(d_ty):
    """Among high-growth covered counties: correlates of receiving no flood-mitigation funding."""
    d = d_ty[d_ty.exp_class == "High growth"].copy()
    d["no_funding"] = (~d.funded).astype(int)
    d["x_svi"], d["x_rural"] = d.svi_2020, (~d.metro.fillna(True)).astype(float)
    d["x_pop"], d["x_plan"], d["x_crs"] = np.log(d.pop2020), d.plan_ok.astype(float), d.crs_any.astype(float)
    d["x_coastal"], d["x_newexp"] = d.coastal.astype(float), np.log1p(d.d_dev_sfha_km2.clip(lower=0))
    X = ["x_svi", "x_rural", "x_pop", "x_plan", "x_crs", "x_coastal", "x_newexp"]
    d = d.dropna(subset=X)
    Z = d[X].copy()
    for k in ("x_svi", "x_pop", "x_newexp"):
        Z[k] = z(Z[k])
    desc = d.groupby("no_funding")[["svi_2020", "x_rural", "pop2020", "x_plan", "x_crs", "d_dev_sfha_km2"]].mean()
    desc.to_csv(TAB / "exp_left_behind_means.csv")
    rows = []
    try:
        m = sm.Logit(d.no_funding, sm.add_constant(Z)).fit(disp=0, maxiter=300, cov_type="cluster",
                                                           cov_kwds={"groups": d.STATEFP.astype(int)})
        me = m.get_margeff(at="overall")
        for i, k in enumerate(m.params.index[1:]):
            rows.append((k, 100 * me.margeff[i], 100 * me.margeff_se[i], me.pvalues[i], len(d), int(d.no_funding.sum())))
    except Exception as e:
        print("left_behind logit failed:", e)
    t = pd.DataFrame(rows, columns=["term", "ame_pp", "se", "p", "n", "n_unfunded"])
    t.to_csv(TAB / "exp_left_behind_logit.csv", index=False)
    return t, desc


def spatial_check(c, counties):
    """Spatial dependence in the primary any-funding model: Moran's I of the outcome and of Pearson residuals
    (queen contiguity among covered counties, 999 permutations). Also re-estimates the model with spatially
    lagged predictors (neighbour means) as a sensitivity test."""
    import libpysal
    from esda.moran import Moran
    d = c[c.nfhl_covered.fillna(False)].copy()
    d["x_new_exp"], d["x_base_exp"] = np.log1p(d.d_dev_sfha_km2.clip(lower=0)), np.log1p(d.dev_sfha_km2_2000)
    d["x_loss"], d["x_dr"], d["x_pop"] = np.log1p(d.claims_paid_1996_2025), d.dr_years, np.log(d.pop2020)
    d["x_svi"], d["x_pen"] = d.svi_2020, d.resPenetrationRateSfha
    d["x_rural"], d["x_coastal"] = (~d.metro.fillna(True)).astype(float), d.coastal.astype(float)
    X = ["x_new_exp", "x_base_exp", "x_loss", "x_dr", "x_pop", "x_svi", "x_pen", "x_rural", "x_coastal"]
    d = d.dropna(subset=X + ["STATEFP"])
    g = counties.set_index("GEOID").loc[d.index].reset_index()
    w = libpysal.weights.Queen.from_dataframe(g, use_index=False, silence_warnings=True)
    w.transform = "r"
    y = (d.mit_obl_2001_2025 > 0).astype(int).values
    Z = d[X].copy()
    for k in X[:7]:
        Z[k] = z(Z[k])
    fe = pd.get_dummies(d.STATEFP, prefix="st", drop_first=True, dtype=float)
    keep = fe.columns[(fe.T @ y > 0).values & (fe.T @ (1 - y) > 0).values]
    m = sm.Logit(y, sm.add_constant(pd.concat([Z, fe[keep]], axis=1))).fit(disp=0, maxiter=300)
    resid = (y - m.predict()) / np.sqrt(m.predict() * (1 - m.predict()))
    mi_y, mi_r = Moran(y, w, permutations=999), Moran(resid, w, permutations=999)
    # sensitivity: add neighbour means of the exposure and loss terms (spatially lagged X)
    lag = pd.DataFrame({f"w_{k}": libpysal.weights.lag_spatial(w, Z[k].values) for k in ("x_new_exp", "x_base_exp", "x_loss")},
                       index=d.index)
    m2 = sm.Logit(y, sm.add_constant(pd.concat([Z, lag, fe[keep]], axis=1))).fit(
        disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": d.STATEFP.astype(int)})
    me = m2.get_margeff(at="overall")
    idx = list(m2.params.index[1:])
    get = lambda k: (100 * me.margeff[idx.index(k)], 100 * me.margeff_se[idx.index(k)], me.pvalues[idx.index(k)])
    out = {"moran_y": mi_y.I, "moran_y_p": mi_y.p_sim, "moran_resid": mi_r.I, "moran_resid_p": mi_r.p_sim,
           "n": len(d), "islands": len(w.islands)}
    for k in ("x_new_exp", "x_base_exp", "x_loss"):
        out[f"slx_{k}"], out[f"slx_{k}_se"], out[f"slx_{k}_p"] = get(k)
    pd.Series(out).to_csv(TAB / "exp_spatial_check.csv")
    return out


def hmgp_by_incident():
    """Share of HMGP flood-mitigation obligations (FY2001-2025) by the incident type of the triggering declaration."""
    dd = pd.read_parquet(PROCESSED.parent / "raw" / "openfema" / "DisasterDeclarationsSummaries_v2.parquet")
    inc = dd[dd.declarationType == "DR"].drop_duplicates("disasterNumber")
    inc = inc.assign(dn=pd.to_numeric(inc.disasterNumber, errors="coerce"),
                     titled=inc.declarationTitle.str.contains("FLOOD", case=False, na=False)).set_index("dn")
    h = pd.read_parquet(PROCESSED / "hma_projects_county.parquet")
    h = h[h.flood_activity & (h.prog == "HMGP") & h.fiscalYear.between(2001, 2025)].copy()
    h["dn"] = pd.to_numeric(h.disasterNumber, errors="coerce")
    h = h.join(inc[["incidentType", "titled"]], on="dn")
    grp = h.incidentType.fillna("Unknown").replace({"Tropical Storm": "Hurricane/tropical/coastal storm",
                                                   "Hurricane": "Hurricane/tropical/coastal storm",
                                                   "Coastal Storm": "Hurricane/tropical/coastal storm",
                                                   "Typhoon": "Hurricane/tropical/coastal storm"})
    t = (100 * h.groupby(grp).obl_real_alloc.sum() / h.obl_real_alloc.sum()).sort_values(ascending=False)
    hur = h.incidentType.isin(["Hurricane", "Tropical Storm", "Coastal Storm", "Typhoon"])
    t.loc["(hurricane-type declarations titled 'flood', % of their $)"] = \
        100 * h.loc[hur & h.titled, "obl_real_alloc"].sum() / h.loc[hur, "obl_real_alloc"].sum()
    t.round(2).to_csv(TAB / "exp_hmgp_by_incident.csv")
    return t


def geography_checks(c, d_ty):
    """Facts quoted in the text about coverage and geography, computed rather than eyeballed from maps."""
    cov = c.nfhl_covered.fillna(False)
    nc = c[~cov]
    g = c[cov].groupby("STUSPS").d_dev_sfha_km2.sum().sort_values(ascending=False)
    u = d_ty[d_ty.type == "High growth, no funding"]
    x = c[cov].dropna(subset=["d_dev_gfp_km2", "d_dev_sfha_km2"])
    gn = c.dropna(subset=["d_dev_gfp_km2"])
    s = pd.read_parquet(PROCESSED.parent / "raw" / "openfema" / "HmaSubapplicationsProjectSiteInventories_v1_paged.parquet")
    s = s[s.primaryActivity.isin(["Acquisition", "Elevation"])]
    zz = s.floodZone.fillna("").str.upper()
    known = zz[zz != ""]
    out = {
        "nc_n": len(nc), "nc_texas": int((nc.STUSPS == "TX").sum()),
        "cov_inland_n": int((cov & ~c.coastal.astype(bool)).sum()),
        "coastal_cty_pct": 100 * c.coastal.astype(bool).mean(),
        "coastal_obl_pct": 100 * c.loc[c.coastal.astype(bool), "mit_obl_2001_2025"].sum() / c.mit_obl_2001_2025.sum(),
        "coastal_pop_pct": 100 * c.loc[c.coastal.astype(bool), "pop2020"].sum() / c.pop2020.sum(),
        "coastal_newdev_pct": 100 * c.loc[c.coastal.astype(bool), "d_dev_sfha_km2"].sum() / c.d_dev_sfha_km2.sum(),
        "nc_west": int((nc.region == "West").sum()), "nc_midwest": int((nc.region == "Midwest").sum()),
        "nc_south": int((nc.region == "South").sum()), "nc_northeast": int((nc.region == "Northeast").sum()),
        "nc_nonmetro_pct": 100 * (~nc.metro.fillna(True)).mean(), "cov_nonmetro_pct": 100 * (~c[cov].metro.fillna(True)).mean(),
        "nc_pop_pct": 100 * nc.pop2020.sum() / c.pop2020.sum(),
        "nc_claims_pct": 100 * nc.claims_paid_1996_2025.sum() / c.claims_paid_1996_2025.sum(),
        "gfp_growth_share_uncovered_pct": 100 * gn[~gn.nfhl_covered.fillna(False)].d_dev_gfp_km2.sum() / gn.d_dev_gfp_km2.sum(),
        "gfp_rho_growth": x.d_dev_gfp_km2.corr(x.d_dev_sfha_km2, method="spearman"),
        "gfp_rho_level": x.dev_gfp_km2_2000.corr(x.dev_sfha_km2_2000, method="spearman"), "gfp_n": len(x),
        "fl_tx_share_pct": 100 * g.reindex(["FL", "TX"]).sum() / g.sum(), "top10_share_pct": 100 * g.head(10).sum() / g.sum(),
        "top10_states": ", ".join(g.head(10).index),
        "unf_top_states": ", ".join(u.STUSPS.value_counts().head(5).index), "unf_coastal_pct": 100 * u.coastal.mean(),
        "mit_zone_known_pct": 100 * len(known) / len(s),
        "mit_in_sfha_pct": 100 * known.str.contains(r"^(?:A|AE|AH|AO|A\d|A1|V|VE)").mean(),
    }
    pd.Series(out).to_csv(TAB / "exp_geography_checks.csv")
    return out


# ------------------------------------------------------------------ figures
def fig_exposure_map(c, counties):
    g = counties.copy()
    for k in ("d_dev_sfha_km2", "rel_growth_sfha", "nfhl_covered"):
        g[k] = g.GEOID.map(c[k])
    g["base"] = g.GEOID.map(c["dev_sfha_km2_2000"])
    states = g.dissolve("STATEFP")
    small = "#d9d9d9"
    panels = [
        ("d_dev_sfha_km2", "a",
         "Developed land added in 1%\nfloodplain, 2000–2025 (km²)", [-np.inf, 0, 1, 5, 20, np.inf], ["No increase", "< 1", "1–5", "5–20", "> 20"],
         ["#bdbdbd"] + SEQ4, None),
        ("rel_growth_sfha", "b",
         "Increase in developed floodplain\nland since 2000 (%)", [-np.inf, 10, 25, 50, np.inf], ["< 10", "10–25", "25–50", "> 50"], SEQ4,
         "Little floodplain development in 2000 (< 0.5 km²)"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.2), gridspec_kw={"hspace": 0.12})
    for ax, (col, title, ltitle, bins, labels, colors, small_lab) in zip(axes, panels):
        cov = g[g.nfhl_covered.fillna(False)].copy()
        g[g.nfhl_covered != True].plot(ax=ax, color="white", edgecolor="#d5d5d5", linewidth=0.05)
        has = cov[col].notna()
        if small_lab:
            cov[~has].plot(ax=ax, color=small, edgecolor="#d0d0d0", linewidth=0.05)
        k = np.digitize(cov.loc[has, col], bins[1:-1], right=True)
        cov[has].plot(ax=ax, color=[colors[i] for i in k], edgecolor="#d0d0d0", linewidth=0.05)
        states.boundary.plot(ax=ax, color="#555555", linewidth=0.3)
        ax.set_axis_off()
        ax.text(0.0, 1.0, title, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
        hand = [mpl.patches.Patch(facecolor=cc, edgecolor="#888888", linewidth=0.3) for cc in colors]
        labs = list(labels)
        if small_lab:
            hand.append(mpl.patches.Patch(facecolor=small, edgecolor="#888888", linewidth=0.3))
            labs.append(small_lab)
        hand.append(mpl.patches.Patch(facecolor="white", edgecolor="#888888", linewidth=0.3))
        labs.append("No adequate FEMA flood map (not analysed)")
        wrap = {"Little floodplain development in 2000 (< 0.5 km²)": "Little floodplain development\nin 2000 (< 0.5 km²)",
                "No adequate FEMA flood map (not analysed)": "No adequate FEMA\nflood map (not analysed)"}
        labs = [wrap.get(l_, l_) for l_ in labs]
        # anchored at the top so both panels' legends start at the same height
        ax.legend(hand, labs, title=ltitle, loc="upper left", bbox_to_anchor=(0.885, 0.56), ncol=1, fontsize=8,
                  title_fontsize=8.5, handlelength=1.1, labelspacing=0.35, alignment="left", borderaxespad=0)
    save(fig, "Fig1_exposure_map")


EXAMPLES = [  # (GEOID, row group) -- high-growth counties chosen to span flood settings and funding status
    ("12086", "coastal"), ("48201", "coastal"), ("12071", "coastal"),
    ("04013", "inland"), ("06067", "inland"), ("22063", "inland"),
    ("06047", "inland"), ("35061", "inland"), ("48135", "inland")]
SHIFT_KM = {"35061": 6}  # move the window east so the Rio Grande corridor is not under the inset


def fig_examples(c, d_ty, counties, km=24):
    """30 m examples of development in the 1% floodplain (A zones) over a greyscale satellite basemap for nine
    high-growth counties: coastal funded, inland funded and inland unfunded. Each panel has a state locator
    (upper right), the FEMA funding status (upper left) and the largest incorporated places in the window."""
    import json
    import contextily as cx
    import matplotlib.patheffects as pe
    import rasterio
    from affine import Affine
    from rasterio.windows import Window
    from shapely.geometry import box
    from config import BIG, RAW
    gj = json.load(open(BIG / "nlcd_grid.json"))
    gtr = Affine(*gj["transform"])
    nlcd = {y: sorted((BIG / "nlcd").glob(f"*LndCov_{y}_CU_C1V2*.tif"))[0] for y in (2000, 2025)}
    states = counties.dissolve("STATEFP")
    places = gpd.read_file(f"zip://{RAW / 'census' / 'cb_2020_us_place_500k.zip'}")
    places = places[places.LSAD != "57"].to_crs(counties.crs)  # incorporated places only (no CDPs)
    cols = {2: (BLUE[200], 0.3), 3: (BLUE[400], 1.0), 4: ("#0b2545", 1.0)}
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    fig, axes = plt.subplots(3, 3, figsize=(7.4, 8.9))
    for ax, (geoid, _), letter in zip(axes.ravel(), EXAMPLES, "abcdefghi"):
        tifs = sorted((BIG / "exposure" / "zones").glob(f"{geoid}__*.tif"))
        zone = None
        for t in tifs:
            with rasterio.open(t) as zs:
                a, tr = zs.read(1), zs.transform
            zone = a if zone is None else np.maximum(zone, a)
        col, row = ~gtr * (tr.c, tr.f)
        win = Window(int(round(col)), int(round(row)), zone.shape[1], zone.shape[0])
        with rasterio.open(nlcd[2000]) as s0, rasterio.open(nlcd[2025]) as s1:
            d0 = np.isin(s0.read(1, window=win), [21, 22, 23, 24])
            d1 = np.isin(s1.read(1, window=win), [21, 22, 23, 24])
        sf = zone == 6
        new = sf & d1 & ~d0
        n = int(km * 1000 / 30)
        cs = np.pad(new.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        if new.shape[0] > n and new.shape[1] > n:
            tot = cs[n:, n:] - cs[:-n, n:] - cs[n:, :-n] + cs[:-n, :-n]
            r0, c0 = np.unravel_index(np.argmax(tot[::10, ::10]), tot[::10, ::10].shape)
            r0, c0 = r0 * 10, c0 * 10
        else:
            r0, c0 = 0, 0
        c0 = int(np.clip(c0 + SHIFT_KM.get(geoid, 0) * 1000 / 30, 0, max(new.shape[1] - n, 0)))
        sl = (slice(r0, r0 + n), slice(c0, c0 + n))
        img = np.zeros(sf[sl].shape, np.uint8)
        img[sf[sl]] = 2                           # SFHA, undeveloped
        img[sf[sl] & d0[sl]] = 3                  # SFHA, developed by 2000
        img[new[sl]] = 4                          # SFHA, developed 2001-2025
        rgba = np.zeros(img.shape + (4,))
        for k, (cc, al) in cols.items():
            rgba[img == k] = (*mpl.colors.to_rgb(cc), al)
        h, w = img.shape
        x0, y0 = gtr * (win.col_off + c0, win.row_off + r0)
        ext = (x0, x0 + w * 30, y0 - h * 30, y0)
        ax.set_xlim(ext[0], ext[1]), ax.set_ylim(ext[2], ext[3])
        # lightened greyscale satellite basemap (Esri World Imagery); attribution given in the caption
        cx.add_basemap(ax, crs=counties.crs, source=cx.providers.Esri.WorldImagery, zoom=12, attribution=False)
        bm = ax.images[-1]
        arr = np.asarray(bm.get_array()).astype(float)
        arr = arr / 255 if arr.max() > 1 else arr
        g = 0.2 + 0.8 * (arr[..., :3] @ [0.299, 0.587, 0.114])
        bm.set_data(np.dstack([g, g, g]))
        ax.imshow(rgba, extent=ext, interpolation="nearest", zorder=5)
        ax.set_xlim(ext[0], ext[1]), ax.set_ylim(ext[2], ext[3])
        # fade the part of the window outside the county and draw the county boundary
        cty = counties[counties.GEOID == geoid]
        cgeom = cty.geometry.union_all()
        gpd.GeoSeries([box(ext[0], ext[2], ext[1], ext[3]).difference(cgeom)], crs=counties.crs).plot(
            ax=ax, color="white", alpha=0.6, lw=0, zorder=5.5)
        cty.boundary.plot(ax=ax, color="#222222", lw=0.9, zorder=5.6)
        ax.set_xlim(ext[0], ext[1]), ax.set_ylim(ext[2], ext[3])
        # state locator, upper right: state outline with a star at the window
        ins = ax.inset_axes([0.70, 0.66, 0.29, 0.33], zorder=8)
        states.loc[[cty.STATEFP.iloc[0]]].plot(ax=ins, color="white", edgecolor="#333333", lw=0.6)
        cty.plot(ax=ins, color=BLUE[300], edgecolor="#333333", lw=0.4)  # whole county
        ins.plot((ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2, marker="*", ms=8, color="#ffd23f",
                 markeredgecolor="#333333", markeredgewidth=0.6)
        ins.set_axis_off(), ins.set_aspect("equal")
        # upper left: floodplain development added 2000-2025 and HMA flood-mitigation obligations FY2001-2025
        obl = d_ty.loc[geoid, "mit_obl_2001_2025"]
        obl = 0 if pd.isna(obl) else obl
        money = "no HMA funding" if obl <= 0 else ("HMA $" + (f"{obl / 1e9:.1f} billion" if obl >= 1e9
                                                              else f"{obl / 1e6:,.0f} million"))
        sub_line = f"+{d_ty.loc[geoid, 'd_dev_sfha_km2']:.1f} km² · {money}"  # county totals, below the title
        # place names: largest incorporated places by area inside the window, clear of inset, label and scale bar
        m = 0.08 * w * 30
        frame = box(ext[0] + m, ext[2] + 0.12 * h * 30, ext[1] - m, ext[3] - m)
        frame = frame.difference(box(ext[0] + 0.66 * w * 30, ext[2] + 0.62 * h * 30, ext[1], ext[3]))
        frame = frame.intersection(cgeom)
        pl = places[places.STATEFP == cty.STATEFP.iloc[0]]
        pl = gpd.GeoDataFrame({"NAME": pl.NAME}, geometry=pl.geometry.intersection(frame), crs=pl.crs)
        pl = pl[pl.area > 0.004 * (w * 30) * (h * 30)]
        pl = pl.assign(a=pl.area).sort_values("a", ascending=False).head(2)
        for nm, pt in zip(pl.NAME, pl.geometry.representative_point()):
            hw = len(nm) * 3.9 * (w * 30 / 166) / 2 + 400  # half label width in metres (7 pt text, ~166 pt panel)
            ax.text(np.clip(pt.x, ext[0] + hw, ext[1] - hw), pt.y, nm, fontsize=7, ha="center", va="center", color="#111111", zorder=6,
                    path_effects=halo)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True), sp.set_color("#888888"), sp.set_linewidth(0.5)
        name = f"{c.loc[geoid, 'NAME']}, {c.loc[geoid, 'STUSPS']}"
        ax.text(0.0, 1.085, f"{letter}   {name}", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
        ax.text(0.0, 1.015, sub_line, transform=ax.transAxes, fontsize=7.5, color="#444444", va="bottom")
        sx, sy = ext[0] + 0.05 * w * 30, ext[2] + 0.05 * h * 30
        ax.plot([sx, sx + 5000], [sy] * 2, color="black", lw=1.6, zorder=6,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ax.text(sx + 2500, sy + 0.015 * h * 30, "5 km", ha="center", va="bottom", fontsize=7, zorder=6,
                path_effects=halo)
    # legend swatches show each class as it appears over the greyscale basemap (mean grey about 0.7)
    over = lambda cc, al: tuple(al * np.array(mpl.colors.to_rgb(cc)) + (1 - al) * 0.6)
    hd = [mpl.lines.Line2D([], [], ls="none", marker="*", ms=9, color="#ffd23f", markeredgecolor="#333333",
                           markeredgewidth=0.6)]
    hd += [mpl.patches.Patch(facecolor=over(*cols[k]), edgecolor="#888888", linewidth=0.3) for k in (4, 3, 2)]
    hd.append(mpl.lines.Line2D([], [], color="#222222", lw=0.9))
    # two rows (filled by column): floodplain classes on top, location and boundary below
    star, new, old, undev, bnd = hd
    fig.legend([new, star, old, bnd, undev],
               ["1% floodplain, developed 2001–2025", "Panel location; county shaded (inset)", "1% floodplain, developed by 2000",
                "County boundary (outside faded)", "1% floodplain, undeveloped"],
               loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.05), handlelength=1.4,
               columnspacing=1.4)
    fig.tight_layout(w_pad=0.6, h_pad=1.6)
    # map-only crops of each panel (used by the graphical abstract)
    fig.canvas.draw()
    for ax, (geoid, _) in zip(axes.ravel(), EXAMPLES):
        ext_in = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(FIG / f"panel_{geoid}.png", dpi=300, bbox_inches=ext_in)
    save(fig, "Fig2_examples")
    return [g_ for g_, _ in EXAMPLES]


def fig_trends_and_ratio(y, c):
    """(a) developed land inside vs outside A zones; (b) insured units in new A-zone buildings by year built;
    (c) buildings acquired or elevated with HMA funds by fiscal year, with ICC claims by year of loss;
    (d) cumulative totals 2001-2024. (b) and (c) are on different time bases, so they are drawn separately and
    compared only as totals in (d)."""
    fl = pd.read_csv(TAB / "exp_flows_by_year.csv", index_col=0).loc[2001:2024]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6))
    lab = lambda ax, k: ax.text(-0.02, 1.03, f"{k}", transform=ax.transAxes, fontsize=12, fontweight="bold",
                                va="bottom", ha="right")
    # (a)
    ax = axes[0, 0]
    idx = 100 * y[["dev_sfha_km2", "dev_out_km2"]] / y[["dev_sfha_km2", "dev_out_km2"]].iloc[0]
    ax.plot(idx.index, idx.dev_sfha_km2, color="#0b2545", marker="o", ms=4, lw=1.6, label="Inside A zones")
    ax.plot(idx.index, idx.dev_out_km2, color=BLUE[300], marker="s", ms=4, lw=1.6, ls="--", label="Outside floodplain")
    ax.set_ylabel("Developed land\n(index, 2000 = 100)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    lab(ax, "a")
    # (b) insured units by year of construction
    ax = axes[0, 1]
    other = fl.new_insured - fl.new_sf
    ax.bar(fl.index, fl.new_sf / 1e3, color="#0b2545", width=0.8, label="Single-family")
    ax.bar(fl.index, other / 1e3, bottom=fl.new_sf / 1e3, color=BLUE[300], width=0.8, label="Other units")
    ax.set_ylabel("Insured units, new A-zone\nbuildings (thousands)")
    ax.set_xlabel("Year of construction")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    lab(ax, "b")
    # (c) mitigation by fiscal year (HMA) and ICC claims by year of loss
    ax = axes[1, 0]
    ax.bar(fl.index, fl.acquired / 1e3, color="#0b2545", width=0.8, label="HMA acquired")
    ax.bar(fl.index, fl.elevated / 1e3, bottom=fl.acquired / 1e3, color=BLUE[400], width=0.8, label="HMA elevated")
    ax.plot(fl.index, fl.icc / 1e3, color=BLUE[200], marker="o", ms=3.5, lw=1.4, label="NFIP ICC claims", zorder=5)
    ax.set_ylabel("Buildings or ICC claims\n(thousands)")
    ax.set_xlabel("Fiscal year (HMA) or year of loss (ICC)")
    top = max((fl.acquired + fl.elevated).max(), fl.icc.max()) / 1e3
    ax.set_ylim(0, top * 1.5)  # headroom so the legend sits above the data
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    lab(ax, "c")
    # (d) cumulative totals 2001-2024
    ax = axes[1, 1]
    tot = fl.sum() / 1e3
    bars = [("All insured\nunits", tot.new_insured, "#0b2545"), ("Single-\nfamily", tot.new_sf, BLUE[550]),
            ("HMA acquired\nor elevated", tot.mitigated, BLUE[400]), ("HMA +\nICC", tot.mitigated + tot.icc, BLUE[200])]
    xs = np.arange(len(bars))
    ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.65)
    for x_, (_, v, _) in zip(xs, bars):
        ax.text(x_, v + 8, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs, [b[0] for b in bars], fontsize=9)
    ax.set_ylabel("Total, 2001–2024 (thousands)")
    ax.set_ylim(0, tot.new_insured * 1.28)
    ax.axvline(1.5, color="#999999", lw=0.6, ls=":")
    ax.text(0.5, tot.new_insured * 1.19, "Insured units (2024)", ha="center", fontsize=9, color=INK)
    ax.text(2.5, tot.new_insured * 1.19, "HMA project records", ha="center", fontsize=9, color=INK)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    lab(ax, "d")
    fig.tight_layout(h_pad=1.6, w_pad=2.0)
    save(fig, "Fig3_trends_flows")


def fig_typology(d, counties):
    order = ["High growth, no funding", "High growth, funded", "Lower growth, funded", "Lower growth, no funding"]
    cols = dict(zip(order, [BLUE[700], BLUE[400], BLUE[100], "#bdbdbd"]))
    g = counties.copy()
    g["cls"] = g.GEOID.map(d["type"])  # not "type": GeoDataFrame.type is the geometry-type property
    states = g.dissolve("STATEFP")
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    g[g.cls.isna()].plot(ax=ax, color="white", edgecolor="#d5d5d5", linewidth=0.05)
    for k in order:
        if (g.cls == k).any():
            g[g.cls == k].plot(ax=ax, color=cols[k], edgecolor="#cccccc", linewidth=0.05)
    states.boundary.plot(ax=ax, color="#555555", linewidth=0.3)
    ax.set_axis_off()
    h = [mpl.patches.Patch(facecolor=cols[k], edgecolor="#888888", linewidth=0.3) for k in order] + \
        [mpl.patches.Patch(facecolor="white", edgecolor="#888888", linewidth=0.3)]
    ax.legend(h, ["High growth, no HMA funding", "High growth, HMA funded", "Lower growth, HMA funded",
                   "Lower growth, no HMA funding",
                  "No adequate FEMA\nflood map (not analysed)"], loc="upper left", bbox_to_anchor=(0.885, 0.62), ncol=1,
              borderaxespad=0, fontsize=8, handlelength=1.1, labelspacing=0.45, alignment="left", title_fontsize=8.5,
              title="Floodplain development growth\n(2000–2025) and HMA flood-\nmitigation funding (2001–2025)")
    save(fig, "Fig5_typology")


def fig_alignment(t):
    lab = {"x_new_exp": "New A-zone development", "x_base_exp": "A-zone development, 2000",
           "x_loss": "NFIP claims paid", "x_dr": "Years with flood declaration", "x_pop": "Population",
           "x_svi": "Social vulnerability", "x_pen": "Insurance take-up (SFHA)"}
    progs = [("All programs", "#0b2545", "o"), ("HMGP", BLUE[550], "s"), ("FMA", BLUE[300], "D"),
             ("PDM/BRIC", BLUE[200], "^")]
    d = t[(t.part == "part1_any") & t.term.isin(lab) & t.outcome.isin([p_ for p_, _, _ in progs])]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    terms = list(lab)
    for j, (prog, col, mk) in enumerate(progs):
        s = d[d.outcome == prog].set_index("term").reindex(terms)
        yy = np.arange(len(terms)) + (j - 1.5) * 0.19
        ax.errorbar(s.coef, yy, xerr=1.96 * s.se, fmt=mk, color=col, ms=5, capsize=2, elinewidth=1,
                    markeredgecolor="white", markeredgewidth=0.5, label=prog)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(len(terms)), [lab[k] for k in terms])
    ax.invert_yaxis()
    ax.set_xlabel("Change in probability of any HMA flood-mitigation\nfunding, per SD (percentage points)")
    ax.legend(loc="lower right", title="Program", alignment="left")
    ax.grid(axis="x", color="#e6e6e6", lw=0.5)
    save(fig, "Fig4_alignment")


def main():
    c, ey = load()
    counties = gpd.read_file(PROCESSED / "counties.gpkg")
    y, rt, meta = national_trends(c, ey)
    ratio, _ = building_ratio(c)
    t_al, _ = alignment(c)
    t_rb = robustness(c)
    print(funding_sources(c).round(3).to_string())
    print(early_late(c, ey).round(3).to_string())
    print(t_rb.round(2).to_string())
    t_pp = period_panel(c, ey)
    t_ty, d_ty = typology(c)
    print(spatial_check(c, counties))
    fig_examples(c, d_ty, counties)
    t_rl = rl_coverage(c, d_ty)
    print(hmgp_by_incident())
    print(geography_checks(c, d_ty))
    t_tp = throughput(c)
    t_lb, lb_desc = left_behind(d_ty)
    print(t_rl.round(2).to_string()); print(t_tp); print(t_lb.round(3).to_string()); print(lb_desc.round(3).to_string())
    fig_exposure_map(c, counties)
    fig_trends_and_ratio(y, c)
    fig_typology(d_ty, counties)
    fig_alignment(t_al)
    pd.set_option("display.width", 220)
    print(meta)
    print(y.round(2).to_string())
    print(rt.round(2).to_string())
    print(ratio.round(2).to_string())
    print(t_al[t_al.part == "part1_any"].round(3).to_string())
    print(t_pp.round(3).to_string())
    print(t_ty.round(1).to_string())


if __name__ == "__main__":
    main()
