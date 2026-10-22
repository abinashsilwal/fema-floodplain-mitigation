"""Collect every number quoted in the manuscript into outputs/tables/numbers.json.

Every number reported in the article is read from this file, so text and results cannot drift apart.
"""
import json

import pandas as pd

from config import PROCESSED, RAW, TAB


def fmt(x, d=0):
    return f"{x:,.{d}f}".replace("-", "−")


def main():
    n = {}
    cov = pd.read_csv(TAB / "exp_coverage.csv", index_col=0).iloc[:, 0]
    n["cov_n"] = fmt(cov.n_counties_covered)
    n["nfhl_counties"] = fmt(cov.n_counties_nfhl)
    n["cov_pop_share"] = fmt(100 * cov.pop2020_share_covered, 1)
    n["cov_dev_share"] = fmt(100 * cov.dev2025_share_covered, 1)
    log = pd.read_csv(RAW / "nfhl_catalog.csv", dtype={"dfirm": str})
    n["nfhl_deliveries"] = fmt((~log.dfirm.str[:2].isin(["02", "15", "60", "66", "69", "72", "78"])).sum())

    y = pd.read_csv(TAB / "exp_national_trends.csv", index_col=0)
    n["dev_sfha_2000"] = fmt(y.dev_sfha_km2[2000])
    n["dev_sfha_2025"] = fmt(y.dev_sfha_km2[2025])
    n["d_dev_sfha"] = fmt(y.dev_sfha_km2[2025] - y.dev_sfha_km2[2000])
    n["growth_sfha_pct"] = fmt(100 * (y.dev_sfha_km2[2025] / y.dev_sfha_km2[2000] - 1), 1)
    n["growth_out_pct"] = fmt(100 * (y.dev_out_km2[2025] / y.dev_out_km2[2000] - 1), 1)
    n["newdev_share_all"] = fmt(100 * (y.dev_sfha_km2[2025] - y.dev_sfha_km2[2000]) / (y.dev_km2[2025] - y.dev_km2[2000]), 1)
    for yr in (2005, 2010, 2015, 2020, 2025):
        n[f"newdev_share_{yr}"] = fmt(y.new_dev_share_sfha[yr], 1)
        n[f"growth_sfha_{yr}"] = fmt(y.growth_sfha_pct[yr], 1)
        n[f"growth_out_{yr}"] = fmt(y.growth_out_pct[yr], 1)
    n["share_dev_in_sfha_2000"] = fmt(y.share_dev_in_sfha[2000], 2)
    n["share_dev_in_sfha_2025"] = fmt(y.share_dev_in_sfha[2025], 2)
    n["devmh_growth_pct"] = fmt(100 * (y.devmh_sfha_km2[2025] / y.devmh_sfha_km2[2000] - 1), 1)

    r = pd.read_csv(TAB / "exp_region_growth.csv", index_col=0)
    for reg in r.index:
        k = reg.lower().replace(" ", "")
        n[f"growth_sfha_{k}"] = fmt(r.loc[reg, "growth_sfha_pct"], 1)
        n[f"growth_all_{k}"] = fmt(r.loc[reg, "growth_all_pct"], 1)
        n[f"newdev_share_{k}"] = fmt(r.loc[reg, "new_dev_share_sfha"], 1)

    b = pd.read_csv(TAB / "exp_building_ratio.csv").set_index("area")
    a = b.loc["CONUS (all counties)"]
    n["ins_new_bldg"] = fmt(a.insured_new_sfha_bldg)
    n["acq_2001_2024"] = fmt(a.acquired)
    n["elev_2001_2024"] = fmt(a.elevated)
    n["mit_2001_2024"] = fmt(a.acquired + a.elevated)
    n["ratio_ins"] = fmt(a.ratio_insured_per_mitigated, 1)
    cv = b.loc["CONUS (covered counties)"]
    n["ratio_ins_cov"] = fmt(cv.ratio_insured_per_mitigated, 1)
    n["ratio_est_cov"] = fmt(cv.ratio_est_per_mitigated, 1)
    n["est_new_struct"] = fmt(cv.est_new_sfha_res_struct)
    for reg in ("Midwest", "Northeast", "South", "West"):
        if reg in b.index:
            n[f"ratio_{reg.lower()}"] = fmt(b.loc[reg, "ratio_insured_per_mitigated"], 1)

    m = pd.read_parquet(PROCESSED / "county_mitigation_loss.parquet")
    n["acq_all"] = fmt(m.mit_acq_all.sum())
    n["elev_all"] = fmt(m.mit_elev_all.sum())
    n["fema_sfha_structures"] = fmt(m.totalResStructuresSfha.sum())
    pol = pd.read_parquet(PROCESSED / "nfip_policies_agg.parquet")
    n["policies_rows"] = "72.6 million"
    p24 = pol[(pol.pol_year == 2024) & (pol.zone == "sfha_a")]
    n["pol_sfha_2024"] = fmt(p24.policies.sum())
    n["pol_sfha_new_share"] = fmt(100 * p24[p24.con_year >= 2001].policies.sum() / p24.policies.sum(), 1)

    cl = pd.read_csv(TAB / "nfip_claims_national.csv", index_col=0).iloc[:, 0]
    n["claims_total"] = fmt(cl.claims_total)
    n["claims_paid_bn"] = fmt(cl.paid_total_bn, 1)
    n["claims_share_new2000"] = fmt(100 * cl.share_claims_new2000, 1)
    n["paid_share_new2000"] = fmt(100 * cl.share_paid_new2000, 1)
    n["paid_new2000_bn"] = fmt(cl.paid_total_bn * cl.share_paid_new2000, 1)
    n["claims_outside_sfha"] = fmt(100 * cl.share_claims_outside_sfha, 1)

    al = pd.read_csv(TAB / "exp_alignment_hurdle.csv")
    for (o, part, t), g in al.groupby(["outcome", "part", "term"]):
        k = f"al_{o.split()[0].lower().replace('/', '')}_{part[:5]}_{t[2:]}"
        n[k] = fmt(g.coef.iloc[0], 1 if part.startswith("part1") else 3)
        n[k + "_se"] = fmt(g.se.iloc[0], 1 if part.startswith("part1") else 3)
        n[k + "_p"] = f"{g.p.iloc[0]:.3f}" if g.p.iloc[0] >= 0.001 else "< 0.001"
        if part.startswith("part1"):  # 95% confidence interval
            n[k + "_lo"] = fmt(g.coef.iloc[0] - 1.96 * g.se.iloc[0], 1)
            n[k + "_hi"] = fmt(g.coef.iloc[0] + 1.96 * g.se.iloc[0], 1)
    first = al[(al.outcome == "All programs") & (al.part == "part1_any")]
    if len(first):
        n["al_n"] = fmt(first.n.iloc[0])
        n["al_npos"] = fmt(first.n_pos.iloc[0])

    pp = pd.read_csv(TAB / "exp_period_panel.csv")
    for r_ in pp.itertuples():
        k = f"pp_{r_.y}_{r_.spec.replace('+', '_')}_{r_.term}"
        n[k] = fmt(r_.coef, 2 if r_.y.startswith("any") else 3)
        n[k + "_se"] = fmt(r_.se, 2 if r_.y.startswith("any") else 3)
        n[k + "_p"] = f"{r_.p:.2f}" if r_.p >= 0.01 else "< 0.01"
    n["pp_counties"] = fmt(pp.counties.iloc[0])

    ty = pd.read_csv(TAB / "exp_typology.csv", index_col=0)
    hi = ty.loc[[i for i in ty.index if i.startswith("High")]]
    n["ty_high_n"] = fmt(hi.counties.sum())
    if "High growth, no funding" in ty.index:
        u = ty.loc["High growth, no funding"]
        n["ty_hi_unf_n"] = fmt(u.counties)
        n["ty_hi_unf_pct"] = fmt(100 * u.counties / hi.counties.sum(), 1)
        n["ty_hi_unf_pop"] = fmt(u.pop2020 / 1e6, 1)
        n["ty_hi_unf_newdev"] = fmt(u.new_dev_sfha_km2)
        n["ty_hi_unf_share"] = fmt(u.share_new_dev, 1)
        n["ty_hi_unf_bldg"] = fmt(u.insured_new_bldg)
        n["ty_hi_unf_ead"] = fmt(u.ead_chg5_median, 1)
    if "High growth, funded" in ty.index:
        n["ty_hi_fun_ead"] = fmt(ty.loc["High growth, funded", "ead_chg5_median"], 1)
    ts = pd.read_csv(TAB / "exp_typology_sensitivity.csv")
    n["ty_sens_min"] = fmt(ts.pct_unfunded.min(), 1)
    n["ty_sens_max"] = fmt(ts.pct_unfunded.max(), 1)

    rb = pd.read_csv(TAB / "exp_robustness.csv")
    n["rb_newexp_min"] = fmt(rb.new_exp.min(), 1)
    n["rb_newexp_max"] = fmt(rb.new_exp.max(), 1)
    n["rb_base_min"] = fmt(rb.base_exp.min(), 1)
    n["rb_base_max"] = fmt(rb.base_exp.max(), 1)
    n["rb_loss_min"] = fmt(rb.loss.min(), 1)
    n["rb_loss_max"] = fmt(rb.loss.max(), 1)

    # earlier funding-timing analysis (03_analysis.py)
    es = pd.read_csv(TAB / "event_study_flood_declarations.csv")
    g = es.set_index(["outcome", "rel_year"])
    n["es_all_t0"] = fmt(g.loc[("All programs", 0), "coef"], 1)
    n["es_all_t0_se"] = fmt(g.loc[("All programs", 0), "se"], 1)
    n["es_all_mean"] = fmt(g.loc[("All programs", 0), "mean_pct"], 1)
    n["es_hmgp_t0"] = fmt(g.loc[("HMGP", 0), "coef"], 1)
    for k in (0, 1, 2):  # dated by submission to FEMA (actual process date)
        n[f"es_sub_l{k}"] = fmt(g.loc[("All programs, submission year", k), "coef"], 1)
        n[f"es_sub_l{k}_se"] = fmt(g.loc[("All programs, submission year", k), "se"], 1)
        n[f"es_hmgp_sub_l{k}"] = fmt(g.loc[("HMGP, submission year", k), "coef"], 1)
    n["es_fma_l1"] = fmt(g.loc[("FMA", 1), "coef"], 1)
    n["es_fma_l3"] = fmt(g.loc[("FMA", 3), "coef"], 1)
    n["es_fma_mean"] = fmt(g.loc[("FMA", 0), "mean_pct"], 1)
    prog = pd.read_csv(TAB / "fig2_program_by_year_bn2024.csv", index_col=0)
    w = prog.loc[2001:2021]
    n["obl_2001_2021_bn"] = fmt(w.values.sum(), 1)
    n["share_hmgp"] = fmt(100 * w["HMGP"].sum() / w.values.sum(), 1)
    n["share_fma"] = fmt(100 * w["FMA"].sum() / w.values.sum(), 1)
    n["share_pdm"] = fmt(100 * w["PDM/BRIC"].sum() / w.values.sum(), 1)

    # ---- extensions (analysis plan v2)
    n["dev_out_growth_pct"] = fmt(100 * (y.dev_out_km2[2025] / y.dev_out_km2[2000] - 1), 1)
    n["dev_all_growth_pct"] = fmt(100 * (y.dev_km2[2025] / y.dev_km2[2000] - 1), 1)
    n["x500_growth_pct"] = fmt(100 * (y.dev_x500_km2[2025] / y.dev_x500_km2[2000] - 1), 1)
    n["dev_x500_2025"] = fmt(y.dev_x500_km2[2025])
    tp = pd.read_csv(TAB / "exp_throughput.csv", index_col=0).iloc[:, 0]
    n["new_per_year"] = fmt(tp.new_insured_per_year, -2 if False else 0)
    n["mit_per_year"] = fmt(tp.mitigated_per_year)
    n["flow_ratio"] = fmt(tp.flow_ratio, 1)
    n["icc_per_year"] = fmt(tp.icc_per_year)
    n["flow_ratio_icc"] = fmt(tp.flow_ratio_with_icc, 1)
    n["new_sf_per_year"] = fmt(tp.new_sf_per_year)
    fly = pd.read_csv(TAB / "exp_flows_by_year.csv", index_col=0).loc[2001:2024]
    n["ratio_sf_acq"] = fmt(fly.new_sf.sum() / fly.acquired.sum(), 1)  # comparable with buyout-only comparisons
    n["acq_2001_2024_flow"] = fmt(fly.acquired.sum())
    n["flow_ratio_sf"] = fmt(tp.flow_ratio_sf, 1)
    n["flow_ratio_sf_icc"] = fmt(tp.new_sf_per_year / (tp.mitigated_per_year + tp.icc_per_year), 1)
    n["new_insured_2001_2024"] = fmt(tp.new_insured_2001_2024)
    n["mitigated_2001_2024"] = fmt(tp.mitigated_2001_2024)
    n["pen_adj_new"] = fmt(tp.penetration_adjusted_new_bldg / 1e6, 2)
    n["scale_acq_bn"] = fmt(tp.scale_usd_bn_at_acq_cost)
    n["scale_elev_bn"] = fmt(tp.scale_usd_bn_at_elev_cost)
    n["actual_obl_bn"] = fmt(tp.actual_flood_obl_2001_2025_bn, 1)
    uc = pd.read_csv(TAB / "exp_unit_cost.csv", index_col=0).iloc[:, 0]
    n["unit_acq_k"] = fmt(uc["Acquisition"] / 1e3)
    n["unit_elev_k"] = fmt(uc["Elevation"] / 1e3)
    rl = pd.read_csv(TAB / "exp_repetitive_loss_national.csv", index_col=0).iloc[:, 0]
    n["rl_total"] = fmt(rl.rl_total)
    n["rl_mitigated"] = fmt(rl.rl_mitigated)
    n["rl_pct"] = fmt(100 * rl.rl_share_mitigated, 1)
    n["srl_total"] = fmt(rl.srl_total)
    n["srl_pct"] = fmt(100 * rl.srl_share_mitigated, 1)
    n["rl_sfha_pct"] = fmt(100 * rl.rl_share_sfha, 1)
    n["rl_unmit_losses"] = fmt(rl.rl_unmitigated_losses)
    rc = pd.read_csv(TAB / "exp_rl_coverage.csv").set_index("group")
    for k, lab in (("hi_unf", "High growth, no funding"), ("hi_fun", "High growth, funded")):
        if lab in rc.index:
            n[f"rl_pct_{k}"] = fmt(rc.loc[lab, "rl_pct_mitigated"], 1)
    lb = pd.read_csv(TAB / "exp_left_behind_logit.csv")
    for r_ in lb.itertuples():
        n[f"lb_{r_.term[2:]}"] = fmt(r_.ame_pp, 1)
        n[f"lb_{r_.term[2:]}_se"] = fmt(r_.se, 1)
        n[f"lb_{r_.term[2:]}_p"] = f"{r_.p:.3f}" if r_.p >= 0.001 else "< 0.001"
    if len(lb):
        n["lb_n"], n["lb_unf"] = fmt(lb.n.iloc[0]), fmt(lb.n_unfunded.iloc[0])
    lm = pd.read_csv(TAB / "exp_left_behind_means.csv", index_col=0)
    n["lb_crs_funded"] = fmt(100 * lm.loc[0, "x_crs"], 0)
    n["lb_crs_unfunded"] = fmt(100 * lm.loc[1, "x_crs"], 0)
    n["lb_rural_funded"] = fmt(100 * lm.loc[0, "x_rural"], 0)
    n["lb_rural_unfunded"] = fmt(100 * lm.loc[1, "x_rural"], 0)
    n["lb_pop_funded_k"] = fmt(lm.loc[0, "pop2020"] / 1e3)
    n["lb_pop_unfunded_k"] = fmt(lm.loc[1, "pop2020"] / 1e3)
    ts2 = ts.set_index("threshold")
    n["ty_unf_pct_tercile"] = fmt(ts2.loc["tercile", "pct_unfunded"], 1)
    n["rb_n_specs"] = str(len(rb) - 1)  # alternatives to the primary model
    alt = rb[rb.group != "Primary"]
    n["rb_n_sigpos"] = str(int((alt.new_exp - 1.96 * alt.new_exp_se > 0).sum()))
    n["rb_newexp_hi_max"] = fmt((alt.new_exp + 1.96 * alt.new_exp_se).max(), 1)
    rc = rb.set_index("case")
    for key, case in (("cov80", "Coverage threshold 80% of developed land mapped"),
                      ("cov95", "Coverage threshold 95% of developed land mapped"),
                      ("pre", "Losses before the funding window only (NFIP 1996–2000)"),
                      ("single", "Single-county projects only (no equal split)"),
                      ("nopen", "Without insurance take-up (adds counties lacking it)")):
        n[f"rb_{key}_new"] = fmt(rc.loc[case, "new_exp"], 1)
        n[f"rb_{key}_loss"] = fmt(rc.loc[case, "loss"], 1)
        n[f"rb_{key}_n"] = fmt(rc.loc[case, "n"])
        n[f"rb_{key}_lo"] = fmt(rc.loc[case, "new_exp"] - 1.96 * rc.loc[case, "new_exp_se"], 1)
        n[f"rb_{key}_hi"] = fmt(rc.loc[case, "new_exp"] + 1.96 * rc.loc[case, "new_exp_se"], 1)
    am = pd.read_csv(TAB / "exp_applications_models.csv")
    for key, mod in (("apA", "A."), ("apA2", "A2."), ("apB", "B."), ("apC", "C."), ("apD", "D."), ("apE", "E."), ("apF", "F.")):
        g = am[am.model.str.startswith(mod)].set_index("term")
        for t_ in ("new_exp", "base_exp", "loss", "dr", "pop"):
            r_ = g.loc[f"x_{t_}"]
            n[f"{key}_{t_}"] = fmt(r_.coef, 1)
            n[f"{key}_{t_}_p"] = "< 0.01" if r_.p < 0.01 else f"{r_.p:.2f}"
            n[f"{key}_{t_}_lo"] = fmt(r_.coef - 1.96 * r_.se, 1)
            n[f"{key}_{t_}_hi"] = fmt(r_.coef + 1.96 * r_.se, 1)
        n[f"{key}_n"] = fmt(g.n.iloc[0])
    asum = pd.read_csv(TAB / "exp_applications_summary.csv", index_col=0).iloc[:, 0]
    for k in ("apps_total", "apps_funded", "apps_not_funded", "apps_undecided", "hi_unfunded_n", "hi_unfunded_applied"):
        n[f"ap_{k}"] = fmt(asum[k])
    for k in ("covered_applied_pct", "hi_applied_pct", "lo_applied_pct", "hi_success_pct", "lo_success_pct"):
        n[f"ap_{k}"] = fmt(asum[k], 0)
    n["ap_hi_unfunded_never"] = fmt(asum["hi_unfunded_n"] - asum["hi_unfunded_applied"])
    # data-quality and timing checks requested by the second external review
    hraw = pd.read_parquet(RAW / "openfema" / "HmaSubapplications_v2.parquet", columns=["fiscalYear", "totalObligatedAmount"])
    n["hma_fy1018_bn"] = fmt(hraw.loc[hraw.fiscalYear.between(2010, 2018), "totalObligatedAmount"].sum() / 1e9, 1)
    el = pd.read_csv(TAB / "exp_early_late.csv", index_col=0)
    for k in ("new_exp", "base_exp", "loss", "dr"):
        r_ = el.loc[f"x_{k}"]
        n[f"el_{k}"] = fmt(r_.ame, 1)
        n[f"el_{k}_p"] = "< 0.01" if r_.p < 0.01 else f"{r_.p:.2f}"
        n[f"el_{k}_lo"], n[f"el_{k}_hi"] = fmt(r_.ame - 1.96 * r_.se, 1), fmt(r_.ame + 1.96 * r_.se, 1)
    qa = pd.read_csv(TAB / "qa_nlcd_summary.csv", index_col=0).iloc[:, 0]
    n["qa_counties"] = fmt(qa["counties"])
    n["qa_loss_share"] = fmt(qa["loss_share_of_gain_2000_2025_pct"], 0)
    n["qa_loss_share_recent"] = fmt(qa["loss_share_of_gain_2020_2025_pct"], 0)
    rr = pd.read_csv(TAB / "exp_new_vs_old_claim_rates_region.csv")
    n["cr_region_min"], n["cr_region_max"] = fmt(100 * rr.rate_ratio.min(), 0), fmt(100 * rr.rate_ratio.max(), 0)
    af = pd.read_csv(TAB / "exp_applications_flow.csv").set_index("step")["count"]
    pick = lambda s_: af[af.index.str.strip().str.startswith(s_)].iloc[0]
    n["ap_pairs"] = fmt(pick("Application-county pairs"))
    n["ap_pairs_cov"] = fmt(pick("pairs in covered counties"))
    n["ap_closed0"] = fmt(pick("closed with no obligation"))
    n["ap_pending"] = fmt(pick("still pending"))
    nv = pd.read_csv(TAB / "exp_never_applied_model.csv", index_col=0)
    for k in ("plan", "crs", "pop", "rural", "loss", "dr", "svi", "new"):
        r_ = nv.loc[f"x_{k}"]
        n[f"nv_{k}"] = fmt(r_.ame, 1)
        n[f"nv_{k}_p"] = "< 0.01" if r_.p < 0.01 else f"{r_.p:.2f}"
    n["nv_n"], n["nv_never"] = fmt(nv.n.iloc[0]), fmt(nv.n_never.iloc[0])
    nm = pd.read_csv(TAB / "exp_never_applied_means.csv", index_col=0)
    n["nv_loss_never_m"] = fmt(nm.loc[1, "claims_paid_1996_2025"] / 1e6, 1)
    n["nv_loss_applied_m"] = fmt(nm.loc[0, "claims_paid_1996_2025"] / 1e6, 0)
    n["nv_crs_never"], n["nv_crs_applied"] = fmt(100 * nm.loc[1, "x_crs"], 0), fmt(100 * nm.loc[0, "x_crs"], 0)
    n["ap_hi_unf_all_n"] = fmt(asum["hi_unfunded_all_n"])
    n["ap_hi_unf_all_never"] = fmt(asum["hi_unfunded_all_never"])
    n["ap_match_funded"] = fmt(asum["match_funded_pct"], 0)
    n["ap_match_not_funded"] = fmt(asum["match_not_funded_pct"], 0)
    dg = pd.read_csv(TAB / "exp_alignment_diagnostics.csv", index_col=0).iloc[:, 0]
    n["diag_auc"] = f"{float(dg['auc']):.2f}"
    n["diag_auc_nofe"] = f"{float(dg['auc_without_state_fe']):.2f}"
    n["diag_r2"] = f"{float(dg['mcfadden_r2']):.2f}"
    n["diag_vif"] = fmt(float(dg["max_vif"]), 1)
    n["diag_corr"] = f"{float(dg['corr_new_base']):.2f}"
    n["diag_states"] = fmt(float(dg["states_in_model"]))
    xw = pd.read_csv(TAB / "exp_ead_crosswalk_check.csv", index_col=0).iloc[:, 0]
    n["aw_rho_chg"] = f"{float(xw['spearman_chg5_centroid_vs_aw']):.2f}"
    n["aw_huc"] = fmt(float(xw["huc10_matched"]))
    n["aw_unf_med"] = fmt(float(xw["aw_unf_median"]), 1)
    n["aw_fun_med"] = fmt(float(xw["aw_fun_median"]), 1)
    n["aw_p"] = f"{float(xw['aw_perm_p']):.2f}"
    et = pd.read_csv(TAB / "exp_ead_unfunded_test.csv", index_col=0).iloc[:, 0]
    for k in ("unf_q25", "unf_q75", "fun_q25", "fun_q75"):
        n[f"ead_{k}"] = fmt(et[k], 0)
    n["ead_perm_p"] = f"{et['perm_p']:.2f}"
    sf = pd.read_csv(TAB / "exp_sample_flow.csv", index_col=0).iloc[:, 0]
    for k in ("covered", "model", "no_takeup", "no_takeup_zero_struct", "no_svi"):
        n[f"smp_{k}"] = fmt(sf[k])
    n["smp_dropped"] = fmt(sf["covered"] - sf["model"])
    n["smp_dropped_newdev_pct"] = fmt(sf["dropped_newdev_pct"], 1)
    n["smp_dropped_pop_med"] = fmt(round(sf["dropped_pop_median"], -2))
    n["smp_kept_pop_med"] = fmt(round(sf["kept_pop_median"], -2))
    cr = pd.read_csv(TAB / "exp_new_vs_old_claim_rates.csv", index_col=0)
    n["cr_new"] = fmt(cr.loc["built_2001_plus", "claims_per_1000_py"], 1)
    n["cr_old"] = fmt(cr.loc["built_before_2001", "claims_per_1000_py"], 1)
    n["pr_new"] = fmt(cr.loc["built_2001_plus", "paid_per_py_usd2024"])
    n["pr_old"] = fmt(cr.loc["built_before_2001", "paid_per_py_usd2024"])
    n["cr_ratio_pct"] = fmt(100 * cr.loc["built_2001_plus", "claims_per_1000_py"] / cr.loc["built_before_2001", "claims_per_1000_py"])
    n["pr_ratio_pct"] = fmt(100 * cr.loc["built_2001_plus", "paid_per_py_usd2024"] / cr.loc["built_before_2001", "paid_per_py_usd2024"])
    # shares of insured A-zone policy-years, claims and payments (2016-2024) from buildings constructed 2001+
    paid = cr.policy_years * cr.paid_per_py_usd2024
    n["cr_new_py_share"] = fmt(100 * cr.loc["built_2001_plus", "policy_years"] / cr.policy_years.sum())
    n["cr_new_claim_share"] = fmt(100 * cr.loc["built_2001_plus", "claims"] / cr.claims.sum())
    n["cr_new_paid_share"] = fmt(100 * paid["built_2001_plus"] / paid.sum())
    sc = pd.read_csv(TAB / "exp_spatial_check.csv", index_col=0).iloc[:, 0]
    n["moran_y"], n["moran_resid"] = f"{sc.moran_y:.2f}", f"{sc.moran_resid:.2f}"
    n["moran_resid_p"] = f"{sc.moran_resid_p:.3f}"
    for k in ("x_new_exp", "x_base_exp", "x_loss"):
        n[f"slx_{k[2:]}"] = fmt(sc[f"slx_{k}"], 1)
        n[f"slx_{k[2:]}_p"] = f"{sc[f'slx_{k}_p']:.2f}" if sc[f"slx_{k}_p"] >= 0.01 else "< 0.01"
    fl = pd.read_csv(TAB / "exp_flows_by_year.csv", index_col=0).loc[2001:2024]
    r = fl.new_insured / fl.mitigated
    n["flow_ratio_min"], n["flow_ratio_max"] = fmt(r.min(), 0), fmt(r.max(), 0)
    n["flow_ratio_min_year"], n["flow_ratio_max_year"] = str(int(r.idxmin())), str(int(r.idxmax()))
    n["mit_max_year"], n["mit_max"] = str(int(fl.mitigated.idxmax())), fmt(fl.mitigated.max())
    ft = pd.read_csv(TAB / "exp_claims_by_flood_type.csv")
    tt = ft.groupby("ftype")[["n", "paid", "pct_claims", "pct_paid"]].sum()
    for k in ("riverine", "coastal", "pluvial", "other"):
        n[f"ft_{k}_claims"] = fmt(tt.loc[k, "pct_claims"], 0)
        n[f"ft_{k}_paid"] = fmt(tt.loc[k, "pct_paid"], 0)
        x = ft[ft.ftype == k]
        n[f"ft_{k}_outside"] = fmt(100 * x[~x.sfha].n.sum() / x.n.sum(), 0)
    hi = pd.read_csv(TAB / "exp_hmgp_by_incident.csv", index_col=0).iloc[:, 0]
    n["hmgp_hurr_pct"] = fmt(hi["Hurricane/tropical/coastal storm"], 0)
    n["hmgp_severe_pct"] = fmt(hi["Severe Storm"], 0)
    n["hmgp_flood_pct"] = fmt(hi["Flood"], 0)
    n["hmgp_bio_pct"] = fmt(hi["Biological"], 1)
    cc = pd.read_csv(TAB / "concentration_top10pct.csv", index_col=0).iloc[:, 0]
    n["conc_obl"] = fmt(float(cc["Obligations"]), 0)
    n["conc_dmg"] = fmt(float(cc["Reported damage"]), 0)
    n["conc_ead"] = fmt(float(cc["Modelled EAD (EPA)"]), 0)
    gc = pd.read_csv(TAB / "exp_geography_checks.csv", index_col=0).iloc[:, 0]
    for k in ("nc_n", "nc_texas", "nc_west", "nc_midwest", "nc_south", "nc_northeast", "gfp_n", "cov_inland_n"):
        n[k] = fmt(float(gc[k]))
    for k in ("nc_nonmetro_pct", "cov_nonmetro_pct", "nc_pop_pct", "nc_claims_pct", "gfp_growth_share_uncovered_pct",
              "fl_tx_share_pct", "top10_share_pct", "unf_coastal_pct", "mit_zone_known_pct", "mit_in_sfha_pct",
              "coastal_cty_pct", "coastal_obl_pct", "coastal_pop_pct", "coastal_newdev_pct"):
        n[k] = fmt(float(gc[k]), 1 if k not in ("nc_nonmetro_pct", "cov_nonmetro_pct", "unf_coastal_pct",
                                                 "mit_zone_known_pct", "mit_in_sfha_pct", "top10_share_pct") else 0)
    n["gfp_rho_growth"] = f"{float(gc['gfp_rho_growth']):.2f}"
    n["gfp_rho_level"] = f"{float(gc['gfp_rho_level']):.2f}"
    n["top10_states"] = gc["top10_states"]
    n["unf_top_states"] = gc["unf_top_states"]
    fs = pd.read_csv(TAB / "exp_funding_sources_intensive.csv").set_index("outcome")
    for key, lab in (("fs_hma", "HMA"), ("fs_all", "HMA + PA 406 + ICC"), ("fs_pa", "PA 406"), ("fs_icc", "ICC")):
        n[f"{key}_new"] = fmt(fs.loc[lab, "new_exp_pct"], 1)
        n[f"{key}_p"] = "< 0.01" if fs.loc[lab, "p_new"] < 0.01 else f"{fs.loc[lab, 'p_new']:.2f}"
        n[f"{key}_any_pct"] = fmt(fs.loc[lab, "pct_any"], 0)
    n["fs_all_dr"] = fmt(fs.loc["HMA + PA 406 + ICC", "dr_pct"], 0)
    n["fs_all_loss"] = fmt(fs.loc["HMA + PA 406 + ICC", "loss_pct"], 0)
    n["fs_all_base"] = fmt(fs.loc["HMA + PA 406 + ICC", "base_pct"], 0)
    ms = pd.read_csv(TAB / "exp_mitigation_sources_national.csv", index_col=0).iloc[:, 0]
    n["src_hma_bn"], n["src_pa_bn"], n["src_icc_bn"] = (fmt(ms[k], 1) for k in
                                                         ("mit_obl_2001_2025", "pa406_2001_2025", "icc_2001_2025"))
    n["src_icc_claims"] = fmt(ms["icc_claims"])
    n["src_any_n"] = fmt(ms["funded_any_fema"])
    uo = pd.read_csv(TAB / "exp_unfunded_other_sources.csv", index_col=0).iloc[:, 0]
    for k in ("with_pa406", "with_icc", "with_hud_buyouts", "with_none"):
        n[f"unf_{k}"] = fmt(uo[k])
    n["unf_none_pop"] = fmt(uo["none_pop_m"], 1)
    n["unf_pa_musd"] = fmt(uo["pa406_total_musd"], 0)
    n["unf_icc_musd"] = fmt(uo["icc_total_musd"], 0)
    # construction year of mitigated buildings vs insured stock and claims (17_within_county.py)
    yb = pd.read_csv(TAB / "wc_year_built.csv").set_index("built_from")
    for cut in (1990, 2001):
        n[f"yb_units_{cut}"] = fmt(yb.loc[cut, "insured_units_pct"], 1)
        n[f"yb_claims_{cut}"] = fmt(yb.loc[cut, "claims_pct"], 1)
        n[f"yb_mit_{cut}"] = fmt(yb.loc[cut, "mitigated_pct"], 1)
    n["yb_n_mit"] = fmt(yb.n_mitigated.iloc[0])
    n["yb_median"] = str(int(yb.mitigated_median_year.iloc[0]))
    wz = pd.read_csv(TAB / "wc_zip_model.csv")
    for out_, lab in (("site_any", "any"), ("site_funded", "fund")):
        r = wz[wz.outcome == out_].set_index("term")
        for t, k in (("z_units_new", "new"), ("z_units_old", "old"), ("z_claims", "claims")):
            n[f"wz_{lab}_{k}"] = fmt(r.loc[t, "coef"], 1)
            n[f"wz_{lab}_{k}_lo"], n[f"wz_{lab}_{k}_hi"] = fmt(r.loc[t, "lo"], 1), fmt(r.loc[t, "hi"], 1)
        n[f"wz_{lab}_nzip"], n[f"wz_{lab}_ncounty"] = fmt(r.n_zip.iloc[0]), fmt(r.n_county.iloc[0])
    for out_, lab in (("site_any_share", "any"), ("site_funded_share", "fund")):
        r = wz[wz.outcome == out_].set_index("term")
        n[f"wz_{lab}_share"] = fmt(r.loc["z_share_new", "coef"], 1)
        n[f"wz_{lab}_share_lo"], n[f"wz_{lab}_share_hi"] = (fmt(r.loc["z_share_new", k], 1) for k in ("lo", "hi"))
        for k in ("coef", "lo", "hi"):  # two decimals where a bound is close to zero
            n[f"wz_{lab}_share_{k}2"] = fmt(r.loc["z_share_new", k], 2)
    # sample sizes for the statistics and reproducibility section
    am = pd.read_csv(TAB / "exp_applications_models.csv").groupby("model")[["n", "n_pos"]].first()
    for pre, k in (("A. ", "apA"), ("A2.", "apA2"), ("C. ", "apC")):
        r = am[am.index.str.startswith(pre)].iloc[0]
        n[f"{k}_n"], n[f"{k}_npos"] = fmt(r.n), fmt(r.n_pos)
    pp_ = pd.read_csv(TAB / "exp_period_panel.csv")
    pp_ = pp_[(pp_.y == "any_obl") & (pp_.spec == "lagged")].iloc[0]
    n["pp_nobs"], n["pp_counties"] = fmt(pp_.nobs), fmt(pp_.counties)
    # coefficient contrasts and site-inventory status checks (20_contrasts.py)
    ct = pd.read_csv(TAB / "exp_contrasts.csv")
    for mdl, mk in (("Logit AME, state-clustered", "logit"), ("LPM, Conley (widest of 50-800 km)", "lpm")):
        for con, ck in (("x_loss - x_new_exp", "loss"), ("x_base_exp - x_new_exp", "base"), ("x_dr - x_new_exp", "dr")):
            r = ct[(ct.model == mdl) & (ct.contrast == con)].iloc[0]
            n[f"ct_{ck}_{mk}"], n[f"ct_{ck}_{mk}_lo"], n[f"ct_{ck}_{mk}_hi"] = (fmt(r[k], 1) for k in ("diff_pp", "lo", "hi"))
    for yv, k in (("site_any", "any"), ("site_funded", "fund")):
        r = ct[ct.model == f"ZIP LPM, county FE ({yv})"].iloc[0]
        n[f"ct_zip_{k}"], n[f"ct_zip_{k}_lo"], n[f"ct_zip_{k}_hi"] = (fmt(r[c_], 1) for c_ in ("diff_pp", "lo", "hi"))
    ss = pd.read_csv(TAB / "wc_site_status.csv").set_index("group")
    n["ss_known_pct"] = fmt(100 * ss.year_known_share.iloc[0], 1)
    n["ss_fma_pct"] = fmt(100 * ss.funded_fma_share.iloc[0])
    n["ss_paid_new"], n["ss_paid_n"] = fmt(ss.loc["Funded with payments > 0", "built_2001_pct"], 1), \
        fmt(ss.loc["Funded with payments > 0", "sites"])
    uf, uu = ss.loc["Unique sites, funded in any subapplication"], ss.loc["Unique sites, never funded"]
    n["ss_unf_new"], n["ss_unf_n"] = fmt(uu.built_2001_pct, 1), fmt(uu.sites)
    n["ss_fund_uniq_new"], n["ss_fund_uniq_n"] = fmt(uf.built_2001_pct, 1), fmt(uf.sites)
    n["ss_unf_new_all"] = fmt(ss.loc["Not funded", "built_2001_pct"], 1)
    n["ss_fund_subapp_new"] = fmt(ss.loc["Funded (obligation > 0)", "subapp_mean_built_2001_pct"], 1)
    n["ss_unf_subapp_new"] = fmt(uu.subapp_mean_built_2001_pct, 1)
    n["ss_rep_keys"], n["ss_rep_rows"] = fmt(ss.repeated_site_keys.iloc[0]), fmt(ss.rows_with_repeated_key.iloc[0])
    o = ss.loc["Funded, other programs"]
    n["ss_nonfma_new"], n["ss_nonfma_n"] = fmt(o.built_2001_pct, 1), fmt(o.sites)
    n["ss_fma_new"] = fmt(ss.loc["Funded, Flood Mitigation Assistance", "built_2001_pct"], 1)
    # counties meeting each map-coverage threshold (spatial eligibility, before complete-case restriction)
    ex = pd.read_parquet(PROCESSED / "exposure_county.parquet")
    shr = ex.set_index("GEOID").dev_mapped_share if "dev_mapped_share" in ex else None
    if shr is not None:
        n["cov80_elig"], n["cov95_elig"] = fmt((shr >= 0.8).sum()), fmt((shr >= 0.95).sum())
    # second stage of the two-part model: log obligations among funded counties
    h2 = pd.read_csv(TAB / "exp_alignment_hurdle.csv")
    h2 = h2[(h2.outcome == "All programs") & (h2.part == "part2_log")].set_index("term")
    for t_, k in (("x_new_exp", "new"), ("x_base_exp", "base"), ("x_loss", "loss"), ("x_dr", "dr")):
        n[f"p2_{k}"], n[f"p2_{k}_p"] = fmt(h2.loc[t_, "coef"], 2), fmt(h2.loc[t_, "p"], 2)
    n["p2_n"] = fmt(h2.n.iloc[0])
    st = pd.read_csv(TAB / "exp_signed_transform.csv")  # 21_signed_transform.py
    for mdl, k in (("Five-year panel, any funding", "panel"), ("County logit, any funding (AME per SD)", "logit")):
        r = st[(st.model == mdl) & st["transform"].str.startswith("asinh")].iloc[0]
        n[f"st_{k}"], n[f"st_{k}_lo"], n[f"st_{k}_hi"] = fmt(r.coef, 1), fmt(r.lo, 1), fmt(r.hi, 1)
    fl = pd.read_csv(TAB / "wc_site_flow.csv", index_col=0)["count"]
    n["sf_records"], n["sf_rep_sites"] = fmt(fl.iloc[0]), fmt(fl["sites listed in more than one subapplication"])
    n["sf_rep_records"], n["sf_removed"] = fmt(fl["records belonging to those sites"]), fmt(fl["records removed as repeat listings"])
    n["sf_unique"] = fmt(fl["unique sites"])
    fpr = pd.read_csv(TAB / "wc_site_fingerprints.csv").set_index("fingerprint")
    lo_ = fpr.loc["Loose: ZIP, year built, structure"]
    n["fp_loose_fund"], n["fp_loose_never"] = fmt(lo_.funded_built_2001_pct, 1), fmt(lo_.never_funded_built_2001_pct, 1)
    nm = fpr.loc["No matching (all records)"]
    n["fp_none_fund"], n["fp_none_never"] = fmt(nm.funded_built_2001_pct, 1), fmt(nm.never_funded_built_2001_pct, 1)
    sp = pd.read_csv(TAB / "exp_spatial_se.csv")  # 19_spatial_se.py
    ne = sp[sp.term == "x_new_exp"]
    cs = ne[ne.se_type == "State-clustered"].iloc[0]
    cz = ne[ne.se_type != "State-clustered"]
    n["sp_new"] = fmt(cs.coef_pp, 1)
    n["sp_state_lo"], n["sp_state_hi"] = fmt(cs.lo, 1), fmt(cs.hi, 1)
    n["sp_conley_lo"], n["sp_conley_hi"] = fmt(cz.lo.min(), 1), fmt(cz.hi.max(), 1)
    n["sp_cut_min"], n["sp_cut_max"] = fmt(cz.cutoff_km.min()), fmt(cz.cutoff_km.max())
    oth = sp[(sp.se_type != "State-clustered") & sp.term.isin(["x_base_exp", "x_loss", "x_dr"])]
    n["sp_other_min_lo"] = fmt(oth.lo.min(), 1)  # lowest lower bound among baseline, losses, declarations
    cw = pd.read_csv(TAB / "wc_claim_rates_within_county.csv").set_index("measure")
    for m_, k in (("claims", "cl"), ("payments", "pd")):
        n[f"crw_{k}"], n[f"crw_{k}_lo"], n[f"crw_{k}_hi"] = (fmt(cw.loc[m_, c_], 2) for c_ in
                                                             ("mh_ratio_within_county", "mh_lo", "mh_hi"))
        n[f"crw_{k}_crude"] = fmt(cw.loc[m_, "crude_ratio"], 2)
    n["crw_counties"] = fmt(cw.counties.iloc[0])
    ym = pd.read_csv(TAB / "wc_year_built_matched.csv").set_index("built_from")
    n["ybm_counties"] = fmt(ym.loc[2001, "counties"])
    n["ybm_units_2001"], n["ybm_claims_2001"], n["ybm_mit_2001"] = (
        fmt(ym.loc[2001, k], 1) for k in ("insured_units_pct_matched", "claims_pct_matched", "mitigated_pct"))
    ws = pd.read_csv(TAB / "wc_zip_summary.csv", index_col=0).value
    n["wz_zips"], n["wz_counties"], n["wz_sites"] = fmt(ws.zips), fmt(ws.counties), fmt(ws.sites)
    n["wz_zip_site"], n["wz_zip_funded"] = fmt(ws.zips_with_site), fmt(ws.zips_with_funded_site)
    n["wz_match"], n["wz_rho"] = fmt(ws.sites_matched_pct), fmt(ws.rho_new_old, 2)
    # development behind accredited levees (zone X, levee-reduced), covered counties
    ecy = pd.read_parquet(PROCESSED / "exposure_county_year.parquet")
    cv = ecy[(ecy.year == 2025) & (ecy.dev_mapped_share >= 0.9)].GEOID
    lv = ecy[ecy.GEOID.isin(cv)].groupby("year").dev_levee_km2.sum()
    # net decreases in developed A-zone land are set to zero before log(1 + x)
    pv = ecy[ecy.GEOID.isin(cv)].pivot(index="GEOID", columns="year", values="dev_sfha_km2")
    n["neg_counties"] = fmt(((pv[2025] - pv[2000]) < 0).sum())
    dd = pv.diff(axis=1).iloc[:, 1:].to_numpy().ravel()
    n["neg_periods_pct"] = fmt(100 * (dd < 0).mean(), 1)
    n["lev_added_km2"], n["lev_growth_pct"] = fmt(lv[2025] - lv[2000]), fmt(100 * (lv[2025] / lv[2000] - 1), 1)
    (TAB / "numbers.json").write_text(json.dumps(n, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(n), "numbers written")


if __name__ == "__main__":
    main()
