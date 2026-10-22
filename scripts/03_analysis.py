"""Descriptive analysis, panel models, cross-sectional alignment, figures.

Primary analysis window: FY2001-2021.
  - 2001 start: allows 5-year lags of Storm Events (reliable from 1996).
  - 2021 end : HMGP awards for later declarations are still being approved (right-censoring)
               and FMA/BRIC selections stalled after FY2023 (CRS IN12642).
"""
import os
import sys
import warnings

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tools.tools as smt

# scipy.linalg fails to load in some Windows MKL builds (delay-load 0xc06d007f); numpy is used for rank checks
smt.matrix_rank = lambda x, method="qr": int(np.linalg.matrix_rank(np.asarray(x)))

from config import FIG, PROCESSED, RAW, TAB

os.environ.setdefault("GDAL_DATA", os.path.join(sys.prefix, "Library", "share", "gdal"))
warnings.filterwarnings("ignore")

Y0, Y1 = 2001, 2021

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 10.5,
    "axes.linewidth": 0.6, "axes.labelsize": 11, "axes.titlesize": 11, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 10, "legend.frameon": False, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 600, "savefig.bbox": "tight", "mathtext.fontset": "dejavusans",
})
# single-hue (blue) ramp, light -> dark; grey only for "none / decrease"
BLUE = {100: "#cde2fb", 200: "#9ec5f4", 300: "#6da7ec", 400: "#3987e5", 500: "#256abf", 550: "#1c5cab",
        650: "#104281", 700: "#0d366b"}
SEQ4 = [BLUE[100], BLUE[300], BLUE[500], BLUE[700]]
INK = "#333333"
mpl.rcParams.update({"axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK,
                     "text.color": INK})


def twfe(y, X, entity, time, cluster, tol=1e-10, max_iter=500):
    """Two-way fixed-effects OLS via alternating demeaning; SE clustered by `cluster`.

    scipy.linalg fails to load in some Windows MKL builds, so this
    avoids linearmodels/scipy and relies on numpy + statsmodels only.
    """
    Z = pd.concat([y.rename("_y"), X], axis=1).astype(float)
    for _ in range(max_iter):
        prev = Z.values.copy()
        Z = Z - Z.groupby(entity.values).transform("mean")
        Z = Z - Z.groupby(time.values).transform("mean")
        if np.abs(Z.values - prev).max() < tol:
            break
    m = sm.OLS(Z["_y"], Z[X.columns]).fit(cov_type="cluster", cov_kwds={"groups": cluster.values})
    ss_res = (m.resid ** 2).sum()
    m.r2_within = 1 - ss_res / (Z["_y"] ** 2).sum()
    return m


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def load():
    p = pd.read_parquet(PROCESSED / "county_year_panel.parquet")
    p = p.sort_values(["GEOID", "year"])
    p["any_flood"] = (p.obl_flood > 0).astype(float)
    p["log_obl_flood"] = np.log1p(p.obl_flood)
    p["ldmg"] = np.log1p(p.flood_dmg_real)
    p["dr_flood_bin"] = (p.dr_flood > 0).astype(float)
    p["log_pop"] = np.log(p["pop"])
    g = p.groupby("GEOID")
    for k in range(1, 6):
        p[f"dr_flood_l{k}"] = g.dr_flood_bin.shift(k)
        p[f"ldmg_l{k}"] = g.ldmg.shift(k)
    for k in range(1, 4):
        p[f"dr_flood_f{k}"] = g.dr_flood_bin.shift(-k)
    for prog in ("hmgp", "fma", "pdm"):
        p[f"any_{prog}"] = (p[f"obl_flood_{prog}"] > 0).astype(float)
    p["any_flood_sub"] = (p.n_flood_sub > 0).astype(float)       # dated by submission to FEMA
    p["any_hmgp_sub"] = (p.n_flood_hmgp_sub > 0).astype(float)
    return p


# ----------------------------------------------------------------- figures
def fig_program_trends():
    h = pd.read_parquet(PROCESSED / "hma_projects_county.parquet")
    h = h[h.flood_activity]
    proj = h.drop_duplicates("subapplicationIdentifier")
    t = proj.pivot_table(index="fiscalYear", columns="prog", values="obl_real", aggfunc="sum").fillna(0) / 1e9
    t = t.reindex(range(1996, 2025), fill_value=0)[["HMGP", "FMA", "PDM/BRIC"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bottom = np.zeros(len(t))
    for col, c in zip(t.columns, [BLUE[700], BLUE[400], BLUE[200]]):
        ax.bar(t.index, t[col], bottom=bottom, color=c, edgecolor="white", linewidth=0.5, width=0.8, label=col)
        bottom += t[col].values
    ax.axvspan(2021.5, 2024.6, color="#f0efec", zorder=0)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(2023, bottom.max() * 0.93, "Incomplete\nreporting", ha="center", va="top", fontsize=9.5)
    ax.set_xlim(1995.4, 2024.6)
    ax.set_ylabel(f"FEMA HMA obligations (billion 2024 USD)")
    ax.set_xlabel("Fiscal year (HMGP: year of disaster declaration)")
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0), title="Program", alignment="left")
    save(fig, "FigS2_program_trends")
    t.round(4).to_csv(TAB / "fig2_program_by_year_bn2024.csv")
    return t


def fig_maps(p, counties):
    w = p[(p.year >= Y0) & (p.year <= Y1)]
    cum = w.groupby("GEOID").agg(obl=("obl_flood", "sum"), dmg=("flood_dmg_real", "sum"),
                                 pop=("pop", "mean"), dr=("dr_flood_bin", "sum"))
    cum["obl_pc"] = cum.obl / cum["pop"]
    cum["dmg_pc"] = cum.dmg / cum["pop"]
    g = counties.merge(cum, left_on="GEOID", right_index=True)
    states = g.dissolve("STATEFP")
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.2), gridspec_kw={"hspace": 0.12})
    for ax, col, lab, ltitle in (
            (axes[0], "obl_pc", "(a)", "Flood-mitigation obligations,\nFY2001–2021 (2024 USD per person)"),
            (axes[1], "dmg_pc", "(b)", "Reported flood property damage,\n2001–2021 (2024 USD per person)")):
        pos = g[g[col] > 0]
        bins = np.quantile(pos[col], [0, .25, .5, .75, 1])
        cls = np.digitize(g[col], bins[1:-1], right=True)
        shades = SEQ4
        colors = np.where(g[col] > 0, [shades[min(c, 3)] for c in cls], "white")
        g.plot(ax=ax, color=colors, edgecolor="#d0d0d0", linewidth=0.05)
        states.boundary.plot(ax=ax, color="#555555", linewidth=0.3)
        ax.set_axis_off()
        ax.text(0.0, 1.0, lab, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
        labels = ["0"] + [f"{bins[i]:,.0f}–{bins[i+1]:,.0f}" for i in range(4)]
        handles = [mpl.patches.Patch(facecolor=c, edgecolor="#888888", linewidth=0.3)
                   for c in ["white"] + shades]
        ax.legend(handles, labels, title=ltitle, loc="upper left", bbox_to_anchor=(0.885, 0.56), fontsize=8,
                  title_fontsize=8.5, handlelength=1.1, labelspacing=0.35, alignment="left", borderaxespad=0)
    save(fig, "FigS3_funding_damage_maps")
    return cum


def run_event_study(p):
    w = p[(p.year >= Y0) & (p.year <= Y1)].set_index(["GEOID", "year"])
    terms = [f"dr_flood_f{k}" for k in (3, 2, 1)] + ["dr_flood_bin"] + [f"dr_flood_l{k}" for k in (1, 2, 3)]
    rows = []
    for y, lab in (("any_flood", "All programs"), ("any_hmgp", "HMGP"), ("any_fma", "FMA"),
                   ("any_pdm", "PDM/BRIC"), ("any_flood_sub", "All programs, submission year"),
                   ("any_hmgp_sub", "HMGP, submission year")):
        d = w[[y] + terms + ["STATEFP"]].dropna()
        ent, tim = d.index.get_level_values(0), d.index.get_level_values(1)
        m = twfe(d[y] * 100, d[terms], ent, tim, d.STATEFP.astype(int))
        for t, k in zip(terms, [-3, -2, -1, 0, 1, 2, 3]):
            rows.append((lab, k, m.params[t], m.bse[t], int(m.nobs), d[y].mean() * 100))
    es = pd.DataFrame(rows, columns=["outcome", "rel_year", "coef", "se", "nobs", "mean_pct"])
    es.to_csv(TAB / "event_study_flood_declarations.csv", index=False)

    # program fiscal year (HMGP = declaration year, so year 0 is partly built in) vs year submitted to FEMA
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.4), sharey="row")
    axes = axes.ravel()
    panels = [("All programs", "All programs, program year"), ("All programs, submission year", None),
              ("HMGP", "HMGP, program year"), ("HMGP, submission year", None)]
    for i, (ax, (key, title)) in enumerate(zip(axes, panels)):
        lab, d = title or key, es[es.outcome == key]
        ax.axhline(0, color=INK, linewidth=0.5)
        ax.axvline(-0.5, color="#999999", linewidth=0.5, linestyle=":")
        ax.errorbar(d.rel_year, d.coef, yerr=1.96 * d.se, fmt="o", color=BLUE[550], ms=4, capsize=2,
                    elinewidth=0.9, markeredgecolor="white", markeredgewidth=0.5)
        ax.set_title(f"({'abcd'[i]}) {lab}", fontsize=11.5, loc="left")
        ax.set_xticks(range(-3, 4))
        if i >= 2:
            ax.set_xlabel("Years relative to flood declaration")
        if i % 2 == 0:
            ax.set_ylabel("Change in probability (pp)")
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    save(fig, "FigS1_event_study")
    return es


def panel_models(p):
    """Two-way FE linear models; SE clustered by state."""
    w = p[(p.year >= Y0) & (p.year <= Y1)].set_index(["GEOID", "year"])
    X = ["dr_flood_bin", "dr_flood_l1", "dr_flood_l2", "dr_flood_l3", "ldmg_l1", "ldmg_l2", "ldmg_l3", "log_pop"]
    specs = [("any_flood", "Any flood-mitigation obligation (×100)", "All"),
             ("log_obl_flood", "log(1 + obligations)", "All"),
             ("any_hmgp", "Any obligation (×100)", "HMGP"),
             ("any_fma", "Any obligation (×100)", "FMA"),
             ("any_pdm", "Any obligation (×100)", "PDM/BRIC")]
    res = []
    for y, lab, prog in specs:
        d = w[[y] + X + ["STATEFP"]].dropna()
        yy = d[y] * (100 if y.startswith("any") else 1)
        ent, tim = d.index.get_level_values(0), d.index.get_level_values(1)
        m = twfe(yy, d[X], ent, tim, d.STATEFP.astype(int))
        for v in X:
            res.append((y, lab, prog, v, m.params[v], m.bse[v], m.pvalues[v], int(m.nobs),
                        m.r2_within, d.index.get_level_values(0).nunique(), yy.mean()))
    r = pd.DataFrame(res, columns=["y", "label", "program", "term", "coef", "se", "p", "nobs", "r2_within",
                                   "n_counties", "y_mean"])
    r.to_csv(TAB / "table3_panel_models.csv", index=False)
    return r


def epa_county(counties):
    e = pd.read_excel(RAW / "epa" / "huc10_EAD_outputs.xlsx", sheet_name="output")
    e = e[(e.Lat != 0) & (e.Long != 0)]
    pts = gpd.GeoDataFrame(e, geometry=gpd.points_from_xy(e.Long, e.Lat), crs="EPSG:4326").to_crs(counties.crs)
    j = gpd.sjoin(pts, counties[["GEOID", "geometry"]], predicate="within", how="left")
    c = j.groupby("GEOID")[["EAD(Base)", "EAD(dT03)", "EAD(dT05)"]].sum(min_count=1)
    c["n_huc10"] = j.groupby("GEOID").size()
    return pts, c


def fig_epa(pts, counties):
    pts = pts.dropna(subset=["EAD(Base)", "EAD(dT05)"]).copy()
    pts["chg"] = 100 * (pts["EAD(dT05)"] - pts["EAD(Base)"]) / pts["EAD(Base)"]
    pts = pts[pts["EAD(Base)"] >= 10_000]  # avoid unstable ratios on tiny baselines
    states = counties.dissolve("STATEFP")
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.2), gridspec_kw={"hspace": 0.12})
    q = np.quantile(pts["EAD(Base)"], [.25, .5, .75])
    qc = np.digitize(pts["EAD(Base)"], q)
    qlab = [f"< {q[0]/1e3:,.0f}", f"{q[0]/1e3:,.0f}–{q[1]/1e3:,.0f}", f"{q[1]/1e3:,.0f}–{q[2]/1e3:,.0f}",
            f"> {q[2]/1e3:,.0f}"]
    for i, (c, lab) in enumerate(zip(SEQ4, qlab)):
        s_ = pts[qc == i]
        axes[0].scatter(s_.geometry.x, s_.geometry.y, s=2.0, color=c, linewidths=0, label=lab)
    leg = dict(loc="upper left", bbox_to_anchor=(0.885, 0.56), markerscale=3, fontsize=8, title_fontsize=8.5,
               labelspacing=0.35, alignment="left", borderaxespad=0)
    axes[0].legend(title="Baseline riverine expected annual\ndamage by HUC10 (USD thousand/yr)", **leg)
    axes[0].text(0.0, 1.0, "(a)", transform=axes[0].transAxes, fontsize=12, fontweight="bold", va="top")
    bins = [-np.inf, -10, 10, 50, np.inf]
    shades = ["#bdbdbd", BLUE[100], BLUE[400], BLUE[700]]
    cls = np.digitize(pts.chg, bins[1:-1])
    for i, (c, lab) in enumerate(zip(shades, ["< −10%", "−10 to 10%", "10 to 50%", "> 50%"])):
        s = pts[cls == i]
        axes[1].scatter(s.geometry.x, s.geometry.y, s=2.0, color=c, linewidths=0, label=lab)
    axes[1].text(0.0, 1.0, "(b)", transform=axes[1].transAxes, fontsize=12, fontweight="bold", va="top")
    axes[1].legend(title="Change in expected annual\ndamage at +5 °C warming", **leg)
    for ax in axes:
        states.boundary.plot(ax=ax, color="#555555", linewidth=0.25)
        ax.set_axis_off()
    save(fig, "FigS4_epa_ead")
    return pts


def cross_section(p, counties, epa_c):
    w = p[(p.year >= Y0) & (p.year <= Y1)]
    pre = p[(p.year >= 1996) & (p.year <= 2000)].groupby("GEOID").agg(dmg_pre=("flood_dmg_real", "sum"))
    c = w.groupby("GEOID").agg(obl=("obl_flood", "sum"), dmg=("flood_dmg_real", "sum"), dr=("dr_flood_bin", "sum"),
                               pop2010=("pop", lambda s: s.iloc[9]))
    c = c.join(pre).join(epa_c)
    c = c.join(counties.set_index("GEOID")[["STATEFP", "region", "coastal", "area_km2"]])
    c = c.join(p.drop_duplicates("GEOID").set_index("GEOID")[["poverty_pct_2019", "med_income_2019"]])
    c["any"] = (c.obl > 0).astype(int)
    c["log_obl"] = np.log(c.obl.where(c.obl > 0))
    c["log_ead"] = np.log1p(c["EAD(Base)"])
    c["ead_chg5"] = 100 * (c["EAD(dT05)"] - c["EAD(Base)"]) / c["EAD(Base)"].where(c["EAD(Base)"] >= 10_000)
    c["log_dmg"] = np.log1p(c.dmg)
    c["log_pop"] = np.log(c.pop2010)
    c["log_inc"] = np.log(c.med_income_2019)
    c.to_parquet(PROCESSED / "county_cross_section.parquet")
    X = ["log_ead", "ead_chg5", "log_dmg", "dr", "log_pop", "poverty_pct_2019", "log_inc", "coastal"]
    d = c.dropna(subset=X + ["STATEFP"]).copy()
    d["coastal"] = d.coastal.astype(float)
    fe = pd.get_dummies(d.STATEFP, prefix="st", drop_first=True, dtype=float)
    Z = (d[X] - d[X].mean()) / d[X].std()  # standardized coefficients
    Z["coastal"] = d.coastal
    out = []
    m1 = sm.Logit(d["any"], sm.add_constant(pd.concat([Z, fe], axis=1))).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d.STATEFP.astype(int)}, maxiter=200)
    me = m1.get_margeff(at="overall")
    mfx = pd.Series(me.margeff, index=m1.params.index[1:])
    mse = pd.Series(me.margeff_se, index=m1.params.index[1:])
    pos = d[d["any"] == 1]
    m2 = sm.OLS(pos.log_obl, sm.add_constant(pd.concat([Z.loc[pos.index], fe.loc[pos.index]], axis=1))).fit(
        cov_type="cluster", cov_kwds={"groups": pos.STATEFP.astype(int)})
    for v in X:
        out.append((v, mfx[v] * 100, mse[v] * 100, me.pvalues[list(m1.params.index[1:]).index(v)],
                    m2.params[v], m2.bse[v], m2.pvalues[v]))
    t = pd.DataFrame(out, columns=["term", "part1_ame_pp", "part1_se", "part1_p", "part2_coef", "part2_se",
                                   "part2_p"])
    t.attrs = {}
    meta = pd.DataFrame({"term": ["N", "N_positive", "pseudo_r2_part1", "r2_part2"],
                         "part1_ame_pp": [len(d), len(pos), m1.prsquared, m2.rsquared]})
    pd.concat([t, meta]).to_csv(TAB / "table4_cross_section_hurdle.csv", index=False)
    return c, t


def descriptives(p, c):
    w = p[(p.year >= Y0) & (p.year <= Y1)]
    rows = []

    def add(lab, s):
        s = s.dropna()
        rows.append((lab, len(s), s.mean(), s.std(), s.median(), s.quantile(.9), 100 * (s == 0).mean()))

    add("Flood-mitigation obligations, county-year (2024 USD thousand)", w.obl_flood / 1e3)
    add("Flood-mitigation projects, county-year (count)", w.n_flood)
    add("Riverine flood events (Storm Events), county-year", w.flood_events)
    add("Reported flood property damage, county-year (2024 USD thousand)", w.flood_dmg_real / 1e3)
    add("Flood-related major disaster declaration (0/1), county-year", w.dr_flood_bin)
    add("Cumulative obligations FY2001–2021, county (2024 USD million)", c.obl / 1e6)
    add("Baseline riverine EAD (EPA), county (USD thousand)", c["EAD(Base)"] / 1e3)
    add("Change in EAD at +5 °C (%), county", c.ead_chg5)
    add("Population 2010, county (thousand)", c.pop2010 / 1e3)
    add("Poverty rate 2019 (%), county", c.poverty_pct_2019)
    t = pd.DataFrame(rows, columns=["variable", "n", "mean", "sd", "median", "p90", "pct_zero"])
    t.to_csv(TAB / "table2_descriptives.csv", index=False)
    return t


def concentration(c):
    """Share of cumulative obligations received by counties ranked by different risk measures."""
    d = c.dropna(subset=["EAD(Base)"]).copy()
    out = {}
    for key, lab in (("obl", "Obligations"), ("dmg", "Reported damage"), ("EAD(Base)", "Modelled EAD (EPA)")):
        s = d.sort_values(key, ascending=False)
        out[lab] = (s.obl.cumsum() / s.obl.sum()).values
    n = np.arange(1, len(d) + 1) / len(d) * 100
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for (lab, y), ls, c in zip(out.items(), ["-", "--", ":"], [BLUE[700], BLUE[500], BLUE[300]]):
        ax.plot(n, y * 100, color=c, linestyle=ls, linewidth=1.4,
                label=lab[0].lower() + lab[1:])
    ax.plot([0, 100], [0, 100], color="#999999", linewidth=0.5)
    ax.set_xlabel("Cumulative share of counties (%)")
    ax.set_ylabel("Cumulative share of flood-mitigation\nobligations, FY2001–2021 (%)")
    ax.set_xlim(0, 100), ax.set_ylim(0, 101)
    ax.legend(loc="lower right", title="Counties ranked by", alignment="left")
    save(fig, "FigS5_concentration")
    k = int(round(0.10 * len(d)))
    summ = {lab: y[k - 1] * 100 for lab, y in out.items()}
    summ["n_counties"] = len(d)
    summ["share_counties_zero_obl"] = 100 * (d.obl == 0).mean()
    pd.Series(summ).to_csv(TAB / "concentration_top10pct.csv")
    return summ


def sensitivity(p, c):
    """Re-estimate headline quantities under alternative coding choices (Table S1)."""
    X = ["dr_flood_bin", "dr_flood_l1", "dr_flood_l2", "dr_flood_l3", "ldmg_l1", "ldmg_l2", "ldmg_l3", "log_pop"]
    base = p.copy()
    for k in ("flood_haz", "flood_any", "flood_single", "all"):
        base[f"any_{k}"] = (base[f"obl_{k}"] > 0).astype(float)
    cases = [
        ("Primary specification", "any_flood", base, Y0, Y1),
        ("Flood defined by hazard field", "any_flood_haz", base, Y0, Y1),
        ("Flood activity or hazard", "any_flood_any", base, Y0, Y1),
        ("Single-county projects only", "any_flood_single", base, Y0, Y1),
        ("All mitigation projects (any hazard)", "any_all", base, Y0, Y1),
        ("Non-coastal counties", "any_flood", base[~base.coastal], Y0, Y1),
        ("Period FY2001–2010", "any_flood", base, 2001, 2010),
        ("Period FY2011–2021", "any_flood", base, 2011, 2021),
        ("Extended to FY2024 (incomplete years)", "any_flood", base, Y0, 2024),
    ]
    rows = []
    for lab, y, d0, a, b in cases:
        w = d0[(d0.year >= a) & (d0.year <= b)].set_index(["GEOID", "year"])
        d = w[[y] + X + ["STATEFP"]].dropna()
        m = twfe(d[y] * 100, d[X], d.index.get_level_values(0), d.index.get_level_values(1),
                 d.STATEFP.astype(int))
        lag = ["dr_flood_l1", "dr_flood_l2", "dr_flood_l3"]
        lag_sum = m.params[lag].sum()
        lag_se = np.sqrt(np.ones(3) @ m.cov_params().loc[lag, lag].values @ np.ones(3))
        rows.append((lab, d[y].mean() * 100, m.params["dr_flood_bin"], m.bse["dr_flood_bin"], lag_sum, lag_se,
                     m.params["ldmg_l1"], m.bse["ldmg_l1"], int(m.nobs)))
    s = pd.DataFrame(rows, columns=["case", "y_mean_pct", "dr_t", "dr_t_se", "dr_lag13_sum", "dr_lag13_se",
                                    "ldmg_l1", "ldmg_l1_se", "nobs"])
    s.to_csv(TAB / "tableS1_sensitivity.csv", index=False)
    return s


def main():
    p = load()
    counties = gpd.read_file(PROCESSED / "counties.gpkg")
    fig_program_trends()
    fig_maps(p, counties)
    es = run_event_study(p)
    r = panel_models(p)
    pts, epa_c = epa_county(counties)
    fig_epa(pts, counties)
    c, t4 = cross_section(p, counties, epa_c)
    t2 = descriptives(p, c)
    conc = concentration(c)
    s1 = sensitivity(p, c)
    pd.set_option("display.width", 220)
    print(s1.round(3).to_string())
    print(t2.round(2).to_string())
    print(es.round(3).to_string())
    print(r[["program", "y", "term", "coef", "se", "p", "nobs"]].round(4).to_string())
    print(t4.round(3).to_string())
    print(conc)
    print("EPA HUC10 assigned to counties:", epa_c.n_huc10.sum(), "counties with >=1 HUC10:", len(epa_c))


if __name__ == "__main__":
    main()
