"""Main-text figures on the risk of new floodplain buildings and where HMA mitigation goes.

Fig4_new_buildings: (a) share of insured A-zone units, A-zone NFIP claims and building sites in funded HMA projects that were
  constructed in or after 1990 and 2001; (b) NFIP claims per 1,000 policy-years for A-zone buildings constructed
  2001+ vs earlier, 2016-2024, by Census region.
Fig5_stages: (a) county-level average marginal effects for funding, applying, applying to pre-disaster programs
  and approval of decided applications; (b) within-county ZIP-code models of flood-mitigation sites.
Inputs: outputs/tables/wc_*.csv, exp_new_vs_old_claim_rates*.csv, exp_alignment_hurdle.csv,
        exp_applications_models.csv (scripts 10, 13, 17).
"""
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import TAB

A = importlib.import_module("03_analysis")  # palette and rc settings
BLUE, INK, save = A.BLUE, A.INK, A.save
NAVY = "#0b2545"
BLACK = "#111111"
GREY = "#8896a6"  # older buildings (slate grey)
TEAL = ["#0b7a5e", "#169c78", "#52b79a", "#92d1bd"]  # HMA outcomes, dark to light


def fig_new_buildings():
    yb = pd.read_csv(TAB / "wc_year_built.csv").set_index("built_from").loc[[1990, 2001]]
    cr = pd.read_csv(TAB / "exp_new_vs_old_claim_rates.csv", index_col=0)
    rr = pd.read_csv(TAB / "exp_new_vs_old_claim_rates_region.csv").set_index("region")
    n_mit = int(yb.n_mitigated.iloc[0])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5.0), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.32})
    series = [("Insured A-zone units\n(policies in effect, 2024)", "insured_units_pct", NAVY),
              ("A-zone NFIP claims\n(2016–2024)", "claims_pct", BLUE[400]),
              (f"Residential building sites in\nfunded HMA projects (FY2020–2025,\nn = {n_mit:,})", "mitigated_pct", "#0a6b6b")]
    x = np.arange(2)
    w = 0.26
    for j, (lab, col_, c) in enumerate(series):
        v = yb[col_].to_numpy()
        a.bar(x + (j - 1) * w, v, w * 0.92, color=c, label=lab)
        for xi, vi in zip(x + (j - 1) * w, v):
            a.text(xi, vi + 0.8, f"{vi:.0f}", ha="center", va="bottom", fontsize=9.5)
    a.set_xticks(x, ["Built 1990 or later", "Built 2001 or later"])
    a.set_ylabel("Share of buildings or claims (%)")
    a.set_ylim(0, 50)
    a.legend(loc="upper right", fontsize=9, handlelength=1.2, labelspacing=0.8)
    a.text(-0.16, 1.02, "a", transform=a.transAxes, fontsize=13, fontweight="bold")

    regs = ["Northeast", "Midwest", "South", "West"]
    new = list(rr.loc[regs, "new_per_1000_py"]) + [cr.loc["built_2001_plus", "claims_per_1000_py"]]
    old = list(rr.loc[regs, "old_per_1000_py"]) + [cr.loc["built_before_2001", "claims_per_1000_py"]]
    y = np.arange(5)
    b.barh(y - 0.19, old, 0.36, color=GREY, label="Built before 2001")
    b.barh(y + 0.19, new, 0.36, color=BLUE[300], label="Built 2001 or later")
    for yi, o, nw in zip(y, old, new):
        b.text(o + 0.3, yi - 0.19, f"{o:.1f}", va="center", fontsize=9.5)
        b.text(nw + 0.3, yi + 0.19, f"{nw:.1f}", va="center", fontsize=9.5)
    b.set_yticks(y, regs + ["CONUS"])
    b.invert_yaxis()
    b.set_xlabel("NFIP claims per 1,000 policy-years,\nA-zone buildings, 2016–2024")
    b.set_xlim(0, max(old) * 1.18)
    b.legend(loc="lower right", fontsize=9.5)
    b.text(-0.2, 1.02, "b", transform=b.transAxes, fontsize=13, fontweight="bold")
    save(fig, "Fig4_new_buildings")


def _dot(ax, y, coef, se):
    """Estimate with its 95% interval and the estimate printed above the dot."""
    lo, hi = coef - 1.96 * se, coef + 1.96 * se
    ax.plot([lo, hi], [y, y], color=BLACK, lw=1.8, solid_capstyle="butt")
    ax.plot(coef, y, "o", ms=6.5, color=BLACK, zorder=5)
    lab = f"{coef:.1f}".replace("-", "−")
    if abs(coef) < 4:  # near the zero line: print to the right of the interval so the line stays unbroken
        ax.text(hi + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]), y, lab, ha="left", va="center", fontsize=11, color=BLACK)
    else:
        ax.text(coef, y - 0.2, lab, ha="center", va="bottom", fontsize=11, color=BLACK)


def _focus_band(ax):
    """Light band behind the first row (new development / newer buildings), the comparison of interest."""
    ax.axhspan(-0.5, 0.5, color="#e3ebf5", zorder=0, lw=0)


@plt.rc_context({"font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14,
                 "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
                 "axes.edgecolor": "black"})
def fig_stages():
    h = pd.read_csv(TAB / "exp_alignment_hurdle.csv")
    ap = pd.read_csv(TAB / "exp_applications_models.csv")
    wz = pd.read_csv(TAB / "wc_zip_model.csv")
    terms = {"x_new_exp": "A-zone development\nadded 2000–2025", "x_base_exp": "A-zone development,\n2000",
             "x_loss": "NFIP claims paid", "x_dr": "Years with a flood-\nrelated declaration"}
    outs = [("Received any\nHMA funding", h[(h.outcome == "All programs") & (h.part == "part1_any")]),
            ("Applied\n(any program)", ap[ap.model.str.startswith("A. ")]),
            ("Applied to FMA or\nPDM/BRIC", ap[ap.model.str.startswith("A2.")]),
            ("Decided application\nfunded", ap[ap.model.str.startswith("C. ")])]
    zterms = {"z_units_new": "Insured units\nbuilt 2001 or later", "z_units_old": "Insured units\nbuilt before 2001",
              "z_claims": "A-zone claims,\n2016–2024"}
    zouts = [("Site in any\napplication", wz[wz.outcome == "site_any"]),
             ("Site in a\nfunded project", wz[wz.outcome == "site_funded"])]
    fig = plt.figure(figsize=(9.6, 8.6))
    top = fig.add_gridspec(1, 4, left=0.25, right=0.985, top=0.82, bottom=0.575, wspace=0.14)
    bot = fig.add_gridspec(1, 2, left=0.25, right=0.985, top=0.355, bottom=0.12, wspace=0.14)
    tk = list(terms)
    for j, (title, d) in enumerate(outs):
        ax = fig.add_subplot(top[0, j])
        s = d.set_index("term").reindex(tk)
        ax.set_xlim(-12, 21)
        _focus_band(ax)
        for i, k in enumerate(tk):
            _dot(ax, i, s.loc[k, "coef"], s.loc[k, "se"])
        ax.axvline(0, color="black", lw=1.1)
        ax.set_ylim(len(tk) - 0.5, -0.5)
        ax.set_yticks(range(len(tk)), [terms[k] if j == 0 else "" for k in tk])
        ax.set_xlim(-12, 21)
        ax.set_xticks([-10, 0, 10, 20])
        ax.tick_params(axis="x", labelsize=13.5)
        ax.set_title(title, fontsize=15, loc="left", pad=8)
        ax.grid(axis="x", color="#d4d4d4", lw=0.7)
        ax.tick_params(axis="y", length=0)
    fig.text(0.01, 0.925, "a", fontsize=22, fontweight="bold")
    fig.text(0.25, 0.915, "Counties: funding 2001–2025, applications 2001–2023", fontsize=15, color="black")
    fig.text(0.62, 0.51, "Change in probability, per SD (percentage points)", ha="center", fontsize=15)
    zk = list(zterms)
    for j, (title, d) in enumerate(zouts):
        ax = fig.add_subplot(bot[0, j])
        s = d.set_index("term").reindex(zk)
        ax.set_xlim(-2, 12)
        _focus_band(ax)
        for i, k in enumerate(zk):
            _dot(ax, i, s.loc[k, "coef"], s.loc[k, "se"])
        ax.axvline(0, color="black", lw=1.1)
        ax.set_ylim(len(zk) - 0.5, -0.5)
        ax.set_yticks(range(len(zk)), [zterms[k] if j == 0 else "" for k in zk])
        ax.set_xlim(-2, 12)
        ax.set_xticks([-2, 0, 2, 4, 5, 6, 8, 10, 12][:0] or [0, 5, 10])
        ax.set_title(title, fontsize=15, loc="left", pad=8)
        ax.grid(axis="x", color="#d4d4d4", lw=0.7)
        ax.tick_params(axis="y", length=0)
    fig.text(0.01, 0.455, "b", fontsize=22, fontweight="bold")
    fig.text(0.25, 0.445, "ZIP codes within counties, 2020–2025 (different predictors from a)", fontsize=15, color="black")
    fig.text(0.62, 0.04, "Change in probability, per SD (percentage points)", ha="center", fontsize=15)
    save(fig, "Fig5_stages")


if __name__ == "__main__":
    fig_new_buildings()
    fig_stages()
