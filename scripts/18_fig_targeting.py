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


def fig_new_buildings():
    yb = pd.read_csv(TAB / "wc_year_built.csv").set_index("built_from").loc[[1990, 2001]]
    cr = pd.read_csv(TAB / "exp_new_vs_old_claim_rates.csv", index_col=0)
    rr = pd.read_csv(TAB / "exp_new_vs_old_claim_rates_region.csv").set_index("region")
    n_mit = int(yb.n_mitigated.iloc[0])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5.0), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.32})
    series = [("Insured A-zone units\n(policies in effect, 2024)", "insured_units_pct", NAVY),
              ("A-zone NFIP claims\n(2016–2024)", "claims_pct", BLUE[400]),
              (f"Building sites in funded HMA\nprojects (FY2020–2025, n = {n_mit:,})", "mitigated_pct", BLUE[200])]
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
    b.barh(y - 0.19, old, 0.36, color=NAVY, label="Built before 2001")
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


def fig_stages():
    h = pd.read_csv(TAB / "exp_alignment_hurdle.csv")
    ap = pd.read_csv(TAB / "exp_applications_models.csv")
    terms = {"x_new_exp": "A-zone development\nadded 2000–2025", "x_base_exp": "A-zone development,\n2000",
             "x_loss": "NFIP claims paid", "x_dr": "Years with a flood-\nrelated declaration"}
    sets = [("Received any HMA funding", h[(h.outcome == "All programs") & (h.part == "part1_any")], NAVY, "o"),
            ("Applied (any program)", ap[ap.model.str.startswith("A. ")], BLUE[550], "s"),
            ("Applied to FMA or PDM/BRIC", ap[ap.model.str.startswith("A2.")], BLUE[300], "D"),
            ("Decided application funded", ap[ap.model.str.startswith("C. ")], BLUE[200], "^")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 5.6), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.55})
    tk = list(terms)
    for j, (lab, d, c, mk) in enumerate(sets):
        s = d.set_index("term").reindex(tk)
        yy = np.arange(len(tk)) + (j - 1.5) * 0.18
        a.errorbar(s.coef, yy, xerr=1.96 * s.se, fmt=mk, color=c, ms=5.5, capsize=2, elinewidth=1,
                   markeredgecolor="white", markeredgewidth=0.5, label=lab)
    a.axvline(0, color=INK, lw=0.6)
    a.set_yticks(range(len(tk)), [terms[k] for k in tk])
    a.invert_yaxis()
    a.set_xlabel("Change in probability, per SD\n(percentage points)")
    a.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=9, handletextpad=0.3,
             columnspacing=1.0, title="Counties (funding 2001–2025; applications 2001–2023)", title_fontsize=9.5)
    a.grid(axis="x", color="#e6e6e6", lw=0.5)
    a.text(-0.5, 1.02, "a", transform=a.transAxes, fontsize=13, fontweight="bold")

    wz = pd.read_csv(TAB / "wc_zip_model.csv")
    zt = {"z_units_new": "Insured A-zone units\nbuilt 2001 or later", "z_units_old": "Insured A-zone units\nbuilt before 2001",
          "z_claims": "A-zone NFIP claims,\n2016–2024"}
    zk = list(zt)
    for j, (out_, lab, c, mk) in enumerate([("site_any", "Site in any application", BLUE[550], "s"),
                                            ("site_funded", "Site in a funded project", BLUE[300], "^")]):
        s = wz[wz.outcome == out_].set_index("term").reindex(zk)
        yy = np.arange(len(zk)) + (j - 0.5) * 0.22
        b.errorbar(s.coef, yy, xerr=1.96 * s.se, fmt=mk, color=c, ms=5.5, capsize=2, elinewidth=1,
                   markeredgecolor="white", markeredgewidth=0.5, label=lab)
    b.axvline(0, color=INK, lw=0.6)
    b.set_yticks(range(len(zk)), [zt[k] for k in zk])
    b.invert_yaxis()
    b.set_xlabel("Change in probability, per SD\n(percentage points)")
    b.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=1, fontsize=9, handletextpad=0.3,
             title="ZIP codes within counties (2020–2025)", title_fontsize=9.5)
    b.grid(axis="x", color="#e6e6e6", lw=0.5)
    b.text(-0.62, 1.02, "b", transform=b.transAxes, fontsize=13, fontweight="bold")
    save(fig, "Fig5_stages")


if __name__ == "__main__":
    fig_new_buildings()
    fig_stages()
