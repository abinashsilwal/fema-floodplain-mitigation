"""Reconciliation checks for numbers that appear in more than one place (text, tables, figures).

Each check states the universe, unit and years it compares and fails loudly if the values disagree.
Output: outputs/tables/reconciliation.csv (one row per check).
"""
import json

import pandas as pd

from config import TAB

n = json.load(open(TAB / "numbers.json", encoding="utf-8"))
num = lambda k: float(str(n[k]).replace(",", "").replace("−", "-"))
tp = pd.read_csv(TAB / "exp_throughput.csv", index_col=0).iloc[:, 0]
br = pd.read_csv(TAB / "exp_building_ratio.csv").set_index("area")
rb = pd.read_csv(TAB / "exp_robustness.csv").set_index("case")
al = pd.read_csv(TAB / "exp_alignment_hurdle.csv")
al = al[(al.outcome == "All programs") & (al.part == "part1_any")].set_index("term")
sf = pd.read_csv(TAB / "exp_sample_flow.csv", index_col=0).iloc[:, 0]
ty = pd.read_csv(TAB / "exp_typology.csv", index_col=0)

checks = []


def check(name, universe, a, b, tol):
    ok = abs(a - b) <= tol
    checks.append((name, universe, a, b, tol, ok))


check("Insured units per year x 24 = total", "CONUS counties; built 2001-2024; insured 2024",
      tp.new_insured_per_year * 24, tp.new_insured_2001_2024, 1)
check("Mitigated buildings per year x 24 = total", "CONUS counties; HMA FY2001-2024",
      tp.mitigated_per_year * 24, tp.mitigated_2001_2024, 1)
check("Flows total = Table 3 CONUS insured units", "CONUS counties; built 2001-2024",
      tp.new_insured_2001_2024, br.loc["CONUS (all counties)", "insured_new_sfha_bldg"], 0.5)
check("Flows total = Table 3 CONUS mitigated", "CONUS counties; FY2001-2024",
      tp.mitigated_2001_2024, br.loc["CONUS (all counties)", ["acquired", "elevated"]].sum(), 0.5)
reg = br.loc[["Midwest", "Northeast", "South", "West"]]
check("Regions sum to covered-county total (insured units)", "covered counties",
      reg.insured_new_sfha_bldg.sum(), br.loc["CONUS (covered counties)", "insured_new_sfha_bldg"], 0.5)
check("Regions sum to covered-county total (mitigated, unrounded)", "covered counties",
      (reg.acquired + reg.elevated).sum(),
      br.loc["CONUS (covered counties)", ["acquired", "elevated"]].sum(), 0.5)
check("Ratio in text = flows ratio", "CONUS counties", num("flow_ratio"), round(tp.flow_ratio, 1), 0.05)
check("Table S2 primary = Table 4 (new exposure AME)", "covered counties, complete cases",
      rb.loc["Primary (covered counties)", "new_exp"], al.loc["x_new_exp", "coef"], 1e-6)
check("Table S2 primary = Table 4 (SE)", "covered counties, complete cases",
      rb.loc["Primary (covered counties)", "new_exp_se"], al.loc["x_new_exp", "se"], 1e-6)
check("Model n = sample-flow n", "covered counties", al.loc["x_new_exp", "n"], sf["model"], 0)
check("Covered - dropped = model n", "covered counties", sf["covered"] - (sf["covered"] - sf["model"]), sf["model"], 0)
check("Typology classes sum to covered counties", "covered counties", ty.counties.sum(), num("cov_n"), 0)
check("Typology classes sum to added A-zone land", "covered counties, km2", ty.new_dev_sfha_km2.sum(),
      num("d_dev_sfha"), 1)

t = pd.DataFrame(checks, columns=["check", "universe", "value_a", "value_b", "tolerance", "ok"])
t.to_csv(TAB / "reconciliation.csv", index=False)
print(t.to_string())
assert t.ok.all(), "reconciliation failed"
