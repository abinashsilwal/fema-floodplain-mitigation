"""Build the CONUS county frame and county-fiscal-year panel.

Inputs : data/raw (OpenFEMA HMA Subapplications v2, Disaster Declarations v2,
         NOAA Storm Events details, Census 2020 counties/coastline, Census
         population estimates, SAIPE 2019, FRED GDPDEF)
Outputs: data/processed/counties.gpkg
         data/processed/hma_projects_county.parquet  (project x county rows)
         data/processed/county_year_panel.parquet
         data/interim/hma_crosswalk_audit.csv
         outputs/tables/hma_inclusion_flow.csv
"""
import glob
import os
import re
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from config import INTERIM, NON_CONUS, PRICE_YEAR, PROCESSED, RAW, TAB

os.environ.setdefault("GDAL_DATA", os.path.join(sys.prefix, "Library", "share", "gdal"))

EQUAL_AREA = "EPSG:5070"  # NAD83 / CONUS Albers

# ---- machine-readable inclusion rules (also written to outputs/tables) ----
PROGRAM_GROUP = {
    "Hazard Mitigation Grant Program": "HMGP",
    "Hazard Mitigation Grant Program Post Fire": "HMGP",
    "Flood Mitigation Assistance": "FMA",
    "Flood Mitigation Assistance Swift Current": "FMA",
    "Repetitive Flood Claims": "FMA",
    "Severe Repetitive Loss": "FMA",
    "Pre-Disaster Mitigation": "PDM/BRIC",
    "Legislative Pre-Disaster Mitigation": "PDM/BRIC",
    "Pre-Disaster Mitigation Congressionally Directed Spending": "PDM/BRIC",
    "Building Resilient Infrastructure and Communities": "PDM/BRIC",
    "Fire Management Assistance Grant - Hazard Mitigation Grant Program": "HMGP",
}
NON_PROJECT_TYPES = {"Plan", "Management cost", "Project scoping", "Technical assistance", "Project scoring"}
NON_PROJECT_ACTIVITIES = {
    "New plan", "Plan update", "Management costs", "Planning related activities", "Technical assistance",
    "Develop or conduct engineering, environmental, feasibility and/or benefit cost analyses",
    "Feasibility, engineering and design studies", "Evaluate facilities to identify mitigation actions",
    "Activities supporting development of applications", "Partnerships",
    "Conduct meetings, outreach and coordination with subapplicants and community residents",
    "Staffing or resources to develop cost share strategy",
}
FLOOD_ACTIVITIES = {"Acquisition", "Elevation", "Floodproofing", "Flood control", "Relocation",
                    "Mitigation reconstruction"}
FLOOD_HAZARDS = {"Flood", "Flooding", "Dam/Levee Break", "Dam/Levee break"}
RIVERINE_STORM_TYPES = {"Flood", "Flash Flood"}
COASTAL_STORM_TYPES = {"Coastal Flood", "Storm Surge/Tide", "Lakeshore Flood"}
CENSUS_REGION = {  # state FIPS -> Census region
    **dict.fromkeys(["09", "23", "25", "33", "44", "50", "34", "36", "42"], "Northeast"),
    **dict.fromkeys(["17", "18", "26", "39", "55", "19", "20", "27", "29", "31", "38", "46"], "Midwest"),
    **dict.fromkeys(["10", "11", "12", "13", "24", "37", "45", "51", "54", "01", "21", "28", "47",
                     "05", "22", "40", "48"], "South"),
    **dict.fromkeys(["04", "08", "16", "30", "32", "35", "49", "56", "06", "41", "53"], "West"),
}


def deflator():
    g = pd.read_csv(RAW / "fred" / "GDPDEF.csv")
    g.columns = ["date", "v"]
    g["year"] = pd.to_datetime(g.date).dt.year
    a = g.groupby("year").v.mean()
    return (a.loc[PRICE_YEAR] / a).rename("defl")


def norm(s):
    s = str(s).upper().replace("&", "AND")
    s = re.sub(r"\bSAINT\b|\bST\.?(?=\s)", "ST", s)
    s = re.sub(r"\bSTE\.?\b|\bSAINTE\b", "STE", s)
    s = re.sub(r"\b(COUNTY|PARISH|BOROUGH|CENSUS AREA|MUNICIPALITY|CITY AND BOROUGH)\b", "", s)
    return re.sub(r"[^A-Z]", "", s)


# ------------------------------------------------------------------ counties
def build_counties():
    c = gpd.read_file("zip://" + str(RAW / "census" / "cb_2020_us_county_500k.zip"))
    c = c[~c.STATEFP.isin(NON_CONUS)].to_crs(EQUAL_AREA)
    c["area_km2"] = c.geometry.area / 1e6
    coast = gpd.read_file("zip://" + str(RAW / "census" / "tl_2020_us_coastline.zip")).to_crs(EQUAL_AREA)
    near = gpd.sjoin(c[["GEOID", "geometry"]], gpd.GeoDataFrame(geometry=coast.buffer(1000)),
                     predicate="intersects")
    c["coastal"] = c.GEOID.isin(near.GEOID)
    c["region"] = c.STATEFP.map(CENSUS_REGION)
    c = c[["GEOID", "STATEFP", "STUSPS", "STATE_NAME", "NAME", "NAMELSAD", "LSAD", "region", "coastal",
           "area_km2", "geometry"]]
    c.to_file(PROCESSED / "counties.gpkg", driver="GPKG")
    return c


def population():
    p0 = pd.read_csv(RAW / "census" / "co-est00int-tot.csv", encoding="latin-1", dtype={"STATE": str, "COUNTY": str})
    p1 = pd.read_csv(RAW / "census" / "co-est2020-alldata.csv", encoding="latin-1", dtype={"STATE": str, "COUNTY": str})
    p2 = pd.read_csv(RAW / "census" / "co-est2024-alldata.csv", encoding="latin-1", dtype={"STATE": str, "COUNTY": str})
    out = []
    for p, yrs in ((p0, range(2000, 2010)), (p1, range(2010, 2021)), (p2, range(2021, 2025))):
        p = p[p.SUMLEV == 50].copy()
        p["GEOID"] = p.STATE.str.zfill(2) + p.COUNTY.str.zfill(3)
        m = p.melt(id_vars="GEOID", value_vars=[f"POPESTIMATE{y}" for y in yrs], var_name="y", value_name="pop")
        m["year"] = m.y.str[-4:].astype(int)
        out.append(m[["GEOID", "year", "pop"]])
    pop = pd.concat(out)
    # CT switched to planning regions in the 2024 vintage: carry 2020 county values forward
    ct = pop[(pop.GEOID.str[:2] == "09") & (pop.year == 2020)]
    for y in range(2021, 2025):
        pop = pd.concat([pop, ct.assign(year=y)])
    # Bedford city VA (51515) merged into Bedford County 2013; keep as reported
    return pop.drop_duplicates(["GEOID", "year"], keep="first")


def saipe():
    s = pd.read_excel(RAW / "census" / "saipe_est19all.xls", header=3, dtype=str)
    s = s[s["County FIPS Code"] != "000"].copy()
    s["GEOID"] = s["State FIPS Code"].str.zfill(2) + s["County FIPS Code"].str.zfill(3)
    s["poverty_pct_2019"] = pd.to_numeric(s["Poverty Percent, All Ages"], errors="coerce")
    s["med_income_2019"] = pd.to_numeric(s["Median Household Income"], errors="coerce")
    return s[["GEOID", "poverty_pct_2019", "med_income_2019"]]


# ------------------------------------------------------------------ HMA
def build_hma(counties, defl):
    d = pd.read_parquet(RAW / "openfema" / "HmaSubapplications_v2.parquet")
    flow = [("All HMA subapplication records (v2, retrieved 2026-10-20)", len(d), d.totalObligatedAmount.sum())]
    d["prog"] = d.program.map(PROGRAM_GROUP)
    d = d[d.program != "Fire Management Assistance Grant - Hazard Mitigation Grant Program"]
    d = d[d.totalObligatedAmount.fillna(0) > 0]
    flow.append(("With obligated federal funds > 0 (excl. FMAG-HMGP)", len(d), d.totalObligatedAmount.sum()))
    d = d[~d.subapplicationType.isin(NON_PROJECT_TYPES) & ~d.primaryActivity.isin(NON_PROJECT_ACTIVITIES)]
    flow.append(("Mitigation projects (plans, management, scoping, TA removed)", len(d), d.totalObligatedAmount.sum()))
    d["flood_activity"] = d.primaryActivity.isin(FLOOD_ACTIVITIES)
    d["flood_hazard"] = d.primaryHazard.isin(FLOOD_HAZARDS)
    d["flood_any"] = d.flood_activity | d.flood_hazard
    d["obl_real"] = d.totalObligatedAmount * d.fiscalYear.map(defl).fillna(1.0)
    # actual process date: fiscal year the subapplication reached FEMA (HMGP fiscalYear is the declaration year;
    # OpenFEMA v2 has no award or obligation date)
    sub = pd.to_datetime(d.dateSubmittedToFema, errors="coerce", utc=True)
    d["sub_fy"] = sub.dt.year + (sub.dt.month >= 10)
    st = counties.drop_duplicates("STATE_NAME").set_index("STATE_NAME").STATEFP
    d["STATEFP"] = d.subapplicantState.map(st)
    d = d[d.STATEFP.notna()]
    flow.append(("Located in CONUS state", len(d), d.totalObligatedAmount.sum()))

    # county-name lookup within state; prefer county over independent city when ambiguous
    look = {}
    for r in counties.sort_values("LSAD").itertuples():  # LSAD '06' county sorts before '25' city
        for k in (norm(r.NAMELSAD), norm(r.NAME)):
            look.setdefault((r.STATEFP, k), r.GEOID)
        if r.LSAD == "25":
            look.setdefault((r.STATEFP, norm(r.NAME) + "CITY"), r.GEOID)
    rows, audit = [], []
    for r in d.itertuples():
        raw = r.benefitingCounties
        names = [x.strip() for x in str(raw).split(";") if x.strip()] if raw is not None else []
        if not names or any(re.search(r"STATE\s*WIDE|ALL COUNTIES", n, re.I) for n in names):
            audit.append((r.subapplicationIdentifier, raw, "statewide_or_missing", 0, len(names)))
            continue
        geo = [look.get((r.STATEFP, norm(n))) for n in names]
        ok = [g for g in geo if g]
        status = "matched" if len(ok) == len(names) else ("partial" if ok else "unmatched")
        audit.append((r.subapplicationIdentifier, raw, status, len(ok), len(names)))
        for g in dict.fromkeys(ok):
            rows.append((r.subapplicationIdentifier, g, len(set(ok))))
    a = pd.DataFrame(audit, columns=["subapplicationIdentifier", "benefitingCounties", "match_status",
                                     "n_matched", "n_listed"])
    a.to_csv(INTERIM / "hma_crosswalk_audit.csv", index=False)
    x = pd.DataFrame(rows, columns=["subapplicationIdentifier", "GEOID", "n_cty"])
    keep = ["subapplicationIdentifier", "prog", "program", "fiscalYear", "disasterNumber", "primaryActivity",
            "primaryHazard", "flood_activity", "flood_hazard", "flood_any", "totalObligatedAmount", "obl_real",
            "status", "STATEFP", "sub_fy"]
    x = x.merge(d[keep], on="subapplicationIdentifier", how="left")
    x["share"] = 1.0 / x.n_cty  # equal split across matched benefiting counties (documented convention)
    x["obl_real_alloc"] = x.obl_real * x.share
    x.to_parquet(PROCESSED / "hma_projects_county.parquet", index=False)

    dm = d.merge(a[["subapplicationIdentifier", "match_status"]], on="subapplicationIdentifier", how="left")
    for lab, m in (("Assigned to >=1 county", dm.match_status.isin(["matched", "partial"])),
                   ("  of which single-county", dm.subapplicationIdentifier.isin(x[x.n_cty == 1].subapplicationIdentifier)),
                   ("Statewide or missing county", dm.match_status == "statewide_or_missing"),
                   ("County name not matched", dm.match_status == "unmatched")):
        flow.append((lab, int(m.sum()), dm.loc[m, "totalObligatedAmount"].sum()))
    fl = dm[dm.flood_activity & dm.match_status.isin(["matched", "partial"])]
    flow.append(("Flood-activity projects assigned to counties (primary)", len(fl), fl.totalObligatedAmount.sum()))
    f = pd.DataFrame(flow, columns=["step", "records", "obligated_nominal_usd"])
    f.to_csv(TAB / "hma_inclusion_flow.csv", index=False)
    print(f.to_string())
    print(a.match_status.value_counts())
    return x


# ------------------------------------------------------------------ NOAA
def parse_damage(s):
    s = s.fillna("0").astype(str).str.strip().str.upper()
    mult = s.str[-1].map({"K": 1e3, "M": 1e6, "B": 1e9, "H": 1e2}).fillna(1.0)
    num = pd.to_numeric(s.str.rstrip("KMBH"), errors="coerce").fillna(0)
    return num * mult


def build_storms(defl):
    out = []
    cols = ["EVENT_ID", "EPISODE_ID", "YEAR", "STATE_FIPS", "CZ_TYPE", "CZ_FIPS", "EVENT_TYPE",
            "DAMAGE_PROPERTY", "DAMAGE_CROPS", "DEATHS_DIRECT"]
    for f in sorted(glob.glob(str(RAW / "noaa_storm_events" / "*.csv.gz"))):
        s = pd.read_csv(f, usecols=cols, low_memory=False)
        s = s[s.EVENT_TYPE.isin(RIVERINE_STORM_TYPES | COASTAL_STORM_TYPES)]
        out.append(s)
    s = pd.concat(out, ignore_index=True)
    s["prop_dmg"] = parse_damage(s.DAMAGE_PROPERTY)
    s["prop_dmg_real"] = s.prop_dmg * s.YEAR.map(defl).fillna(1.0)
    riv = s[s.EVENT_TYPE.isin(RIVERINE_STORM_TYPES) & (s.CZ_TYPE == "C")].copy()
    riv["GEOID"] = riv.STATE_FIPS.astype(int).astype(str).str.zfill(2) + riv.CZ_FIPS.astype(int).astype(str).str.zfill(3)
    g = riv.groupby(["GEOID", "YEAR"]).agg(
        flood_events=("EVENT_ID", "nunique"), flood_episodes=("EPISODE_ID", "nunique"),
        flood_dmg_real=("prop_dmg_real", "sum"), flood_deaths=("DEATHS_DIRECT", "sum")).reset_index()
    print("Storm Events riverine (Flood+Flash Flood, county-coded):", len(riv), "events;",
          "share with CZ_TYPE=C among riverine:",
          round((s[s.EVENT_TYPE.isin(RIVERINE_STORM_TYPES)].CZ_TYPE == "C").mean(), 4))
    return g.rename(columns={"YEAR": "year"})


def build_declarations():
    d = pd.read_parquet(RAW / "openfema" / "DisasterDeclarationsSummaries_v2.parquet")
    d = d[(d.declarationType == "DR") & (d.fipsCountyCode != "000")]
    d["GEOID"] = d.fipsStateCode.str.zfill(2) + d.fipsCountyCode.str.zfill(3)
    d["year"] = d.fyDeclared.astype(int)
    titled = d.declarationTitle.str.contains("FLOOD", case=False, na=False)
    # Primary: flood-related includes hurricanes, tropical and coastal storms (storm surge and rain flooding);
    # these titles rarely mention "flood" but generate most HMGP flood-mitigation funding.
    d["flood_dr"] = d.incidentType.isin(["Flood", "Hurricane", "Tropical Storm", "Coastal Storm", "Typhoon"]) | titled
    d["flood_dr_narrow"] = (d.incidentType == "Flood") | titled  # earlier definition, kept as sensitivity
    g = d.groupby(["GEOID", "year"]).disasterNumber.nunique().rename("dr_any").reset_index()
    for col in ("flood_dr", "flood_dr_narrow"):
        k = "dr_flood" if col == "flood_dr" else "dr_flood_narrow"
        g = g.merge(d[d[col]].groupby(["GEOID", "year"]).disasterNumber.nunique().rename(k).reset_index(),
                    on=["GEOID", "year"], how="left")
    return g.fillna({"dr_flood": 0, "dr_flood_narrow": 0})


# ------------------------------------------------------------------ panel
def main():
    defl = deflator()
    counties = build_counties()
    hma = build_hma(counties, defl)
    storms = build_storms(defl)
    decl = build_declarations()
    pop = population()

    years = range(1996, 2025)
    panel = pd.MultiIndex.from_product([counties.GEOID, years], names=["GEOID", "year"]).to_frame(index=False)
    panel = panel.merge(pd.DataFrame(counties.drop(columns="geometry")), on="GEOID")

    def agg(mask, name, when="fiscalYear"):
        h = hma[mask].groupby(["GEOID", when]).agg(
            **{f"obl_{name}": ("obl_real_alloc", "sum"), f"n_{name}": ("subapplicationIdentifier", "nunique")})
        return h.reset_index().rename(columns={when: "year"})

    for name, mask in {"flood": hma.flood_activity, "flood_haz": hma.flood_hazard, "flood_any": hma.flood_any,
                       "all": pd.Series(True, index=hma.index),
                       "flood_hmgp": hma.flood_activity & (hma.prog == "HMGP"),
                       "flood_fma": hma.flood_activity & (hma.prog == "FMA"),
                       "flood_pdm": hma.flood_activity & (hma.prog == "PDM/BRIC"),
                       "flood_single": hma.flood_activity & (hma.n_cty == 1)}.items():
        panel = panel.merge(agg(mask, name), on=["GEOID", "year"], how="left")
    for name, mask in {"flood_sub": hma.flood_activity,
                       "flood_hmgp_sub": hma.flood_activity & (hma.prog == "HMGP")}.items():  # by submission year
        panel = panel.merge(agg(mask & hma.sub_fy.notna(), name, "sub_fy"), on=["GEOID", "year"], how="left")
    panel = panel.merge(storms, on=["GEOID", "year"], how="left").merge(decl, on=["GEOID", "year"], how="left")
    zero = [c for c in panel.columns if c.startswith(("obl_", "n_", "flood_", "dr_"))]
    panel[zero] = panel[zero].fillna(0)
    panel = panel.merge(pop, on=["GEOID", "year"], how="left")
    first = pop[pop.year == 2000].set_index("GEOID")["pop"]
    panel["pop"] = panel["pop"].fillna(panel.GEOID.map(first))  # 1996-1999 use 2000 estimate
    panel = panel.merge(saipe(), on="GEOID", how="left")
    panel.to_parquet(PROCESSED / "county_year_panel.parquet", index=False)
    print(panel.shape, "counties:", panel.GEOID.nunique(), "missing pop:", panel["pop"].isna().mean().round(4))
    print(panel.groupby("year")[["obl_flood", "flood_events", "dr_flood"]].sum().tail(8))


if __name__ == "__main__":
    main()
