"""County tables for mitigation (buildings acquired/elevated, obligations), insured losses (NFIP claims),
insured new construction (NFIP policies) and covariates (penetration, SVI, rurality).

Output: data/processed/county_mitigation_loss.parquet (one row per county) and
        data/processed/county_period_mitigation_loss.parquet (county x 5-year period, 2001-2025)
"""
import numpy as np
import pandas as pd

from config import PROCESSED, RAW, TAB
from importlib import import_module

PERIODS = [(2001, 2005), (2006, 2010), (2011, 2015), (2016, 2020), (2021, 2025)]
SFHA = r"^(A|AE|AH|AO|AR|A\d|A99)"  # rated/current zone codes for A-zones


def period_of(y):
    y = pd.to_numeric(y, errors="coerce")
    out = pd.Series(pd.NA, index=y.index, dtype="object")
    for a, b in PERIODS:
        out[(y >= a) & (y <= b)] = f"{a}-{b}"
    return out


def defl():
    return import_module("02_build_county_panel").deflator()


def mitigation(d):
    """Buildings and obligations from county-assigned flood-mitigation projects (equal split over counties)."""
    h = pd.read_parquet(PROCESSED / "hma_projects_county.parquet")
    v = pd.read_parquet(RAW / "openfema" / "HmaSubapplications_v2.parquet",
                        columns=["subapplicationIdentifier", "numberBuildingLocations"])
    h = h[h.flood_activity].merge(v, on="subapplicationIdentifier", how="left")
    h["bldg"] = h.numberBuildingLocations.fillna(0) * h.share
    h["acq"] = np.where(h.primaryActivity.isin(["Acquisition", "Relocation"]), h.bldg, 0)
    h["elev"] = np.where(h.primaryActivity.isin(["Elevation", "Mitigation reconstruction"]), h.bldg, 0)
    h["period"] = period_of(h.fiscalYear)
    for p in ("HMGP", "FMA", "PDM/BRIC"):
        h[f"obl_{p}"] = np.where(h.prog == p, h.obl_real_alloc, 0)
    agg = dict(obl=("obl_real_alloc", "sum"), bldg=("bldg", "sum"), acq=("acq", "sum"), elev=("elev", "sum"),
               obl_hmgp=("obl_HMGP", "sum"), obl_fma=("obl_FMA", "sum"), obl_pdm=("obl_PDM/BRIC", "sum"),
               n_proj=("subapplicationIdentifier", "nunique"))
    per = h.dropna(subset=["period"]).groupby(["GEOID", "period"]).agg(**agg).reset_index()
    tot = h[(h.fiscalYear >= 2001) & (h.fiscalYear <= 2025)].groupby("GEOID").agg(**agg)
    tot.columns = [f"mit_{c}_2001_2025" for c in tot.columns]
    allyrs = h.groupby("GEOID").agg(mit_acq_all=("acq", "sum"), mit_elev_all=("elev", "sum"))
    # buildings for the building comparison: same years as insured construction (2001-2024)
    b24 = h[h.fiscalYear.between(2001, 2024)].groupby("GEOID").agg(mit_acq_2001_2024=("acq", "sum"),
                                                                    mit_elev_2001_2024=("elev", "sum"))
    allyrs = allyrs.join(b24, how="outer")
    # sensitivity outcomes: window before HMGP right-censoring / 2025 freeze; closed or completed projects only
    w21 = h[(h.fiscalYear >= 2001) & (h.fiscalYear <= 2021)].groupby("GEOID").obl_real_alloc.sum().rename("mit_obl_2001_2021")
    closed = h[(h.fiscalYear >= 2001) & (h.fiscalYear <= 2025) & h.status.isin(["Closed", "Completed", "Closeout Initiated"])]
    cl = closed.groupby("GEOID").obl_real_alloc.sum().rename("mit_obl_closed_2001_2025")
    # allocation check: single-county projects only (no equal split across benefiting counties)
    one_cty = h[(h.n_cty == 1) & h.fiscalYear.between(2001, 2025)]
    cl = pd.concat([cl, one_cty.groupby("GEOID").obl_real_alloc.sum().rename("mit_obl_single_2001_2025")], axis=1)
    # federal obligation per mitigated building, single-county acquisition/elevation projects (for scale comparison)
    one = h[(h.n_cty == 1) & (h.bldg > 0) & h.primaryActivity.isin(["Acquisition", "Elevation"]) &
            (h.fiscalYear >= 2001) & (h.fiscalYear <= 2025)]
    unit = (one.obl_real / one.numberBuildingLocations).groupby(one.primaryActivity).median()
    unit.rename("median_obl_per_building_2024usd").to_csv(TAB / "exp_unit_cost.csv")
    return per, tot.join(allyrs, how="outer").join(w21, how="outer").join(cl, how="outer")


def claims(d):
    c = pd.read_parquet(RAW / "openfema" / "NfipClaims_v3_paged.parquet")
    c["GEOID"] = c.countyCode.astype(str).str.zfill(5)
    zone = c.ratedFloodZone.fillna(c.floodZoneCurrent).fillna("").str.upper()
    c["sfha"] = zone.str.match(SFHA)
    c["con_year"] = pd.to_datetime(c.originalConstructionDate, errors="coerce", utc=True).dt.year
    c["new2000"] = c.con_year >= 2000
    c["paid"] = c[["amountPaidOnBuildingClaim", "amountPaidOnContentsClaim"]].astype(float).fillna(0).sum(axis=1)
    c["paid_real"] = c.paid * c.yearOfLoss.map(d).fillna(1.0)
    c["period"] = period_of(c.yearOfLoss)
    c = c[c.yearOfLoss.between(1996, 2025)]
    g = lambda m: c[m].groupby("GEOID")
    tot = pd.DataFrame({
        "claims_n_1996_2025": c.groupby("GEOID").size(),
        "claims_paid_1996_2025": c.groupby("GEOID").paid_real.sum(),
        "claims_paid_1996_2000": g(c.yearOfLoss <= 2000).paid_real.sum(),
        "claims_paid_sfha_2001_2025": g(c.sfha & (c.yearOfLoss >= 2001)).paid_real.sum(),
        "claims_n_new2000_sfha": g(c.sfha & c.new2000).size(),
        "claims_paid_new2000_sfha": g(c.sfha & c.new2000).paid_real.sum(),
    })
    # flood type from NFIP causeOfDamage (OpenFEMA data dictionary): 2 stream/river/lake overflow, 3 alluvial fan,
    # A closed-basin lake -> riverine; 1 tidal overflow -> coastal; 4 accumulation of rainfall/snowmelt -> pluvial;
    # 0, 7, 8, 9, B, C, D (other, erosion, earth movement, expedited handling) -> other/unknown
    cod = c.causeOfDamage.astype(str).str.strip().str.upper()
    c["ftype"] = np.select([cod.isin(["2", "3", "A"]), cod.eq("1"), cod.eq("4")], ["riverine", "coastal", "pluvial"],
                           "other")
    ft = c.groupby(["ftype", "sfha"]).agg(n=("paid", "size"), paid=("paid_real", "sum")).reset_index()
    ft["pct_claims"] = 100 * ft.n / ft.n.sum()
    ft["pct_paid"] = 100 * ft.paid / ft.paid.sum()
    ft.to_csv(TAB / "exp_claims_by_flood_type.csv", index=False)
    by_type = c.pivot_table(index="GEOID", columns="ftype", values="paid_real", aggfunc="sum", fill_value=0)
    by_type.columns = [f"claims_paid_{k}" for k in by_type.columns]
    # fair comparison of new vs older A-zone buildings: claim and payment rates per policy-year, 2016-2024
    pol = pd.read_parquet(PROCESSED / "nfip_policies_agg.parquet")
    pp = pol[pol.pol_year.between(2016, 2024) & (pol.zone == "sfha_a") & pol.con_year.notna()]
    w = c[c.yearOfLoss.between(2016, 2024) & c.sfha & c.con_year.notna()]
    rates = {}
    for lab, pm, cm in (("built_2001_plus", pp.con_year >= 2001, w.con_year >= 2001),
                        ("built_before_2001", pp.con_year < 2001, w.con_year < 2001)):
        py = pp.loc[pm, "policies"].sum()
        rates[lab] = {"policy_years": py, "claims": int(cm.sum()), "claims_per_1000_py": 1000 * cm.sum() / py,
                      "paid_per_py_usd2024": w.loc[cm, "paid_real"].sum() / py}
    pd.DataFrame(rates).T.to_csv(TAB / "exp_new_vs_old_claim_rates.csv")
    # same comparison within Census regions (newer buildings are not spread like older ones)
    import geopandas as gpd
    reg = gpd.read_file(PROCESSED / "counties.gpkg", columns=["GEOID", "region"]).set_index("GEOID").region
    rr = []
    for r_name in sorted(reg.dropna().unique()):
        gs = set(reg[reg == r_name].index)
        pr, wr = pp[pp.GEOID.isin(gs)], w[w.GEOID.isin(gs)]
        new_py, old_py = pr.loc[pr.con_year >= 2001, "policies"].sum(), pr.loc[pr.con_year < 2001, "policies"].sum()
        new_c, old_c = int((wr.con_year >= 2001).sum()), int((wr.con_year < 2001).sum())
        rr.append((r_name, 1000 * new_c / new_py, 1000 * old_c / old_py, (new_c / new_py) / (old_c / old_py)))
    pd.DataFrame(rr, columns=["region", "new_per_1000_py", "old_per_1000_py", "rate_ratio"]) \
        .to_csv(TAB / "exp_new_vs_old_claim_rates_region.csv", index=False)
    per = c.dropna(subset=["period"]).groupby(["GEOID", "period"]).agg(
        claims_n=("paid", "size"), claims_paid=("paid_real", "sum")).reset_index()
    nat = {"claims_total": len(c), "paid_total_bn": c.paid_real.sum() / 1e9,
           "share_claims_new2000": c.new2000.mean(), "share_paid_new2000": c.loc[c.new2000, "paid_real"].sum() / c.paid_real.sum(),
           "share_sfha_claims_new2000": c.loc[c.sfha, "new2000"].mean(),
           "share_claims_outside_sfha": (~c.sfha).mean()}
    return per, tot.join(by_type, how="left"), nat


def policies():
    p = pd.read_parquet(PROCESSED / "nfip_policies_agg.parquet")
    p = p[(p.pol_year == 2024) & (p.zone == "sfha_a")]
    out = pd.DataFrame({
        "pol_sfha_2024": p.groupby("GEOID").policies.sum(),
        "pol_sfha_built2000plus": p[p.con_year >= 2000].groupby("GEOID").policies.sum(),
        "pol_sfha_built2001_2024": p[(p.con_year >= 2001) & (p.con_year <= 2024)].groupby("GEOID").policies.sum(),
    })
    return out


def covariates():
    pen = pd.read_parquet(RAW / "openfema" / "NfipResidentialPenetrationRates_v1.parquet")
    pen = pen.rename(columns={"fipsCode": "GEOID"}).set_index("GEOID")[
        ["resPenetrationRateSfha", "totalResStructuresSfha", "totalResStructures", "resContractsInForceSfha"]]
    svi = pd.read_csv(RAW / "cdc" / "SVI_2020_US_county.csv", dtype={"FIPS": str})
    svi = svi.assign(GEOID=svi.FIPS.str.zfill(5)).set_index("GEOID")[["RPL_THEMES"]].rename(columns={"RPL_THEMES": "svi_2020"})
    svi["svi_2020"] = svi.svi_2020.where(svi.svi_2020 >= 0)
    r = pd.read_csv(RAW / "usda" / "Ruralurbancontinuumcodes2023.csv", encoding="latin-1", dtype={"FIPS": str})
    r = r[r.Attribute == "RUCC_2023"].assign(GEOID=lambda x: x.FIPS.str.zfill(5))
    r = r.set_index("GEOID")[["Value"]].rename(columns={"Value": "rucc_2023"}).astype(float)
    r["metro"] = r.rucc_2023 <= 3
    return pen.join(svi, how="outer").join(r, how="outer")


def repetitive_loss():
    """NFIP repetitive-loss (RL) and severe repetitive-loss (SRL) properties and how many are mitigated."""
    m = pd.read_parquet(RAW / "openfema" / "NfipMultipleLossProperties_v1_paged.parquet")
    m["GEOID"] = m.fipsCountyCode.astype(str).str.zfill(5)
    m["srl"] = m.nfipSrl.fillna(False) | m.fmaSrl.fillna(False)
    m["mit"] = m.mitigatedIndicator.fillna(False)
    m["sfha"] = m.floodZone.fillna("").str.upper().str.match(SFHA)
    out = m.groupby("GEOID").agg(rl_n=("mit", "size"), rl_mitigated=("mit", "sum"), srl_n=("srl", "sum"),
                                 srl_mitigated=("srl", lambda s: (s & m.loc[s.index, "mit"]).sum()),
                                 rl_sfha=("sfha", "sum"), rl_losses=("totalLosses", "sum"))
    nat = {"rl_total": len(m), "rl_mitigated": int(m.mit.sum()), "rl_share_mitigated": m.mit.mean(),
           "srl_total": int(m.srl.sum()), "srl_share_mitigated": m.loc[m.srl, "mit"].mean(),
           "rl_share_sfha": m.sfha.mean(), "rl_unmitigated_losses": int(m.loc[~m.mit, "totalLosses"].sum())}
    pd.Series(nat).to_csv(TAB / "exp_repetitive_loss_national.csv")
    return out


def eligibility_capacity():
    """Approved county hazard mitigation plan (HMA eligibility) and CRS participation (capacity); current snapshots."""
    norm = import_module("02_build_county_panel").norm
    p = pd.read_parquet(RAW / "openfema" / "HazardMitigationPlanStatuses_v1_paged.parquet")
    p = p[p.jurisdictionType.str.startswith("County", na=False) & (p.mppGeoid.str.len() == 5)]
    plan = p.groupby("mppGeoid").jurisdictionStatus.apply(lambda s: s.isin(["Approved", "APA"]).any()).rename("plan_ok")
    import geopandas as gpd
    cty = gpd.read_file(PROCESSED / "counties.gpkg")[["GEOID", "STUSPS", "NAME", "NAMELSAD"]]
    look = {}
    for r in cty.itertuples():
        for k in (norm(r.NAMELSAD), norm(r.NAME)):
            look.setdefault((r.STUSPS, k), r.GEOID)
    s = pd.read_parquet(RAW / "openfema" / "NfipCommunityStatusBook_v1_paged.parquet")
    s["crs"] = pd.to_numeric(s.classRating, errors="coerce").between(1, 9)
    s["firm_year"] = pd.to_datetime(s.initialFloodInsuranceRateMap, errors="coerce", utc=True).dt.year
    rows = []
    for r in s.itertuples():
        for nm in str(r.county).split("/"):
            g = look.get((r.state, norm(nm)))
            if g:
                rows.append((g, r.crs, r.firm_year, bool(r.participatingInNFIP)))
    t = pd.DataFrame(rows, columns=["GEOID", "crs", "firm_year", "nfip"])
    cap = t.groupby("GEOID").agg(crs_any=("crs", "max"), first_firm_year=("firm_year", "min"),
                                 nfip_communities=("nfip", "sum"))
    return cap.join(plan, how="outer")


def other_mitigation(d):
    """Flood mitigation outside HMA, FY/loss years 2001-2025, by county:
    - NFIP Increased Cost of Compliance (ICC) payments from claims (often used as the local share of HMA projects,
      so ICC buildings may also be HMA buildings);
    - FEMA Public Assistance Section 406 hazard mitigation (mitigationAmount) on flood-related declarations,
      fiscal year of declaration; statewide records (county code 000) excluded;
    - HUD CDBG-DR buyouts (census tracts, reported through March 2020, tracts with < 11 buyouts suppressed):
      presence only, used to check the 'no FEMA funding' label."""
    c = pd.read_parquet(RAW / "openfema" / "NfipClaims_v3_paged.parquet",
                        columns=["countyCode", "yearOfLoss", "amountPaidOnIncreasedCostOfComplianceClaim"])
    c = c[c.yearOfLoss.between(2001, 2025)]
    c["icc"] = pd.to_numeric(c.amountPaidOnIncreasedCostOfComplianceClaim, errors="coerce").fillna(0)
    c = c[c.icc > 0].assign(GEOID=lambda x: x.countyCode.astype(str).str.zfill(5))
    c["icc_real"] = c.icc * c.yearOfLoss.map(d).fillna(1.0)
    icc = c.groupby("GEOID").agg(icc_2001_2025=("icc_real", "sum"), icc_n_2001_2025=("icc", "size"))
    pa = pd.read_parquet(RAW / "openfema" / "PublicAssistanceFundedProjectsDetails_v2_paged.parquet")
    dec = pd.read_parquet(RAW / "openfema" / "DisasterDeclarationsSummaries_v2.parquet",
                          columns=["disasterNumber", "incidentType", "declarationTitle"]).drop_duplicates("disasterNumber")
    dec["flood_dr"] = dec.incidentType.isin(["Flood", "Hurricane", "Tropical Storm", "Coastal Storm", "Typhoon"]) | \
        dec.declarationTitle.str.contains("FLOOD", case=False, na=False)
    pa = pa.merge(dec[["disasterNumber", "flood_dr"]], on="disasterNumber", how="left")
    dt = pd.to_datetime(pa.declarationDate, errors="coerce", utc=True)
    pa["fy"] = dt.dt.year + (dt.dt.month >= 10)
    pa = pa[pa.flood_dr.fillna(False) & pa.fy.between(2001, 2025) & (pd.to_numeric(pa.countyCode, errors="coerce") > 0)]
    pa["GEOID"] = pa.stateNumberCode.astype(str).str.zfill(2) + pa.countyCode.astype(str).str.zfill(3)
    pa["m_real"] = pd.to_numeric(pa.mitigationAmount, errors="coerce").fillna(0) * pa.fy.map(d).fillna(1.0)
    pa406 = pa.groupby("GEOID").m_real.sum().rename("pa406_2001_2025")
    import geopandas as gpd
    hud = gpd.read_file(f"zip://{RAW / 'hud' / 'cdbg_dr_buyouts_tract.zip'}")
    hud = pd.Series(True, index=pd.Index(hud.CNTY_FIPS.unique(), name="GEOID"), name="hud_buyout_any")
    return icc.join(pa406, how="outer").join(hud, how="outer")


def main():
    d = defl()
    mper, mtot = mitigation(d)
    cper, ctot, nat = claims(d)
    out = mtot.join(ctot, how="outer").join(policies(), how="outer").join(covariates(), how="left") \
        .join(repetitive_loss(), how="left").join(eligibility_capacity(), how="left") \
        .join(other_mitigation(d), how="outer")
    out.index.name = "GEOID"
    out.reset_index().to_parquet(PROCESSED / "county_mitigation_loss.parquet", index=False)
    mper.merge(cper, on=["GEOID", "period"], how="outer").to_parquet(
        PROCESSED / "county_period_mitigation_loss.parquet", index=False)
    pd.Series(nat).to_csv(PROCESSED.parent.parent / "outputs" / "tables" / "nfip_claims_national.csv")
    print(pd.Series(nat).round(4).to_string())
    print(out.describe().T[["count", "mean", "50%", "max"]].round(2).to_string())
    print("national: acquisitions(all yrs)", out.mit_acq_all.sum().round(0), " elevations", out.mit_elev_all.sum().round(0),
          " SFHA res structures (FEMA)", out.totalResStructuresSfha.sum(), " insured SFHA built>=2000 (2024)",
          out.pol_sfha_built2000plus.sum())


if __name__ == "__main__":
    main()
