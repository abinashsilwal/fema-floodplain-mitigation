# Analysis plan v2 (fixed before national results were inspected)

Date: 2026-10-21. Written while the national NFHL run was still processing, so that the analyses,
thresholds and claims below are committed before the main results are seen. Any later change must be
logged in DECISION_LOG.md with the reason.

Working title: *Emerging riverine flood exposure and federal mitigation investment across the contiguous United States*

## 1. Questions and primary outcomes

| # | Question | Primary quantity | Unit / period |
|---|---|---|---|
| Q1 | How much developed land was added inside the present-day 1% floodplain? | km² of NLCD developed classes (21–24) inside NFHL A-zones, 2000→2025; share of new development in SFHA | covered counties, 5-yr steps |
| Q2 | How does new floodplain construction compare with federally mitigated buildings? | annual flow: new insured A-zone buildings (built 2001–2024) vs buildings acquired + elevated (FY2001–2025) | national, region, county |
| Q3 | Does federal flood-mitigation investment align with emerging exposure, baseline exposure, losses or declarations, by program? | AME (pp per SD) of emerging exposure on any funding; OLS on log funding | covered counties, FY2001–2025 |
| Q4 | Does funding follow earlier exposure growth? | coefficient on lagged exposure growth; lead (placebo) must be ~0 | 5-yr county panel, county+period FE |
| Q5 | Where does rapid exposure growth meet no federal mitigation, and who lives there? | counties/population/new buildings in "high growth, no funding"; SVI, rurality, plan status, CRS | covered counties |
| Q6 | How much of the highest-loss existing stock has been mitigated? | share of NFIP repetitive-loss (RL/SRL) properties with mitigatedIndicator = True | national, county |
| Q7 | What is the scale of the gap in program throughput? | ratio of annual new insured SFHA buildings to annual mitigated buildings; dollars at observed per-building federal cost (scale comparison only) | national |

## 2. Extensions added for contribution (all use public data already downloaded)

1. **Mitigation coverage of the highest-loss stock (Q6).** NFIP Multiple Loss Properties (240,651 RL/SRL
   properties; `mitigatedIndicator`). Report share mitigated nationally, by region, and by county exposure-growth
   class; compare with new insured construction. Descriptive; no causal claim.
2. **Throughput gap (Q7).** Primary metric in buildings per year (flows), not dollars, because new construction
   must meet NFIP elevation standards and does not need the same mitigation. Dollar figure = new buildings ×
   median federal obligation per mitigated building (single-activity projects), reported only as a scale
   comparison with an explicit caveat.
3. **Who is left behind (Q5).** Among high-growth counties, model "no funding" on SVI, nonmetro status,
   population, approved hazard mitigation plan (eligibility), and CRS participation (capacity). Plan status and
   CRS are current snapshots, so they are reported as correlates, not causes.

## 3. Safeguards against over-claiming (each is reported, not optional)

| Risk | Safeguard |
|---|---|
| Unmapped areas (no adequate NFHL) | Report share of population, development and NFIP claims in non-covered counties; GFPLAIN250m complete-coverage sensitivity for all counties |
| Fixed present-day map (map endogeneity) | Subsets: median panel date ≤ 2010; first FIRM date (Community Status Book) before 1990 |
| Coastal A-zones mixed with riverine | Non-coastal subset; V-zones reported separately |
| 0.2% zone and edge development | Second exposure tier (0.2% zone) and 1% + 0.2% combined sensitivity |
| Insured-only building counts | Report as lower bound; penetration-adjusted bound (new insured / SFHA take-up, capped); NLCD area remains primary |
| New construction is code-compliant (exposure ≠ vulnerability) | Report post-2000 buildings' share of claims and payouts alongside their share of policies |
| Statewide projects not assigned to counties (~9% of $) | Report; national totals include them; sensitivity allocating statewide $ by county population |
| HMGP right-censoring / 2025 program freeze | Sensitivity restricted to FY2001–2021 |
| Obligations ≠ completed projects | Sensitivity with closed/completed projects only |
| Association ≠ causation | No causal language; lagged panel + lead placebo; all coefficients reported, including nulls |
| Researcher degrees of freedom | Thresholds fixed here (top tercile = high growth; ≥ 90% developed-land coverage; 0.5 km² baseline for % growth); alternatives in supplement |

## 4. Claims we will NOT make

- That federal mitigation *caused* or *failed to cause* changes in development or losses.
- That unmapped counties have no floodplain exposure.
- That every new floodplain building needs federal mitigation, or that the dollar figure is a funding need.
- Any projection of future exposure or of future funding.

## 5. Stop/revise rules

- If < 75% of CONUS population lives in covered counties, the national headline is reframed as "mapped counties"
  and the GFPLAIN result is promoted to the main text.
- If the floodplain-vs-outside growth contrast or the building ratio reverses under the map-vintage or
  non-coastal subsets, the headline is softened and the instability is reported.
- If the lead (placebo) coefficient in the period panel is as large as the lag, Q4 is reported as uninformative.
