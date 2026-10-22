# FEMA mitigation grants and new floodplain development

Code and derived data for the article "FEMA mitigation grants track past losses but not new floodplain
development" (Silwal et al., in preparation).

The study combines Annual NLCD Collection 1.2 land cover (2000-2025), the FEMA National Flood Hazard Layer, and
FEMA's national records of Hazard Mitigation Assistance (HMA) subapplications, project site inventories, NFIP
policies and NFIP claims for the contiguous United States. It measures developed land added inside mapped 1%
annual-chance floodplains, compares newer and older insured floodplain buildings, and relates HMA funding,
applications and project sites to new development, existing exposure and past losses.

## Repository layout

```
scripts/            analysis pipeline (run in the order below)
data/processed/     derived county, county-year and ZIP-code tables (parquet) and county boundaries
data/raw/           provenance.csv (source, date and checksum of every input) and two small inputs
outputs/tables/     model results and summary tables (CSV); numbers.json holds every reported number
docs/               analysis plan, written before national results were examined
```

## Data

Only derived data are included. All inputs are public; `data/raw/provenance.csv` lists, for every input file, the
path the scripts expect (relative to `data/raw`, or to the large-data folder for entries starting `bigdata/`), the
source URL, the download date and the SHA-256 checksum.

| Source | Use |
|---|---|
| USGS Annual NLCD Collection 1.2 (doi:10.5066/P94UXNTS) | developed land, 2000-2025 |
| FEMA National Flood Hazard Layer, county and community deliveries | 1% annual-chance A zones, V zones, 0.2% zone, levee areas |
| OpenFEMA HMA Subapplications v2; HMA Subapplications Project Site Inventories v1 | funding, applications, proposed project sites |
| OpenFEMA FIMA NFIP Redacted Policies v2 and Claims v3 | insured units, claims and payments |
| OpenFEMA Disaster Declarations Summaries v2; Public Assistance Funded Projects Details v2 | declarations; Section 406 mitigation |
| OpenFEMA NFIP Multiple Loss Properties, Community Status Book, Hazard Mitigation Plan Statuses | repetitive loss, CRS, plan status |
| U.S. EPA riverine flood risk (doi:10.23719/1523021); USGS Watershed Boundary Dataset | projected damage by watershed |
| GFPLAIN250m (Nardi et al., 2019) | terrain-based floodplain check |
| U.S. Census Bureau, CDC/ATSDR SVI 2020, USDA RUCC 2023, BEA GDP deflator | boundaries, covariates, prices |

Two small public inputs are included so that the reporting step runs from this repository: the NFHL delivery catalogue (`data/raw/nfhl_catalog.csv`) and the OpenFEMA HMA Subapplications file (`data/raw/openfema/HmaSubapplications_v2.parquet`).

The full NFIP policies file (about 29 GB) can be downloaded from OpenFEMA; set the `NFIP_POLICIES_CSV`
environment variable to its path. Large rasters and vectors are read from `~/fema_flood_bigdata` (override with
`FLOOD_BIGDATA`).

The NFIP tables here are aggregates of FEMA's public, redacted records (counts and sums by county, ZIP code,
flood zone, construction period and year); no record-level data are redistributed.

## Running the pipeline

Create the environment with `conda env create -f environment.yml`, then run from the repository root:

| Step | Script | Purpose |
|---|---|---|
| 1 | (see `data/raw/provenance.csv`) | obtain the public input files |
| 2 | `02_build_county_panel.py` | county-year panel of funding, declarations and covariates |
| 3 | `06_floodplain_exposure.py --zones --counts --county-totals --gfplain` | NFHL rasterization and developed land by flood zone |
| 4 | `07_exposure_table.py` | county exposure tables |
| 5 | `08_nfip_policies.py`, `08b_nfip_policies_zip.py` | NFIP policy aggregates by county and ZIP code |
| 6 | `09_mitigation_loss_tables.py` | mitigation, loss and claim-rate tables |
| 7 | `03_analysis.py` | shared estimators, palette and event study |
| 8 | `10_exposure_analysis.py` | county models, robustness, typology, Figs. 1-3 and 6 (written to `outputs/figures`) |
| 9 | `13_applications.py` | applications and awards |
| 10 | `14_ead_crosswalk.py`, `16_qa_nlcd_transitions.py` | watershed-area EAD check; land-cover transition check |
| 11 | `17_within_county.py`, `20_contrasts.py`, `19_spatial_se.py`, `21_signed_transform.py` | building-age, within-county, contrast, spatial-error and transformation checks |
| 12 | `18_fig_targeting.py` | Figs. 4 and 5 |
| 13 | `11_numbers.py`, `12_reconcile.py` | write every reported number; consistency checks |

Steps 3 and 5 are slow and need about 40 GB of disk space (about 11 GB of land-cover, floodplain and
watershed data plus the 29 GB NFIP policies file). The derived tables in `data/processed` allow steps 6-13 to be
rerun without them, except for the scripts that read the NFIP policy CSV or the rasters directly.

## Licence

Code: MIT licence (see `LICENSE`). Derived data: CC BY 4.0. The underlying federal datasets are in the public
domain.

## Citation

Silwal, A. et al. FEMA mitigation grants track past losses but not new floodplain development (in preparation).
