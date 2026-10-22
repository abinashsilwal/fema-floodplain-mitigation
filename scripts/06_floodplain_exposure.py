"""Developed land inside FEMA flood zones, by county and year (Annual NLCD 1.2 x NFHL).

Three resumable passes (each writes one file per unit and skips units already done):
  --zones          download each NFHL delivery, rasterize flood-zone classes for its county on the
                   NLCD 30 m grid -> BIG/exposure/zones/{GEOID}__{DFIRM}.tif, delete the zip
                   (needs only the grid definition, so it can run while NLCD is downloading)
  --counts         NLCD class counts by zone inside the county for every zone raster and year
                   -> BIG/exposure/nfhl/{GEOID}__{DFIRM}.parquet
  --county-totals  NLCD class counts for every CONUS county and year -> BIG/exposure/totals/{GEOID}.parquet
`--pilot` restricts every pass to the 40 pilot counties.

Zone classes (burned low -> high priority, so the SFHA wins where polygons overlap):
  0 not mapped / area not included, 1 X minimal, 2 X levee-reduced, 3 X 0.2% annual chance,
  4 open water zone, 5 V/VE coastal high hazard, 6 SFHA A-zones (A, AE, AH, AO, AR, A99, A1-30)
The NFHL is a fixed present-day mask applied to every year (exposure change, not hazard change).
"""
import argparse
import json
import os
import random
import sys
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("GDAL_DATA", os.path.join(sys.prefix, "Library", "share", "gdal"))

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import requests
import urllib3
from affine import Affine
from rasterio import features
from rasterio.crs import CRS
from rasterio.windows import Window, from_bounds

from config import BIG, PROCESSED, RAW

urllib3.disable_warnings()
YEARS = [2000, 2005, 2010, 2015, 2020, 2025]
NLCD_DIR = BIG / "nlcd"
OUT_Z = BIG / "exposure" / "zones"
OUT_N = BIG / "exposure" / "nfhl"
OUT_T = BIG / "exposure" / "totals"
TMP = BIG / "nfhl_tmp"
PORTAL = "https://hazards.fema.gov/femaportal/NFHL/"
KEEP = [11, 21, 22, 23, 24]
LUT = np.zeros(256, np.uint8)  # NLCD code -> 0..5 (0 = other)
for i, c in enumerate(KEEP, start=1):
    LUT[c] = i
CLS = ["other", "water", "dev_open", "dev_low", "dev_med", "dev_high"]
ZONES = ["unmapped", "x_minimal", "x_levee", "x_500", "open_water", "coastal_v", "sfha_a"]
PILOT = ["20045", "20091", "29095", "48201", "48453", "48439", "12086", "12095", "22033", "37183", "37129",
         "13121", "47037", "21111", "39049", "18097", "17031", "26163", "27053", "55025", "19153", "31055",
         "40109", "08031", "49035", "04013", "32003", "06037", "06067", "41051", "53033", "16001", "30063",
         "38017", "46099", "36061", "42101", "25025", "51059", "45019"]


def grid():
    g = json.load(open(BIG / "nlcd_grid.json"))
    return CRS.from_wkt(g["crs_wkt"]), Affine(*g["transform"]), g["width"], g["height"]


def window_for(bounds):
    _, tr, W, H = grid()
    w = from_bounds(*bounds, transform=tr).round_offsets(op="floor").round_lengths(op="ceil")
    w = w.intersection(Window(0, 0, W, H))
    return w, rasterio.windows.transform(w, tr)


def nlcd_paths():
    out = {}
    for y in YEARS:
        f = sorted(NLCD_DIR.glob(f"*LndCov_{y}_CU_C1V2*.tif"))
        if f:
            out[y] = f[0]
    return out


def counties():
    crs = grid()[0]
    return gpd.read_file(PROCESSED / "counties.gpkg").to_crs(crs)


# ------------------------------------------------------------------ zones
def zone_code(df):
    z = df.FLD_ZONE.fillna("").str.upper().str.strip()
    sub = df.ZONE_SUBTY.fillna("").str.upper()
    code = np.zeros(len(df), np.uint8)
    code[(z == "X") | (z == "B") | (z == "C")] = 1
    code[(z == "X") & sub.str.contains("LEVEE")] = 2
    code[((z == "X") & sub.str.contains("0.2 PCT")) | (z == "B")] = 3
    code[z == "OPEN WATER"] = 4
    code[z.isin(["V", "VE"]) | z.str.match(r"^V\d")] = 5
    # AR zones are "AR" or "AR/AE" etc.; plain startswith("AR") would also catch "AREA NOT INCLUDED"
    code[z.isin(["A", "AE", "AH", "AO", "A99"]) | z.str.match(r"^A\d") | z.str.match(r"^AR($|/)")] = 6
    return code  # 'AREA NOT INCLUDED', 'D' and blanks stay 0


def fetch_zip(href, dest, tries=40, deadline_s=600):
    """Short socket timeouts plus a per-attempt deadline: the portal throttles bursts of parallel
    requests by stalling connections, and long timeouts left workers hung for hours."""
    url = PORTAL + href.replace(" ", "%20")
    last = None
    for i in range(tries):
        try:
            t0 = time.time()
            with requests.get(url, timeout=(20, 60), verify=False, stream=True,
                              headers={"User-Agent": "Mozilla/5.0 (research)", "Connection": "close"}) as r:
                if r.status_code != 200:
                    raise IOError(f"HTTP {r.status_code}")
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        if time.time() - t0 > deadline_s:
                            raise TimeoutError("attempt exceeded deadline")
            zipfile.ZipFile(dest).testzip()
            return
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:80]}"
        time.sleep(2 + 4 * random.random())  # the portal resets bursts; quick jittered retries work best
    raise RuntimeError(f"download failed after {tries}: {last}")


def nfhl_zones(dfirm, href, targets):
    marker = OUT_Z / f"{dfirm}.json"
    if marker.exists():
        return dfirm, "skip"
    from shapely import wkb
    crs = grid()[0]
    TMP.mkdir(parents=True, exist_ok=True)
    zpath = TMP / f"{dfirm}.zip"
    fetch_zip(href, zpath)
    try:
        names = zipfile.ZipFile(zpath).namelist()
        base = f"/vsizip/{zpath.as_posix()}"
        shp = next((n for n in names if n.upper().endswith("S_FLD_HAZ_AR.SHP")), None)
        if shp is None:
            raise RuntimeError("no S_FLD_HAZ_AR")
        fz = pyogrio.read_dataframe(f"{base}/{shp}", columns=["FLD_ZONE", "ZONE_SUBTY"]).to_crs(crs)
        fz["code"] = zone_code(fz)
        fz = fz[fz.geometry.notna() & ~fz.geometry.is_empty].sort_values("code")
        eff = {}
        pan = next((n for n in names if n.upper().endswith("S_FIRM_PAN.SHP")), None)
        if pan:
            p = pyogrio.read_dataframe(f"{base}/{pan}", read_geometry=False, columns=["EFF_DATE"])
            yrs = pd.to_datetime(p.EFF_DATE.astype(str), errors="coerce", format="mixed").dt.year.dropna()
            if len(yrs):
                eff = {"eff_median": int(yrs.median()), "eff_min": int(yrs.min()), "eff_max": int(yrs.max())}
        done = []
        for geoid, gw in targets:
            geom = wkb.loads(gw)
            sub = fz[fz.intersects(geom.envelope)]
            if sub.empty:
                continue
            w, tr = window_for(geom.bounds)
            shape = (int(w.height), int(w.width))
            zone = features.rasterize(((g, int(c)) for g, c in zip(sub.geometry, sub.code)),
                                      out_shape=shape, transform=tr, fill=0, dtype="uint8")
            prof = dict(driver="GTiff", width=shape[1], height=shape[0], count=1, dtype="uint8", crs=crs,
                        transform=tr, compress="deflate", tiled=True, blockxsize=256, blockysize=256, nodata=None)
            with rasterio.open(OUT_Z / f"{geoid}__{dfirm}.tif", "w", **prof) as dst:
                dst.write(zone, 1)
                dst.update_tags(dfirm=dfirm, **{k: str(v) for k, v in eff.items()})
            done.append(geoid)
        json.dump({"dfirm": dfirm, "counties": done, "n_polys": int(len(fz)), **eff}, open(marker, "w"))
        return dfirm, f"ok {len(fz)} polys -> {done}"
    finally:
        zpath.unlink(missing_ok=True)


def plan_zones(cty, pilot):
    cat = pd.read_csv(RAW / "nfhl_catalog.csv", dtype={"dfirm": str})
    cat = cat[~cat.dfirm.str[:2].isin(["02", "15", "60", "66", "69", "72", "78"])]
    geo = cty.set_index("GEOID")
    jobs, covered = [], set()
    for r in cat[cat.dfirm.str.upper().str.endswith("C")].itertuples():
        g = r.dfirm[:5]
        if g in geo.index:
            covered.add(g)
            if not pilot or g in pilot:
                jobs.append((r.dfirm, r.href, [(g, geo.geometry[g].wkb)]))
    for r in cat[~cat.dfirm.str.upper().str.endswith("C")].itertuples():
        cand = [g for g in geo.index if g[:2] == r.dfirm[:2] and (not pilot or g in pilot)]
        if cand:
            jobs.append((r.dfirm, r.href, [(g, geo.geometry[g].wkb) for g in cand]))
    return jobs, covered


# ------------------------------------------------------------------ counts
def zone_counts(geoid, tifs, geom_wkb):
    """Merge every delivery's zone raster for the county (pixel max = highest-priority zone), then count."""
    from shapely import wkb
    out = OUT_N / f"{geoid}.parquet"
    if out.exists():
        return geoid, "skip"
    geom = wkb.loads(geom_wkb)
    zone, effs, dfirms = None, [], []
    for t in tifs:
        with rasterio.open(t) as zs:
            a = zs.read(1)
            tr = zs.transform
            tags = zs.tags()
        zone = a if zone is None else np.maximum(zone, a)  # all rasters share the county window
        dfirms.append(tags.get("dfirm", t.stem.split("__")[1]))
        if "eff_median" in tags:
            effs.append(float(tags["eff_median"]))
    _, gtr, _, _ = grid()
    col, row = ~gtr * (tr.c, tr.f)
    w = Window(int(round(col)), int(round(row)), zone.shape[1], zone.shape[0])
    cmask = features.geometry_mask([geom], out_shape=zone.shape, transform=tr, invert=True)
    rows = []
    for y, path in nlcd_paths().items():
        with rasterio.open(path) as src:
            a = LUT[src.read(1, window=w)]
        key = zone[cmask].astype(np.int32) * 6 + a[cmask]
        cc = np.bincount(key, minlength=42).reshape(7, 6)
        rows += [(geoid, y, ZONES[i], CLS[j], int(cc[i, j])) for i in range(7) for j in range(6) if cc[i, j]]
    df = pd.DataFrame(rows, columns=["GEOID", "year", "zone", "nlcd", "pixels"])
    df["dfirm"] = ";".join(dfirms)
    df["n_deliveries"] = len(dfirms)
    df["eff_median"] = float(np.median(effs)) if effs else np.nan
    df.to_parquet(out, index=False)
    return geoid, "ok"


def county_totals(geoid, geom_wkb):
    from shapely import wkb
    out = OUT_T / f"{geoid}.parquet"
    if out.exists():
        return geoid, "skip"
    geom = wkb.loads(geom_wkb)
    w, tr = window_for(geom.bounds)
    mask = features.geometry_mask([geom], out_shape=(int(w.height), int(w.width)), transform=tr, invert=True)
    rows = []
    for y, path in nlcd_paths().items():
        with rasterio.open(path) as src:
            a = LUT[src.read(1, window=w)]
        cc = np.bincount(a[mask], minlength=6)
        rows += [(geoid, y, CLS[i], int(n)) for i, n in enumerate(cc)]
    pd.DataFrame(rows, columns=["GEOID", "year", "nlcd", "pixels"]).to_parquet(out, index=False)
    return geoid, "ok"


GFPLAIN = BIG / "gfplain" / "NA" / "NA.TIF"  # Nardi et al. (2019) geomorphic floodplain, 250 m; 0 = floodplain
OUT_G = BIG / "exposure" / "gfplain"


def gfplain_counts(geoid, geom_wkb):
    """Developed land inside the GFPLAIN250m floodplain (complete-coverage sensitivity to NFHL gaps)."""
    from rasterio.warp import Resampling, reproject
    from shapely import wkb
    out = OUT_G / f"{geoid}.parquet"
    if out.exists():
        return geoid, "skip"
    geom = wkb.loads(geom_wkb)
    crs = grid()[0]
    w, tr = window_for(geom.bounds)
    shape = (int(w.height), int(w.width))
    fp = np.full(shape, 255, np.uint8)
    with rasterio.open(GFPLAIN) as src:
        reproject(rasterio.band(src, 1), fp, dst_transform=tr, dst_crs=crs, resampling=Resampling.nearest,
                  dst_nodata=255)
    inside = fp == 0
    cmask = features.geometry_mask([geom], out_shape=shape, transform=tr, invert=True)
    rows = []
    for y, path in nlcd_paths().items():
        with rasterio.open(path) as src:
            a = LUT[src.read(1, window=w)]
        for lab, m in (("gfplain", cmask & inside), ("outside", cmask & ~inside)):
            cc = np.bincount(a[m], minlength=6)
            rows += [(geoid, y, lab, CLS[i], int(n)) for i, n in enumerate(cc) if n]
    pd.DataFrame(rows, columns=["GEOID", "year", "zone", "nlcd", "pixels"]).to_parquet(out, index=False)
    return geoid, "ok"


def run(fn, jobs, workers, label):
    n_err = 0
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, *j): j[0] for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                _, s = f.result()
            except Exception as e:
                s = f"ERR {e}"
                n_err += 1
            if s.startswith("ERR") or i % 25 == 0 or i == len(jobs):
                print(f"{label} {i}/{len(jobs)} {futs[f]} {s[:160]}", flush=True)
    print(f"{label} finished, errors: {n_err}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", action="store_true")
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--county-totals", action="store_true")
    ap.add_argument("--gfplain", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for d in (OUT_Z, OUT_N, OUT_T, OUT_G):
        d.mkdir(parents=True, exist_ok=True)
    cty = counties()
    pilot = set(PILOT) if a.pilot else None
    geo = cty.set_index("GEOID").geometry
    if a.zones:
        jobs, covered = plan_zones(cty, pilot)
        print(f"zone jobs: {len(jobs)}; counties with a county-wide NFHL delivery: {len(covered)}", flush=True)
        run(nfhl_zones, jobs, a.workers, "zones")
    if a.counts:
        by = {}
        for t in sorted(OUT_Z.glob("*__*.tif")):
            by.setdefault(t.stem.split("__")[0], []).append(t)
        if pilot:
            by = {g: v for g, v in by.items() if g in pilot}
        run(zone_counts, [(g, v, geo[g].wkb) for g, v in by.items()], a.workers, "counts")
    if a.gfplain:
        sel = [g for g in geo.index if not pilot or g in pilot]
        run(gfplain_counts, [(g, geo[g].wkb) for g in sel], a.workers, "gfplain")
    if a.county_totals:
        sel = [g for g in geo.index if not pilot or g in pilot]
        run(county_totals, [(g, geo[g].wkb) for g in sel], a.workers, "totals")


if __name__ == "__main__":
    main()
