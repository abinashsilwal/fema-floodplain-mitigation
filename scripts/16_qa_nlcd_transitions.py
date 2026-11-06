"""QA: gross versus net change in developed land inside A zones, 2000-2025 and 2020-2025.

The exposure measure is net change in developed area (NLCD classes 21-24). If pixels developed in an earlier year
are classified as undeveloped later, gross new development exceeds the net figure. This counts, inside A zones, pixels
gained (undeveloped -> developed) and lost (developed -> undeveloped) for a random sample of covered counties plus
the Figure 2 counties. Output: outputs/tables/qa_nlcd_transitions.csv (per county) and qa_nlcd_summary.csv.
"""
import importlib
import json

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.windows import Window

from config import BIG, TAB

E = importlib.import_module("10_exposure_analysis")
DEV = [21, 22, 23, 24]


def county_transitions(geoid, gtr, nl):
    zone = None
    for t in sorted((BIG / "exposure" / "zones").glob(f"{geoid}__*.tif")):
        with rasterio.open(t) as zs:
            a, tr = zs.read(1), zs.transform
        zone = a if zone is None else np.maximum(zone, a)
    if zone is None:
        return None
    col, row = ~gtr * (tr.c, tr.f)
    win = Window(int(round(col)), int(round(row)), zone.shape[1], zone.shape[0])
    dev = {}
    for y, f in nl.items():
        with rasterio.open(f) as s:
            dev[y] = np.isin(s.read(1, window=win), DEV)
    a_ = zone == 6
    px = 900 / 1e6  # km2 per 30 m pixel
    out = {"GEOID": geoid}
    for y0, y1 in ((2000, 2025), (2020, 2025)):
        gain = (a_ & dev[y1] & ~dev[y0]).sum() * px
        loss = (a_ & dev[y0] & ~dev[y1]).sum() * px
        out[f"gain_{y0}_{y1}"], out[f"loss_{y0}_{y1}"], out[f"net_{y0}_{y1}"] = gain, loss, gain - loss
    return out


def main(n=80, seed=7):
    c, _ = E.load()
    cov = c.index[c.nfhl_covered.fillna(False)]
    rng = np.random.default_rng(seed)
    sample = list(rng.choice(cov, size=n, replace=False)) + [g for g, _ in E.EXAMPLES]
    gtr = Affine(*json.load(open(BIG / "nlcd_grid.json"))["transform"])
    nl = {y: sorted((BIG / "nlcd").glob(f"*LndCov_{y}_CU_C1V2*.tif"))[0] for y in (2000, 2020, 2025)}
    rows = [r for r in (county_transitions(g, gtr, nl) for g in dict.fromkeys(sample)) if r]
    t = pd.DataFrame(rows).set_index("GEOID")
    t.round(4).to_csv(TAB / "qa_nlcd_transitions.csv")
    s = {"counties": len(t)}
    for p_ in ("2000_2025", "2020_2025"):
        s[f"loss_share_of_gain_{p_}_pct"] = 100 * t[f"loss_{p_}"].sum() / t[f"gain_{p_}"].sum()
        s[f"gross_over_net_{p_}"] = t[f"gain_{p_}"].sum() / t[f"net_{p_}"].sum()
    pd.Series(s).to_csv(TAB / "qa_nlcd_summary.csv")
    print(pd.Series(s).round(3).to_string())


if __name__ == "__main__":
    main()
