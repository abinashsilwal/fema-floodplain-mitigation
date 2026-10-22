"""Spatial (Conley) standard errors for the primary county model.

The primary model (10_exposure_analysis.alignment) is a logit with state fixed effects and state-clustered errors,
which allow correlation within states but not between neighbouring counties in different states. Here the
linear probability version of the same model (any HMA flood-mitigation funding, fiscal years 2001-2025; same
covariates and state fixed effects) is estimated with (a) state-clustered errors and (b) Conley (1999) spatial HAC
errors with a Bartlett kernel at several distance cut-offs between county centroids (EPSG:5070).
Output: outputs/tables/exp_spatial_se.csv
"""
import importlib

import geopandas as gpd
import numpy as np
import pandas as pd

from config import PROCESSED, TAB

E = importlib.import_module("10_exposure_analysis")
X = ["x_new_exp", "x_base_exp", "x_loss", "x_dr", "x_pop", "x_svi", "x_pen", "x_rural", "x_coastal"]
CUTOFFS_KM = [50, 100, 200, 400, 800]


def design():
    """Same sample and covariates as the primary model in 10_exposure_analysis.alignment()."""
    c, _ = E.load()
    d = c[c.nfhl_covered.fillna(False)].copy()
    d["x_new_exp"] = np.log1p(d.d_dev_sfha_km2.clip(lower=0))
    d["x_base_exp"] = np.log1p(d.dev_sfha_km2_2000)
    d["x_loss"] = np.log1p(d.claims_paid_1996_2025)
    d["x_dr"] = d.dr_years
    d["x_pop"] = np.log(d.pop2020)
    d["x_svi"] = d.svi_2020
    d["x_rural"] = (~d.metro.fillna(True)).astype(float)
    d["x_coastal"] = d.coastal.astype(float)
    d["x_pen"] = d.resPenetrationRateSfha
    d = d.dropna(subset=X + ["STATEFP"])
    for k in X:
        if k not in ("x_rural", "x_coastal"):
            d[k] = E.z(d[k])
    d["y"] = (d.mit_obl_2001_2025 > 0).astype(float)
    return d


def main():
    d = design()
    g = gpd.read_file(PROCESSED / "counties.gpkg", columns=["GEOID"]).to_crs(5070).set_index("GEOID")
    xy = np.column_stack([g.geometry.centroid.x.reindex(d.index if d.index.name == "GEOID" else d.GEOID),
                          g.geometry.centroid.y.reindex(d.index if d.index.name == "GEOID" else d.GEOID)]) / 1000
    # state fixed effects by within-state demeaning (Frisch-Waugh)
    st = d.STATEFP.to_numpy()
    W = d[["y"] + X] - d.groupby("STATEFP")[["y"] + X].transform("mean")
    y, Xm = W.y.to_numpy(), W[X].to_numpy()
    XtXi = np.linalg.inv(Xm.T @ Xm)
    b = XtXi @ Xm.T @ y
    e = y - Xm @ b
    n, k, G = len(y), len(X), d.STATEFP.nunique()
    dof = (n - 1) / (n - k - G)
    u = Xm * e[:, None]  # scores
    rows = []
    # state-clustered
    meat = np.zeros((k, k))
    for s in np.unique(st):
        sg = u[st == s].sum(axis=0)
        meat += np.outer(sg, sg)
    V = XtXi @ meat @ XtXi * G / (G - 1) * dof
    rows.append(("State-clustered", np.nan, V))
    # Conley spatial HAC, Bartlett kernel
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    for cut in CUTOFFS_KM:
        K = np.clip(1 - D / cut, 0, None)
        V = XtXi @ (u.T @ K @ u) @ XtXi * dof
        rows.append((f"Conley {cut} km", cut, V))
    out = []
    for lab, cut, V in rows:
        se = np.sqrt(np.diag(V))
        for i, t in enumerate(X):
            out.append({"se_type": lab, "cutoff_km": cut, "term": t, "coef_pp": 100 * b[i], "se_pp": 100 * se[i],
                        "lo": 100 * (b[i] - 1.96 * se[i]), "hi": 100 * (b[i] + 1.96 * se[i]), "n": n})
    out = pd.DataFrame(out)
    out.to_csv(TAB / "exp_spatial_se.csv", index=False)
    print(out[out.term.isin(["x_new_exp", "x_base_exp", "x_loss", "x_dr"])].round(2).to_string())


if __name__ == "__main__":
    main()
