"""Sensitivity of results to the treatment of net decreases in developed A-zone land.

The main models use log(1 + x) with net decreases set to 0. Here net change enters as asinh(x), which keeps
decreases negative, in (1) the five-year county panel and (2) the primary county logit model.
Output: outputs/tables/exp_signed_transform.csv
"""
import importlib

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import TAB

E = importlib.import_module("10_exposure_analysis")
S = importlib.import_module("19_spatial_se")


def main():
    c, ey = E.load()
    rows = []
    for lab, tf in (("log(1 + x), decreases set to 0", None), ("asinh(x), decreases kept", np.arcsinh)):
        t = E.period_panel(c, ey, tf=tf, out_name="_tmp_panel.csv")
        r = t[(t.y == "any_obl") & (t.spec == "lagged") & (t.term == "l_exp")].iloc[0]
        rows.append({"model": "Five-year panel, any funding", "transform": lab, "coef": r.coef, "se": r.se, "p": r.p,
                     "n": r.nobs})
    (TAB / "_tmp_panel.csv").unlink()
    d = S.design()
    raw = E.load()[0].loc[d.index, "d_dev_sfha_km2"]
    for lab, x in (("log(1 + x), decreases set to 0", d.x_new_exp), ("asinh(x), decreases kept", E.z(np.arcsinh(raw)))):
        dd = d.assign(x_new_exp=x)
        fe = pd.get_dummies(dd.STATEFP, prefix="st", drop_first=True, dtype=float)
        y = dd.y.astype(int)
        keep = fe.columns[(fe.T @ y > 0).values & (fe.T @ (1 - y) > 0).values]
        Xs = sm.add_constant(pd.concat([dd[S.X], fe[keep]], axis=1))
        m = sm.Logit(y, Xs).fit(disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": dd.STATEFP.astype(int)})
        me = m.get_margeff(at="overall")
        i = list(m.params.index[1:]).index("x_new_exp")
        rows.append({"model": "County logit, any funding (AME per SD)", "transform": lab, "coef": 100 * me.margeff[i],
                     "se": 100 * me.margeff_se[i], "p": me.pvalues[i], "n": len(dd)})
    out = pd.DataFrame(rows)
    out["lo"], out["hi"] = out.coef - 1.96 * out.se, out.coef + 1.96 * out.se
    out.to_csv(TAB / "exp_signed_transform.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
