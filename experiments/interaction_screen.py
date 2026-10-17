"""Which cross-domain interactions have testable support? (input to the Civilization Dynamics Engine design)

For every ordered pair of domain variables (X -> Y) estimate, on a country panel of non-overlapping
5-year periods 1950-2019:

    Y[p+1] = b * X[p] + c * Y[p] + country FE + period FE + e

(Y and X are 5-year average annual changes / growth rates, standardised.) b is a *predictive*
association conditional on persistence, country and period effects — not a causal effect.

Inference: (1) country-clustered SEs; (2) a placebo distribution obtained by permuting X across
countries within each period (keeps common time structure, breaks the country link), 200 draws;
(3) Holm correction across all ordered pairs. A link is reported as SUPPORTED only if the placebo
p-value survives Holm at 5% and the sign is stable across the two halves of the sample.

Usage: python experiments/interaction_screen.py <database> <output.json>
"""

from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chronos.analysis.causal import _demean_twoway
from chronos.analysis.data import VarSpec, merged_panel
from chronos.storage.db import connect
from chronos.validation.inference import holm

VARS = [
    VarSpec("gdp_pc_growth", "gdp_per_capita", "owid", "growth"),
    VarSpec("pop_growth", "population", "owid", "growth"),
    VarSpec("energy_pc_growth", "energy_per_capita", "owid", "growth"),
    VarSpec("co2_pc_growth", "co2_per_capita", "owid", "growth"),
    VarSpec("life_exp_change", "e_pelifeex", "vdem", "diff"),
    VarSpec("democracy_change", "v2x_polyarchy", "vdem", "diff"),
    VarSpec("education_change", "e_peaveduc", "vdem", "diff"),
    VarSpec("urbanisation_change", "e_miurbani", "vdem", "diff"),
]
LEVELS = [VarSpec("log_gdp_pc", "gdp_per_capita", "owid", "log")]


def period_panel(con: Any) -> pd.DataFrame:
    df = merged_panel(con, VARS + LEVELS, min_year=1950, max_year=2019)
    df["period"] = (df["year"] // 5) * 5
    cols = [v.name for v in VARS + LEVELS]
    g = df.groupby(["unit", "period"])[cols]
    agg = g.mean()
    n = g.count()
    agg = agg.where(n >= 3)  # need at least 3 of 5 years
    return agg.reset_index()


def estimate(d: pd.DataFrame, x: str, y: str) -> tuple[float, float, int, int] | None:
    import statsmodels.api as sm

    s = d.sort_values(["unit", "period"]).copy()
    s["y_next"] = s.groupby("unit")[y].shift(-1)
    s["per_next"] = s.groupby("unit")["period"].shift(-1)
    s = s[(s["per_next"] - s["period"]) == 5]
    s = s[["unit", "period", "y_next", x, y]].dropna()
    if x == y:
        return None
    if len(s) < 200 or s["unit"].nunique() < 30:
        return None
    for c in ("y_next", x, y):
        s[c] = (s[c] - s[c].mean()) / s[c].std()
    w = _demean_twoway(s, ["y_next", x, y], "unit", "period", iters=200, tol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.OLS(w["y_next"].to_numpy(), w[[x, y]].to_numpy()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(s["unit"])[0]})
    return float(fit.params[0]), float(fit.tvalues[0]), len(s), int(s["unit"].nunique())


def placebo(d: pd.DataFrame, x: str, y: str, n_perm: int, rng: np.random.Generator) -> list[float]:
    out = []
    for _ in range(n_perm):
        p = d.copy()
        p[x] = p.groupby("period")[x].transform(lambda v: v.sample(frac=1.0, random_state=int(rng.integers(1e9))).to_numpy())
        r = estimate(p, x, y)
        if r:
            out.append(r[1])
    return out


def main(db: str, out: str, n_perm: int = 200) -> None:
    con = connect(db)
    d = period_panel(con)
    rng = np.random.default_rng(20261017)
    names = [v.name for v in VARS]
    rows = []
    for x, y in itertools.permutations(names, 2):
        r = estimate(d, x, y)
        if r is None:
            rows.append({"x": x, "y": y, "status": "insufficient data"})
            continue
        b, t, n, k = r
        null = np.array(placebo(d, x, y, n_perm, rng))
        p_plac = float((np.sum(np.abs(null) >= abs(t)) + 1) / (len(null) + 1))
        early = estimate(d[d["period"] < 1985], x, y)
        late = estimate(d[d["period"] >= 1985], x, y)
        rows.append({"x": x, "y": y, "beta_std": b, "t_cluster": t, "n_obs": n, "n_countries": k,
                     "placebo_p": p_plac, "placebo_t_sd": float(null.std()) if len(null) else None,
                     "beta_1950_1984": early[0] if early else None, "beta_1985_2019": late[0] if late else None})
        print(f"{x:>20} -> {y:<20} b={b:+.3f} t={t:+.2f} placebo p={p_plac:.3f} n={n}", flush=True)
    tested = [r for r in rows if "placebo_p" in r]
    for r, a in zip(tested, holm([r["placebo_p"] for r in tested]), strict=True):
        r["placebo_p_holm"] = a
        stable = (r["beta_1950_1984"] is not None and r["beta_1985_2019"] is not None
                  and np.sign(r["beta_1950_1984"]) == np.sign(r["beta_1985_2019"]))
        r["sign_stable_across_halves"] = bool(stable)
        r["status"] = "SUPPORTED (predictive)" if (a is not None and a < 0.05 and stable) else \
                      "suggestive" if r["placebo_p"] < 0.05 else "not supported"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"design": __doc__, "n_perm": n_perm, "rows": rows}, indent=1, default=str))
    print({s: sum(r["status"] == s for r in rows) for s in ("SUPPORTED (predictive)", "suggestive", "not supported", "insufficient data")})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200)
