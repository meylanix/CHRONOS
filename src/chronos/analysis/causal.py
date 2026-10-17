"""Causal hypothesis engine: separates association, predictive dependence and causal estimates.

The engine runs a ladder of increasingly demanding analyses on a hypothesis "X affects Y":

  T1 association            correlation in levels and in within-unit changes
  T2 predictive dependence  Granger tests X->Y and Y->X (prediction, NOT causation)
  T3 conditional association two-way fixed-effects regression with controls, clustered SEs
  T4 quasi-experimental     event study around discrete treatment dates (stacked design with
                            not-yet/never-treated controls), pre-trend test, permutation placebo

A higher tier is never reported as reached unless its diagnostics pass, and each tier states
what it does NOT establish. Threats to validity are listed explicitly. Results can be attached
to the hypothesis registry (pre-registration + multiple-testing correction).
"""

from __future__ import annotations

import json
import math
import sqlite3
import warnings
import zlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from chronos.errors import DataError
from chronos.storage.db import transaction, utc_now

TIER_TEXT = {
    0: ("not supported by these tests", "No test supported the hypothesis with the expected sign. This is not proof "
                                         "of no effect (limited power, measurement error, specification)."),
    1: ("association", "X and Y co-move. Says nothing about direction, confounding or mechanism."),
    2: ("predictive dependence", "Past X helps predict Y (Granger). A common driver or anticipation can produce this."),
    3: ("conditional association", "Association survives unit and time fixed effects and controls. Unobserved "
                                   "time-varying confounders and reverse causality remain possible."),
    4: ("quasi-experimental estimate", "Effect estimated around discrete treatment dates with parallel pre-trends "
                                        "and a placebo check. Causal only under the design's identifying assumptions "
                                        "(no anticipation, no concurrent shocks, comparable controls)."),
}


# ----------------------------------------------------------------------------- T1


def association(df: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
    d = df[["unit", "year", x, y]].dropna().sort_values(["unit", "year"])
    if len(d) < 20:
        return {"n": len(d), "note": "fewer than 20 joint observations"}
    lv = stats.pearsonr(d[x], d[y])
    sp = stats.spearmanr(d[x], d[y])
    g = d.groupby("unit")
    dx, dy = g[x].diff(), g[y].diff()
    ok = dx.notna() & dy.notna()
    res: dict[str, Any] = {"n": len(d), "n_units": int(d["unit"].nunique()),
                           "pearson_levels": float(lv.statistic), "p_levels": float(lv.pvalue),
                           "spearman_levels": float(sp.statistic)}
    if ok.sum() >= 20:
        dc = stats.pearsonr(dx[ok], dy[ok])
        res.update(pearson_changes=float(dc.statistic), p_changes=float(dc.pvalue), n_changes=int(ok.sum()))
        if abs(res["pearson_levels"]) > 0.5 and abs(res["pearson_changes"]) < 0.5 * abs(res["pearson_levels"]):
            res["warning"] = ("Strong correlation in levels but much weaker in changes: likely shared trends "
                              "(spurious-regression risk).")
    res["caveat"] = "p-values assume independent observations; panel and serial dependence make them too small."
    return res


# ----------------------------------------------------------------------------- T2


def stationarity(x: np.ndarray) -> dict[str, Any]:
    from statsmodels.tsa.stattools import adfuller, kpss

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return {"n": len(x), "note": "too short"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = float(adfuller(x, autolag="AIC")[1])
        kpss_p = float(kpss(x, regression="c", nlags="auto")[1])
    return {"n": len(x), "adf_p": adf_p, "kpss_p": kpss_p,
            "looks_stationary": adf_p < 0.05 and kpss_p > 0.05,
            "note": "ADF H0: unit root; KPSS H0: stationary (KPSS p-values are truncated to [0.01, 0.1])."}


def granger(df_unit: pd.DataFrame, x: str, y: str, maxlag: int = 3) -> dict[str, Any]:
    """Granger test on one unit's (already stationary-transformed) series, lag chosen by AIC."""
    from statsmodels.tsa.api import VAR

    d = df_unit.sort_values("year")[[y, x]].dropna()
    if len(d) < 25 + 3 * maxlag:
        return {"n": len(d), "note": "series too short for Granger test"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = VAR(d.to_numpy())
            sel = model.select_order(maxlag)
            lag = max(1, int(sel.aic))
            fit = model.fit(lag)
            xy = fit.test_causality(0, [1], kind="f")  # x (col 1) -> y (col 0)
            yx = fit.test_causality(1, [0], kind="f")
        except (ValueError, np.linalg.LinAlgError) as exc:
            return {"n": len(d), "note": f"VAR failed: {exc}"}
    return {"n": len(d), "lag": lag, "p_x_to_y": float(xy.pvalue), "p_y_to_x": float(yx.pvalue)}


def panel_granger(df: pd.DataFrame, x: str, y: str, maxlag: int = 3, min_obs: int = 40) -> dict[str, Any]:
    per = []
    for u, g in df.groupby("unit"):
        if g[[x, y]].dropna().shape[0] < min_obs:
            continue
        r = granger(g, x, y, maxlag)
        if "p_x_to_y" in r:
            per.append({"unit": u, **r})
    if not per:
        return {"n_units": 0, "note": f"no unit has >= {min_obs} joint observations"}
    pxy = np.array([r["p_x_to_y"] for r in per])
    pyx = np.array([r["p_y_to_x"] for r in per])
    n = len(per)
    k_xy, k_yx = int((pxy < 0.05).sum()), int((pyx < 0.05).sum())
    return {
        "n_units": n,
        "share_units_x_to_y_p05": k_xy / n, "binomial_p_x_to_y": float(stats.binomtest(k_xy, n, 0.05, alternative="greater").pvalue),
        "share_units_y_to_x_p05": k_yx / n, "binomial_p_y_to_x": float(stats.binomtest(k_yx, n, 0.05, alternative="greater").pvalue),
        "note": "Under no predictive dependence about 5% of units reject by chance. Units are not independent "
                "(common shocks), so the binomial p-value is optimistic.",
        "per_unit": per[:200],
    }


# ----------------------------------------------------------------------------- T3


def _demean_twoway(df: pd.DataFrame, cols: Sequence[str], a: str, b: str, iters: int = 100, tol: float = 1e-10) -> pd.DataFrame:
    """Within transformation for two crossed fixed effects (alternating projections; unbalanced panels OK)."""
    out = df[list(cols)].astype(float).copy()
    for _ in range(iters):
        prev = out.copy()
        out = out - out.groupby(df[a]).transform("mean")
        out = out - out.groupby(df[b]).transform("mean")
        if float(np.nanmax(np.abs((out - prev).to_numpy()))) < tol:
            break
    return out


def fixed_effects(df: pd.DataFrame, y: str, xs: Sequence[str], cluster: str = "unit") -> dict[str, Any]:
    import statsmodels.api as sm

    d = df[["unit", "year", y, *xs]].dropna()
    n_units = d["unit"].nunique()
    if len(d) < 50 or n_units < 5:
        return {"n": len(d), "n_units": int(n_units), "note": "too few observations/units for FE regression"}
    w = _demean_twoway(d, [y, *xs], "unit", "year")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.OLS(w[y].to_numpy(), w[list(xs)].to_numpy()).fit(cov_type="cluster",
                                                                   cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    coefs = {}
    for i, name in enumerate(xs):
        coefs[name] = {"coef": float(fit.params[i]), "se": float(fit.bse[i]), "p": float(fit.pvalues[i]),
                       "ci95": [float(fit.conf_int()[i][0]), float(fit.conf_int()[i][1])]}
    return {"n": len(d), "n_units": int(n_units), "n_years": int(d["year"].nunique()), "coefficients": coefs,
            "few_clusters_warning": n_units < 30}


# ----------------------------------------------------------------------------- T4


def first_crossing_years(df: pd.DataFrame, var: str, threshold: float, sustain: int = 5) -> dict[str, int]:
    """Treatment date = first year var >= threshold after being below it, sustained for `sustain` years.
    Units already above the threshold in their first observed year are excluded (always-treated)."""
    out = {}
    for u, g in df[["unit", "year", var]].dropna().sort_values("year").groupby("unit"):
        v, yrs = g[var].to_numpy(), g["year"].to_numpy()
        if not len(v) or v[0] >= threshold:
            continue
        for i in range(1, len(v)):
            if v[i] >= threshold and v[i - 1] < threshold and np.all(v[i: i + sustain] >= threshold) and len(v[i: i + sustain]) == sustain:
                out[str(u)] = int(yrs[i])
                break
    return out


def event_study(df: pd.DataFrame, y: str, treat_year: dict[str, int], window: tuple[int, int] = (-5, 10),
                n_perm: int = 50, seed: int = 0, max_controls: int = 40) -> dict[str, Any]:
    """Stacked event study (Cengiz et al. 2019 style): one stack per treated unit; controls are units not
    treated within [g+window_lo, g+window_hi]. Coefficients relative to k=-1, two-way FE within stack."""
    lo, hi = window
    base = df[["unit", "year", y]].dropna()
    units = set(base["unit"].unique())
    treated = {u: g for u, g in treat_year.items() if u in units}
    if len(treated) < 5:
        return {"n_treated": len(treated), "note": "fewer than 5 treated units with outcome data"}

    def estimate(assign: dict[str, int]) -> tuple[dict[int, tuple[float, float]], float | None, int, float | None, float | None]:
        import statsmodels.api as sm

        stacks = []
        for s_id, (u, g) in enumerate(sorted(assign.items())):
            rows = base[(base["year"] >= g + lo) & (base["year"] <= g + hi)]
            ctrl_units = sorted(c for c in units if c != u and not (c in assign and g + lo <= assign[c] <= g + hi + 1))
            if len(ctrl_units) > max_controls:  # deterministic subsample keeps the stacked regression tractable
                pick = np.random.default_rng(zlib.crc32(f"{u}:{g}:{seed}".encode())).choice(len(ctrl_units), max_controls, replace=False)
                ctrl_units = [ctrl_units[i] for i in sorted(pick)]
            part = rows[rows["unit"].isin([u, *ctrl_units])].copy()
            if (part["unit"] == u).sum() < (hi - lo) // 2:
                continue
            part["stack"] = s_id
            part["rel"] = np.where(part["unit"] == u, part["year"] - g, np.nan)
            stacks.append(part)
        if len(stacks) < 5:
            return {}, None, 0, None, None
        st = pd.concat(stacks, ignore_index=True)
        ks = [k for k in range(lo, hi + 1) if k != -1]
        for k in ks:
            st[f"k{k}"] = (st["rel"] == k).astype(float)
        st["fe_unit"] = st["stack"].astype(str) + ":" + st["unit"].astype(str)
        st["fe_year"] = st["stack"].astype(str) + ":" + st["year"].astype(str)
        cols = [f"k{k}" for k in ks]
        w = _demean_twoway(st, [y, *cols], "fe_unit", "fe_year", iters=50, tol=1e-8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.OLS(w[y].to_numpy(), w[cols].to_numpy()).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(st["unit"])[0]})
        coefs = {k: (float(fit.params[i]), float(fit.bse[i])) for i, k in enumerate(ks)}
        pre = [i for i, k in enumerate(ks) if k < -1]
        post = [i for i, k in enumerate(ks) if k >= 0]
        pre_p = None
        if pre:
            R = np.zeros((len(pre), len(ks)))
            for r, i in enumerate(pre):
                R[r, i] = 1
            try:
                pre_p = float(fit.f_test(R).pvalue)
            except (ValueError, np.linalg.LinAlgError):
                pre_p = None
        avg_post = float(np.mean([fit.params[i] for i in post]))
        Rp = np.zeros((1, len(ks)))
        Rp[0, post] = 1 / len(post)
        try:
            post_p = float(fit.t_test(Rp).pvalue)
        except (ValueError, np.linalg.LinAlgError):
            post_p = None
        return coefs, pre_p, len(stacks), avg_post, post_p

    coefs, pre_p, n_stacks, avg_post, post_p = estimate(treated)
    if not coefs:
        return {"n_treated": len(treated), "note": "too few usable stacks"}
    rng = np.random.default_rng(seed)
    all_units = sorted(units)
    years = list(treated.values())
    null = []
    for _ in range(n_perm):
        fake_units = rng.choice(all_units, size=len(treated), replace=False)
        fake = {str(u): int(g) for u, g in zip(fake_units, rng.permutation(years), strict=True)}
        _, _, ns, ap, _ = estimate(fake)
        if ns and ap is not None:
            null.append(ap)
    perm_p = float((np.sum(np.abs(null) >= abs(avg_post)) + 1) / (len(null) + 1)) if null and avg_post is not None else None
    return {
        "n_treated_units": n_stacks, "window": list(window),
        "dynamic_effects": {str(k): {"coef": c, "se": s} for k, (c, s) in coefs.items()},
        "average_post_effect": avg_post, "post_effect_p": post_p,
        "pretrend_joint_p": pre_p, "permutation_p": perm_p, "n_permutations": len(null),
        "parallel_pretrends_not_rejected": pre_p is not None and pre_p > 0.10,
    }


# ----------------------------------------------------------------------------- ladder


def assess(df: pd.DataFrame, x: str, y: str, controls: Sequence[str] = (), expected_sign: int | None = None,
           treat_year: dict[str, int] | None = None, window: tuple[int, int] = (-5, 10), maxlag: int = 3,
           n_perm: int = 50, granger_min_obs: int = 40) -> dict[str, Any]:
    """Run the evidence ladder. `df` has columns unit, year, x, y, controls (already transformed)."""
    report: dict[str, Any] = {"hypothesis": f"{x} -> {y}", "controls": list(controls), "expected_sign": expected_sign}
    report["T1_association"] = association(df, x, y)
    report["stationarity"] = {v: _panel_stationarity(df, v) for v in (x, y)}
    report["T2_predictive"] = panel_granger(df, x, y, maxlag, min_obs=granger_min_obs)
    report["T3_fixed_effects"] = {
        "without_controls": fixed_effects(df, y, [x]),
        "with_controls": fixed_effects(df, y, [x, *controls]) if controls else None,
    }
    if treat_year:
        report["T4_event_study"] = event_study(df, y, treat_year, window, n_perm=n_perm)
    tier, reasons = _tier(report, x, expected_sign)
    fe0 = report["T3_fixed_effects"]["without_controls"]
    if report["T1_association"].get("n", 0) < 20 and fe0.get("n", 0) < 50:
        label, text = "insufficient data", "Too few joint observations for any test."
    else:
        label, text = TIER_TEXT[tier]
    report["evidence_tier"] = {"tier": tier, "label": label, "does_not_show": text, "reasons": reasons,
                               "note": "Tier = strongest design whose criteria are met; criteria are reported separately."}
    report["threats_to_validity"] = _threats(report, x, controls)
    return report


def _panel_stationarity(df: pd.DataFrame, v: str) -> dict[str, Any]:
    res = [stationarity(g[v].to_numpy()) for _, g in df.groupby("unit") if g[v].notna().sum() >= 30]
    res = [r for r in res if "adf_p" in r]
    if not res:
        return {"note": "no unit with >= 30 observations"}
    return {"n_units": len(res), "share_adf_rejects_unit_root": float(np.mean([r["adf_p"] < 0.05 for r in res])),
            "share_kpss_not_rejecting_stationarity": float(np.mean([r["kpss_p"] > 0.05 for r in res]))}


def _sign_ok(coef: float, expected: int | None) -> bool:
    return expected is None or (coef * expected > 0)


def _tier(r: dict[str, Any], x: str, expected: int | None) -> tuple[int, list[str]]:
    reasons, tier = [], 0
    a = r["T1_association"]
    if a.get("p_changes") is not None and a["p_changes"] < 0.05 and _sign_ok(a["pearson_changes"], expected):
        tier = 1
        reasons.append(f"T1: within-unit changes correlate (r={a['pearson_changes']:.2f}).")
    else:
        reasons.append("T1 not met: no significant association in changes with the expected sign.")
    g = r["T2_predictive"]
    if g.get("n_units") and g["binomial_p_x_to_y"] < 0.05:
        tier = max(tier, 2)
        reasons.append(f"T2: {g['share_units_x_to_y_p05']:.0%} of units show X->Y Granger dependence.")
        if g["binomial_p_y_to_x"] < 0.05:
            reasons.append("T2 caveat: Y->X dependence is also present (feedback / reverse causality).")
    fe = r["T3_fixed_effects"]["with_controls"] or r["T3_fixed_effects"]["without_controls"]
    if fe and "coefficients" in fe:
        c = fe["coefficients"][x]
        if c["p"] < 0.05 and _sign_ok(c["coef"], expected) and not fe.get("few_clusters_warning"):
            tier = max(tier, 3)
            reasons.append(f"T3: FE coefficient {c['coef']:.4g} (p={c['p']:.3f}, {fe['n_units']} clusters).")
        else:
            reasons.append(f"T3 not met (coef {c['coef']:.4g}, p={c['p']:.3f}"
                           f"{', too few clusters' if fe.get('few_clusters_warning') else ''}).")
    es = r.get("T4_event_study")
    if es and "average_post_effect" in es:
        ok = (es["post_effect_p"] is not None and es["post_effect_p"] < 0.05 and es["parallel_pretrends_not_rejected"]
              and es["permutation_p"] is not None and es["permutation_p"] < 0.05
              and _sign_ok(es["average_post_effect"], expected))
        if ok:
            tier = 4
            reasons.append(f"T4: post-treatment effect {es['average_post_effect']:.4g}, pre-trends p={es['pretrend_joint_p']:.2f}, "
                           f"placebo p={es['permutation_p']:.3f}.")
        else:
            reasons.append(f"T4 not met (post p={es.get('post_effect_p')}, pre-trend p={es.get('pretrend_joint_p')}, "
                           f"placebo p={es.get('permutation_p')}).")
    return tier, reasons


def _threats(r: dict[str, Any], x: str, controls: Sequence[str]) -> list[str]:
    t = ["Omitted time-varying confounders cannot be excluded by observational data alone.",
         "Historical series are revised and partly modelled (measurement error, back-casting); "
         "measurement error in X attenuates estimates."]
    if r["T1_association"].get("warning"):
        t.append(r["T1_association"]["warning"])
    g = r["T2_predictive"]
    if g.get("n_units") and g.get("binomial_p_y_to_x", 1) < 0.05:
        t.append("Reverse dependence (Y predicts X) detected.")
    fe = r["T3_fixed_effects"]["without_controls"]
    if fe.get("few_clusters_warning"):
        t.append("Fewer than 30 clusters: cluster-robust SEs are unreliable.")
    if not controls:
        t.append("No controls specified.")
    es = r.get("T4_event_study")
    if es and es.get("pretrend_joint_p") is not None and es["pretrend_joint_p"] <= 0.10:
        t.append("Pre-treatment trends differ (parallel-trends assumption rejected).")
    t.append("Specification choices (lags, transforms, sample) were made by the analyst; see the hypothesis "
             "registry for the number of tests run on this question.")
    return t


# ----------------------------------------------------------------------------- registry


def register_hypothesis(con: sqlite3.Connection, hyp_id: str, statement: str, spec: dict[str, Any],
                        kind: str = "variable_relation", claim_id: str | None = None) -> None:
    exists = con.execute("SELECT spec_json FROM hypothesis WHERE hyp_id=?", (hyp_id,)).fetchone()
    if exists:
        if json.loads(exists[0]) != spec:
            raise DataError(f"Hypothesis {hyp_id} is already registered with a different specification; "
                            "register a new id instead of changing a pre-registered one.")
        return
    with transaction(con):
        con.execute("INSERT INTO hypothesis(hyp_id,statement,kind,claim_id,spec_json,registered_utc) VALUES (?,?,?,?,?,?)",
                    (hyp_id, statement, kind, claim_id, json.dumps(spec, sort_keys=True), utc_now()))


def record_test(con: sqlite3.Connection, hyp_id: str, method: str, p_value: float | None, estimate: float | None,
                tier: str, result: dict[str, Any], experiment_id: str | None = None) -> int:
    with transaction(con):
        cur = con.execute(
            """INSERT INTO hypothesis_test(hyp_id,method,p_value,estimate,evidence_tier,result_json,experiment_id,run_utc)
               VALUES (?,?,?,?,?,?,?,?)""",
            (hyp_id, method, p_value, estimate, tier, json.dumps(result, default=str), experiment_id, utc_now()))
        con.execute("UPDATE hypothesis SET status='tested' WHERE hyp_id=?", (hyp_id,))
    return int(cur.lastrowid or 0)


def registry(con: sqlite3.Connection) -> list[dict[str, Any]]:
    """All tests with Benjamini–Hochberg adjusted p-values across the whole registry."""
    rows = [dict(r) for r in con.execute(
        """SELECT t.test_id, t.hyp_id, h.statement, h.registered_utc, t.method, t.p_value, t.estimate,
                  t.evidence_tier, t.run_utc FROM hypothesis_test t JOIN hypothesis h USING(hyp_id) ORDER BY t.test_id""")]
    ps = [(i, r["p_value"]) for i, r in enumerate(rows) if r["p_value"] is not None and not math.isnan(r["p_value"])]
    m = len(ps)
    if m:
        order = sorted(ps, key=lambda t: t[1])
        adj = [0.0] * m
        running = 1.0
        for rank in range(m, 0, -1):
            i, p = order[rank - 1]
            running = min(running, p * m / rank)
            adj[rank - 1] = running
        for (i, _), a in zip(order, adj, strict=True):
            rows[i]["p_bh_adjusted"] = a
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["hyp_id"]] = counts.get(r["hyp_id"], 0) + 1
    for r in rows:
        r["n_tests_on_hypothesis"] = counts[r["hyp_id"]]
    return rows
