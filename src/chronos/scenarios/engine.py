"""Conditional scenario engine (world Kaya identity + TCRE).

For each scenario the engine reports: assumptions (with the historical value of each derived
parameter), mechanisms (linked to the knowledge graph with their evidence levels), projected
values at each horizon (central path and the spread implied by the assumption ranges),
critical dependencies (one-at-a-time sensitivity), wildcards with historical shock sizes,
signposts evaluated against data, and validation status per horizon.

The spread within a scenario comes from sampling the stated parameter ranges. It is NOT a
probability distribution over futures and scenarios receive NO probabilities.
"""

from __future__ import annotations

import math
import sqlite3
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from chronos.errors import ChronosError, DataError
from chronos.storage import repository as repo
from chronos.timeseries import TimeSeries

PARAMS = ("pop_growth", "gdp_pc_growth", "energy_intensity_change", "carbon_intensity_change")
OUTPUTS = ("population", "gdp_per_capita", "gdp", "energy", "co2", "cumulative_co2", "warming")
UNITS = {"population": "people", "gdp_per_capita": "int-$ (2011) per person", "gdp": "int-$ (2011)",
         "energy": "TWh", "co2": "Mt CO2", "cumulative_co2": "Mt CO2", "warming": "degC (GHG-attributed)"}


@dataclass
class Period:
    until: int
    low: float
    central: float
    high: float


@dataclass
class Param:
    name: str
    periods: list[Period]
    basis: str = ""
    derive: str | None = None
    spread: float | None = None
    derived_info: dict[str, Any] | None = None

    def value(self, year: int, u: float | None) -> float:
        p = next((p for p in self.periods if year <= p.until), self.periods[-1])
        return p.central if u is None else _tri_ppf(u, p.low, p.central, p.high)


@dataclass
class Scenario:
    id: str
    name: str
    summary: str
    params: dict[str, Param]
    assumptions: list[str] = field(default_factory=list)
    mechanisms: list[str] = field(default_factory=list)
    mechanism_narrative: str = ""
    critical_dependencies: list[str] = field(default_factory=list)
    wildcards: list[str] = field(default_factory=list)
    signposts: list[dict[str, Any]] = field(default_factory=list)


def _tri_ppf(u: float, a: float, c: float, b: float) -> float:
    """Inverse CDF of the triangular distribution (a=low, c=mode, b=high)."""
    if b <= a:
        return c
    fc = (c - a) / (b - a)
    if u < fc:
        return a + math.sqrt(u * (b - a) * (c - a))
    return b - math.sqrt((1 - u) * (b - a) * (b - c))


def load_spec(path: str | Path) -> tuple[dict[str, Any], list[Scenario]]:
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ChronosError(f"Cannot read scenario file {path}: {exc}") from exc
    model = raw["model"]
    out = []
    for s in raw.get("scenario", []):
        params = {}
        for name in PARAMS:
            if name not in s.get("params", {}):
                raise ChronosError(f"Scenario {s['id']}: missing parameter {name}")
            p = s["params"][name]
            periods = [Period(int(x["until"]), float(x["low"]), float(x["central"]), float(x["high"])) for x in p["periods"]]
            for per in periods:
                if not per.low <= per.central <= per.high:
                    raise ChronosError(f"Scenario {s['id']} {name}: need low <= central <= high")
            params[name] = Param(name, periods, p.get("basis", ""), p.get("derive"), p.get("spread"))
        out.append(Scenario(s["id"], s["name"], s.get("summary", ""), params, s.get("assumptions", []),
                            s.get("mechanisms", []), s.get("mechanism_narrative", ""), s.get("critical_dependencies", []),
                            s.get("wildcards", []), s.get("signposts", [])))
    if not out:
        raise ChronosError("No [[scenario]] entries")
    return model, out


# ----------------------------------------------------------------------------- base state


@dataclass
class Base:
    year: int                       # last year where population, GDP, energy and CO2 are all observed
    values: dict[str, float]
    series: dict[str, TimeSeries]
    notes: list[str]


def load_base(con: sqlite3.Connection, model: dict[str, Any]) -> Base:
    series = {}
    for key, spec in model["base"].items():
        try:
            series[key] = repo.load_series(con, spec["indicator"], spec["entity"], spec.get("source"))
        except DataError as exc:
            raise DataError(f"Scenario base needs {key}: {exc}") from exc
    core = ["population", "gdp", "energy", "co2"]
    common = set.intersection(*[set(series[k].years.tolist()) for k in core])
    if not common:
        raise DataError("No year in which population, GDP, energy and CO2 are all observed")
    y = max(common)
    vals = {k: series[k].at(y) for k in core}
    for k in ("cumulative_co2", "warming"):
        if k in series and series[k].get(y) is not None:
            vals[k] = series[k].at(y)
    notes = [f"Base year {y}: latest year in which population, GDP, energy and CO2 are all observed."]
    for k, s in series.items():
        if s.last_year > y:
            notes.append(f"{k}: observed up to {s.last_year}; years {y + 1}-{s.last_year} are used to check the scenario paths (hindcast).")
    return Base(y, vals, series, notes)


def _cagr(s: TimeSeries, years: int) -> tuple[float, int, int] | None:
    end = s.last_year
    starts = [int(x) for x in s.years if x <= end - years]
    if not starts:
        return None
    y0 = starts[-1]
    return ((s.at(end) / s.at(y0)) ** (1 / (end - y0)) - 1) * 100, y0, end


def derive_param(base: Base, spec: str) -> dict[str, Any]:
    kind, var, n = spec.split(":")
    if kind != "cagr":
        raise ChronosError(f"Unknown derive kind {kind!r}")
    years = int(n)
    S = base.series
    if var in ("population",):
        s = S["population"]
    elif var == "gdp_per_capita":
        common = sorted(set(S["gdp"].years.tolist()) & set(S["population"].years.tolist()))
        s = TimeSeries.from_pairs([(y, S["gdp"].at(y) / S["population"].at(y)) for y in common])
    elif var == "energy_intensity":
        common = sorted(set(S["energy"].years.tolist()) & set(S["gdp"].years.tolist()))
        s = TimeSeries.from_pairs([(y, S["energy"].at(y) / S["gdp"].at(y)) for y in common])
    elif var == "carbon_intensity":
        common = sorted(set(S["co2"].years.tolist()) & set(S["energy"].years.tolist()))
        s = TimeSeries.from_pairs([(y, S["co2"].at(y) / S["energy"].at(y)) for y in common])
    else:
        raise ChronosError(f"Unknown derive variable {var!r}")
    r = _cagr(s, years)
    if r is None:
        raise DataError(f"Not enough data to derive {spec}")
    return {"value": r[0], "from_year": r[1], "to_year": r[2], "variable": var}


def apply_derivations(scen: Scenario, base: Base) -> None:
    for p in scen.params.values():
        if p.derive:
            info = derive_param(base, p.derive)
            first = p.periods[0]
            spread = p.spread if p.spread is not None else (first.high - first.low) / 2
            p.periods[0] = Period(first.until, info["value"] - spread, info["value"], info["value"] + spread)
            p.derived_info = info


# ----------------------------------------------------------------------------- simulation


def simulate(base: Base, scen: Scenario, end_year: int, u: dict[str, float] | None = None,
             tcre: float = 0.45, anchor: bool = True) -> dict[str, np.ndarray]:
    """Annual Kaya path from the base year.

    anchor=True: in years after the base year that are already observed (e.g. population, energy and
    CO2 for 2023-2024), observed values replace simulated ones and the implied intensities carry the
    path forward; unobserved quantities (e.g. GDP after 2022) follow the scenario's assumptions.
    anchor=False: pure assumption path (used for the hindcast check).
    """
    years = np.arange(base.year, end_year + 1)
    n = len(years)
    P, y, e, c = (np.empty(n) for _ in range(4))
    P[0] = base.values["population"]
    y[0] = base.values["gdp"] / P[0]
    e[0] = base.values["energy"] / base.values["gdp"]
    c[0] = base.values["co2"] / base.values["energy"]
    S = base.series

    def obs(key: str, t: int) -> float | None:
        return S[key].get(t) if anchor and key in S else None

    for i in range(1, n):
        t = int(years[i])

        def q(k: str) -> float | None:
            return None if u is None else u[k]

        P[i] = obs("population", t) or P[i - 1] * (1 + scen.params["pop_growth"].value(t, q("pop_growth")) / 100)
        y[i] = y[i - 1] * (1 + scen.params["gdp_pc_growth"].value(t, q("gdp_pc_growth")) / 100)
        e_obs = obs("energy", t)
        e[i] = e_obs / (P[i] * y[i]) if e_obs else e[i - 1] * (1 + scen.params["energy_intensity_change"].value(t, q("energy_intensity_change")) / 100)
        co2_obs = obs("co2", t)
        c[i] = co2_obs / (P[i] * y[i] * e[i]) if co2_obs else c[i - 1] * (1 + scen.params["carbon_intensity_change"].value(t, q("carbon_intensity_change")) / 100)
    gdp = P * y
    energy = gdp * e
    co2 = energy * c
    cum = base.values.get("cumulative_co2", float("nan")) + np.concatenate(([0.0], np.cumsum(co2[1:])))
    # Warming: last observed value, then TCRE x additional cumulative CO2 (non-CO2 forcing held fixed).
    w_series = S.get("warming") if anchor else None
    warming = np.full(n, np.nan)
    w_year: int = base.year
    w_val: float = base.values.get("warming", float("nan"))
    if w_series is not None:
        for i, yr in enumerate(years.tolist()):
            v = w_series.get(yr)
            if v is not None:
                warming[i] = v
                w_year, w_val = yr, float(v)
    k0 = int(np.searchsorted(years, w_year))
    warming[k0:] = w_val + tcre * (cum[k0:] - cum[k0]) / 1e6  # Mt -> per 1000 Gt
    return {"year": years, "population": P, "gdp_per_capita": y, "gdp": gdp, "energy": energy, "co2": co2,
            "cumulative_co2": cum, "warming": warming}


def lhs(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube sample in [0,1)^k."""
    cut = (np.arange(n)[:, None] + rng.random((n, k))) / n
    for j in range(k):
        cut[:, j] = cut[rng.permutation(n), j]
    return cut


@dataclass
class ScenarioRun:
    scenario: Scenario
    central: dict[str, np.ndarray]
    samples: dict[str, np.ndarray]     # output -> (n_samples, n_years)
    years: np.ndarray


def run_scenario(base: Base, scen: Scenario, model: dict[str, Any]) -> ScenarioRun:
    end = int(model["end_year"])
    tc = model.get("tcre", {"low": 0.45, "central": 0.45, "high": 0.45})
    central = simulate(base, scen, end, None, tc["central"])
    rng = np.random.default_rng(int(model.get("seed", 0)))
    n = int(model.get("samples", 300))
    keys = [*PARAMS, "tcre"]
    U = lhs(n, len(keys), rng)
    samples = {k: np.empty((n, len(central["year"]))) for k in OUTPUTS}
    for i in range(n):
        u = dict(zip(keys, U[i], strict=True))
        sim = simulate(base, scen, end, u, _tri_ppf(u["tcre"], tc["low"], tc["central"], tc["high"]))
        for k in OUTPUTS:
            samples[k][i] = sim[k]
    return ScenarioRun(scen, central, samples, central["year"])


# ----------------------------------------------------------------------------- reporting pieces


def horizon_table(run: ScenarioRun, reference_year: int, horizons: list[int]) -> list[dict[str, Any]]:
    rows = []
    for h in horizons:
        yr = reference_year + h
        idx = int(np.searchsorted(run.years, yr))
        if idx >= len(run.years) or run.years[idx] != yr:
            continue
        row: dict[str, Any] = {"horizon": h, "year": yr}
        for k in OUTPUTS:
            col = run.samples[k][:, idx]
            row[k] = {"central": float(run.central[k][idx]), "p05": float(np.quantile(col, 0.05)),
                      "p50": float(np.quantile(col, 0.5)), "p95": float(np.quantile(col, 0.95))}
        rows.append(row)
    return rows


def sensitivity(base: Base, scen: Scenario, model: dict[str, Any], year: int,
                outputs: tuple[str, ...] = ("gdp_per_capita", "co2", "warming")) -> dict[str, list[dict[str, Any]]]:
    """One-at-a-time swing: each parameter at its low vs high range bound, others central."""
    tc = model.get("tcre", {"low": 0.45, "central": 0.45, "high": 0.45})
    end = int(model["end_year"])
    res: dict[str, list[dict[str, Any]]] = {o: [] for o in outputs}
    for k in [*PARAMS, "tcre"]:
        if k == "tcre":
            sim_lo = simulate(base, scen, end, None, tc["low"])
            sim_hi = simulate(base, scen, end, None, tc["high"])
        else:
            sim_lo = _simulate_one(base, scen, end, k, 0.0, tc["central"])
            sim_hi = _simulate_one(base, scen, end, k, 1.0, tc["central"])
        idx = int(np.searchsorted(sim_lo["year"], year))
        if idx >= len(sim_lo["year"]):
            continue
        for o in outputs:
            res[o].append({"parameter": k, "at_low": float(sim_lo[o][idx]), "at_high": float(sim_hi[o][idx]),
                           "swing": float(abs(sim_hi[o][idx] - sim_lo[o][idx]))})
    for o in outputs:
        res[o].sort(key=lambda r: -r["swing"])
    return res


def _simulate_one(base: Base, scen: Scenario, end: int, which: str, u_val: float, tcre: float) -> dict[str, np.ndarray]:
    """Central path with one parameter set to quantile u_val of its range in every period."""
    clone = Scenario(scen.id, scen.name, scen.summary, dict(scen.params))
    p = scen.params[which]
    clone.params[which] = Param(p.name, [Period(x.until, x.low, _tri_ppf(u_val, x.low, x.central, x.high), x.high)
                                         for x in p.periods], p.basis)
    return simulate(base, clone, end, None, tcre)


def hindcast(base: Base, scen: Scenario, model: dict[str, Any]) -> list[dict[str, Any]]:
    """Assumption check: the pure assumption path (no anchoring) vs values observed after the base year."""
    tc = model.get("tcre", {"central": 0.45})
    pure = simulate(base, scen, int(model["end_year"]), None, tc["central"], anchor=False)
    run = ScenarioRun(scen, pure, {}, pure["year"])
    out = []
    mapping = {"population": "population", "energy": "energy", "co2": "co2"}
    for out_key, base_key in mapping.items():
        s = base.series.get(base_key)
        if s is None:
            continue
        for yr in [int(y) for y in s.years if y > base.year]:
            idx = int(np.searchsorted(run.years, yr))
            if idx < len(run.years) and run.years[idx] == yr:
                obs, sim = s.at(yr), float(run.central[out_key][idx])
                out.append({"variable": out_key, "year": yr, "observed": obs, "scenario_central": sim,
                            "deviation_percent": (sim / obs - 1) * 100})
    return out


def evaluate_signposts(con: sqlite3.Connection, signposts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check each rule {indicator, source, entity|kind, metric, start, end, op, threshold|low/high} against data."""
    from chronos.state.civilization import popweighted

    res = []
    for sp in signposts:
        sp = {"label": sp.get("label", sp.get("indicator", "signpost")), **sp}
        row: dict[str, Any] = {"label": sp["label"], "window": [sp["start"], sp["end"]], "rule": _rule_text(sp)}
        try:
            if sp.get("kind") == "popweight":
                s = popweighted(con, sp["indicator"], sp["source"])
            else:
                s = repo.load_series(con, sp["indicator"], sp["entity"], sp.get("source"))
        except (DataError, ChronosError):
            row.update(status="NO DATA", detail="indicator not loaded")
            res.append(row)
            continue
        if s.last_year < sp["end"] or s.get(sp["start"]) is None and sp["metric"] != "level":
            row.update(status="PENDING", detail=f"data available to {s.last_year}; window ends {sp['end']}")
            res.append(row)
            continue
        v_end = s.get(sp["end"])
        if v_end is None:
            row.update(status="PENDING", detail=f"no observation for {sp['end']}")
            res.append(row)
            continue
        if sp["metric"] == "level":
            val = v_end
        elif sp["metric"] == "change":
            val = v_end - s.at(sp["start"])
        else:
            val = ((v_end / s.at(sp["start"])) ** (1 / (sp["end"] - sp["start"])) - 1) * 100
        ok = _check(sp, val)
        row.update(status="CONSISTENT" if ok else "INCONSISTENT", observed=val,
                   detail="observed value meets the scenario's rule" if ok else "observed value contradicts the scenario's rule")
        res.append(row)
    return res


def _rule_text(sp: dict[str, Any]) -> str:
    if sp["op"] == "between":
        return f"{sp['metric']} in [{sp['low']}, {sp['high']}]"
    return f"{sp['metric']} {sp['op']} {sp['threshold']}"


def _check(sp: dict[str, Any], v: float) -> bool:
    op = sp["op"]
    if op == "between":
        return bool(sp["low"] <= v <= sp["high"])
    t = sp["threshold"]
    return bool({">=": v >= t, "<=": v <= t, ">": v > t, "<": v < t}[op])


def historical_shocks(base: Base, top: int = 5, since: int = 1900) -> dict[str, list[dict[str, Any]]]:
    """Largest one-year declines in annual world series since `since` (context for wildcards)."""
    out = {}
    for key in ("co2", "energy", "population"):
        s = base.series.get(key)
        if s is None:
            continue
        run = s.tail_regular_run().between(since, 10_000)
        if len(run) < 10 or not run.is_positive:
            continue
        ch = np.diff(np.log(run.values))
        order = [i for i in np.argsort(ch)[:top] if ch[i] < 0]
        if order:
            out[key] = [{"year": int(run.years[i + 1]), "change_percent": float((math.exp(ch[i]) - 1) * 100)} for i in order]
    return out


def validation_status(evidence: dict[str, Any] | None, horizons: list[int]) -> dict[int, dict[str, Any]]:
    """Map horizons to empirical extrapolation-error evidence (from panel backtests), or UNVALIDATED."""
    out: dict[int, dict[str, Any]] = {}
    for h in horizons:
        e = (evidence or {}).get(str(h))
        if not e or e.get("n_forecasts", 0) < 30:
            out[h] = {"status": "UNVALIDATED",
                      "note": "No adequate historical test exists for this horizon with the loaded data; values are "
                              "arithmetic consequences of assumptions only."}
        else:
            out[h] = {"status": "EMPIRICAL ERROR SCALE AVAILABLE", **e}
    return out
