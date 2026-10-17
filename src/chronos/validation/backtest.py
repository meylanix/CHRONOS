"""Walk-forward (rolling-origin) validation with leakage protection and honest verdicts.

For every origin o and horizon h a model receives series.truncate(o) only. Prediction
intervals are empirical: for origin o they are built from the model's own past h-step
errors whose outcomes were known by o (origin' + h <= o) — no future residuals are used.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from chronos.forecasting.models import Model
from chronos.timeseries import Panel, TimeSeries
from chronos.validation import metrics as M


@dataclass(frozen=True)
class BacktestConfig:
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    min_history: int = 25            # observations required before the first origin
    origin_step: int = 1             # years between origins
    first_origin: int | None = None
    last_origin: int | None = None
    levels: tuple[float, ...] = (0.5, 0.8, 0.95)
    min_residuals: int = 10          # past errors needed before an empirical interval is issued
    exclude_statuses: tuple[str, ...] = ("filler",)  # never score against filler values


@dataclass
class Record:
    model: str
    horizon: int
    origin: int
    forecast: float
    actual: float
    scale: float | None              # in-sample mean |y_t - y_{t-h}| at the origin (for MASE)
    native_lo: float | None = None
    native_hi: float | None = None
    native_level: float | None = None
    intervals: dict[float, tuple[float, float]] = field(default_factory=dict)


def _scale(history: TimeSeries, horizon: int) -> float | None:
    st = history.step()
    if horizon % st:
        return None
    k = horizon // st
    run = history.tail_regular_run()
    if len(run) <= k:
        return None
    s = float(np.mean(np.abs(run.values[k:] - run.values[:-k])))
    return s if s > 0 else None


def _origins(series: TimeSeries, cfg: BacktestConfig, h: int) -> list[int]:
    years = series.years
    if len(years) <= cfg.min_history:
        return []
    start = int(years[cfg.min_history - 1])
    if cfg.first_origin is not None:
        start = max(start, cfg.first_origin)
    end = series.last_year - h
    if cfg.last_origin is not None:
        end = min(end, cfg.last_origin)
    st = series.step()
    step = max(cfg.origin_step, st)
    cand = [int(y) for y in years if start <= y <= end]
    out, last = [], None
    for y in cand:
        if last is None or y - last >= step:
            out.append(y)
            last = y
    return out


def run_forecasts(series: TimeSeries, models: Sequence[Model], cfg: BacktestConfig,
                  panel: Panel | None = None) -> list[Record]:
    status = dict(zip(series.years.tolist(), series.status, strict=True)) if series.status else {}
    records: list[Record] = []
    for h in cfg.horizons:
        for o in _origins(series, cfg, h):
            actual = series.get(o + h)
            if actual is None or status.get(o + h) in cfg.exclude_statuses:
                continue
            hist = series.truncate(o)
            sc = _scale(hist, h)
            ptrunc = panel.truncate(o) if panel is not None else None
            for m in models:
                f = m.predict(hist, h, ptrunc if m.needs_panel else None)
                if f is None or not math.isfinite(f.value):
                    continue
                records.append(Record(m.name, h, o, f.value, actual, sc, f.lo, f.hi, f.level))
    _attach_empirical_intervals(records, cfg, positive=series.is_positive)
    return records


def _attach_empirical_intervals(records: list[Record], cfg: BacktestConfig, positive: bool) -> None:
    groups: dict[tuple[str, int], list[Record]] = defaultdict(list)
    for r in records:
        groups[(r.model, r.horizon)].append(r)
    for (_, h), recs in groups.items():
        recs.sort(key=lambda r: r.origin)
        use_log = positive and all(r.forecast > 0 for r in recs)
        errs = [math.log(r.actual / r.forecast) if use_log else r.actual - r.forecast for r in recs]
        for i, r in enumerate(recs):
            pool = [errs[j] for j in range(i) if recs[j].origin + h <= r.origin]
            if len(pool) < cfg.min_residuals:
                continue
            for lv in cfg.levels:
                qlo, qhi = np.quantile(pool, [(1 - lv) / 2, 1 - (1 - lv) / 2])
                if use_log:
                    r.intervals[lv] = (r.forecast * math.exp(qlo), r.forecast * math.exp(qhi))
                else:
                    r.intervals[lv] = (r.forecast + qlo, r.forecast + qhi)


def summarize(records: list[Record], models: Sequence[Model], cfg: BacktestConfig) -> dict[str, Any]:
    by: dict[tuple[str, int], list[Record]] = defaultdict(list)
    for r in records:
        by[(r.model, r.horizon)].append(r)
    baselines = [m.name for m in models if m.is_baseline]
    out: dict[str, Any] = {"horizons": {}}
    for h in cfg.horizons:
        present = [m.name for m in models if (m.name, h) in by]
        if not present:
            continue
        # Like-for-like table on the origins common to all models with adequate coverage; a model that
        # can forecast only at few origins (e.g. needs a long history) is reported separately.
        n_own = {n: len({r.origin for r in by[(n, h)]}) for n in present}
        top = max(n_own.values())
        names = [n for n in present if n_own[n] >= 0.6 * top]
        low = [n for n in present if n not in names]
        common = set.intersection(*[{r.origin for r in by[(n, h)]} for n in names])
        hres: dict[str, Any] = {"n_common_origins": len(common), "models": {}, "low_coverage_models": {}}
        for n in names:
            recs = sorted((r for r in by[(n, h)] if r.origin in common), key=lambda r: r.origin)
            hres["models"][n] = _model_metrics(recs, cfg, h)
        for n in low:
            recs = sorted(by[(n, h)], key=lambda r: r.origin)
            hres["low_coverage_models"][n] = {**_model_metrics(recs, cfg, h),
                                              "note": f"forecasts at only {n_own[n]} of {top} origins; not in the main table"}
        hres["comparisons"] = _comparisons(by, present, baselines, h)
        hres["stability"] = _stability(by, names, common, h)
        hres["verdict"] = _verdict(hres, models, baselines)
        out["horizons"][str(h)] = hres
    return out


def _model_metrics(recs: list[Record], cfg: BacktestConfig, h: int) -> dict[str, Any]:
    if not recs:
        return {"n": 0}
    a = np.array([r.actual for r in recs])
    p = np.array([r.forecast for r in recs])
    e = a - p
    scaled = [abs(r.actual - r.forecast) / r.scale for r in recs if r.scale]
    res: dict[str, Any] = {
        "n": len(recs),
        "n_nonoverlapping": M.nonoverlapping_count([r.origin for r in recs], h),
        "first_origin": recs[0].origin, "last_origin": recs[-1].origin,
        "MAE": M.mae(e), "RMSE": M.rmse(e), "MAPE_percent": M.mape(a, p),
        "MASE": float(np.mean(scaled)) if scaled else None,
        "mean_abs_log_ratio": M.mean_abs_log_ratio(a, p),
        "bias_mean_error": float(np.mean(e)),
    }
    cal = {}
    for lv in cfg.levels:
        with_iv = [r for r in recs if lv in r.intervals]
        if not with_iv:
            continue
        lo = np.array([r.intervals[lv][0] for r in with_iv])
        hi = np.array([r.intervals[lv][1] for r in with_iv])
        y = np.array([r.actual for r in with_iv])
        cal[str(lv)] = {"n": len(with_iv), "nominal": lv, "empirical_coverage": float(np.mean((y >= lo) & (y <= hi))),
                        "mean_width": float(np.mean(hi - lo)), "interval_score": M.interval_score(lo, hi, y, 1 - lv)}
    res["empirical_interval_calibration"] = cal
    native = [r for r in recs if r.native_lo is not None and r.native_hi is not None]
    if native:
        y = np.array([r.actual for r in native])
        lo = np.array([r.native_lo for r in native])
        hi = np.array([r.native_hi for r in native])
        res["native_interval"] = {"nominal": native[0].native_level, "n": len(native),
                                  "empirical_coverage": float(np.mean((y >= lo) & (y <= hi))),
                                  "mean_width": float(np.mean(hi - lo))}
    return res


def _comparisons(by: dict[tuple[str, int], list[Record]], names: list[str], baselines: list[str],
                 h: int) -> dict[str, Any]:
    """Each candidate vs each baseline on the origins where both produced a forecast."""
    err = {n: {r.origin: r.actual - r.forecast for r in by[(n, h)]} for n in names}
    out = {}
    for n in names:
        if n in baselines:
            continue
        for b in baselines:
            if b not in err:
                continue
            origins = sorted(set(err[n]) & set(err[b]))
            if not origins:
                continue
            e1 = [err[n][o] for o in origins]
            e2 = [err[b][o] for o in origins]
            dm = M.diebold_mariano(e1, e2, h)
            mae_n, mae_b = M.mae(e1), M.mae(e2)
            out[f"{n}__vs__{b}"] = {
                "n_paired_origins": len(origins),
                "relative_MAE": mae_n / mae_b if mae_b > 0 else None,
                "skill_vs_baseline": 1 - mae_n / mae_b if mae_b > 0 else None,
                "share_of_origins_better": float(np.mean(np.abs(e1) < np.abs(e2))) if origins else None,
                "diebold_mariano": dm,
            }
    return out


def _stability(by: dict[tuple[str, int], list[Record]], names: list[str], common: set[int], h: int,
               blocks: int = 3) -> dict[str, Any]:
    origins = sorted(common)
    if len(origins) < blocks * 5:
        return {"note": "too few origins for sub-period stability analysis"}
    parts = np.array_split(np.array(origins), blocks)
    res: dict[str, Any] = {"blocks": []}
    winners = []
    for part in parts:
        oset = set(part.tolist())
        maes = {n: M.mae([r.actual - r.forecast for r in by[(n, h)] if r.origin in oset]) for n in names}
        best = min(maes, key=lambda k: maes[k])
        winners.append(best)
        res["blocks"].append({"origins": [int(part[0]), int(part[-1])], "MAE": maes, "best": best})
    res["same_best_model_in_all_blocks"] = len(set(winners)) == 1
    return res


def _verdict(hres: dict[str, Any], models: Sequence[Model], baselines: list[str]) -> dict[str, Any]:
    ms = hres["models"]
    ranked = sorted((n for n in ms if ms[n].get("n")), key=lambda n: ms[n]["MAE"])
    best_baseline = next((n for n in ranked if n in baselines), None)
    notes = []
    candidates = {}
    for m in models:
        if m.is_baseline or (m.name not in ms and m.name not in hres.get("low_coverage_models", {})):
            continue
        cmp = hres["comparisons"].get(f"{m.name}__vs__{best_baseline}") if best_baseline else None
        beats = bool(cmp and cmp["relative_MAE"] is not None and cmp["relative_MAE"] < 1)
        p = cmp["diebold_mariano"].get("p_value") if cmp else None
        significant = beats and p is not None and not math.isnan(p) and p < 0.05
        candidates[m.name] = {"beats_best_baseline": beats, "dm_p_value": p, "significant_at_5pct": significant}
        if not beats:
            notes.append(f"{m.name} does NOT beat the best simple baseline ({best_baseline}); prefer the baseline.")
        elif not significant:
            notes.append(f"{m.name} has lower MAE than {best_baseline}, but the difference is not statistically "
                         f"significant (DM p={p:.2f})." if p is not None and not math.isnan(p) else
                         f"{m.name} has lower MAE than {best_baseline}; significance could not be assessed.")
    n_nonover = min((ms[n].get("n_nonoverlapping", 0) for n in ranked), default=0)
    if n_nonover < 10:
        notes.append(f"Only {n_nonover} non-overlapping forecast windows: results are statistically fragile.")
    for n in ranked:
        for lv, c in ms[n].get("empirical_interval_calibration", {}).items():
            if c["n"] >= 20 and abs(c["empirical_coverage"] - c["nominal"]) > 0.15:
                notes.append(f"{n}: {float(lv):.0%} empirical interval covered {c['empirical_coverage']:.0%} — miscalibrated.")
        nat = ms[n].get("native_interval")
        if nat and nat["n"] >= 20 and abs(nat["empirical_coverage"] - (nat["nominal"] or 0)) > 0.15:
            notes.append(f"{n}: native {nat['nominal']:.0%} band covered only {nat['empirical_coverage']:.0%} of outcomes.")
    return {"ranking_by_MAE": ranked, "best_model": ranked[0] if ranked else None, "best_baseline": best_baseline,
            "candidates": candidates, "notes": notes}


def revision_risk(series: TimeSeries, first_origin: int | None) -> dict[str, Any]:
    retrieved = str(series.meta.get("retrieved_utc") or "")
    as_of = series.meta.get("as_of_date")
    real_time = bool(as_of and first_origin is not None and int(str(as_of)[:4]) <= first_origin)
    if series.data_kind == "synthetic":
        return {"risk": "n/a (synthetic data)"}
    if real_time:
        return {"risk": "low", "detail": f"archival vintage as of {as_of} used"}
    return {
        "risk": "PRESENT",
        "detail": (f"Backtest uses the vintage retrieved {retrieved[:10] or 'unknown'}. Historical statistics are "
                   "routinely revised and back-cast; a forecaster at each origin would have seen different numbers. "
                   "No archival (real-time) vintages are stored for this series, so revision leakage cannot be "
                   "excluded; reported skill may be optimistic."),
    }


def walk_forward(series: TimeSeries, models: Sequence[Model], cfg: BacktestConfig | None = None,
                 panel: Panel | None = None) -> dict[str, Any]:
    cfg = cfg or BacktestConfig()
    if series.data_kind == "synthetic":
        banner = "SYNTHETIC DATA — results test the machinery only and say nothing about the real world."
    else:
        banner = None
    records = run_forecasts(series, models, cfg, panel)
    summary = summarize(records, models, cfg)
    first = min((r.origin for r in records), default=None)
    return {
        "series": {"indicator": series.name, "entity": series.entity, "data_kind": series.data_kind,
                   "n_observations": len(series), "first_year": series.first_year if len(series) else None,
                   "last_year": series.last_year if len(series) else None, "unit": series.unit,
                   "provenance": {k: series.meta.get(k) for k in ("source_id", "source_dataset", "source_variable",
                                                                    "vintage_id", "retrieved_utc", "source_version")}},
        "banner": banner,
        "config": asdict(cfg),
        "models": {m.name: {"baseline": m.is_baseline, "description": m.description} for m in models},
        "revision_leakage": revision_risk(series, first),
        "summary": summary,
        "n_records": len(records),
        "records": [{"model": r.model, "h": r.horizon, "origin": r.origin, "forecast": r.forecast, "actual": r.actual,
                     **({f"lo{lv}": iv[0] for lv, iv in r.intervals.items()}),
                     **({f"hi{lv}": iv[1] for lv, iv in r.intervals.items()})} for r in records],
    }


def point_experiment(series: TimeSeries, models: Sequence[Model], origin: int, target: int,
                     panel: Panel | None = None) -> dict[str, Any]:
    """'Use data known by 1950 to forecast 1970': one origin, one target. n=1, not evidence of skill."""
    if target <= origin:
        raise ValueError("target must be after origin")
    hist = series.truncate(origin)
    actual = series.get(target)
    rows = []
    for m in models:
        f = m.predict(hist, target - origin, panel.truncate(origin) if (panel is not None and m.needs_panel) else None)
        if f is None:
            rows.append({"model": m.name, "forecast": None, "note": "not enough data at origin"})
            continue
        row: dict[str, Any] = {"model": m.name, "baseline": m.is_baseline, "forecast": f.value}
        if actual is not None:
            row.update(error=actual - f.value, abs_pct_error=abs(actual - f.value) / abs(actual) * 100 if actual else None)
        if f.details.get("analogs"):
            row["analogs"] = f.details["analogs"]
        rows.append(row)
    return {"origin": origin, "target": target, "actual": actual, "n_history_obs": len(hist),
            "history_last_year": hist.last_year if len(hist) else None, "results": rows,
            "warning": "Single origin (n=1): illustrates the procedure; it is NOT evidence that any model has skill.",
            "revision_leakage": revision_risk(series, origin)}
