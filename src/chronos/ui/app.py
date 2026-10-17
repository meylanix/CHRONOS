"""CHRONOS research interface (Streamlit).

Run:  chronos ui            (or: streamlit run src/chronos/ui/app.py -- --db chronos.sqlite)
Every view shows where its numbers come from; synthetic data is labelled on every chart.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chronos.config import load_settings
from chronos.errors import ChronosError
from chronos.storage import repository as repo
from chronos.storage.db import connect
from chronos.ui.style import (
    CATEGORICAL,
    FRESHNESS,
    INK,
    INK_SECONDARY,
    MUTED,
    ColorKeeper,
    evidence_color,
    layout,
    rgba,
)

ROOT = Path(__file__).resolve().parents[3]


def _db_from_argv() -> str:
    argv = sys.argv[1:]
    if "--db" in argv:
        return argv[argv.index("--db") + 1]
    return os.environ.get("CHRONOS_DB") or load_settings().db_path


st.set_page_config(page_title="CHRONOS", page_icon="⏳", layout="wide")
if "db" not in st.session_state:
    st.session_state.db = _db_from_argv()
if "colors" not in st.session_state:
    st.session_state.colors = {}
color = ColorKeeper(st.session_state.colors)


def con():  # fresh connection per call: Streamlit reruns scripts in worker threads
    return connect(st.session_state.db)


def db_version() -> float:
    p = Path(st.session_state.db)
    return p.stat().st_mtime if p.exists() else 0.0


def chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def md_safe(text: Any) -> str:
    """Escape '$' so Streamlit does not render units such as 'international-$' as LaTeX."""
    return str(text).replace("$", "\\$")


def badge(status: str) -> str:
    icon, _ = FRESHNESS.get(status, ("•", MUTED))
    return f"{icon} {status}"


# ----------------------------------------------------------------------------- cached computations


@st.cache_data(show_spinner=False)
def catalog(db: str, v: float) -> pd.DataFrame:
    with_db = connect(db)
    df = pd.DataFrame(repo.catalog(with_db))
    with_db.close()
    return df


@st.cache_data(show_spinner="Running walk-forward backtest…")
def run_backtest(db: str, v: float, indicator: str, entity: str, source: str, horizons: tuple[int, ...],
                 models: tuple[str, ...]) -> dict[str, Any]:
    from chronos.forecasting.models import build_models
    from chronos.validation.backtest import BacktestConfig, walk_forward

    c = connect(db)
    s = repo.load_series(c, indicator, entity, source)
    c.close()
    return walk_forward(s, build_models(list(models)), BacktestConfig(horizons=horizons))


@st.cache_data(show_spinner="Building the civilization state model…")
def run_state(db: str, v: float, year: int) -> dict[str, Any]:
    from chronos.state.civilization import build_state, load_catalog

    c = connect(db)
    out = build_state(c, load_catalog(ROOT / "config" / "indicators.toml"), year)
    c.close()
    return out


@st.cache_data(show_spinner="Running scenarios (and historical extrapolation backtests)…")
def run_scenarios(db: str, v: float, evidence: bool) -> dict[str, Any]:
    from chronos.scenarios.report import build_report

    c = connect(db)
    out = build_report(c, str(ROOT / "scenarios" / "world_2026.toml"), with_evidence=evidence)
    c.close()
    return json.loads(json.dumps(out, default=str))


@st.cache_data(show_spinner="Assembling the research dossier…")
def run_dossier(db: str, v: float, year: int, with_scen: bool) -> tuple[dict[str, Any], str]:
    from chronos.research.transitions import build_dossier, dossier_markdown

    c = connect(db)
    scen = run_scenarios(db, v, True) if with_scen else None
    d = build_dossier(c, year, scen)
    c.close()
    return json.loads(json.dumps(d, default=str)), dossier_markdown(d)


@st.cache_data(show_spinner="Building state vectors for the analog engine…")
def analog_table(db: str, v: float, min_year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    from chronos.analysis.analogs import build_state_table

    c = connect(db)
    t, info = build_state_table(c, min_year=min_year)
    c.close()
    return t, info


@st.cache_data(show_spinner="Backtesting the analog engine…")
def analog_backtest(db: str, v: float, horizon: int) -> dict[str, Any]:
    from chronos.analysis.analogs import AnalogEngine, backtest

    t, _ = analog_table(db, v, 1820)
    return json.loads(json.dumps(backtest(AnalogEngine(t), horizon, range(1960, 2011, 10)), default=str))


# ----------------------------------------------------------------------------- pages


def page_overview() -> None:
    from chronos.ingestion.registry import SOURCES

    st.title("CHRONOS — Historical Intelligence & Civilization Forecasting Engine")
    st.caption("A research instrument, not an oracle: every number is traceable to a source vintage, every forecast "
               "is compared with simple baselines on data it could not see, and every causal statement carries an "
               "explicit evidence level.")
    c = con()
    n = {k: c.execute(q).fetchone()[0] for k, q in {
        "real series": "SELECT COUNT(*) FROM series WHERE data_kind='real'",
        "synthetic series": "SELECT COUNT(*) FROM series WHERE data_kind='synthetic'",
        "observations": "SELECT COUNT(*) FROM observation",
        "vintages": "SELECT COUNT(*) FROM vintage",
        "events": "SELECT COUNT(*) FROM event",
        "claims": "SELECT COUNT(*) FROM claim",
        "experiments": "SELECT COUNT(*) FROM experiment"}.items()}
    cols = st.columns(len(n))
    for col, (k, v) in zip(cols, n.items(), strict=True):
        col.metric(k, f"{v:,}")
    if n["synthetic series"]:
        st.warning("This database contains SYNTHETIC series (demo/test data). They are labelled in every view and are "
                   "never mixed with real series.")
    st.subheader("Sources and connector status")
    stored = dict(c.execute("SELECT source_id, COUNT(*) FROM series GROUP BY source_id").fetchall())
    st.dataframe(pd.DataFrame([{"source": e.source_id, "status": e.status, "series stored": stored.get(e.source_id, 0),
                                "license": e.license, "command": e.command, "notes": e.notes} for e in SOURCES.values()]),
                 use_container_width=True, hide_index=True)
    st.caption("LIVE_VERIFIED = run end-to-end against the real source during development. IMPLEMENTED_UNVERIFIED = written "
               "from official documentation; the host was blocked in the development sandbox. FILE_IMPORT = parser for an "
               "official file you download. NOT_IMPLEMENTED = catalogued only.")
    st.subheader("Latest vintages")
    st.dataframe(pd.read_sql_query("SELECT vintage_id, source_id, source_dataset, retrieved_utc, source_version, "
                                   "substr(content_sha256,1,16) AS sha256, location FROM vintage ORDER BY vintage_id DESC LIMIT 20", c),
                 use_container_width=True, hide_index=True)
    c.close()


def _series_picker(key: str, df: pd.DataFrame, multi: bool = True) -> tuple[str, str, list[str]] | None:
    if df.empty:
        st.info("No data stored yet. Run `chronos fetch owid-github --dataset co2` (see README).")
        return None
    opts = sorted({f"{r.indicator} · {r.source_id}" for r in df.itertuples()})
    default = next((i for i, o in enumerate(opts) if o.startswith("gdp_per_capita · owid")), 0)
    choice = st.selectbox("Indicator · source", opts, index=default, key=f"{key}_ind")
    ind, src = choice.split(" · ")
    ents = sorted(df[(df.indicator == ind) & (df.source_id == src)].entity.unique())
    pref = [e for e in ("World", "United States", "China", "India") if e in ents] or ents[:1]
    if multi:
        sel = st.multiselect("Entities (max 8)", ents, default=pref[:3], key=f"{key}_ent", max_selections=8)
    else:
        single = [e for e in ("United States", "United Kingdom", "World") if e in ents] or ents[:1]
        sel = [st.selectbox("Entity", ents, index=ents.index(single[0]) if single else 0, key=f"{key}_ent1")]
    return ind, src, sel


def page_explorer() -> None:
    st.header("Historical time series")
    df = catalog(st.session_state.db, db_version())
    picked = _series_picker("ex", df)
    if not picked:
        return
    ind, src, ents = picked
    c = con()
    fig = go.Figure()
    prov: list[dict[str, Any]] = []
    uniform_status: list[str] = []
    for e in ents:
        s = repo.load_series(c, ind, e, src)
        col = color(e)
        st_arr = np.array(s.status or ["observed"] * len(s))
        fig.add_trace(go.Scatter(x=s.years, y=s.values, mode="lines", name=e, line={"color": col, "width": 2},
                                 hovertemplate=f"{e} %{{x}}: %{{y:,.4g}}<extra></extra>"))
        flagged = st_arr != "observed"
        if flagged.all():
            uniform_status.append(f"{e}: all values '{st_arr[0]}'")
        elif flagged.any():
            fig.add_trace(go.Scatter(x=s.years[flagged], y=s.values[flagged], mode="markers", name=f"{e}: not 'observed'",
                                     marker={"symbol": "circle-open", "size": 8, "color": col},
                                     text=st_arr[flagged], hovertemplate="%{x}: %{y:,.4g} (%{text})<extra></extra>",
                                     showlegend=True))
        prov.append({"entity": e, "data_kind": s.data_kind, "unit": s.unit, "first": s.first_year, "last": s.last_year,
                     "n": len(s), **{k: s.meta.get(k) for k in ("source_name", "source_dataset", "source_variable",
                                                                "vintage_id", "retrieved_utc", "source_version", "source_license")},
                     "statuses": dict(zip(*np.unique(st_arr, return_counts=True), strict=True))})
    kinds = {p["data_kind"] for p in prov}
    title = f"{ind} ({src})" + ("  — SYNTHETIC DATA" if "synthetic" in kinds else "")
    log = st.toggle("Log scale", value=ind in ("gdp", "gdp_per_capita", "population", "co2"))
    layout(fig, title, 460, y_title=prov[0]["unit"] if prov else None)
    if log:
        fig.update_yaxes(type="log")
    chart(fig)
    st.caption("Open circles mark individual values whose status is not 'observed' (imputed, derived, estimated, projected "
               "or filler)." + (" " + "; ".join(uniform_status) + "." if uniform_status else ""))
    st.subheader("Provenance")
    st.dataframe(pd.DataFrame(prov).astype(str), use_container_width=True, hide_index=True)
    with st.expander("Revisions between stored vintages"):
        for e in ents:
            rec = repo.find_series(c, ind, e, src)
            revs = repo.series_revisions(c, rec.series_id)
            st.write(f"**{e}**: " + (json.dumps(revs) if revs else "one stored value set — no revisions observed yet"))
    c.close()


def page_knowledge() -> None:
    from chronos.knowledge import store

    st.header("Historical knowledge graph")
    st.caption("Event A → claimed mechanism → Event B. Evidence levels (0–5) are curator assessments; a sequence of events "
               "is never treated as proof of causation.")
    c = con()
    events = store.list_events(c)
    claims = store.list_claims(c)
    if not events:
        st.info("No events. Run `chronos kg import data/curated/tech_transitions.json`.")
        c.close()
        return
    lanes = [("transition:printing", "Printing"), ("transition:electrification", "Electrification"),
             ("transition:internet", "Computers / Internet"), ("transition:genai", "Generative AI")]
    other = [e for e in events if not any(t in e["tags"] for t, _ in lanes)]
    if other:
        lanes.append(("__other__", "Other events"))
    ev_by_id = {e["event_id"]: e for e in events}
    cols = st.columns(2)
    for i, (tag, label) in enumerate(lanes):
        evs = other if tag == "__other__" else [e for e in events if tag in e["tags"]]
        if not evs:
            continue
        fig = go.Figure()
        ys = {e["event_id"]: k for k, e in enumerate(sorted(evs, key=lambda e: e["start_year"]))}
        for e in evs:
            y = ys[e["event_id"]]
            end = e["end_year"] or e["start_year"]
            if e["end_year"]:
                fig.add_trace(go.Scatter(x=[e["start_year"], e["end_year"]], y=[y, y], mode="lines",
                                         line={"color": rgba(MUTED, 0.45), "width": 4}, hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=[e["start_year"]], y=[y], mode="markers", marker={"size": 9, "color": INK},
                                     error_x={"type": "constant", "value": e["year_uncertainty"], "color": MUTED} if e["year_uncertainty"] else None,
                                     hovertext=f"{e['title']} ({e['start_year']}{'–' + str(e['end_year']) if e['end_year'] else ''}); "
                                               f"refs: {len(e['references'])}", hoverinfo="text", showlegend=False))
            fig.add_trace(go.Scatter(x=[end], y=[y], mode="text", text=[" " + e["title"][:52]], textposition="middle right",
                                     textfont={"color": INK_SECONDARY, "size": 12}, hoverinfo="skip", showlegend=False))
        for cl in claims:
            if cl["source_event_id"] in ys and cl["target_event_id"] in ys:
                a, b = ev_by_id[cl["source_event_id"]], ev_by_id[cl["target_event_id"]]
                fig.add_annotation(x=b["start_year"], y=ys[b["event_id"]], ax=a["start_year"], ay=ys[a["event_id"]],
                                   xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=2,
                                   arrowcolor=evidence_color(cl["evidence_level"]), opacity=0.9,
                                   hovertext=f"{cl['relation']} (L{cl['evidence_level']}): {cl['statement']}")
        lo, hi = min(e["start_year"] for e in evs), max((e["end_year"] or e["start_year"]) for e in evs)
        span = [lo - 3 - 0.03 * (hi - lo), hi + 3 + 0.9 * max(hi - lo, 10)]
        layout(fig, label, 90 + 55 * len(evs))
        fig.update_xaxes(range=span)
        fig.update_yaxes(visible=False)
        fig.update_layout(hovermode="closest")
        with cols[i % 2]:
            chart(fig)
    st.caption("Arrows: claims within a lane, darker blue = higher evidence level. Grey bars: event duration; whiskers: dating uncertainty.")
    st.subheader("Claims")
    rel = st.multiselect("Relation", sorted({c_["relation"] for c_ in claims}), default=None)
    rows = [{"claim": cl["claim_id"], "from": cl["source_event_id"], "to": cl["target_event_id"], "relation": cl["relation"],
             "mechanism": cl["mechanism_id"], "level": cl["evidence_level"], "evidence": cl["evidence_label"],
             "status": cl["status"], "refs": len(cl["references"]), "alternatives": len(cl["alternatives"]),
             "statement": cl["statement"]} for cl in claims if not rel or cl["relation"] in rel]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    pick = st.selectbox("Claim detail", [r["claim"] for r in rows]) if rows else None
    if pick:
        d = store.get_claim(c, pick)
        st.markdown(f"**{d['statement']}**  \nrelation: `{d['relation']}` · evidence level **{d['evidence_level']}** "
                    f"({d['evidence_label']}) · status **{d['status']}** · curator: {d['curator'] or '—'}")
        st.dataframe(pd.DataFrame(d["references"]), use_container_width=True, hide_index=True)
        for a in d["alternatives"]:
            st.markdown(f"- alternative explanation: {a['description']}" + (f" _(ref {a['ref_id']})_" if a["ref_id"] else ""))
    with st.expander("Audit: weak spots the rules do not forbid"):
        st.dataframe(pd.DataFrame(store.audit(c)), use_container_width=True, hide_index=True)
    with st.expander("Documented facts"):
        st.dataframe(pd.DataFrame(store.list_facts(c))[["statement", "value", "unit", "year", "ref_id", "verification", "uncertainty"]],
                     use_container_width=True, hide_index=True)
    with st.expander("References"):
        st.dataframe(pd.DataFrame(store.list_references(c)), use_container_width=True, hide_index=True)
    c.close()


def page_analogs() -> None:
    from chronos.analysis.analogs import AnalogEngine, resolve_code

    st.header("Multidimensional historical analogs")
    st.caption("Analogs are matched on several groups at once (economy, demography, energy, institutions, position relative "
               "to the frontier). A match in one group only is flagged as partial.")
    try:
        table, info = analog_table(st.session_state.db, db_version(), 1820)
    except ChronosError as exc:
        st.info(f"{exc}")
        return
    names = sorted(set(table["name"]))
    c1, c2, c3, c4 = st.columns(4)
    ent = c1.selectbox("Entity", names, index=names.index("China") if "China" in names else 0)
    year = c2.number_input("Year", 1850, 2025, 2022)
    k = c3.slider("Analogs", 3, 20, 10)
    h = c4.selectbox("Show what followed (years)", [0, 5, 10, 20], index=2)
    eng = AnalogEngine(table)
    try:
        res = eng.find(resolve_code(table, ent), int(year), k=k)
    except ChronosError as exc:
        st.warning(str(exc))
        return
    outs = eng.outcomes(res["analogs"], "log_gdp_pc", h) if h else [None] * len(res["analogs"])
    rows = []
    for a, o in zip(res["analogs"], outs, strict=True):
        rows.append({"analog": f"{a['name']} {a['year']}", "distance": round(a["distance"], 3), "coverage": round(a["coverage"], 2),
                     **{f"Δ {g}": (round(v, 2) if v is not None else None) for g, v in a["group_distance"].items()},
                     "comparability": a["comparability"],
                     **({f"GDP/person next {h}y %": round((np.exp(o) - 1) * 100, 1) if o is not None else None} if h else {})})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(res["method"] + " Δ group = RMS standardised difference within the group (SD units). "
               "Outcomes after the target year are shown for context; they are not a forecast unless the backtest below supports it.")
    groups = [g for g in rows[0] if g.startswith("Δ ")] if rows else []
    if rows:
        z = [[r[g] if r[g] is not None else np.nan for g in groups] for r in rows]
        fig = go.Figure(go.Heatmap(z=z, x=[g[2:] for g in groups], y=[r["analog"] for r in rows],
                                   colorscale=[[0, "#cde2fb"], [1, "#104281"]], zmin=0, zmax=2,
                                   hovertemplate="%{y} · %{x}: %{z:.2f} SD<extra></extra>", colorbar={"title": "SD"}))
        layout(fig, "Difference from target by dimension group (blank = no data)", 120 + 28 * len(rows))
        fig.update_yaxes(autorange="reversed", gridcolor=None)
        fig.update_layout(hovermode="closest")
        chart(fig)
    if st.button("Backtest: do analogs predict the next 10 years of GDP per person better than baselines?"):
        bt = analog_backtest(st.session_state.db, db_version(), 10)
        st.markdown(f"**{bt.get('verdict')}**")
        st.dataframe(pd.DataFrame([{"method": m, **v} for m, v in bt.get("methods", {}).items()]), hide_index=True)
        st.dataframe(pd.DataFrame([{"vs": b, **v} for b, v in bt.get("analog_vs", {}).items()]), hide_index=True)
        for n in bt.get("notes", []):
            st.caption(n)


def page_backtest() -> None:
    from chronos.forecasting.models import MODEL_FACTORY
    from chronos.validation.backtest import point_experiment

    st.header("Scientific validation: walk-forward backtests")
    st.caption("At every origin each model sees only data up to that year. Prediction intervals are built from the model's "
               "own past errors whose outcomes were known by the origin. Simple baselines are always included.")
    df = catalog(st.session_state.db, db_version())
    picked = _series_picker("bt", df, multi=False)
    if not picked:
        return
    ind, src, ents = picked
    c1, c2 = st.columns([1, 2])
    horizons = c1.multiselect("Horizons (years)", [1, 5, 10, 20, 30], default=[1, 5, 10, 20])
    default_models = ["naive", "drift", "log_drift", "v01_log_trend", "mean", "damped_trend", "analog_knn", "analog_knn_sep10"]
    models = c2.multiselect("Models", sorted(MODEL_FACTORY), default=default_models)
    if not horizons or not models:
        return
    res = run_backtest(st.session_state.db, db_version(), ind, ents[0], src, tuple(sorted(horizons)), tuple(models))
    if res.get("banner"):
        st.error(res["banner"])
    rl = res["revision_leakage"]
    (st.warning if rl["risk"] == "PRESENT" else st.info)(f"Revision leakage: {rl['risk']}. {rl.get('detail', '')}")
    hs = list(res["summary"]["horizons"])
    if not hs:
        info = res["series"]
        st.warning(f"Not enough contiguous annual history for these horizons: {info['n_observations']} observations "
                   f"{info['first_year']}–{info['last_year']} (benchmark-year series such as World GDP before 1950 cannot be "
                   "walk-forward tested). Choose an entity with a long annual series.")
        return
    h = st.radio("Horizon", hs, horizontal=True, index=min(2, len(hs) - 1))
    hr = res["summary"]["horizons"][h]
    v = hr["verdict"]
    st.markdown(f"**Best model: `{v['best_model']}`** · best baseline: `{v['best_baseline']}` · {hr['n_common_origins']} common origins")
    for n in v["notes"]:
        st.markdown(f"- {n}")
    rows = []
    for name, m in hr["models"].items():
        cal = m.get("empirical_interval_calibration", {})
        rows.append({"model": name, "baseline": res["models"][name]["baseline"], "MAE": m.get("MAE"), "RMSE": m.get("RMSE"),
                     "MASE": m.get("MASE"), "MAPE %": m.get("MAPE_percent"), "|log ratio|": m.get("mean_abs_log_ratio"),
                     "bias": m.get("bias_mean_error"), "non-overlapping windows": m.get("n_nonoverlapping"),
                     "cov50": cal.get("0.5", {}).get("empirical_coverage"), "cov80": cal.get("0.8", {}).get("empirical_coverage"),
                     "cov95": cal.get("0.95", {}).get("empirical_coverage")})
    mt = pd.DataFrame(rows).sort_values("MAE")
    st.dataframe(mt, use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Bar(x=mt["MAE"], y=mt["model"], orientation="h",
                               marker={"color": [CATEGORICAL[1] if b else CATEGORICAL[0] for b in mt["baseline"]],
                                       "cornerradius": 4},
                               hovertemplate="%{y}: MAE %{x:,.4g}<extra></extra>"))
        layout(fig, f"MAE at {h}-year horizon (orange = baseline, blue = candidate)", 360)
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(hovermode="closest")
        chart(fig)
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0.4, 1.0], y=[0.4, 1.0], mode="lines", line={"color": MUTED, "dash": "dot", "width": 1},
                                 name="perfect calibration", hoverinfo="skip"))
        for name, m in hr["models"].items():
            cal = m.get("empirical_interval_calibration", {})
            if cal:
                fig.add_trace(go.Scatter(x=[float(k) for k in cal], y=[c_["empirical_coverage"] for c_ in cal.values()],
                                         mode="lines+markers", name=name, line={"width": 2, "color": color("model:" + name)},
                                         marker={"size": 8}))
        layout(fig, "Interval calibration: nominal vs empirical coverage", 360, "empirical", "nominal")
        chart(fig)
    model = st.selectbox("Forecast vs actual for model", list(hr["models"]))
    recs = pd.DataFrame([r for r in run_backtest(st.session_state.db, db_version(), ind, ents[0], src,
                                                 tuple(sorted(horizons)), tuple(models))["records"]
                         if r["model"] == model and str(r["h"]) == h])
    if not recs.empty:
        recs["target"] = recs["origin"] + int(h)
        fig = go.Figure()
        if "lo0.8" in recs:
            iv = recs.dropna(subset=["lo0.8"])
            fig.add_trace(go.Scatter(x=list(iv.target) + list(iv.target[::-1]), y=list(iv["hi0.8"]) + list(iv["lo0.8"][::-1]),
                                     fill="toself", fillcolor=rgba(CATEGORICAL[0], 0.15), line={"width": 0},
                                     name="80% empirical interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=recs.target, y=recs.actual, mode="lines", name="actual", line={"color": INK, "width": 2}))
        fig.add_trace(go.Scatter(x=recs.target, y=recs.forecast, mode="lines", name=f"{model} forecast made {h}y earlier",
                                 line={"color": CATEGORICAL[0], "width": 2}))
        layout(fig, f"{model}: {h}-year-ahead forecasts vs outcomes", 380)
        chart(fig)
    with st.expander("Diebold–Mariano comparisons (candidate vs each baseline, paired origins)"):
        st.dataframe(pd.DataFrame([{"comparison": k, "n": v_.get("n_paired_origins"), "relative MAE": v_["relative_MAE"],
                                    "share better": v_["share_of_origins_better"], "DM p": v_["diebold_mariano"].get("p_value")}
                                   for k, v_ in hr["comparisons"].items()]), use_container_width=True, hide_index=True)
    st.subheader("Single-origin experiment")
    c1, c2 = st.columns(2)
    o = c1.number_input("Data known by year", 1800, 2020, 1950)
    t = c2.number_input("Forecast year", 1801, 2030, 1970)
    if t > o:
        from chronos.forecasting.models import build_models

        cc = con()
        s = repo.load_series(cc, ind, ents[0], src)
        cc.close()
        pe = point_experiment(s, build_models(list(models)), int(o), int(t))
        st.caption(pe["warning"])
        st.dataframe(pd.DataFrame([{k: r.get(k) for k in ("model", "forecast", "error", "abs_pct_error")} for r in pe["results"]]),
                     hide_index=True)
        st.write(f"Actual value in {t}: {pe['actual']}")


def page_patterns() -> None:
    from chronos.analysis import patterns

    st.header("Pattern detection: regimes and similar trajectories")
    df = catalog(st.session_state.db, db_version())
    picked = _series_picker("pt", df, multi=False)
    if not picked:
        return
    ind, src, ents = picked
    c1, c2, c3 = st.columns(3)
    signal = c1.selectbox("Signal", ["growth", "level", "diff"])
    model = c2.selectbox("Break model", ["mean", "meanvar"], help="meanvar: regimes may also differ in volatility")
    min_size = c3.slider("Minimum regime length", 3, 30, 8)
    c = con()
    s = repo.load_series(c, ind, ents[0], src)
    try:
        yrs, x = patterns.signal_of(s, signal)
    except ValueError as exc:
        st.warning(str(exc))
        c.close()
        return
    res = patterns.structural_breaks(yrs, x, 5, min_size, model)
    scale = 100 if signal == "growth" else 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yrs, y=x * scale, mode="lines", name=signal, line={"color": MUTED, "width": 1}))
    for seg in res["segments"]:
        fig.add_trace(go.Scatter(x=[seg["start_year"], seg["end_year"]], y=[seg["mean"] * scale] * 2, mode="lines",
                                 line={"color": CATEGORICAL[0], "width": 3}, name="regime mean", showlegend=False,
                                 hovertemplate=f"{seg['start_year']}–{seg['end_year']}: mean {seg['mean'] * scale:.2f}<extra></extra>"))
    layout(fig, f"{ents[0]}: {signal} of {ind}" + (" (%/yr)" if signal == "growth" else "") + f" — breaks {res['breaks']}", 420)
    chart(fig)
    st.caption(res.get("note", ""))
    st.dataframe(pd.DataFrame(res["segments"]), hide_index=True)
    st.subheader("Who moved like this?")
    c1, c2, c3 = st.columns(3)
    a = c1.number_input("Window start", int(s.first_year), int(s.last_year), max(int(s.first_year), 1950))
    b = c2.number_input("Window end", int(s.first_year), int(s.last_year), min(int(s.last_year), 1975))
    before = c3.checkbox("Only windows that ended before the query window", value=False)
    if b > a:
        try:
            panel = repo.load_panel(c, ind, src)
            sim = patterns.similar_trajectories(panel, ents[0], int(a), int(b), 10, "growth" if s.is_positive else "shape",
                                                before_only=before)
            st.dataframe(pd.DataFrame(sim["matches"]), hide_index=True)
            st.caption(sim["note"])
        except (ValueError, ChronosError) as exc:
            st.warning(str(exc))
    c.close()


def page_causal() -> None:
    from chronos.analysis import causal
    from chronos.analysis.data import VarSpec, merged_panel

    st.header("Causal hypothesis engine")
    st.caption("A ladder: association → predictive (Granger) → conditional association (two-way fixed effects) → "
               "quasi-experimental event study. Each tier states what it does not show.")
    c1, c2, c3 = st.columns(3)
    x = c1.text_input("X (indicator:source:transform)", "v2x_polyarchy:vdem:level")
    y = c2.text_input("Y", "gdp_per_capita:owid:growth")
    ctr = c3.text_input("Controls (comma-separated)", "population:owid:growth")
    c4, c5, c6 = st.columns(3)
    treat = c4.text_input("Event-study treatment (spec>=threshold, optional)", "v2x_regime:vdem:level>=2")
    outcome = c5.text_input("Event-study outcome", "gdp_per_capita:owid:log")
    years = c6.slider("Years", 1800, 2025, (1950, 2022))
    if not st.button("Run the evidence ladder"):
        return

    def spec(sv: str) -> VarSpec:
        ind, src, *tr = sv.split(":")
        t = tr[0] if tr else "level"
        return VarSpec(f"{ind}__{t}", ind, src, t)

    try:
        X, Y = spec(x), spec(y)
        C = [spec(v) for v in ctr.split(",") if v.strip()]
        specs = [X, Y, *C]
        T = OUT = None
        if treat.strip():
            tv, thr = treat.rsplit(">=", 1)
            T = spec(tv)
            OUT = spec(outcome) if outcome.strip() else Y
            specs += [T, OUT]
        cc = con()
        with st.spinner("Estimating…"):
            df = merged_panel(cc, specs, min_year=years[0], max_year=years[1])
            rep = causal.assess(df, X.name, Y.name, [v.name for v in C], 1, None)
            if T is not None and OUT is not None:
                tyears = causal.first_crossing_years(df, T.name, float(thr))
                rep["T4_event_study"] = causal.event_study(df, OUT.name, tyears, (-5, 10), n_perm=30)
                tier, reasons = causal._tier(rep, X.name, 1)
                rep["evidence_tier"].update(tier=tier, label=causal.TIER_TEXT[tier][0], does_not_show=causal.TIER_TEXT[tier][1],
                                            reasons=reasons)
        cc.close()
    except (ChronosError, ValueError) as exc:
        st.error(str(exc))
        return
    et = rep["evidence_tier"]
    st.markdown(f"### Evidence tier {et['tier']}: {et['label']}\n_Does not show:_ {et['does_not_show']}")
    for r in et["reasons"]:
        st.markdown(f"- {r}")
    st.markdown("**Threats to validity**")
    for t_ in rep["threats_to_validity"]:
        st.markdown(f"- {t_}")
    es = rep.get("T4_event_study")
    if es and es.get("dynamic_effects"):
        ks = sorted(int(k) for k in es["dynamic_effects"])
        co = [es["dynamic_effects"][str(k)]["coef"] for k in ks]
        se = [es["dynamic_effects"][str(k)]["se"] for k in ks]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=co, mode="markers+lines", name="effect", line={"color": CATEGORICAL[0], "width": 2},
                                 marker={"size": 8}, error_y={"type": "data", "array": [1.96 * s_ for s_ in se], "color": MUTED}))
        fig.add_hline(y=0, line={"color": MUTED, "width": 1})
        fig.add_vline(x=-0.5, line={"color": MUTED, "width": 1, "dash": "dot"})
        layout(fig, f"Event study: effect on {OUT.name if OUT else ''} by years since treatment (95% CI, ref. k=-1)", 380)
        chart(fig)
        st.caption(f"pre-trend joint p = {es.get('pretrend_joint_p')}, post-effect p = {es.get('post_effect_p')}, "
                   f"placebo (permutation) p = {es.get('permutation_p')}, treated units = {es.get('n_treated_units')}")
    with st.expander("Full report (JSON)"):
        rep["T2_predictive"].pop("per_unit", None)
        st.json(json.loads(json.dumps(rep, default=str)))


def page_state() -> None:
    st.header("Civilization State Model")
    year = st.number_input("Reference year", 2000, 2100, 2026)
    stt = run_state(st.session_state.db, db_version(), int(year))
    st.caption(" ".join(stt["principles"]))
    st.write("Status counts: " + ", ".join(f"{badge(k)}: {v}" for k, v in stt["status_counts"].items()))
    for dom, entries in stt["domains"].items():
        st.subheader(dom.replace("_", " ").title())
        for e in entries:
            with st.container(border=True):
                left, right = st.columns([2, 3])
                left.markdown(f"**{md_safe(e['label'])}**  \n{badge(e['status'])}")
                if e["status"] == "NO DATA":
                    left.caption(f"Load with: `{e.get('load_hint')}`")
                    continue
                if e["kind"] == "fact":
                    for f in e["facts"]:
                        right.markdown(f"- {md_safe(f['statement'])} _({f['year']}, {badge(f['status'])}; {md_safe(f['reference'][:80])})_")
                    continue
                left.markdown(f"**{e['value']:,.4g}** {md_safe(e.get('unit') or '')}  \n{e['note']}")
                ch = e.get("change")
                if ch:
                    txt = (f"{ch['cagr_percent']:+.2f} %/yr" if ch.get("cagr_percent") is not None else f"{ch['absolute_change']:+.3g}")
                    left.caption(f"Change {ch['from_year']}–{ch['to_year']}: {txt}")
                if e.get("nowcast"):
                    n = e["nowcast"]
                    left.caption(f"{n['label']}: {n['value']:,.4g} (80%: {n['lo80']:,.4g}–{n['hi80']:,.4g})")
                hist = e.get("history")
                if hist:
                    fig = go.Figure(go.Scatter(x=hist["years"], y=hist["values"], mode="lines",
                                               line={"color": CATEGORICAL[0], "width": 2},
                                               hovertemplate="%{x}: %{y:,.4g}<extra></extra>"))
                    if e.get("nowcast"):
                        n = e["nowcast"]
                        fig.add_trace(go.Scatter(x=[n["year"]] * 2, y=[n["lo80"], n["hi80"]], mode="lines",
                                                 line={"color": MUTED, "width": 2}, hoverinfo="skip"))
                        fig.add_trace(go.Scatter(x=[n["year"]], y=[n["value"]], mode="markers",
                                                 marker={"symbol": "circle-open", "size": 9, "color": INK_SECONDARY},
                                                 hovertemplate="nowcast %{x}: %{y:,.4g}<extra></extra>"))
                    layout(fig, None, 170)
                    fig.update_layout(showlegend=False, margin={"l": 0, "r": 0, "t": 5, "b": 0})
                    with right:
                        chart(fig)
                src = e.get("source", {})
                right.caption(f"Source: {src.get('source_name') or src.get('source_id')} · {src.get('source_dataset') or src.get('derivation') or ''} "
                              f"· retrieved {str(src.get('retrieved_utc'))[:10]} · license: {src.get('source_license')}")


OUT_LABELS = {"population": "World population", "gdp_per_capita": "GDP per person (2011 int-$)", "co2": "CO₂ emissions (Mt)",
              "warming": "Warming attributed to GHG (°C)", "energy": "Primary energy (TWh)", "gdp": "World GDP (2011 int-$)"}


def page_scenarios() -> None:
    st.header("Conditional future scenarios (2026–2126)")
    ev = st.toggle("Include historical extrapolation-error backtests (slower)", value=True)
    rep = run_scenarios(st.session_state.db, db_version(), ev)
    st.warning(rep["disclaimer"])
    for n in rep["base"]["notes"]:
        st.caption(n)
    st.subheader("How far can we check this? (validation by horizon)")
    val = rep["validation"]["gdp_per_capita"]
    st.dataframe(pd.DataFrame([{"horizon (years)": h, **{k: v for k, v in d.items() if k not in ("indicator",)}} for h, d in val.items()]),
                 use_container_width=True, hide_index=True)
    var = st.selectbox("Output", list(OUT_LABELS), format_func=lambda k: OUT_LABELS[k], index=3)
    show_band = st.checkbox("Show 5–95% assumption spread", value=True)
    fig = go.Figure()
    base_key = {"population": "population", "co2": "co2", "warming": "warming", "energy": "energy", "gdp": "gdp"}.get(var)
    c = con()
    if base_key:
        spec = {"population": ("population", "World"), "co2": ("co2", "World"), "warming": ("temperature_change_from_ghg", "World"),
                "energy": ("primary_energy_consumption", "World"), "gdp": ("gdp", "World")}[base_key]
        try:
            s = repo.load_series(c, spec[0], spec[1], "owid").between(1950, 3000)
            fig.add_trace(go.Scatter(x=s.years, y=s.values, mode="lines", name="observed", line={"color": INK, "width": 2}))
        except ChronosError:
            pass
    c.close()
    for i, s_ in enumerate(rep["scenarios"]):
        if "error" in s_:
            continue
        col = CATEGORICAL[i % len(CATEGORICAL)]
        yrs = s_["path"]["years"]
        if show_band:
            fig.add_trace(go.Scatter(x=yrs + yrs[::-1], y=s_["path"]["p95"][var] + s_["path"]["p05"][var][::-1], fill="toself",
                                     fillcolor=rgba(col, 0.12), line={"width": 0}, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=yrs, y=s_["path"][var], mode="lines", name=s_["name"], line={"color": col, "width": 2}))
    layout(fig, OUT_LABELS[var] + " — conditional paths (not predictions)", 480)
    fig.add_vline(x=rep["model"]["reference_year"], line={"color": MUTED, "width": 1, "dash": "dot"})
    chart(fig)
    st.caption("Lines: each scenario's central path; shaded: spread from the scenario's own parameter ranges. Dotted line: reference year.")
    names = [s_["name"] for s_ in rep["scenarios"] if "error" not in s_]
    pick = st.selectbox("Scenario details", names)
    s_ = next(x for x in rep["scenarios"] if x.get("name") == pick)
    st.markdown(f"**{s_['name']}** — {s_['summary']}")
    st.markdown("**Assumptions**\n" + "\n".join(f"- {a}" for a in s_["assumptions"]))
    prow = []
    for k, p in s_["parameters"].items():
        for per in p["periods"]:
            prow.append({"parameter (%/yr)": k, "until": per["until"], "low": per["low"], "central": per["central"], "high": per["high"],
                         "basis": p["basis"], "derived from data": json.dumps(p["derived_from_data"]) if p["derived_from_data"] else ""})
    st.dataframe(pd.DataFrame(prow), use_container_width=True, hide_index=True)
    st.markdown(f"**Mechanisms.** {s_['mechanisms']['narrative']}")
    st.dataframe(pd.DataFrame([{"mechanism": m["name"], "max historical evidence level": m["max_evidence_level"],
                                "claims": ", ".join(c_["claim_id"] for c_ in m["historical_claims"])} for m in s_["mechanisms"]["linked"]]),
                 hide_index=True, use_container_width=True)
    yr = st.radio("Sensitivity year", list(s_["critical_dependencies"]["computed"]), horizontal=True)
    out = st.radio("Sensitivity of", ["warming", "co2", "gdp_per_capita"], horizontal=True)
    sens = s_["critical_dependencies"]["computed"][yr][out]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=[r["parameter"] for r in sens], x=[r["at_high"] - r["at_low"] for r in sens], orientation="h",
                         marker={"color": CATEGORICAL[0], "cornerradius": 4},
                         hovertemplate="%{y}: high − low = %{x:,.4g}<extra></extra>"))
    layout(fig, f"One-at-a-time swing in {out} ({yr}): parameter at its high minus at its low bound", 300)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    chart(fig)
    st.markdown("**Critical dependencies (stated):** " + "; ".join(s_["critical_dependencies"]["stated"]))
    st.markdown("**Signposts** (machine-checked against stored data)")
    st.dataframe(pd.DataFrame(s_["signposts"]), use_container_width=True, hide_index=True)
    st.markdown("**Events that could change the trajectory (no probabilities):** " + "; ".join(s_["wildcards"]))
    shocks = rep.get("historical_shocks", {})
    if shocks:
        st.caption("Largest one-year declines since 1900 in the data: " + "; ".join(
            f"{k}: " + ", ".join(f"{x['year']} ({x['change_percent']:.1f}%)" for x in v) for k, v in shocks.items()))
    with st.expander("Assumption check against years observed after the base year"):
        st.dataframe(pd.DataFrame(s_["hindcast_check"]), hide_index=True)


def page_research() -> None:
    st.header("Research query")
    st.markdown("> " + "Compare the consequences of the printing press, the spread of the internet and the emergence of "
                "generative AI. Which common economic and social mechanisms can be identified? What do these mechanisms "
                "suggest about human development by 2040?")
    c1, c2 = st.columns(2)
    year = c1.number_input("Outlook year", 2027, 2126, 2040)
    with_scen = c2.checkbox("Include scenario outlook", value=True)
    try:
        d, md = run_dossier(st.session_state.db, db_version(), int(year), with_scen)
    except ChronosError as exc:
        st.info(str(exc))
        return
    st.download_button("Download dossier (Markdown)", md, file_name=f"chronos_dossier_{year}.md")
    st.markdown(md_safe(md))


PAGES = {
    "Overview & sources": page_overview,
    "Time series": page_explorer,
    "Knowledge graph": page_knowledge,
    "Analogs": page_analogs,
    "Backtesting": page_backtest,
    "Patterns": page_patterns,
    "Causal hypotheses": page_causal,
    "Civilization state": page_state,
    "Scenarios": page_scenarios,
    "Research query": page_research,
}

with st.sidebar:
    st.markdown("### ⏳ CHRONOS")
    st.session_state.db = st.text_input("Database", st.session_state.db)
    page = st.radio("View", list(PAGES))
    st.caption("No probabilities are assigned to scenarios. Analog bands and scenario spreads are not calibrated "
               "probabilities. LLM-based extraction: NOT IMPLEMENTED.")

if not Path(st.session_state.db).exists():
    st.error(f"Database {st.session_state.db} not found. Create it with `chronos init` and load data (see README).")
else:
    PAGES[page]()
