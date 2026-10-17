"""Plain-text rendering of results for the terminal (JSON is available via --json / --json-out)."""

from __future__ import annotations

from typing import Any


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if abs(x) >= 1e5 or (abs(x) < 1e-3 and x != 0):
            return f"{x:.3e}"
        return f"{x:.{nd}g}"
    return str(x)


def table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    cells = [[fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) if cells else len(c) for i, c in enumerate(cols)]
    line = "  ".join(c.ljust(w) for c, w in zip(cols, widths, strict=True))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(v.ljust(w) for v, w in zip(row, widths, strict=True)) for row in cells]
    return "\n".join([line, sep, *body])


def backtest_text(res: dict[str, Any]) -> str:
    s = res["series"]
    out = [f"WALK-FORWARD BACKTEST  {s['indicator']} / {s['entity']}  ({s['data_kind']}, {s['n_observations']} obs "
           f"{s['first_year']}–{s['last_year']}, unit: {s['unit'] or 'n/a'})"]
    if res.get("banner"):
        out.append("!! " + res["banner"])
    prov = s["provenance"]
    out.append(f"source: {prov.get('source_id')} / {prov.get('source_dataset')} / {prov.get('source_variable')}; "
               f"vintage {prov.get('vintage_id')} retrieved {str(prov.get('retrieved_utc'))[:10]}")
    out.append(f"revision leakage: {res['revision_leakage']['risk']}")
    for h, hr in res["summary"]["horizons"].items():
        out.append(f"\n── horizon {h} years: {hr['n_common_origins']} common origins")
        rows = []
        for name, m in hr["models"].items():
            cal = m.get("empirical_interval_calibration", {}).get("0.8", {})
            rows.append({"model": name + (" (baseline)" if res["models"][name]["baseline"] else ""),
                         "n_nonoverlap": m.get("n_nonoverlapping"), "MAE": m.get("MAE"), "RMSE": m.get("RMSE"),
                         "MASE": m.get("MASE"), "MAPE%": m.get("MAPE_percent"),
                         "cov80": cal.get("empirical_coverage")})
        rows.sort(key=lambda r: r["MAE"] if r["MAE"] is not None else float("inf"))
        out.append(table(rows, ["model", "n_nonoverlap", "MAE", "RMSE", "MASE", "MAPE%", "cov80"]))
        for name, m in hr.get("low_coverage_models", {}).items():
            out.append(f"  (low coverage) {name}: MAE {fmt(m.get('MAE'))} on {m.get('n')} own origins — {m['note']}")
        v = hr["verdict"]
        out.append(f"best: {v['best_model']}   best baseline: {v['best_baseline']}")
        for n in v["notes"]:
            out.append(f"  • {n}")
        st = hr.get("stability", {})
        if "same_best_model_in_all_blocks" in st:
            out.append(f"  • stability: best model per sub-period = {[b['best'] for b in st['blocks']]}")
    return "\n".join(out)
