"""Research dossier: compare information-technology transitions and derive conditional 2040 outlooks.

The dossier is assembled only from what is stored: knowledge-graph events, claims, facts and
references, plus scenario-engine output. It separates documented facts, curator-assessed
interpretations and assumptions, lists counter-hypotheses, and reports data gaps. It writes no
sentence that is not traceable to a stored record or to an explicit rule in this module.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from chronos.errors import DataError
from chronos.knowledge import store

TRANSITIONS = {
    "printing": {"label": "Movable-type printing (c. 1450)", "tag": "transition:printing"},
    "internet": {"label": "Computers, the Internet and the Web (1970s-2010s)", "tag": "transition:internet"},
    "genai": {"label": "Generative AI (2017/2022-)", "tag": "transition:genai"},
}
COMPARATOR = {"electrification": {"label": "Electrification (comparator)", "tag": "transition:electrification"}}

QUESTION = ("Compare the consequences of the printing press, the spread of the internet and the emergence of "
            "generative AI. Which common economic and social mechanisms can be identified? What do these "
            "mechanisms suggest about human development by 2040?")


def _events_by_transition(con: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for key, t in {**TRANSITIONS, **COMPARATOR}.items():
        out[key] = sorted(store.list_events(con, tag=t["tag"], kind="real"), key=lambda e: (e["start_year"], e["event_id"]))
    return out


def build_dossier(con: sqlite3.Connection, horizon_year: int = 2040, scenario_report: dict[str, Any] | None = None,
                  question: str = QUESTION) -> dict[str, Any]:
    events = _events_by_transition(con)
    if not any(events[k] for k in TRANSITIONS):
        raise DataError("No transition events in the knowledge graph. Run: chronos kg import data/curated/tech_transitions.json")
    ev_index = {e["event_id"]: k for k, evs in events.items() for e in evs}
    claims = [c for c in store.list_claims(con) if c["data_kind"] == "real"]
    mechanisms = {m["mechanism_id"]: m for m in store.list_mechanisms(con)}

    # 1-2. timelines
    timelines = {k: [{"year": e["start_year"], "end_year": e["end_year"], "uncertainty": e["year_uncertainty"],
                      "precision": e["date_precision"], "date": e["date_detail"], "event_id": e["event_id"],
                      "title": e["title"], "role": next((t for t in e["tags"] if t in ("anchor", "diffusion", "outcome", "comparison")), ""),
                      "references": [r["ref_id"] for r in e["references"]]} for e in evs] for k, evs in events.items()}

    # 3. mechanisms by transition
    matrix: dict[str, dict[str, Any]] = {}
    for c in claims:
        mid = c["mechanism_id"]
        if not mid:
            continue
        trans = {t for t in (ev_index.get(c["source_event_id"]), ev_index.get(c["target_event_id"])) if t}
        for t in trans:
            cell = matrix.setdefault(mid, {}).setdefault(t, {"max_level": 0, "claims": []})
            cell["max_level"] = max(cell["max_level"], c["evidence_level"])
            cell["claims"].append({"claim_id": c["claim_id"], "relation": c["relation"], "level": c["evidence_level"],
                                   "status": c["status"], "statement": c["statement"]})

    # 4. similarities / differences
    core = list(TRANSITIONS)
    shared, partial, unique = [], [], []
    for mid, cells in matrix.items():
        present = [t for t in core if t in cells and cells[t]["max_level"] >= 2]
        item = {"mechanism_id": mid, "name": mechanisms.get(mid, {}).get("name", mid), "present_in": present,
                "levels": {t: cells[t]["max_level"] for t in core if t in cells}}
        if len(present) == 3:
            shared.append(item)
        elif len(present) == 2:
            partial.append(item)
        elif len(present) == 1:
            unique.append(item)
    facts = {k: store.list_facts(con, tag=TRANSITIONS[k]["tag"]) for k in core}
    diffusion = _diffusion_comparison(facts)

    # 5. facts vs interpretations
    interpretations = sorted(({"claim_id": c["claim_id"], "statement": c["statement"], "relation": c["relation"],
                               "evidence_level": c["evidence_level"], "evidence_label": c["evidence_label"],
                               "status": c["status"], "transitions": sorted({ev_index.get(c["source_event_id"]),
                                                                             ev_index.get(c["target_event_id"])} - {None}),
                               "supports": [r["ref_id"] for r in c["references"] if r["stance"] == "supports"]}
                              for c in claims), key=lambda c: -c["evidence_level"])

    # 6. counter-hypotheses
    counter = []
    for c in claims:
        opp = [r for r in c["references"] if r["stance"] in ("contradicts", "qualifies")]
        if c["alternatives"] or opp:
            counter.append({"claim_id": c["claim_id"], "statement": c["statement"],
                            "alternatives": [a["description"] for a in c["alternatives"]],
                            "contradicting_or_qualifying": [{"ref_id": r["ref_id"], "stance": r["stance"],
                                                              "locator": r["locator"]} for r in opp]})
    one_sided = [c["claim_id"] for c in claims if c["evidence_level"] >= 3 and not c["alternatives"]
                 and not any(r["stance"] in ("contradicts", "qualifies") for r in c["references"])]

    # 7. scenarios to horizon_year
    outlook = _outlook(scenario_report, horizon_year) if scenario_report else {"status": "NOT RUN",
                                                                                "note": "scenario engine output not supplied"}

    # 8. evidence vs assumption
    stances = {c["claim_id"]: {r["stance"] for r in c["references"]} for c in claims}
    contested = [i for i in interpretations if i["status"] in ("contested", "refuted") or "contradicts" in stances[i["claim_id"]]]
    contested_ids = {i["claim_id"] for i in contested}
    established = [i for i in interpretations if i["evidence_level"] >= 4 and i["claim_id"] not in contested_ids]
    qualified = [i for i in interpretations if "qualifies" in stances[i["claim_id"]] and i["claim_id"] not in contested_ids]
    speculative = [i for i in interpretations if i["evidence_level"] <= 2]

    # 7b. what the mechanisms suggest: registered forward-looking hypotheses with falsification conditions
    from chronos.scenarios.engine import evaluate_signposts

    outlook_h = []
    for h in store.list_outlook_hypotheses(con):
        if h["data_kind"] != "real":
            continue
        item = {k: h[k] for k in ("oh_id", "statement", "mechanism_id", "based_on_claims", "basis_evidence_levels",
                                  "falsified_if", "check_by_year", "scope", "notes")}
        item["mechanism"] = mechanisms.get(h["mechanism_id"], {}).get("name") if h["mechanism_id"] else None
        item["status"] = (evaluate_signposts(con, [h["signpost"]])[0]["status"] if h["signpost"]
                          else "NOT OPERATIONALISED (no machine-checkable indicator)")
        outlook_h.append(item)
    assumptions = [
        "Mechanism transfer: that a mechanism documented for printing or the web operates similarly for generative AI "
        "is an ASSUMPTION; contexts differ (literacy, institutions, market structure, speed).",
        "Diffusion metrics are not commensurable across transitions (cities with presses vs share of population online "
        "vs share of US adults who tried a tool); speed comparisons are qualitative.",
        f"Anything stated about {horizon_year} is conditional on a scenario's assumptions; no scenario receives a probability.",
    ]

    # 9. references
    cited = {r for e in sum(events.values(), []) for r in (x["ref_id"] for x in e["references"])}
    cited |= {r["ref_id"] for c in claims for r in c["references"]}
    cited |= {f["ref_id"] for fs in facts.values() for f in fs}
    refs = [r for r in store.list_references(con) if r["ref_id"] in cited]

    # 10. data gaps
    gaps = _gaps(con, events, facts, scenario_report)

    return {
        "question": question, "horizon_year": horizon_year,
        "transitions": {k: v["label"] for k, v in {**TRANSITIONS, **COMPARATOR}.items()},
        "timelines": timelines, "mechanism_matrix": matrix,
        "similarities": {"shared_by_all_three": shared, "shared_by_two": partial, "specific_to_one": unique},
        "differences": {"diffusion": diffusion, "mechanisms_specific_to_one": unique},
        "facts": {k: [{"statement": f["statement"], "value": f["value"], "unit": f["unit"], "year": f["year"],
                       "ref_id": f["ref_id"], "uncertainty": f["uncertainty"], "verification": f["verification"]}
                      for f in fs] for k, fs in facts.items()},
        "interpretations": interpretations, "counter_hypotheses": counter, "one_sided_claims": one_sided,
        "conclusions": {"supported_by_evidence": established, "contested": contested, "qualified": qualified,
                        "speculative": speculative, "assumptions": assumptions},
        "outlook_hypotheses": outlook_h,
        "outlook": outlook, "references": refs, "data_gaps": gaps,
        "method": ("Assembled from stored knowledge-graph records and scenario output only. Evidence levels are curator "
                   "assessments (see each reference's verification status); CHRONOS statistics do not raise them."),
    }


def _diffusion_comparison(facts: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    want = {"printing": "f_print_cities", "internet": "f_itu_2019", "genai": "f_genai_adoption_us"}
    rows = []
    for t, fid in want.items():
        f = next((x for x in facts.get(t, []) if x["fact_id"] == fid), None)
        if f:
            rows.append({"transition": t, "statement": f["statement"], "year": f["year"], "ref_id": f["ref_id"]})
        else:
            rows.append({"transition": t, "statement": "NO DATA", "year": None, "ref_id": None})
    return rows


def _outlook(rep: dict[str, Any], year: int) -> dict[str, Any]:
    ref = rep["model"]["reference_year"]
    out: dict[str, Any] = {"status": "CONDITIONAL", "disclaimer": rep["disclaimer"], "year": year, "scenarios": []}
    gdp_val = rep["validation"].get("gdp_per_capita", {})
    h = year - ref
    nearest = min((int(k) for k in gdp_val), key=lambda k: abs(k - h), default=None) if gdp_val else None
    if nearest is not None:
        entry = gdp_val[nearest] if nearest in gdp_val else gdp_val[str(nearest)]
        out["empirical_error_scale_nearest_horizon"] = {"horizon": nearest, **entry}
    for s in rep["scenarios"]:
        if "error" in s:
            continue
        yrs = s["path"]["years"]
        if year not in yrs:
            continue
        i = yrs.index(year)
        out["scenarios"].append({
            "id": s["id"], "name": s["name"],
            "values": {k: {"central": s["path"][k][i], "p05": s["path"]["p05"][k][i], "p95": s["path"]["p95"][k][i]}
                       for k in ("population", "gdp_per_capita", "co2", "warming")},
            "mechanisms_assumed": [m["name"] for m in s["mechanisms"]["linked"]],
            "key_assumptions": s["assumptions"][:3],
            "signposts": [{"label": sp["label"], "status": sp["status"]} for sp in s["signposts"]],
        })
    return out


def _gaps(con: sqlite3.Connection, events: dict[str, list[dict[str, Any]]], facts: dict[str, list[dict[str, Any]]],
          rep: dict[str, Any] | None) -> list[str]:
    gaps = []
    for k, t in TRANSITIONS.items():
        if not events.get(k):
            gaps.append(f"No events stored for {t['label']}.")
        if not facts.get(k):
            gaps.append(f"No quantitative facts stored for {t['label']}.")
    n_series = con.execute("SELECT COUNT(*) FROM series WHERE indicator IN ('IT.NET.USER.ZS','internet_users_share')").fetchone()[0]
    if not n_series:
        gaps.append("No internet-adoption time series loaded (World Bank IT.NET.USER.ZS); diffusion curves cannot be fitted.")
    gaps.append("No machine-readable series for printing diffusion or book output is loaded (Buringh & van Zanden data "
                "are on OWID's chart API, unreachable in the development sandbox).")
    gaps.append("Generative-AI outcomes are observed for 2-4 years only; long-run effects are unobservable today.")
    gaps.append("Knowledge-graph references are metadata-checked, not content-checked by a domain expert.")
    if rep is None:
        gaps.append("Scenario engine was not run (needs OWID co2/energy data).")
    return gaps


def dossier_markdown(d: dict[str, Any]) -> str:
    L = [f"# CHRONOS research dossier\n\n**Question.** {d['question']}\n", f"_{d['method']}_\n"]
    L.append("## 1–2. Timelines (documented events)\n")
    for k, rows in d["timelines"].items():
        L.append(f"### {d['transitions'][k]}")
        for r in rows:
            when = r["date"] or (f"{r['year']}–{r['end_year']}" if r["end_year"] else f"{r['year']}")
            unc = f" ±{r['uncertainty']}y" if r["uncertainty"] else ""
            L.append(f"- **{when}{unc}** — {r['title']} _(refs: {', '.join(r['references']) or 'none'})_")
        L.append("")
    L.append("## 3–4. Mechanisms: where they appear and how well supported\n")
    L.append("| Mechanism | Printing | Internet | Gen. AI |\n|---|---|---|---|")
    for mid, cells in d["mechanism_matrix"].items():
        name = next((x["name"] for grp in d["similarities"].values() for x in grp if x["mechanism_id"] == mid), mid)
        L.append(f"| {name} | " + " | ".join(str(cells[t]["max_level"]) if t in cells else "—" for t in ("printing", "internet", "genai")) + " |")
    L.append("\nCells: highest curator-assessed evidence level (0–5) among claims linking that mechanism to the transition.\n")
    for title, key in (("Shared by all three (level ≥ 2)", "shared_by_all_three"), ("Shared by two", "shared_by_two"),
                       ("Specific to one", "specific_to_one")):
        items = d["similarities"][key]
        L.append(f"**{title}:** " + ("; ".join(f"{i['name']} ({', '.join(i['present_in'])})" for i in items) if items else "none recorded"))
    L.append("\n**Diffusion (not commensurable — different units):**")
    for r in d["differences"]["diffusion"]:
        L.append(f"- {r['transition']}: {r['statement']} _(ref: {r['ref_id']})_")
    L.append("\n## 5. Documented facts\n")
    for k, fs in d["facts"].items():
        for f in fs:
            L.append(f"- [{k}] {f['statement']} _(ref: {f['ref_id']}; {f['verification']}{'; ' + f['uncertainty'] if f['uncertainty'] else ''})_")
    L.append("\n## 5. Interpretations (claims), strongest first\n")
    for c in d["interpretations"]:
        L.append(f"- **L{c['evidence_level']} {c['evidence_label']}**, {c['status']}: {c['statement']} "
                 f"_(claim {c['claim_id']}; supporting: {', '.join(c['supports']) or 'none'})_")
    L.append("\n## 6. Counter-hypotheses and qualifications\n")
    for c in d["counter_hypotheses"]:
        L.append(f"- **{c['statement']}**")
        for a in c["alternatives"]:
            L.append(f"  - alternative: {a}")
        for r in c["contradicting_or_qualifying"]:
            L.append(f"  - {r['stance']}: {r['ref_id']}{' (' + r['locator'] + ')' if r['locator'] else ''}")
    if d["one_sided_claims"]:
        L.append(f"\n_Claims at level ≥ 3 with no recorded counter-evidence or alternative (curation gap): {', '.join(d['one_sided_claims'])}_")
    o = d["outlook"]
    L.append(f"\n## 7. Conditional outlook to {d['horizon_year']}\n")
    if o["status"] != "CONDITIONAL":
        L.append(f"NOT RUN: {o.get('note')}")
    else:
        L.append(f"> {o['disclaimer']}\n")
        es = o.get("empirical_error_scale_nearest_horizon")
        if es and es.get("status") != "UNVALIDATED":
            L.append(f"Historical error scale (country GDP/person extrapolations, {es['horizon']}-year horizon): median |log error| "
                     f"{es.get('median_abs_log_error', float('nan')):.2f}, 80% interval coverage {es.get('empirical_80pct_coverage', float('nan')):.0%}.\n")
        L.append("| Scenario | World population | GDP per person (2011 int-$) | CO₂ (Mt) | Warming (°C, GHG-attributed) |\n|---|---|---|---|---|")
        def cell(v: dict[str, Any], k: str, nd: int = 0, scale: float = 1.0) -> str:
            return f"{v[k]['central'] / scale:,.{nd}f} [{v[k]['p05'] / scale:,.{nd}f}–{v[k]['p95'] / scale:,.{nd}f}]"

        for s in o["scenarios"]:
            v = s["values"]
            L.append(f"| {s['name']} | {cell(v, 'population', 2, 1e9)} bn | {cell(v, 'gdp_per_capita')} | {cell(v, 'co2')} | {cell(v, 'warming', 2)} |")
        L.append("\nBrackets: 5–95% spread implied by each scenario's own parameter ranges (not probabilities).\n")
        for s in o["scenarios"]:
            L.append(f"- **{s['name']}** assumes: {'; '.join(s['mechanisms_assumed'])}. Signposts: "
                     + "; ".join(f"{x['label']} [{x['status']}]" for x in s["signposts"]))
    L.append(f"\n### What the mechanisms suggest for {d['horizon_year']}: registered hypotheses (not predictions of known accuracy)\n")
    for h in d.get("outlook_hypotheses", []):
        L.append(f"- **{h['oh_id']}** — {h['statement']}  \n  mechanism: {h['mechanism'] or '—'}; based on claims "
                 f"{', '.join(f'{c} (L{lv})' for c, lv in zip(h['based_on_claims'], h['basis_evidence_levels'], strict=True))}; "
                 f"check by {h['check_by_year']}; status: **{h['status']}**  \n  falsified if: {h['falsified_if']}")
    if not d.get("outlook_hypotheses"):
        L.append("- none registered")
    c = d["conclusions"]
    L.append("\n## 8. What follows from the evidence — and what does not\n")
    L.append("**Supported by quantitative or identified designs (in their own historical context):**")
    L += [f"- {i['statement']} (L{i['evidence_level']})" for i in c["supported_by_evidence"]] or ["- none"]
    L.append("\n**Contested (contradicting evidence recorded):**")
    L += [f"- {i['statement']}" for i in c["contested"]] or ["- none"]
    L.append("\n**Qualified (limits of scope or external validity recorded):**")
    L += [f"- {i['statement']}" for i in c["qualified"]] or ["- none"]
    L.append("\n**Speculative (level ≤ 2):**")
    L += [f"- {i['statement']}" for i in c["speculative"]] or ["- none"]
    L.append("\n**Assumptions required for any statement about the future:**")
    L += [f"- {a}" for a in c["assumptions"]]
    L.append("\n## 9. References\n")
    for r in d["references"]:
        bits = [r["authors"] or "", f"({r['year']})" if r["year"] else "", r["title"], r["venue"] or r["publisher"] or "",
                f"doi:{r['doi']}" if r["doi"] else "", r["url"] or ""]
        L.append(f"- `{r['ref_id']}` " + ". ".join(b for b in bits if b) + f" — _{r['verification']}_")
    L.append("\n## 10. Data gaps\n")
    L += [f"- {g}" for g in d["data_gaps"]]
    return "\n".join(L) + "\n"
