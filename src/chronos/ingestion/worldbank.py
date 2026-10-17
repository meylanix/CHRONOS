"""World Bank Indicators API v2.

Request:  https://api.worldbank.org/v2/country/{countries}/indicator/{code}?format=json&per_page=N&page=P&date=A:B
Response: [ {page, pages, per_page, total, sourceid, lastupdated, ...},
            [ {indicator{id,value}, country{id,value}, countryiso3code, date, value, unit, obs_status, decimal}, ... ] ]
Errors:   [ {"message": [ {"id": "...", "key": "...", "value": "..."} ]} ]
Status: IMPLEMENTED_UNVERIFIED (host unreachable from the development sandbox).
"""

from __future__ import annotations

from typing import Any

from chronos.config import Settings
from chronos.errors import DataError, SourceError
from chronos.ingestion.http import FetchResult, fetch
from chronos.ingestion.registry import entry
from chronos.storage.repository import IngestBatch, ObsRow, SeriesMeta, SeriesPayload

BASE = "https://api.worldbank.org/v2"
PER_PAGE = 20000


def _check(payload: Any, url: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "message" in payload[0]:
        msgs = "; ".join(f"{m.get('key')}: {m.get('value')}" for m in payload[0]["message"])
        raise SourceError(f"World Bank API error for {url}: {msgs}")
    if not (isinstance(payload, list) and len(payload) == 2 and isinstance(payload[0], dict)):
        raise SourceError(f"Unexpected World Bank response shape from {url}")
    return payload[0], payload[1] or []


def parse_pages(pages: list[dict[str, Any] | list[Any]], indicator_meta: dict[str, Any] | None,
                local_indicator: str | None) -> tuple[list[SeriesPayload], str | None]:
    by_country: dict[tuple[str, str | None], list[ObsRow]] = {}
    code = title = None
    last_updated = None
    for raw in pages:
        meta, rows = _check(raw, "page")
        last_updated = meta.get("lastupdated", last_updated)
        for r in rows:
            if r.get("value") is None:
                continue
            try:
                year = int(str(r["date"])[:4])
            except (KeyError, ValueError):
                continue
            code = r.get("indicator", {}).get("id", code)
            title = r.get("indicator", {}).get("value", title)
            key = (r.get("country", {}).get("value", "?"), r.get("countryiso3code") or None)
            status = "estimated" if r.get("obs_status") == "E" else "observed"
            by_country.setdefault(key, []).append(ObsRow(year, float(r["value"]), status))
    if code is None:
        raise DataError("World Bank returned no non-missing observations")
    unit = (indicator_meta or {}).get("unit") or None
    desc = (indicator_meta or {}).get("sourceNote")
    out = [
        SeriesPayload(SeriesMeta(local_indicator or code, name, "worldbank", "WDI", code, iso3, title, desc, unit),
                      sorted(rows, key=lambda r: r.year))
        for (name, iso3), rows in sorted(by_country.items())
    ]
    return out, last_updated


def fetch_indicator(settings: Settings, indicator: str, countries: str = "all", date: str | None = None,
                    local_indicator: str | None = None) -> IngestBatch:
    url = f"{BASE}/country/{countries}/indicator/{indicator}"
    params: dict[str, Any] = {"format": "json", "per_page": PER_PAGE, "page": 1}
    if date:
        params["date"] = date
    first: FetchResult = fetch(url, settings, "worldbank", params=params, suffix=".json")
    meta, _ = _check(first.json(), url)
    pages = [first.json()]
    for p in range(2, int(meta.get("pages", 1)) + 1):
        pages.append(fetch(url, settings, "worldbank", params={**params, "page": p}, suffix=".json").json())
    try:
        ind_raw = fetch(f"{BASE}/indicator/{indicator}", settings, "worldbank", params={"format": "json"},
                        suffix=".json").json()
        ind_meta = ind_raw[1][0] if isinstance(ind_raw, list) and len(ind_raw) == 2 and ind_raw[1] else None
    except SourceError:
        ind_meta = None
    payloads, last_updated = parse_pages(pages, ind_meta, local_indicator)
    return IngestBatch(entry("worldbank").info(), "WDI", first.url, first.sha256, payloads,
                       source_version=f"lastupdated:{last_updated}" if last_updated else None)
