"""GDELT DOC 2.0 API, timeline volume mode (contemporary news coverage, not history).

Request: https://api.gdeltproject.org/api/v2/doc/doc?query=Q&mode=timelinevol&format=json
         &startdatetime=YYYYMMDDHHMMSS&enddatetime=YYYYMMDDHHMMSS
Response: {"timeline": [{"series": "...", "data": [{"date": "20240101T000000Z", "value": 0.12}, ...]}]}
The value is the share (%) of monitored online news articles matching the query.
CHRONOS stores annual means; a year with fewer than 330 daily points is marked 'estimated'.
Status: IMPLEMENTED_UNVERIFIED (host unreachable from the development sandbox).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from chronos.config import Settings
from chronos.errors import DataError
from chronos.ingestion.http import fetch
from chronos.ingestion.registry import entry
from chronos.storage.repository import IngestBatch, ObsRow, SeriesMeta, SeriesPayload

URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def parse_timeline(payload: dict[str, Any], indicator: str, query: str) -> list[SeriesPayload]:
    timeline = payload.get("timeline") if isinstance(payload, dict) else None
    if not timeline:
        raise DataError("GDELT response contains no timeline")
    points: dict[int, list[float]] = defaultdict(list)
    for item in timeline[0].get("data", []):
        try:
            points[int(str(item["date"])[:4])].append(float(item["value"]))
        except (KeyError, ValueError, TypeError):
            continue
    if not points:
        raise DataError("GDELT timeline has no usable points")
    rows = [ObsRow(y, sum(v) / len(v), "observed" if len(v) >= 330 else "estimated", f"{len(v)} daily points")
            for y, v in sorted(points.items())]
    return [SeriesPayload(SeriesMeta(indicator, "World (online news monitored by GDELT)", "gdelt", "DOC2/timelinevol",
                                     query, None, f"News coverage share: {query}",
                                     "Annual mean of daily GDELT timelinevol values", "% of monitored articles"),
                          rows)]


def fetch_timeline(settings: Settings, query: str, start: str, end: str, indicator: str) -> IngestBatch:
    params = {"query": query, "mode": "timelinevol", "format": "json", "startdatetime": start, "enddatetime": end}
    fr = fetch(URL, settings, "gdelt", params=params, suffix=".json")
    return IngestBatch(entry("gdelt").info(), "DOC2/timelinevol", fr.url, fr.sha256,
                       parse_timeline(fr.json(), indicator, query))
