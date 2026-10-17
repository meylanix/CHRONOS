"""Our World in Data Chart API (any chart on ourworldindata.org/grapher/<slug>).

Data:     https://ourworldindata.org/grapher/<slug>.csv?v=1&csvType=full&useColumnShortNames=true
Metadata: https://ourworldindata.org/grapher/<slug>.metadata.json?v=1&csvType=full&useColumnShortNames=true
CSV layout: Entity, Code, Year (or Day), <one column per indicator>.
The metadata JSON is parsed defensively (keys: chart{title,citation}, columns{<short name>:
{titleShort/titleLong, unit, citationShort/citationLong, lastUpdated, ...}}).
Status: IMPLEMENTED_UNVERIFIED (host unreachable from the development sandbox).
"""

from __future__ import annotations

import io
import re
from typing import Any

import pandas as pd

from chronos.config import Settings
from chronos.errors import DataError
from chronos.ingestion.http import FetchResult, fetch, read_local
from chronos.ingestion.registry import entry
from chronos.storage.repository import IngestBatch, ObsRow, SeriesMeta, SeriesPayload

BASE = "https://ourworldindata.org/grapher"
PARAMS = {"v": "1", "csvType": "full", "useColumnShortNames": "true"}
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,200}$")


def parse(data: FetchResult, metadata: dict[str, Any] | None, slug: str, column: str | None,
          entities: list[str] | None, indicator: str | None) -> IngestBatch:
    df = pd.read_csv(io.BytesIO(data.content))
    if "Entity" not in df.columns or "Year" not in df.columns:
        raise DataError(f"Chart {slug!r} is not an annual Entity/Year chart (columns: {list(df.columns)[:6]})")
    value_cols = [c for c in df.columns if c not in ("Entity", "Code", "Year")]
    numeric = [c for c in value_cols if pd.api.types.is_numeric_dtype(df[c])]
    if column and column not in value_cols:
        raise DataError(f"Column {column!r} not found; available: {value_cols}")
    if not column and len(numeric) != 1:
        raise DataError(f"Chart has {len(numeric)} numeric columns {numeric}; choose one with --column "
                        "(v0.1 silently took the first one).")
    col = column or numeric[0]
    if entities:
        absent = sorted(set(entities) - set(df["Entity"].unique()))
        if absent:
            raise DataError(f"Entities not in chart: {absent}")
        df = df[df["Entity"].isin(entities)]
    cmeta = ((metadata or {}).get("columns") or {}).get(col, {}) if isinstance(metadata, dict) else {}
    chart = (metadata or {}).get("chart", {}) if isinstance(metadata, dict) else {}
    title = cmeta.get("titleLong") or cmeta.get("titleShort") or chart.get("title") or col
    unit = cmeta.get("unit") or cmeta.get("shortUnit") or None
    citation = cmeta.get("citationLong") or cmeta.get("citationShort") or chart.get("citation")
    payloads = []
    for ent, g in df.groupby("Entity", sort=True):
        sub = g[["Year", col]].dropna()
        if sub.empty:
            continue
        code = g["Code"].dropna().iloc[0] if "Code" in g and g["Code"].notna().any() else None
        payloads.append(SeriesPayload(
            SeriesMeta(indicator or slug, str(ent), "owid-grapher", f"grapher/{slug}", col, code, title,
                       f"Citation: {citation}" if citation else None, unit),
            [ObsRow(int(y), float(v)) for y, v in zip(sub["Year"], sub[col], strict=True)]))
    version = cmeta.get("lastUpdated") or (metadata or {}).get("dateDownloaded") if isinstance(metadata, dict) else None
    return IngestBatch(entry("owid-grapher").info(), f"grapher/{slug}", data.url, data.sha256, payloads,
                       source_version=f"lastUpdated:{version}" if version else None)


def fetch_chart(settings: Settings, slug: str, column: str | None = None, entities: list[str] | None = None,
                indicator: str | None = None, data_file: str | None = None) -> IngestBatch:
    if not SLUG_RE.match(slug):
        raise DataError(f"Invalid OWID chart slug {slug!r}")
    if data_file:
        return parse(read_local(data_file), None, slug, column, entities, indicator)
    data = fetch(f"{BASE}/{slug}.csv", settings, "owid-grapher", params=PARAMS, suffix=".csv")
    meta = fetch(f"{BASE}/{slug}.metadata.json", settings, "owid-grapher", params=PARAMS, suffix=".json").json()
    return parse(data, meta, slug, column, entities, indicator)
