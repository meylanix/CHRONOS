"""Official Our World in Data datasets published on GitHub (owid/co2-data, owid/energy-data).

These wide CSVs (country, year, iso_code, <indicators>...) come with a codebook that gives
title, description, unit and original source of each column. That metadata is stored with
every series, so the original publisher (e.g. Maddison Project Database for `gdp`) stays visible.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from chronos.config import Settings
from chronos.errors import DataError
from chronos.ingestion.http import FetchResult, fetch, read_local
from chronos.ingestion.registry import entry
from chronos.storage.repository import IngestBatch, ObsRow, SeriesMeta, SeriesPayload

RAW = "https://raw.githubusercontent.com/owid"


@dataclass(frozen=True)
class OwidDataset:
    key: str
    repo: str
    data_url: str
    codebook_url: str
    default_columns: tuple[str, ...]


DATASETS = {
    "co2": OwidDataset(
        "co2", "owid/co2-data", f"{RAW}/co2-data/master/owid-co2-data.csv",
        f"{RAW}/co2-data/master/owid-co2-codebook.csv",
        ("population", "gdp", "co2", "co2_per_capita", "co2_per_gdp", "co2_per_unit_energy", "cumulative_co2",
         "total_ghg", "temperature_change_from_ghg"),
    ),
    "energy": OwidDataset(
        "energy", "owid/energy-data", f"{RAW}/energy-data/master/owid-energy-data.csv",
        f"{RAW}/energy-data/master/owid-energy-codebook.csv",
        ("primary_energy_consumption", "energy_per_capita", "energy_per_gdp", "fossil_share_energy",
         "renewables_share_energy", "low_carbon_share_energy", "electricity_generation"),
    ),
}

ID_COLUMNS = ("country", "year", "iso_code")


def _codebook(fr: FetchResult) -> dict[str, dict[str, str]]:
    cb = pd.read_csv(io.BytesIO(fr.content), dtype=str).fillna("")
    return {r["column"]: {k: r.get(k, "") for k in ("title", "description", "unit", "source")} for _, r in cb.iterrows()}


def parse(data: FetchResult, codebook: FetchResult | None, ds: OwidDataset, columns: list[str] | None,
          entities: list[str] | None, derive_gdp_per_capita: bool = True) -> IngestBatch:
    df = pd.read_csv(io.BytesIO(data.content))
    missing_ids = [c for c in ID_COLUMNS[:2] if c not in df.columns]
    if missing_ids:
        raise DataError(f"OWID {ds.repo}: missing id columns {missing_ids}; file layout changed?")
    wanted = list(columns or ds.default_columns)
    unknown = [c for c in wanted if c not in df.columns]
    if unknown:
        raise DataError(f"Columns not in {ds.repo}: {unknown}")
    if entities:
        absent = sorted(set(entities) - set(df["country"].unique()))
        if absent:
            raise DataError(f"Entities not in {ds.repo}: {absent}")
        df = df[df["country"].isin(entities)]
    cb = _codebook(codebook) if codebook else {}
    this_year = datetime.now(UTC).year
    payloads: list[SeriesPayload] = []
    for entity, g in df.groupby("country", sort=True):
        code = next((c for c in g["iso_code"].dropna().unique()), None) if "iso_code" in g else None
        for col in wanted:
            sub = g[["year", col]].dropna()
            if sub.empty:
                continue
            meta = cb.get(col, {})
            payloads.append(SeriesPayload(
                SeriesMeta(indicator=col, entity=str(entity), source_id="owid", source_dataset=ds.repo,
                           source_variable=col, entity_code=code, title=meta.get("title") or col,
                           description=(meta.get("description", "") + " | Original source: " + meta.get("source", "")).strip(" |"),
                           unit=meta.get("unit") or None),
                [ObsRow(int(y), float(v), "projected" if int(y) > this_year else "observed")
                 for y, v in zip(sub["year"], sub[col], strict=True)],
            ))
        if derive_gdp_per_capita and ds.key == "co2" and {"gdp", "population"} <= set(g.columns):
            sub = g[["year", "gdp", "population"]].dropna()
            sub = sub[sub["population"] > 0]
            if not sub.empty:
                payloads.append(SeriesPayload(
                    SeriesMeta(indicator="gdp_per_capita", entity=str(entity), source_id="owid", source_dataset=ds.repo,
                               source_variable="gdp/population", entity_code=code,
                               title="GDP per capita (CHRONOS-derived)",
                               description="Derived by CHRONOS as OWID gdp (Maddison Project Database 2023) divided by "
                                           "OWID population (composite of several sources). Can differ from Maddison's "
                                           "own per-capita series because the population denominators differ.",
                               unit="international-$ in 2011 prices per person"),
                    [ObsRow(int(y), float(gdp) / float(p), "derived")
                     for y, gdp, p in zip(sub["year"], sub["gdp"], sub["population"], strict=True)],
                ))
    version = f"etag:{data.etag}" if data.etag else None
    return IngestBatch(entry("owid").info(), ds.repo, data.url, data.sha256, payloads, source_version=version,
                       notes=f"codebook sha256={codebook.sha256[:16] if codebook else 'n/a'}")


def fetch_dataset(settings: Settings, key: str, columns: list[str] | None = None, entities: list[str] | None = None,
                  data_file: str | None = None, codebook_file: str | None = None) -> IngestBatch:
    if key not in DATASETS:
        raise DataError(f"Unknown OWID GitHub dataset {key!r}; choose from {sorted(DATASETS)}")
    ds = DATASETS[key]
    data = read_local(data_file) if data_file else fetch(ds.data_url, settings, "owid", suffix=".csv")
    if codebook_file:
        codebook: FetchResult | None = read_local(codebook_file)
    elif data_file:
        codebook = None
    else:
        codebook = fetch(ds.codebook_url, settings, "owid", suffix=".csv")
    return parse(data, codebook, ds, columns, entities)
