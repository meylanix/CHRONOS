"""Importers for official downloadable files that cannot be fetched by API.

Maddison Project Database 2023, Penn World Table 11.0 and Correlates of War NMC are
distributed as files (DataverseNL / correlatesofwar.org). Their layouts are encoded below;
the parsers check required columns and fail loudly if a release changes the layout.
They were NOT tested against the real files in the development sandbox (hosts blocked).
"""

from __future__ import annotations

import io
from collections.abc import Mapping

import pandas as pd

from chronos.errors import DataError
from chronos.ingestion.http import FetchResult, read_local
from chronos.ingestion.registry import entry
from chronos.storage.repository import IngestBatch, ObsRow, SeriesMeta, SeriesPayload


def _find_sheet(fr: FetchResult, required: set[str], preferred: tuple[str, ...]) -> pd.DataFrame:
    if fr.url.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(fr.content))
        if not required <= set(df.columns):
            raise DataError(f"CSV lacks columns {sorted(required - set(df.columns))}")
        return df
    book = pd.ExcelFile(io.BytesIO(fr.content))
    order = [s for s in preferred if s in book.sheet_names] + [s for s in book.sheet_names if s not in preferred]
    for sheet in order:
        df = book.parse(sheet)
        if required <= set(df.columns):
            return df
    raise DataError(f"No sheet with columns {sorted(required)} (sheets: {book.sheet_names})")


def long_table(df: pd.DataFrame, entity_col: str, code_col: str | None, year_col: str,
               variables: Mapping[str, tuple[str, str]], source_id: str, dataset: str,
               prefix: str) -> list[SeriesPayload]:
    payloads = []
    for ent, g in df.groupby(entity_col, sort=True):
        code = str(g[code_col].iloc[0]) if code_col and code_col in g else None
        for var, (title, unit) in variables.items():
            if var not in g.columns:
                continue
            sub = g[[year_col, var]].dropna()
            sub = sub[pd.to_numeric(sub[var], errors="coerce").notna()]
            if sub.empty:
                continue
            payloads.append(SeriesPayload(
                SeriesMeta(f"{prefix}{var}", str(ent), source_id, dataset, var, code, title, None, unit),
                [ObsRow(int(y), float(v)) for y, v in zip(sub[year_col], sub[var], strict=True)]))
    return payloads


MADDISON_VARS = {"gdppc": ("Real GDP per capita (Maddison Project Database 2023)", "2011 int-$"),
                 "pop": ("Population (Maddison Project Database 2023)", "thousands")}

PWT_VARS = {
    "rgdpe": ("Expenditure-side real GDP at chained PPPs", "mil. 2021US$ (see PWT legend)"),
    "rgdpna": ("Real GDP at constant national prices", "mil. 2021US$ (see PWT legend)"),
    "pop": ("Population", "millions"),
    "emp": ("Persons engaged", "millions"),
    "hc": ("Human capital index (years of schooling and returns)", "index"),
    "ctfp": ("TFP level at current PPPs (USA=1)", "index"),
    "rtfpna": ("TFP at constant national prices (2021=1)", "index"),
    "labsh": ("Share of labour compensation in GDP", "share"),
    "csh_x": ("Share of merchandise exports at current PPPs", "share"),
}

COW_NMC_VARS = {
    "milex": ("Military expenditure", "thousands of current USD (pre-1914: GBP); see NMC codebook"),
    "milper": ("Military personnel", "thousands"),
    "irst": ("Iron and steel production", "thousands of tons"),
    "pec": ("Primary energy consumption", "thousands of coal-ton equivalents"),
    "tpop": ("Total population", "thousands"),
    "upop": ("Urban population", "thousands"),
    "cinc": ("Composite Index of National Capability", "share of system total"),
}


def import_maddison(path: str) -> IngestBatch:
    fr = read_local(path)
    df = _find_sheet(fr, {"countrycode", "country", "year"}, ("Full data",))
    if not ({"gdppc", "pop"} & set(df.columns)):
        raise DataError("Maddison sheet lacks gdppc/pop columns (2020 release used cgdppc/rgdpnapc: use import-csv)")
    payloads = long_table(df, "country", "countrycode", "year", MADDISON_VARS, "maddison", "MPD2023", "maddison_")
    return IngestBatch(entry("maddison").info(), "MPD2023", fr.url, fr.sha256, payloads, source_version="2023")


def import_pwt(path: str, version: str = "11.0") -> IngestBatch:
    fr = read_local(path)
    df = _find_sheet(fr, {"countrycode", "country", "year"}, ("Data",))
    payloads = long_table(df, "country", "countrycode", "year", PWT_VARS, "pwt", f"PWT{version}", "pwt_")
    return IngestBatch(entry("pwt").info(), f"PWT{version}", fr.url, fr.sha256, payloads, source_version=version)


def import_cow_nmc(path: str) -> IngestBatch:
    fr = read_local(path)
    df = _find_sheet(fr, {"stateabb", "ccode", "year"}, ())
    df = df.replace(-9, pd.NA)  # NMC codes missing values as -9
    version = str(df["version"].dropna().iloc[0]) if "version" in df and df["version"].notna().any() else None
    payloads = long_table(df, "stateabb", "ccode", "year", COW_NMC_VARS, "cow", "NMC", "cow_")
    return IngestBatch(entry("cow").info(), "NMC", fr.url, fr.sha256, payloads, source_version=version)
