"""V-Dem Country-Year dataset (official vdemdata GitHub package data, or an official CSV).

Loading the full RData needs ~4 GB of RAM (28k rows x 4.6k columns). Pass --file with the
official CSV from v-dem.net to read only the selected columns instead.
"""

from __future__ import annotations

import io
import re
import tempfile
from pathlib import Path

import pandas as pd

from chronos.config import Settings
from chronos.errors import DataError, SourceError
from chronos.ingestion.http import FetchResult, fetch, read_local
from chronos.ingestion.registry import entry
from chronos.storage.repository import IngestBatch, ObsRow, SeriesMeta, SeriesPayload

RDATA_URL = "https://raw.githubusercontent.com/vdeminstitute/vdemdata/master/data/vdem.RData"
DESCRIPTION_URL = "https://raw.githubusercontent.com/vdeminstitute/vdemdata/master/DESCRIPTION"

VARIABLES: dict[str, tuple[str, str]] = {
    "v2x_polyarchy": ("Electoral democracy index", "index 0-1"),
    "v2x_libdem": ("Liberal democracy index", "index 0-1"),
    "v2x_regime": ("Regimes of the World (0 closed autocracy .. 3 liberal democracy)", "ordinal 0-3"),
    "v2x_rule": ("Rule of law index", "index 0-1"),
    "v2x_corr": ("Political corruption index", "index 0-1"),
    "v2x_freexp_altinf": ("Freedom of expression and alternative sources of information", "index 0-1"),
    "e_pelifeex": ("Life expectancy (external variable, third-party source)", "years"),
    "e_gdppc": ("GDP per capita (external variable: Fariss et al. latent estimates)", "thousands 2011 USD (see codebook)"),
    "e_pop": ("Population (external variable: Fariss et al. latent estimates)", "thousands (see codebook)"),
    "e_miurbani": ("Urbanization rate (external variable, Clio-Infra/Maddison)", "share"),
    "e_peaveduc": ("Average years of education (external variable)", "years"),
    "e_civil_war": ("Civil war (external variable)", "0/1"),
    "e_miinteco": ("Armed conflict, international (external variable)", "0/1"),
}


def _frame_from(fr: FetchResult, columns: list[str]) -> pd.DataFrame:
    keep = ["country_name", "country_text_id", "year", *columns]
    if fr.url.lower().endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(fr.content), usecols=lambda c: c in keep, low_memory=False)
        except ValueError as exc:
            raise DataError(f"V-Dem CSV lacks expected columns: {exc}") from exc
    try:
        import pyreadr
    except ImportError as exc:
        raise SourceError("Reading V-Dem RData needs the optional dependency pyreadr (pip install pyreadr)") from exc
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vdem.RData"
        path.write_bytes(fr.content)
        result = pyreadr.read_r(str(path))
    if "vdem" not in result:
        raise DataError(f"RData does not contain object 'vdem' (found {list(result)})")
    df = result["vdem"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise DataError(f"V-Dem data lacks columns {missing}")
    return df[keep].copy()


def parse(fr: FetchResult, columns: list[str] | None, version: str | None) -> IngestBatch:
    cols = columns or list(VARIABLES)
    df = _frame_from(fr, cols)
    payloads = []
    for (name, code), g in df.groupby(["country_name", "country_text_id"], sort=True):
        for col in cols:
            sub = g[["year", col]].dropna()
            if sub.empty:
                continue
            title, unit = VARIABLES.get(col, (col, ""))
            payloads.append(SeriesPayload(
                SeriesMeta(col, str(name), "vdem", "V-Dem Country-Year", col, str(code), title,
                           "V-Dem; e_* variables are third-party data distributed with V-Dem (see V-Dem codebook).",
                           unit or None),
                [ObsRow(int(y), float(v)) for y, v in zip(sub["year"], sub[col], strict=True)]))
    return IngestBatch(entry("vdem").info(), "V-Dem Country-Year", fr.url, fr.sha256, payloads, source_version=version)


def fetch_vdem(settings: Settings, columns: list[str] | None = None, file: str | None = None) -> IngestBatch:
    if file:
        return parse(read_local(file), columns, version=None)
    desc = fetch(DESCRIPTION_URL, settings, "vdem", suffix=".txt", keep_raw=False).text()
    m = re.search(r"^Version:\s*(\S+)", desc, flags=re.M)
    fr = fetch(RDATA_URL, settings, "vdem", suffix=".RData")
    return parse(fr, columns, version=f"vdemdata {m.group(1)}" if m else None)
