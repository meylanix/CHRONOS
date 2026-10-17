"""Seshat Global History Databank, Equinox-2020 release (official GitHub copy).

Two datasets are created so that imputed and coded values are never confused:
  seshat_<cc>      from sheet ImpSCDat (single imputation; Seshat: "use only as predictor
                   variables"). Status: 'imputed' where AggrSCWarAgriRelig has no value,
                   'filler' where uniq == 'n' (repeated value, not new data), else 'observed'.
  seshat_raw_<cc>  from sheet AggrSCWarAgriRelig, coded values only (no imputation).
Plus seshat_spc / seshat_miltech from SPC_MilTech (computed from the imputed data).
Time resolution: one value per NGA and century. Negative years are BCE.
"""

from __future__ import annotations

import io

import pandas as pd

from chronos.config import Settings
from chronos.errors import DataError
from chronos.ingestion.http import FetchResult, fetch, read_local
from chronos.ingestion.registry import entry
from chronos.storage.repository import IngestBatch, ObsRow, SeriesMeta, SeriesPayload

EQUINOX_URL = "https://raw.githubusercontent.com/seshatdb/Equinox_Data/main/Equinox_on_GitHub_June9_2022.xlsx"

# Complexity characteristics (Turchin et al. 2018, PNAS 115(2):E144-E151). Scale CCs are log10.
CCS: dict[str, tuple[str, str]] = {
    "Pop": ("Polity population", "log10(people)"),
    "Terr": ("Polity territory", "log10(km^2)"),
    "Cap": ("Population of the largest settlement", "log10(people)"),
    "Hier": ("Hierarchical complexity (mean of hierarchy levels)", "levels"),
    "Gov": ("Government (share of governance characteristics present)", "index 0-1"),
    "Infra": ("Infrastructure (share of characteristics present)", "index 0-1"),
    "Info": ("Information systems (share of characteristics present)", "index 0-1"),
    "Money": ("Most sophisticated monetary instrument (normalised)", "index 0-1"),
}


def _dedupe(df: pd.DataFrame, warnings: list[str], sheet: str) -> pd.DataFrame:
    dup = df.duplicated(["NGA", "Time"], keep=False)
    if dup.any():
        d = df[dup].copy()
        d["_rank"] = (d["uniq"] != "y").astype(int)
        keep = d.sort_values(["NGA", "Time", "_rank"]).drop_duplicates(["NGA", "Time"], keep="first").index
        dropped = d.index.difference(keep)
        for i in dropped:
            warnings.append(f"{sheet}: {df.at[i, 'NGA']} {df.at[i, 'Time']}: two polities in one century; "
                            f"kept one row, dropped polity {df.at[i, 'PolID']}")
        df = df.drop(index=dropped)
    return df


def parse(fr: FetchResult) -> IngestBatch:
    try:
        book = pd.ExcelFile(io.BytesIO(fr.content))
        imp = book.parse("ImpSCDat")
        agg = book.parse("AggrSCWarAgriRelig")
        spc = book.parse("SPC_MilTech")
        polities = book.parse("Polities")
    except (ValueError, KeyError) as exc:
        raise DataError(f"Equinox workbook layout not recognised: {exc}") from exc
    for name, df, cols in (("ImpSCDat", imp, ["NGA", "PolID", "Time", *CCS, "uniq", "Dupl"]),
                           ("SPC_MilTech", spc, ["NGA", "PolID", "Time", "SPC", "MilTech", "uniq"])):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise DataError(f"Sheet {name} lacks columns {missing}")
    warnings: list[str] = []
    imp = _dedupe(imp, warnings, "ImpSCDat")
    spc = _dedupe(spc, warnings, "SPC_MilTech")
    agg = _dedupe(agg, [], "AggrSCWarAgriRelig")
    names = dict(zip(polities["PolID"], polities["PolName"], strict=False))
    agg_idx = agg.set_index(["NGA", "PolID", "Time"])
    payloads: list[SeriesPayload] = []

    def note(row: pd.Series) -> str:
        n = f"polity {row['PolID']} ({names.get(row['PolID'], '?')})"
        return n + ("; polity also controls another NGA (Dupl=y)" if row.get("Dupl") == "y" else "")

    for nga, g in imp.groupby("NGA", sort=True):
        g = g.sort_values("Time")
        for cc, (title, unit) in CCS.items():
            rows = []
            for _, r in g.iterrows():
                key = (r["NGA"], r["PolID"], r["Time"])
                coded = agg_idx[cc].get(key) if cc in agg_idx.columns and key in agg_idx.index else None
                if r["uniq"] == "n":
                    status = "filler"
                elif coded is None or pd.isna(coded):
                    status = "imputed"
                else:
                    status = "observed"
                rows.append(ObsRow(int(r["Time"]), float(r[cc]), status, note(r)))
            payloads.append(SeriesPayload(
                SeriesMeta(f"seshat_{cc.lower()}", str(nga), "seshat", "Equinox2020/ImpSCDat", cc, None,
                           f"{title} [imputed set]", "Seshat Equinox-2020 ImpSCDat (single imputation; use as "
                           "predictor only).", unit, "century"), rows))
    for nga, g in agg.groupby("NGA", sort=True):
        for cc, (title, unit) in CCS.items():
            if cc not in g.columns:
                continue
            sub = g[["Time", cc, "uniq", "PolID", "Dupl"]].dropna(subset=[cc]).sort_values("Time")
            if sub.empty:
                continue
            payloads.append(SeriesPayload(
                SeriesMeta(f"seshat_raw_{cc.lower()}", str(nga), "seshat", "Equinox2020/AggrSCWarAgriRelig", cc, None,
                           f"{title} [coded, not imputed]", "Seshat Equinox-2020 aggregated complexity "
                           "characteristics without imputation.", unit, "century"),
                [ObsRow(int(r["Time"]), float(r[cc]), "filler" if r["uniq"] == "n" else "observed", note(r))
                 for _, r in sub.iterrows()]))
    for nga, g in spc.groupby("NGA", sort=True):
        g = g.sort_values("Time")
        for col, title, unit in (("SPC", "Social complexity (PC1 of complexity characteristics)", "PC score"),
                                 ("MilTech", "Military technology (sum of 6 warfare characteristics)", "count")):
            payloads.append(SeriesPayload(
                SeriesMeta(f"seshat_{col.lower()}", str(nga), "seshat", "Equinox2020/SPC_MilTech", col, None, title,
                           "Computed by Seshat from the imputed data (ImpSCDat).", unit, "century"),
                [ObsRow(int(r["Time"]), float(r[col]), "filler" if r["uniq"] == "n" else "derived", note(r))
                 for _, r in g.iterrows()]))
    return IngestBatch(entry("seshat").info(), "Equinox2020", fr.url, fr.sha256, payloads,
                       source_version="Equinox-2020 (GitHub file of 2022-06-09)", warnings=warnings)


def fetch_equinox(settings: Settings, file: str | None = None) -> IngestBatch:
    fr = read_local(file) if file else fetch(EQUINOX_URL, settings, "seshat", suffix=".xlsx")
    return parse(fr)
