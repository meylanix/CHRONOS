"""Shared helpers to pull stored panels into pandas, joined across sources by ISO3 code."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from chronos.errors import DataError


def panel_frame(con: sqlite3.Connection, indicator: str, source_id: str, include_filler: bool = False) -> pd.DataFrame:
    """Latest-vintage observations of one indicator: columns entity, entity_code, year, value, status."""
    q = """SELECT s.entity, s.entity_code, o.year, o.value, o.status FROM series s
           JOIN series_vintage sv ON sv.series_id = s.series_id
           JOIN observation o ON o.series_id = s.series_id AND o.vintage_id = sv.rows_vintage_id
           WHERE s.indicator=? AND s.source_id=?
             AND sv.vintage_id = (SELECT MAX(vintage_id) FROM series_vintage WHERE series_id = s.series_id)"""
    if not include_filler:
        q += " AND o.status <> 'filler'"
    rows = con.execute(q, (indicator, source_id)).fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=["entity", "entity_code", "year", "value", "status"])


@dataclass(frozen=True)
class VarSpec:
    name: str
    indicator: str
    source_id: str
    transform: str = "level"  # level | log | growth (annual log change) | diff


def merged_panel(con: sqlite3.Connection, specs: Sequence[VarSpec], key: str = "entity_code",
                 min_year: int | None = None, max_year: int | None = None) -> pd.DataFrame:
    """Long panel (entity key, year) x variables. key='entity_code' joins sources by ISO3."""
    frames = []
    names: dict[str, str] = {}
    for sp in specs:
        df = panel_frame(con, sp.indicator, sp.source_id)
        if df.empty:
            raise DataError(f"No data for {sp.indicator} from {sp.source_id}")
        if key == "entity_code":
            df = df[df["entity_code"].notna() & ~df["entity_code"].astype(str).str.startswith("OWID_")]
        for c, n in zip(df[key], df["entity"], strict=False):
            names.setdefault(c, n)
        s = df.set_index([key, "year"])["value"].astype(float).sort_index()
        s = s[~s.index.duplicated()]
        if sp.transform in ("log", "growth"):
            s = np.log(s.where(s > 0))
        if sp.transform in ("growth", "diff"):
            prev = s.copy()
            prev.index = pd.MultiIndex.from_arrays([prev.index.get_level_values(0), prev.index.get_level_values(1) + 1])
            s = s - prev.reindex(s.index)
        frames.append(s.rename(sp.name))
    out = pd.concat(frames, axis=1).reset_index().rename(columns={key: "unit", "year": "year"})
    if min_year is not None:
        out = out[out["year"] >= min_year]
    if max_year is not None:
        out = out[out["year"] <= max_year]
    out["unit_name"] = out["unit"].map(names)
    return out
