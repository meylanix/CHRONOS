"""Immutable annual time series and panels with explicit truncation (leakage guard).

Forecasting code receives `series.truncate(origin)`, a copy that physically contains
no observation after the origin. Tests additionally verify that forecasts are invariant
to any modification of post-origin data (see tests/unit/test_leakage.py).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimeSeries:
    years: np.ndarray
    values: np.ndarray
    name: str = ""
    entity: str = ""
    unit: str | None = None
    data_kind: str = "real"
    meta: Mapping[str, Any] = field(default_factory=dict)
    # status per year ('observed', 'imputed', 'filler', ...); same length as years
    status: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        years = np.asarray(self.years, dtype=np.int64)
        values = np.asarray(self.values, dtype=float)
        if years.shape != values.shape or years.ndim != 1:
            raise ValueError("years and values must be 1-D arrays of equal length")
        order = np.argsort(years, kind="stable")
        years, values = years[order], values[order]
        if len(years) and np.any(np.diff(years) == 0):
            raise ValueError("duplicate years in series")
        if not np.all(np.isfinite(values)):
            raise ValueError("series values must be finite (drop missing values before construction)")
        status = self.status
        if status is not None:
            if len(status) != len(order):
                raise ValueError("status length mismatch")
            status = tuple(status[i] for i in order)
        object.__setattr__(self, "years", years)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "status", status)

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[int, float]], **kw: Any) -> TimeSeries:
        items = sorted((int(y), float(v)) for y, v in pairs)
        return cls(np.array([y for y, _ in items], dtype=np.int64), np.array([v for _, v in items]), **kw)

    @classmethod
    def from_mapping(cls, mapping: Mapping[int, float], **kw: Any) -> TimeSeries:
        return cls.from_pairs(mapping.items(), **kw)

    def __len__(self) -> int:
        return int(self.years.size)

    @property
    def first_year(self) -> int:
        return int(self.years[0])

    @property
    def last_year(self) -> int:
        return int(self.years[-1])

    def as_dict(self) -> dict[int, float]:
        return {int(y): float(v) for y, v in zip(self.years, self.values, strict=True)}

    def get(self, year: int) -> float | None:
        idx = np.searchsorted(self.years, year)
        if idx < len(self.years) and self.years[idx] == year:
            return float(self.values[idx])
        return None

    def at(self, year: int) -> float:
        """Value at `year`; raises KeyError if the year is not observed (use get() for optional access)."""
        v = self.get(year)
        if v is None:
            raise KeyError(f"{self.name}/{self.entity}: no observation for {year}")
        return v

    def has_window(self, start: int, end: int, step: int = 1) -> bool:
        """True if every year start..end (inclusive, at the series step) is present."""
        needed = np.arange(start, end + 1, step)
        return bool(np.all(np.isin(needed, self.years)))

    def truncate(self, origin: int) -> TimeSeries:
        """Copy with observations at years <= origin only."""
        mask = self.years <= origin
        status = tuple(s for s, m in zip(self.status, mask, strict=True) if m) if self.status else None
        return TimeSeries(self.years[mask].copy(), self.values[mask].copy(), self.name, self.entity,
                          self.unit, self.data_kind, dict(self.meta), status)

    def between(self, start: int, end: int) -> TimeSeries:
        mask = (self.years >= start) & (self.years <= end)
        status = tuple(s for s, m in zip(self.status, mask, strict=True) if m) if self.status else None
        return TimeSeries(self.years[mask].copy(), self.values[mask].copy(), self.name, self.entity,
                          self.unit, self.data_kind, dict(self.meta), status)

    @property
    def is_positive(self) -> bool:
        return bool(len(self) and np.all(self.values > 0))

    def step(self) -> int:
        """Most common spacing between observations (1 for annual, 100 for Seshat centuries)."""
        if len(self) < 2:
            return 1
        diffs = np.diff(self.years)
        vals, counts = np.unique(diffs, return_counts=True)
        return int(vals[np.argmax(counts)])

    def tail_regular_run(self, step: int | None = None) -> TimeSeries:
        """Consecutive observations (at `step`) ending at the last observation."""
        if len(self) < 2:
            return self
        st = step or self.step()
        diffs = np.diff(self.years)
        bad = np.where(diffs != st)[0]
        lo = int(bad[-1] + 1) if len(bad) else 0
        return TimeSeries(self.years[lo:], self.values[lo:], self.name, self.entity, self.unit,
                          self.data_kind, dict(self.meta), self.status[lo:] if self.status else None)

    def longest_regular_run(self) -> TimeSeries:
        """Longest run of consecutive observations at the dominant step."""
        if len(self) < 2:
            return self
        st = self.step()
        breaks = np.where(np.diff(self.years) != st)[0]
        bounds = np.concatenate(([0], breaks + 1, [len(self)]))
        lengths = np.diff(bounds)
        k = int(np.argmax(lengths))
        lo, hi = int(bounds[k]), int(bounds[k + 1])
        return TimeSeries(self.years[lo:hi], self.values[lo:hi], self.name, self.entity, self.unit,
                          self.data_kind, dict(self.meta), self.status[lo:hi] if self.status else None)


@dataclass(frozen=True)
class Panel:
    """Several entities' series for one indicator (e.g. GDP per capita for 160 countries)."""

    indicator: str
    series: Mapping[str, TimeSeries]

    def truncate(self, origin: int) -> Panel:
        out = {}
        for k, s in self.series.items():
            t = s.truncate(origin)
            if len(t):
                out[k] = t
        return Panel(self.indicator, out)

    def entities(self) -> list[str]:
        return sorted(self.series)

    def value(self, entity: str, year: int) -> float | None:
        s = self.series.get(entity)
        return s.get(year) if s is not None else None

    @property
    def data_kinds(self) -> set[str]:
        return {s.data_kind for s in self.series.values()}
