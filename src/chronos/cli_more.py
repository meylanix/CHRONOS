"""CLI commands for the later phases (state model, scenarios, research dossiers, UI)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def register(add: Callable[..., Any]) -> None:
    return None
