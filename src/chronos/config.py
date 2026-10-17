"""Settings: defaults < TOML file < environment variables < explicit CLI flags.

TOML file location: $CHRONOS_CONFIG, else ./config/chronos.toml if it exists.
Environment variables (all optional):
  CHRONOS_DB, CHRONOS_CACHE_DIR, CHRONOS_HTTP_TIMEOUT, CHRONOS_HTTP_RETRIES,
  CHRONOS_USER_AGENT, CHRONOS_LOG_LEVEL, CHRONOS_LOG_FORMAT, CHRONOS_REFERENCE_YEAR.
No secrets are required by any implemented connector; none are read from files in the repo.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from chronos.errors import ConfigError


@dataclass(frozen=True)
class Settings:
    db_path: str = "chronos.sqlite"
    cache_dir: str = "data/cache"
    http_timeout: float = 60.0
    http_retries: int = 3
    user_agent: str = "CHRONOS/0.2 (historical-research; +https://github.com/meylanix/chronos)"
    log_level: str = "INFO"
    log_format: str = "text"  # "text" or "json"
    # The "present" for staleness checks of the civilization-state model.
    reference_year: int = 2026
    scenario_dir: str = "scenarios"
    indicator_catalog: str = "config/indicators.toml"
    extra: dict[str, Any] = field(default_factory=dict)


_ENV_MAP: dict[str, tuple[str, type]] = {
    "CHRONOS_DB": ("db_path", str),
    "CHRONOS_CACHE_DIR": ("cache_dir", str),
    "CHRONOS_HTTP_TIMEOUT": ("http_timeout", float),
    "CHRONOS_HTTP_RETRIES": ("http_retries", int),
    "CHRONOS_USER_AGENT": ("user_agent", str),
    "CHRONOS_LOG_LEVEL": ("log_level", str),
    "CHRONOS_LOG_FORMAT": ("log_format", str),
    "CHRONOS_REFERENCE_YEAR": ("reference_year", int),
    "CHRONOS_SCENARIO_DIR": ("scenario_dir", str),
    "CHRONOS_INDICATOR_CATALOG": ("indicator_catalog", str),
}


def _from_toml(path: Path) -> dict[str, Any]:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Cannot read config {path}: {exc}") from exc
    section = raw.get("chronos", raw)
    known = {f.name for f in fields(Settings)} - {"extra"}
    values = {k: v for k, v in section.items() if k in known}
    values["extra"] = {k: v for k, v in raw.items() if k != "chronos"}
    return values


def load_settings(config_path: str | None = None, **overrides: Any) -> Settings:
    settings = Settings()
    path_str = config_path or os.environ.get("CHRONOS_CONFIG")
    path = Path(path_str) if path_str else Path("config/chronos.toml")
    if path_str and not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    if path.exists():
        settings = replace(settings, **_from_toml(path))
    env_values: dict[str, Any] = {}
    for env_key, (attr, cast) in _ENV_MAP.items():
        if env_key in os.environ:
            try:
                env_values[attr] = cast(os.environ[env_key])
            except ValueError as exc:
                raise ConfigError(f"Invalid value for {env_key}: {os.environ[env_key]!r}") from exc
    settings = replace(settings, **env_values)
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(settings, **clean) if clean else settings
