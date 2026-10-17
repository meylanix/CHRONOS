"""SQLite connection and ordered migrations.

Migrations are numbered SQL files in storage/migrations plus Python steps. Each one is
applied once inside a transaction and recorded in schema_migrations. The schema avoids
SQLite-only features where practical, so moving to PostgreSQL means porting these files.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

from chronos.logsetup import get_logger

log = get_logger("storage")

PythonMigration = Callable[[sqlite3.Connection], None]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _sql_migrations() -> list[tuple[int, str, str]]:
    pkg = resources.files("chronos.storage") / "migrations"
    out = []
    for entry in pkg.iterdir():
        name = entry.name
        if name.endswith(".sql") and name[:4].isdigit():
            out.append((int(name[:4]), name, entry.read_text(encoding="utf-8")))
    return sorted(out)


def _python_migrations() -> list[tuple[int, str, PythonMigration]]:
    from chronos.storage.legacy import import_v01_tables

    return [(1001, "1001_import_v01_tables", import_v01_tables)]


def migrate(con: sqlite3.Connection) -> list[str]:
    con.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_utc TEXT NOT NULL)"
    )
    done = {row[0] for row in con.execute("SELECT version FROM schema_migrations")}
    applied: list[str] = []
    steps: list[tuple[int, str, str | PythonMigration]] = [*_sql_migrations(), *_python_migrations()]
    for version, name, body in sorted(steps, key=lambda s: s[0]):
        if version in done:
            continue
        try:
            con.execute("BEGIN")
            if isinstance(body, str):
                for statement in _split_sql(body):
                    con.execute(statement)
            else:
                body(con)
            con.execute("INSERT INTO schema_migrations VALUES (?,?,?)", (version, name, utc_now()))
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        applied.append(name)
        log.info("applied migration", extra={"ctx": {"migration": name}})
    return applied


def _split_sql(script: str) -> list[str]:
    # Migration files contain no semicolons inside literals, so a plain split is safe.
    cleaned = "\n".join(line for line in script.splitlines() if not line.strip().startswith("--"))
    return [s.strip() for s in cleaned.split(";") if s.strip()]


def connect(path: str | Path, migrate_schema: bool = True) -> sqlite3.Connection:
    target = Path(path)
    if str(target) != ":memory:":
        target.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(target), isolation_level=None)  # explicit transactions
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    if migrate_schema:
        migrate(con)
    return con


@contextmanager
def open_db(path: str | Path) -> Iterator[sqlite3.Connection]:
    """Context manager that really closes the connection (sqlite3's own one only commits)."""
    con = connect(path)
    try:
        yield con
    finally:
        con.close()


@contextmanager
def transaction(con: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    if con.in_transaction:
        # Nested use: run inside the caller's transaction via a savepoint.
        con.execute("SAVEPOINT nested")
        try:
            yield con
            con.execute("RELEASE nested")
        except Exception:
            con.execute("ROLLBACK TO nested")
            con.execute("RELEASE nested")
            raise
        return
    con.execute("BEGIN")
    try:
        yield con
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
