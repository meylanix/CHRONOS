"""Reproducible experiment records: config + code version + data fingerprint + results hash."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from chronos import __version__
from chronos.storage.db import transaction, utc_now


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def git_commit() -> str | None:
    root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        if sha.returncode != 0:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return sha.stdout.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def results_hash(results: Any) -> str:
    return hashlib.sha256(_canon(results).encode()).hexdigest()


def record(con: sqlite3.Connection, kind: str, config: dict[str, Any], fingerprint: dict[str, Any],
           results: dict[str, Any], seed: int | None = None, notes: str | None = None) -> str:
    base = hashlib.sha256(_canon({"kind": kind, "config": config, "data": fingerprint}).encode()).hexdigest()[:12]
    now = utc_now()
    exp_id = f"{base}-{now.replace(':', '').replace('-', '')[:15]}"
    with transaction(con):
        con.execute(
            """INSERT OR REPLACE INTO experiment(experiment_id,kind,created_utc,chronos_version,git_commit,config_json,
                 data_fingerprint,seed,results_json,results_sha256,notes) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (exp_id, kind, now, __version__, git_commit(), _canon(config), _canon(fingerprint), seed,
             json.dumps(results, default=str), results_hash(results), notes))
    return exp_id


def list_experiments(con: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    return [dict(r) for r in con.execute(
        """SELECT experiment_id, kind, created_utc, chronos_version, git_commit, results_sha256, notes
           FROM experiment ORDER BY created_utc DESC LIMIT ?""", (limit,))]


def get_experiment(con: sqlite3.Connection, exp_id: str) -> dict[str, Any] | None:
    r = con.execute("SELECT * FROM experiment WHERE experiment_id=? OR experiment_id LIKE ?",
                    (exp_id, exp_id + "%")).fetchone()
    if r is None:
        return None
    d = dict(r)
    for k in ("config_json", "data_fingerprint", "results_json"):
        d[k] = json.loads(d[k])
    return d
