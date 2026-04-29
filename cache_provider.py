from __future__ import annotations

import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class CachedRun:
    run_id: str
    issue_url: str
    result_markdown: str
    actions_html: str
    logs: str
    status: str
    created_at: float


class RunCache(Protocol):
    def store(
        self,
        issue_url: str,
        result_markdown: str,
        actions_html: str,
        logs: str,
        status: str,
    ) -> None: ...

    def get(self, run_id: str) -> CachedRun | None: ...

    def list_recent(self) -> list[CachedRun]: ...

    def get_latest_for_url(self, issue_url: str) -> CachedRun | None: ...


class InMemoryRunCache:
    def __init__(self, max_runs: int = 50) -> None:
        self.max_runs = max_runs
        self._runs: OrderedDict[str, CachedRun] = OrderedDict()
        self._lock = threading.Lock()

    def store(
        self,
        issue_url: str,
        result_markdown: str,
        actions_html: str,
        logs: str,
        status: str,
    ) -> None:
        run_id = build_cache_key(issue_url)
        cached = CachedRun(
            run_id=run_id,
            issue_url=issue_url,
            result_markdown=result_markdown,
            actions_html=actions_html,
            logs=logs,
            status=status,
            created_at=time.time(),
        )
        with self._lock:
            self._runs[run_id] = cached
            self._runs.move_to_end(run_id)
            while len(self._runs) > self.max_runs:
                self._runs.popitem(last=False)

    def get(self, run_id: str) -> CachedRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_recent(self) -> list[CachedRun]:
        with self._lock:
            runs = list(self._runs.values())
        runs.reverse()
        return runs

    def get_latest_for_url(self, issue_url: str) -> CachedRun | None:
        for run in self.list_recent():
            if run.issue_url == issue_url:
                return run
        return None


class SQLiteRunCache:
    def __init__(self, path: str, max_runs: int = 50) -> None:
        self.max_runs = max_runs
        self.db_path = resolve_cache_db_path(path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cached_runs (
                    run_id TEXT PRIMARY KEY,
                    issue_url TEXT NOT NULL,
                    result_markdown TEXT NOT NULL,
                    actions_html TEXT NOT NULL,
                    logs TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'Unknown',
                    created_at REAL NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_cached_runs_created_at ON cached_runs(created_at DESC)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_cached_runs_issue_url ON cached_runs(issue_url, created_at DESC)"
            )
            try:
                self._connection.execute(
                    "ALTER TABLE cached_runs ADD COLUMN status TEXT NOT NULL DEFAULT 'Unknown'"
                )
            except sqlite3.OperationalError:
                pass
            self._connection.commit()

    def store(
        self,
        issue_url: str,
        result_markdown: str,
        actions_html: str,
        logs: str,
        status: str,
    ) -> None:
        run_id = build_cache_key(issue_url)
        created_at = time.time()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO cached_runs (run_id, issue_url, result_markdown, actions_html, logs, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, issue_url, result_markdown, actions_html, logs, status, created_at),
            )
            overflow = self._connection.execute(
                "SELECT COUNT(*) AS count FROM cached_runs"
            ).fetchone()["count"] - self.max_runs
            if overflow > 0:
                self._connection.execute(
                    """
                    DELETE FROM cached_runs
                    WHERE run_id IN (
                        SELECT run_id FROM cached_runs
                        ORDER BY created_at ASC
                        LIMIT ?
                    )
                    """,
                    (overflow,),
                )
            self._connection.commit()

    def get(self, run_id: str) -> CachedRun | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT run_id, issue_url, result_markdown, actions_html, logs, status, created_at
                FROM cached_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        return row_to_cached_run(row) if row is not None else None

    def list_recent(self) -> list[CachedRun]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT run_id, issue_url, result_markdown, actions_html, logs, status, created_at
                FROM cached_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (self.max_runs,),
            ).fetchall()
        return [row_to_cached_run(row) for row in rows]

    def get_latest_for_url(self, issue_url: str) -> CachedRun | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT run_id, issue_url, result_markdown, actions_html, logs, status, created_at
                FROM cached_runs
                WHERE issue_url = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (issue_url,),
            ).fetchone()
        return row_to_cached_run(row) if row is not None else None


def create_run_cache(
    provider: str,
    path: str,
    max_runs: int = 50,
) -> RunCache:
    normalized = provider.strip().lower()
    if normalized == "memory":
        return InMemoryRunCache(max_runs=max_runs)
    if normalized == "sqlite":
        return SQLiteRunCache(path=path, max_runs=max_runs)
    raise ValueError(f"Unsupported cache provider: {provider}")


def resolve_cache_db_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists() and candidate.is_dir():
        return candidate / "run-cache.sqlite3"
    if candidate.suffix:
        return candidate
    return candidate / "run-cache.sqlite3"


def row_to_cached_run(row: sqlite3.Row) -> CachedRun:
    return CachedRun(
        run_id=row["run_id"],
        issue_url=row["issue_url"],
        result_markdown=row["result_markdown"],
        actions_html=row["actions_html"],
        logs=row["logs"],
        status=row["status"],
        created_at=float(row["created_at"]),
    )


def format_cached_run_label(cached_run: CachedRun) -> str:
    timestamp = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(cached_run.created_at)
    )
    return f"{timestamp} | {cached_run.status} | {cached_run.issue_url}"


def build_cache_key(issue_url: str) -> str:
    return f"{int(time.time() * 1000)}::{issue_url}"
