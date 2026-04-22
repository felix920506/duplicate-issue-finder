from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class CachedRun:
    run_id: str
    issue_url: str
    result_markdown: str
    actions_html: str
    logs: str
    download_path: str | None
    created_at: float


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
        download_path: str | None,
    ) -> None:
        run_id = self._build_cache_key(issue_url)
        cached = CachedRun(
            run_id=run_id,
            issue_url=issue_url,
            result_markdown=result_markdown,
            actions_html=actions_html,
            logs=logs,
            download_path=download_path,
            created_at=time.time(),
        )
        with self._lock:
            self._runs[run_id] = cached
            self._runs.move_to_end(run_id)
            while len(self._runs) > self.max_runs:
                self._runs.popitem(last=False)

    def get(self, run_id: str) -> CachedRun | None:
        with self._lock:
            cached = self._runs.get(run_id)
            if cached is not None:
                self._runs.move_to_end(run_id)
            return cached

    def list_recent(self) -> list[CachedRun]:
        with self._lock:
            runs = list(self._runs.values())
        runs.reverse()
        return runs

    @staticmethod
    def format_label(cached_run: CachedRun) -> str:
        timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(cached_run.created_at)
        )
        return f"{timestamp} | {cached_run.issue_url}"

    @staticmethod
    def _build_cache_key(issue_url: str) -> str:
        return f"{int(time.time() * 1000)}::{issue_url}"
