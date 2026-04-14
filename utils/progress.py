"""Persistent progress reporting for batch runs.

Writes a small JSON snapshot to disk every N patents so that a run that
crashes or is interrupted can be inspected (and the user knows where they
stand) without having to parse the full output CSV.

Resume itself is handled in main.py by reading the existing output CSV's
publication_number column and skipping rows already written. This module
just records progress; the file it produces is informational.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("progress");


def _short_err(err: str | None) -> str:
    if not err:
        return "no_addresses_found";
    first_line = str(err).strip().splitlines()[0];
    return first_line[:200];


class ProgressReport:
    """Append-only progress accumulator with periodic disk flush.

    Not thread-safe — must be called from the single output coroutine.
    """

    def __init__(
        self,
        path: Path,
        run_id: str,
        total: int,
        *,
        already_done: int = 0,
        flush_every: int = 25,
    ) -> None:
        self.path = path;
        self.run_id = run_id;
        self.total = total;
        self.already_done = already_done;
        self.flush_every = max(1, int(flush_every));

        self.started_at = time.time();
        self.completed = 0;
        self.successes = 0;
        self.failures = 0;
        self.error_counts: Counter[str] = Counter();
        self.recent_failures: list[dict] = [];
        self.last_patent_id = "";

        self.path.parent.mkdir(parents=True, exist_ok=True);
        self.flush();  # initial snapshot so the file always exists

    def record(self, patent_id: str, *, success: bool, error: str | None) -> None:
        self.completed += 1;
        self.last_patent_id = patent_id;
        if success:
            self.successes += 1;
        else:
            self.failures += 1;
            err_key = _short_err(error);
            self.error_counts[err_key] += 1;
            self.recent_failures.append({
                "patent_id": patent_id,
                "error": err_key,
                "ts": datetime.now(timezone.utc).isoformat(),
            });
            self.recent_failures = self.recent_failures[-20:];

        if self.completed % self.flush_every == 0 or self.completed >= self.total:
            self.flush();

    def flush(self) -> None:
        elapsed = max(time.time() - self.started_at, 1e-9);
        rate_per_s = self.completed / elapsed if self.completed else 0.0;
        remaining = max(self.total - self.completed, 0);
        eta_s = (remaining / rate_per_s) if rate_per_s > 0 else None;
        eta_utc = (
            datetime.fromtimestamp(time.time() + eta_s, tz=timezone.utc).isoformat()
            if eta_s is not None
            else None
        );

        data = {
            "run_id": self.run_id,
            "started_at": datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_target_this_run": self.total,
            "already_done_before_resume": self.already_done,
            "completed_this_run": self.completed,
            "successes_this_run": self.successes,
            "failures_this_run": self.failures,
            "remaining": remaining,
            "progress_pct": round(self.completed / self.total * 100, 2) if self.total else 100.0,
            "elapsed_s": round(elapsed, 1),
            "throughput_per_min": round(rate_per_s * 60, 2),
            "eta_seconds": round(eta_s, 1) if eta_s is not None else None,
            "eta_utc": eta_utc,
            "last_patent_id": self.last_patent_id,
            "failure_breakdown": dict(self.error_counts.most_common(20)),
            "recent_failures": list(reversed(self.recent_failures)),
        };

        tmp = self.path.with_suffix(self.path.suffix + ".tmp");
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False));
        tmp.replace(self.path);

    def log_summary(self) -> None:
        logger.info(
            f"Progress: {self.completed}/{self.total} "
            f"(ok={self.successes}, fail={self.failures}) "
            f"@ {round(self.completed / max(time.time() - self.started_at, 1e-9) * 60, 1)}/min"
        );
