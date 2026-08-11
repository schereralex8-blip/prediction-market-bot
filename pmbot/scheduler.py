"""A once-a-day job that runs inside the web process.

Why in-process rather than a separate scheduled service: the journal is SQLite
on a mounted volume, and a volume attaches to exactly one service. A cron
service would get its own empty disk -- refreshing game logs nowhere useful and
stamping closing lines on an empty journal -- while the real one never updated.
Nothing would error; the numbers would just quietly be wrong.

Behaviour worth knowing:

* **State lives on the volume**, so a redeploy at 16:05 doesn't re-run a job
  that already went at 16:00.
* **A missed run catches up.** If the container was down at 16:00 and comes up
  at 18:00, the job runs then. For a daily data refresh, late beats skipped.
* **A failing job never kills the thread.** It records the error and waits for
  tomorrow, because a scheduler that dies on one bad night is worse than the
  bad night.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Callable

log = logging.getLogger("pmbot.scheduler")


def parse_time(text: str) -> dtime:
    """'16:00' -> 16:00 UTC. Accepts 'HH:MM' or 'HH'."""
    cleaned = text.strip()
    parts = cleaned.split(":")
    if len(parts) > 2:
        # Silently taking the first two fields of a malformed schedule means
        # discovering the real firing time weeks later.
        raise ValueError(f"cannot read {text!r} as a time; expected HH:MM, e.g. 16:00")
    try:
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        raise ValueError(f"cannot read {text!r} as a time; expected HH:MM, e.g. 16:00") from None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"{text!r} is not a valid time of day")
    return dtime(hour=hour, minute=minute)


@dataclass
class JobState:
    last_run_date: str | None = None  # UTC date the job last *started*
    last_started_at: str | None = None
    last_finished_at: str | None = None
    last_status: str | None = None  # "ok" | "failed"
    last_duration_seconds: float | None = None
    last_summary: str | None = None
    runs: int = 0

    @classmethod
    def load(cls, path: Path) -> "JobState":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("scheduler state at %s is unreadable; starting fresh", path)
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(self), indent=1))
            tmp.replace(path)
        except OSError as exc:  # a read-only volume shouldn't stop the job running
            log.warning("could not persist scheduler state: %s", exc)


class DailyScheduler:
    """Runs ``job`` once per UTC day, at or after ``at``."""

    def __init__(
        self,
        job: Callable[[], str],
        at: dtime,
        state_path: str | Path,
        tick_seconds: float = 30.0,
        name: str = "daily",
    ) -> None:
        self.job = job
        self.at = at
        self.state_path = Path(state_path)
        self.tick_seconds = tick_seconds
        self.name = name
        self.state = JobState.load(self.state_path)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def next_run_after(self, now: datetime) -> datetime:
        """When the job is next due, given what has already run."""
        today = datetime.combine(now.date(), self.at, tzinfo=timezone.utc)
        if self.state.last_run_date == now.date().isoformat():
            return today + timedelta(days=1)  # already went today
        return today if now < today else now  # due now if the window has passed

    def due(self, now: datetime) -> bool:
        if self.state.last_run_date == now.date().isoformat():
            return False
        scheduled = datetime.combine(now.date(), self.at, tzinfo=timezone.utc)
        return now >= scheduled

    # ------------------------------------------------------------------
    def run_once(self, now: datetime | None = None) -> JobState:
        """Run the job, recording the outcome whichever way it goes."""
        now = now or datetime.now(timezone.utc)
        started = now
        # Claim the day up front: if the job crashes hard, it must not spin.
        self.state.last_run_date = now.date().isoformat()
        self.state.last_started_at = started.isoformat(timespec="seconds")
        self.state.runs += 1
        self.state.save(self.state_path)

        log.info("scheduled job %r starting", self.name)
        try:
            summary = self.job()
            self.state.last_status = "ok"
            self.state.last_summary = summary
        except Exception as exc:  # noqa: BLE001 - one bad night must not kill the loop
            log.exception("scheduled job %r failed", self.name)
            self.state.last_status = "failed"
            self.state.last_summary = f"{type(exc).__name__}: {exc}"

        finished = datetime.now(timezone.utc)
        self.state.last_finished_at = finished.isoformat(timespec="seconds")
        self.state.last_duration_seconds = round((finished - started).total_seconds(), 1)
        self.state.save(self.state_path)
        log.info(
            "scheduled job %r %s in %.1fs",
            self.name, self.state.last_status, self.state.last_duration_seconds or 0.0,
        )
        return self.state

    # ------------------------------------------------------------------
    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self._loop, name=f"pmbot-{self.name}", daemon=True)
        self._thread = thread
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        log.info(
            "scheduled job %r armed for %02d:%02d UTC daily (next: %s)",
            self.name, self.at.hour, self.at.minute,
            self.next_run_after(datetime.now(timezone.utc)).isoformat(timespec="minutes"),
        )
        while not self._stop.is_set():
            now = datetime.now(timezone.utc)
            if self.due(now):
                self.run_once(now)
            # Short ticks rather than one long sleep: the container can be
            # suspended or the clock stepped, and a day-long sleep would miss.
            self._stop.wait(self.tick_seconds)

    # ------------------------------------------------------------------
    def status(self) -> dict[str, object]:
        now = datetime.now(timezone.utc)
        return {
            "name": self.name,
            "schedule_utc": f"{self.at.hour:02d}:{self.at.minute:02d}",
            "next_run": self.next_run_after(now).isoformat(timespec="seconds"),
            "running": bool(self._thread and self._thread.is_alive()),
            **asdict(self.state),
        }
