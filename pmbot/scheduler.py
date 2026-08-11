"""Scheduled passes that run inside the web process.

Why in-process rather than a separate scheduled service: the journal is SQLite
on a mounted volume, and a volume attaches to exactly one service. A cron
service would get its own empty disk -- refreshing game logs nowhere useful and
stamping closing lines on an empty journal -- while the real one never updated.
Nothing would error; the numbers would just quietly be wrong.

A schedule is a **list**, because the passes want different times:

    16:00                 the full pass: fetch, close, scan
    23:00=close           closing lines only, near lock, for CLV

Each entry is tracked separately, so one can run and the other can fail
without either affecting the next day.

Behaviour worth knowing:

* **State lives on the volume**, so a redeploy at 16:05 doesn't re-run an
  entry that already went at 16:00.
* **A missed entry catches up.** Down at 16:00, up at 18:00 → it runs then.
  For a data refresh, late beats skipped.
* **A failing entry never kills the thread.** It records the error and waits
  for tomorrow, because a scheduler that dies on one bad night is worse than
  the bad night.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

log = logging.getLogger("pmbot.scheduler")

STEPS: tuple[str, ...] = ("fetch", "close", "scan")
STEP_ALIASES: dict[str, tuple[str, ...]] = {"full": STEPS, "all": STEPS}
STATE_VERSION = 2


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


@dataclass(frozen=True)
class ScheduledRun:
    """One entry: a UTC time and which steps run at it."""

    at: dtime
    steps: tuple[str, ...] = STEPS

    @property
    def clock(self) -> str:
        return f"{self.at.hour:02d}:{self.at.minute:02d}"

    @property
    def label(self) -> str:
        """Stable key for state on disk. Changing the steps starts a new key,
        which is correct: it is a different job."""
        kind = "full" if set(self.steps) == set(STEPS) else "+".join(self.steps)
        return f"{self.clock} {kind}"

    def skips(self) -> dict[str, bool]:
        return {f"skip_{step}": step not in self.steps for step in STEPS}


def parse_schedule(spec: str | Sequence[str]) -> list[ScheduledRun]:
    """Read ``"16:00, 23:00=close"`` into a sorted list of entries.

    Each item is ``TIME`` (all steps) or ``TIME=STEP[+STEP...]``. ``full`` and
    ``all`` are aliases for every step.
    """
    items: list[str] = []
    if isinstance(spec, str):
        items = [part for part in spec.split(",")]
    else:
        for entry in spec:
            items.extend(entry.split(","))

    runs: dict[str, ScheduledRun] = {}
    for raw in items:
        text = raw.strip()
        if not text:
            continue
        time_part, _, steps_part = text.partition("=")
        at = parse_time(time_part)
        steps = _parse_steps(steps_part) if steps_part.strip() else STEPS
        run = ScheduledRun(at=at, steps=steps)
        # Two identical entries are a config slip, not a request to run twice.
        runs[run.label] = run
    if not runs:
        raise ValueError("the schedule is empty; expected something like '16:00, 23:00=close'")
    return sorted(runs.values(), key=lambda r: (r.at, r.label))


def _parse_steps(text: str) -> tuple[str, ...]:
    names = [n.strip().lower() for n in text.replace("+", " ").split() if n.strip()]
    resolved: list[str] = []
    for name in names:
        if name in STEP_ALIASES:
            resolved.extend(STEP_ALIASES[name])
        elif name in STEPS:
            resolved.append(name)
        else:
            raise ValueError(
                f"unknown schedule step {name!r}; expected some of "
                f"{', '.join(STEPS)} (or 'full')"
            )
    # Keep canonical order so labels are stable regardless of how they're typed.
    return tuple(step for step in STEPS if step in resolved)


@dataclass
class JobState:
    last_run_date: str | None = None  # UTC date this entry last *started*
    last_started_at: str | None = None
    last_finished_at: str | None = None
    last_status: str | None = None  # "ok" | "failed"
    last_duration_seconds: float | None = None
    last_summary: str | None = None
    runs: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "JobState":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ScheduleState:
    """Per-entry state, keyed by the entry's label."""

    runs: dict[str, JobState] = field(default_factory=dict)

    def for_label(self, label: str) -> JobState:
        return self.runs.setdefault(label, JobState())

    @classmethod
    def load(cls, path: Path, first_label: str | None = None) -> "ScheduleState":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("scheduler state at %s is unreadable; starting fresh", path)
            return cls()

        if isinstance(data, dict) and "runs" in data and isinstance(data["runs"], dict):
            return cls(runs={k: JobState.from_dict(v) for k, v in data["runs"].items()})

        # A file from when the schedule was a single job. Carry it onto the
        # first entry so upgrading doesn't fire an extra pass on the same day.
        if isinstance(data, dict) and "last_run_date" in data and first_label:
            log.info("migrating single-job scheduler state onto %r", first_label)
            return cls(runs={first_label: JobState.from_dict(data)})
        return cls()

    def save(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {"version": STATE_VERSION,
                     "runs": {k: asdict(v) for k, v in self.runs.items()}},
                    indent=1,
                )
            )
            tmp.replace(path)
        except OSError as exc:  # a read-only volume shouldn't stop jobs running
            log.warning("could not persist scheduler state: %s", exc)


class DailySchedule:
    """Runs each entry once per UTC day, at or after its time."""

    def __init__(
        self,
        runs: Sequence[ScheduledRun],
        job: Callable[[ScheduledRun], str],
        state_path: str | Path,
        tick_seconds: float = 30.0,
    ) -> None:
        if not runs:
            raise ValueError("a schedule needs at least one entry")
        self.runs = list(runs)
        self.job = job
        self.state_path = Path(state_path)
        self.tick_seconds = tick_seconds
        self.state = ScheduleState.load(self.state_path, first_label=self.runs[0].label)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def next_run_after(self, now: datetime, run: ScheduledRun) -> datetime:
        today = datetime.combine(now.date(), run.at, tzinfo=timezone.utc)
        if self.state.for_label(run.label).last_run_date == now.date().isoformat():
            return today + timedelta(days=1)
        return today if now < today else now

    def next_run(self, now: datetime) -> datetime | None:
        upcoming = [self.next_run_after(now, run) for run in self.runs]
        return min(upcoming) if upcoming else None

    def due(self, now: datetime) -> list[ScheduledRun]:
        """Every entry that should have run by now and hasn't today."""
        today = now.date().isoformat()
        return [
            run
            for run in self.runs
            if self.state.for_label(run.label).last_run_date != today
            and now >= datetime.combine(now.date(), run.at, tzinfo=timezone.utc)
        ]

    # ------------------------------------------------------------------
    def run_once(self, run: ScheduledRun, now: datetime | None = None) -> JobState:
        """Run one entry, recording the outcome whichever way it goes."""
        now = now or datetime.now(timezone.utc)
        state = self.state.for_label(run.label)
        # Claim the day up front: if the job crashes hard, it must not spin.
        state.last_run_date = now.date().isoformat()
        state.last_started_at = now.isoformat(timespec="seconds")
        state.runs += 1
        self.state.save(self.state_path)

        log.info("scheduled run %r starting", run.label)
        try:
            state.last_summary = self.job(run)
            state.last_status = "ok"
        except Exception as exc:  # noqa: BLE001 - one bad night must not kill the loop
            log.exception("scheduled run %r failed", run.label)
            state.last_status = "failed"
            state.last_summary = f"{type(exc).__name__}: {exc}"

        finished = datetime.now(timezone.utc)
        state.last_finished_at = finished.isoformat(timespec="seconds")
        state.last_duration_seconds = round((finished - now).total_seconds(), 1)
        self.state.save(self.state_path)
        log.info(
            "scheduled run %r %s in %.1fs",
            run.label, state.last_status, state.last_duration_seconds or 0.0,
        )
        return state

    # ------------------------------------------------------------------
    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self._loop, name="pmbot-schedule", daemon=True)
        self._thread = thread
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        now = datetime.now(timezone.utc)
        log.info(
            "schedule armed: %s (next: %s)",
            ", ".join(run.label for run in self.runs),
            (self.next_run(now) or now).isoformat(timespec="minutes"),
        )
        while not self._stop.is_set():
            for run in self.due(datetime.now(timezone.utc)):
                if self._stop.is_set():
                    break
                self.run_once(run)
            # Short ticks rather than one long sleep: the container can be
            # suspended or the clock stepped, and a day-long sleep would miss.
            self._stop.wait(self.tick_seconds)

    # ------------------------------------------------------------------
    def status(self) -> dict[str, object]:
        now = datetime.now(timezone.utc)
        entries = []
        for run in self.runs:
            state = self.state.for_label(run.label)
            entries.append({
                "label": run.label,
                "schedule_utc": run.clock,
                "steps": list(run.steps),
                "next_run": self.next_run_after(now, run).isoformat(timespec="seconds"),
                **asdict(state),
            })
        finished = [e for e in entries if e.get("last_finished_at")]
        # Tie-break on start time: two entries can finish within the same
        # second, and "most recent" should still mean the later one.
        latest = (
            max(finished, key=lambda e: (e["last_finished_at"], e["last_started_at"] or ""))
            if finished
            else None
        )
        next_run = self.next_run(now)
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "schedule": ", ".join(run.label for run in self.runs),
            "next_run": next_run.isoformat(timespec="seconds") if next_run else None,
            "runs": entries,
            # Flattened "most recent outcome", for compact displays.
            "last_status": latest["last_status"] if latest else None,
            "last_finished_at": latest["last_finished_at"] if latest else None,
            "last_summary": latest["last_summary"] if latest else None,
            "last_label": latest["label"] if latest else None,
        }


def describe(runs: Iterable[ScheduledRun]) -> str:
    return ", ".join(run.label for run in runs)
