"""The daily job: when it fires, when it doesn't, and what it survives."""

import json
from datetime import datetime, time as dtime, timezone

import pytest

from pmbot.scheduler import DailyScheduler, JobState, parse_time


def at(hour: int, minute: int = 0, day: int = 11) -> datetime:
    return datetime(2026, 8, day, hour, minute, tzinfo=timezone.utc)


@pytest.fixture
def scheduler(tmp_path):
    calls = []

    def job() -> str:
        calls.append(1)
        return f"run {len(calls)}"

    sched = DailyScheduler(job=job, at=dtime(16, 0), state_path=tmp_path / "scheduler.json")
    sched.calls = calls  # type: ignore[attr-defined]
    return sched


class TestParseTime:
    @pytest.mark.parametrize(
        "text,expected", [("16:00", (16, 0)), ("06:30", (6, 30)), ("9", (9, 0)), (" 23:59 ", (23, 59))]
    )
    def test_valid_times(self, text, expected):
        parsed = parse_time(text)
        assert (parsed.hour, parsed.minute) == expected

    @pytest.mark.parametrize("text", ["", "nope", "25:00", "16:70", "16:00:00:00", "-1"])
    def test_rubbish_is_rejected_loudly(self, text):
        with pytest.raises(ValueError):
            parse_time(text)


class TestWhenItFires:
    def test_not_before_the_hour(self, scheduler):
        assert scheduler.due(at(15, 59)) is False

    def test_on_the_hour(self, scheduler):
        assert scheduler.due(at(16, 0)) is True

    def test_after_the_hour(self, scheduler):
        assert scheduler.due(at(20, 0)) is True

    def test_not_twice_in_a_day(self, scheduler):
        scheduler.run_once(at(16, 0))
        assert scheduler.due(at(16, 1)) is False
        assert scheduler.due(at(23, 59)) is False

    def test_again_tomorrow(self, scheduler):
        scheduler.run_once(at(16, 0))
        assert scheduler.due(at(16, 0, day=12)) is True

    def test_a_missed_window_catches_up(self, scheduler):
        """Container down at 16:00, up at 18:00. Late beats skipped."""
        assert scheduler.due(at(18, 0)) is True

    def test_next_run_is_today_before_the_window(self, scheduler):
        assert scheduler.next_run_after(at(9, 0)) == at(16, 0)

    def test_next_run_is_tomorrow_once_it_has_run(self, scheduler):
        scheduler.run_once(at(16, 0))
        assert scheduler.next_run_after(at(17, 0)) == at(16, 0, day=12)


class TestRunning:
    def test_a_successful_run_is_recorded(self, scheduler):
        state = scheduler.run_once(at(16, 0))
        assert state.runs == 1
        assert state.last_status == "ok"
        assert state.last_summary == "run 1"
        assert state.last_run_date == "2026-08-11"
        assert state.last_duration_seconds is not None

    def test_a_failing_job_is_recorded_not_raised(self, tmp_path):
        """One bad night must not take the scheduler down with it."""
        def boom() -> str:
            raise RuntimeError("odds feed exploded")

        sched = DailyScheduler(job=boom, at=dtime(16, 0), state_path=tmp_path / "s.json")
        state = sched.run_once(at(16, 0))
        assert state.last_status == "failed"
        assert "odds feed exploded" in state.last_summary

    def test_a_failed_run_still_claims_the_day(self, tmp_path):
        """Otherwise a job that fails fast spins in a tight retry loop."""
        def boom() -> str:
            raise RuntimeError("nope")

        sched = DailyScheduler(job=boom, at=dtime(16, 0), state_path=tmp_path / "s.json")
        sched.run_once(at(16, 0))
        assert sched.due(at(16, 1)) is False

    def test_state_survives_a_restart(self, tmp_path):
        path = tmp_path / "scheduler.json"
        first = DailyScheduler(job=lambda: "done", at=dtime(16, 0), state_path=path)
        first.run_once(at(16, 0))

        second = DailyScheduler(job=lambda: "done", at=dtime(16, 0), state_path=path)
        assert second.state.runs == 1
        assert second.due(at(17, 0)) is False  # a redeploy must not re-run it

    def test_state_is_written_where_it_can_be_read(self, scheduler, tmp_path):
        scheduler.run_once(at(16, 0))
        stored = json.loads((tmp_path / "scheduler.json").read_text())
        assert stored["last_status"] == "ok" and stored["runs"] == 1

    def test_a_corrupt_state_file_does_not_stop_the_job(self, tmp_path):
        path = tmp_path / "scheduler.json"
        path.write_text("{not json at all")
        sched = DailyScheduler(job=lambda: "ok", at=dtime(16, 0), state_path=path)
        assert sched.state == JobState()
        assert sched.due(at(16, 0)) is True

    def test_unknown_state_keys_are_ignored(self, tmp_path):
        path = tmp_path / "scheduler.json"
        path.write_text(json.dumps({"runs": 3, "from_a_future_version": True}))
        assert DailyScheduler(job=lambda: "ok", at=dtime(16, 0), state_path=path).state.runs == 3


class TestThread:
    def test_it_runs_on_its_own_thread_and_stops(self, tmp_path):
        calls = []
        sched = DailyScheduler(
            job=lambda: calls.append(1) or "ok",
            at=dtime(0, 0),  # always due
            state_path=tmp_path / "s.json",
            tick_seconds=0.05,
        )
        thread = sched.start()
        for _ in range(100):
            if calls:
                break
            import time as _t

            _t.sleep(0.05)
        sched.stop()
        assert calls, "the job never fired"
        assert not thread.is_alive()

    def test_status_reports_what_a_deployment_needs(self, scheduler):
        scheduler.run_once(at(16, 0))
        status = scheduler.status()
        assert status["schedule_utc"] == "16:00"
        assert status["last_status"] == "ok"
        assert status["runs"] == 1
        assert "next_run" in status


class TestServeWiring:
    """`serve` has to arm the job from flags or the environment."""

    def build(self, monkeypatch, tmp_path, **env):
        import argparse

        from pmbot.cli import build_daily_scheduler
        from pmbot.config import Settings

        for key in ("PMBOT_DAILY_AT", "PMBOT_DAILY_SPORTS", "PMBOT_DAILY_LOG_BETS"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        settings = Settings()
        settings.data.db = str(tmp_path / "j.sqlite3")
        args = argparse.Namespace(daily=None, daily_sport=None, daily_log=False)
        return build_daily_scheduler(args, settings)

    def test_nothing_armed_without_a_time(self, monkeypatch, tmp_path):
        assert self.build(monkeypatch, tmp_path) is None

    def test_the_environment_arms_it(self, monkeypatch, tmp_path):
        sched = self.build(monkeypatch, tmp_path, PMBOT_DAILY_AT="16:30")
        assert sched is not None
        assert sched.at == dtime(16, 30)

    def test_state_lands_next_to_the_journal_on_the_volume(self, monkeypatch, tmp_path):
        sched = self.build(monkeypatch, tmp_path, PMBOT_DAILY_AT="16:00")
        assert sched.state_path == tmp_path / "scheduler.json"

    def test_a_bad_sport_is_rejected_at_startup(self, monkeypatch, tmp_path):
        """Better to refuse to boot than to skip a sport silently every night."""
        with pytest.raises(ValueError, match="unknown sport"):
            self.build(monkeypatch, tmp_path, PMBOT_DAILY_AT="16:00", PMBOT_DAILY_SPORTS="nba,quidditch")

    def test_a_bad_time_is_rejected_at_startup(self, monkeypatch, tmp_path):
        with pytest.raises(ValueError):
            self.build(monkeypatch, tmp_path, PMBOT_DAILY_AT="teatime")
