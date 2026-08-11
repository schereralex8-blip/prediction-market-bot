"""The daily job: when it fires, when it doesn't, and what it survives."""

import json
from datetime import datetime, time as dtime, timezone

import pytest

from pmbot.scheduler import (
    DailySchedule,
    JobState,
    ScheduledRun,
    ScheduleState,
    parse_schedule,
    parse_time,
)


def at(hour: int, minute: int = 0, day: int = 11) -> datetime:
    return datetime(2026, 8, day, hour, minute, tzinfo=timezone.utc)


def one(hour=16, minute=0, steps=None) -> ScheduledRun:
    return ScheduledRun(at=dtime(hour, minute), steps=steps or ("fetch", "close", "scan"))


@pytest.fixture
def scheduler(tmp_path):
    calls = []

    def job(run: ScheduledRun) -> str:
        calls.append(run.label)
        return f"run {len(calls)}"

    sched = DailySchedule(runs=[one()], job=job, state_path=tmp_path / "scheduler.json")
    sched.calls = calls  # type: ignore[attr-defined]
    return sched


FULL = one()


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
        assert scheduler.due(at(15, 59)) == []

    def test_on_the_hour(self, scheduler):
        assert scheduler.due(at(16, 0)) == [FULL]

    def test_after_the_hour(self, scheduler):
        assert scheduler.due(at(20, 0)) == [FULL]

    def test_not_twice_in_a_day(self, scheduler):
        scheduler.run_once(FULL, at(16, 0))
        assert scheduler.due(at(16, 1)) == []
        assert scheduler.due(at(23, 59)) == []

    def test_again_tomorrow(self, scheduler):
        scheduler.run_once(FULL, at(16, 0))
        assert scheduler.due(at(16, 0, day=12)) == [FULL]

    def test_a_missed_window_catches_up(self, scheduler):
        """Container down at 16:00, up at 18:00. Late beats skipped."""
        assert scheduler.due(at(18, 0)) == [FULL]

    def test_next_run_is_today_before_the_window(self, scheduler):
        assert scheduler.next_run_after(at(9, 0), FULL) == at(16, 0)

    def test_next_run_is_tomorrow_once_it_has_run(self, scheduler):
        scheduler.run_once(FULL, at(16, 0))
        assert scheduler.next_run_after(at(17, 0), FULL) == at(16, 0, day=12)


class TestRunning:
    def test_a_successful_run_is_recorded(self, scheduler):
        state = scheduler.run_once(FULL, at(16, 0))
        assert state.runs == 1
        assert state.last_status == "ok"
        assert state.last_summary == "run 1"
        assert state.last_run_date == "2026-08-11"
        assert state.last_duration_seconds is not None

    def test_a_failing_job_is_recorded_not_raised(self, tmp_path):
        """One bad night must not take the scheduler down with it."""
        def boom(run) -> str:
            raise RuntimeError("odds feed exploded")

        sched = DailySchedule(runs=[one()], job=boom, state_path=tmp_path / "s.json")
        state = sched.run_once(one(), at(16, 0))
        assert state.last_status == "failed"
        assert "odds feed exploded" in state.last_summary

    def test_a_failed_run_still_claims_the_day(self, tmp_path):
        """Otherwise a job that fails fast spins in a tight retry loop."""
        def boom(run) -> str:
            raise RuntimeError("nope")

        sched = DailySchedule(runs=[one()], job=boom, state_path=tmp_path / "s.json")
        sched.run_once(one(), at(16, 0))
        assert sched.due(at(16, 1)) == []

    def test_state_survives_a_restart(self, tmp_path):
        path = tmp_path / "scheduler.json"
        first = DailySchedule(runs=[one()], job=lambda r: "done", state_path=path)
        first.run_once(one(), at(16, 0))

        second = DailySchedule(runs=[one()], job=lambda r: "done", state_path=path)
        assert second.state.for_label(one().label).runs == 1
        assert second.due(at(17, 0)) == []  # a redeploy must not re-run it

    def test_state_is_written_where_it_can_be_read(self, scheduler, tmp_path):
        scheduler.run_once(FULL, at(16, 0))
        stored = json.loads((tmp_path / "scheduler.json").read_text())["runs"][FULL.label]
        assert stored["last_status"] == "ok" and stored["runs"] == 1

    def test_a_corrupt_state_file_does_not_stop_the_job(self, tmp_path):
        path = tmp_path / "scheduler.json"
        path.write_text("{not json at all")
        sched = DailySchedule(runs=[one()], job=lambda r: "ok", state_path=path)
        assert sched.state == ScheduleState()
        assert sched.due(at(16, 0)) == [FULL]

    def test_unknown_state_keys_are_ignored(self, tmp_path):
        path = tmp_path / "scheduler.json"
        path.write_text(json.dumps({"runs": {FULL.label: {"runs": 3, "from_the_future": True}}}))
        sched = DailySchedule(runs=[one()], job=lambda r: "ok", state_path=path)
        assert sched.state.for_label(FULL.label).runs == 3


class TestThread:
    def test_it_runs_on_its_own_thread_and_stops(self, tmp_path):
        calls = []
        sched = DailySchedule(
            runs=[one(0, 0)],  # always due
            job=lambda r: calls.append(1) or "ok",
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
        scheduler.run_once(FULL, at(16, 0))
        status = scheduler.status()
        assert status["schedule"] == "16:00 full"
        assert status["last_status"] == "ok"
        assert status["runs"][0]["runs"] == 1
        assert status["next_run"]


class TestParseSchedule:
    def test_a_bare_time_runs_every_step(self):
        runs = parse_schedule("16:00")
        assert len(runs) == 1
        assert runs[0].steps == ("fetch", "close", "scan")
        assert runs[0].label == "16:00 full"

    def test_a_list_of_times(self):
        assert [r.clock for r in parse_schedule("16:00, 23:00")] == ["16:00", "23:00"]

    def test_steps_can_be_narrowed_per_entry(self):
        """The point of the list: a full pass early, closes near lock."""
        runs = parse_schedule("16:00, 23:00=close")
        assert runs[1].steps == ("close",)
        assert runs[1].label == "23:00 close"
        assert runs[1].skips() == {"skip_fetch": True, "skip_close": False, "skip_scan": True}

    def test_several_steps_join_with_plus(self):
        assert parse_schedule("23:30=close+scan")[0].steps == ("close", "scan")

    def test_step_order_is_canonical_however_it_is_typed(self):
        assert parse_schedule("9:00=scan+close")[0].label == parse_schedule("9:00=close+scan")[0].label

    def test_full_and_all_are_aliases(self):
        assert parse_schedule("16:00=full")[0].steps == parse_schedule("16:00=all")[0].steps

    def test_entries_sort_by_time(self):
        assert [r.clock for r in parse_schedule("23:00=close, 09:00, 16:00")] == [
            "09:00", "16:00", "23:00"
        ]

    def test_a_repeated_entry_is_a_slip_not_a_double_run(self):
        assert len(parse_schedule("16:00, 16:00=full")) == 1

    def test_the_same_time_with_different_steps_is_two_entries(self):
        assert len(parse_schedule("16:00, 16:00=close")) == 2

    def test_a_list_argument_works_too(self):
        assert len(parse_schedule(["16:00", "23:00=close"])) == 2

    def test_whitespace_and_stray_commas_survive(self):
        assert len(parse_schedule(" 16:00 , , 23:00=close ")) == 2

    def test_an_empty_schedule_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            parse_schedule(" , ")

    def test_an_unknown_step_is_rejected(self):
        with pytest.raises(ValueError, match="unknown schedule step"):
            parse_schedule("16:00=gamble")


class TestMultipleEntries:
    def build(self, tmp_path):
        fired = []

        def job(run):
            fired.append(run.label)
            return f"{run.label} done"

        sched = DailySchedule(
            runs=parse_schedule("16:00, 23:00=close"),
            job=job,
            state_path=tmp_path / "scheduler.json",
        )
        sched.fired = fired  # type: ignore[attr-defined]
        return sched

    def test_each_entry_fires_at_its_own_time(self, tmp_path):
        sched = self.build(tmp_path)
        assert [r.clock for r in sched.due(at(16, 30))] == ["16:00"]
        sched.run_once(sched.runs[0], at(16, 30))
        assert [r.clock for r in sched.due(at(16, 45))] == []
        assert [r.clock for r in sched.due(at(23, 5))] == ["23:00"]

    def test_entries_are_tracked_separately(self, tmp_path):
        sched = self.build(tmp_path)
        sched.run_once(sched.runs[0], at(16, 0))
        assert sched.state.for_label("16:00 full").runs == 1
        assert sched.state.for_label("23:00 close").runs == 0

    def test_both_catch_up_after_a_long_outage(self, tmp_path):
        """Down all day, up at 23:30: both are overdue and both should run."""
        sched = self.build(tmp_path)
        assert len(sched.due(at(23, 30))) == 2

    def test_next_run_is_the_soonest_entry(self, tmp_path):
        sched = self.build(tmp_path)
        assert sched.next_run(at(9, 0)) == at(16, 0)
        sched.run_once(sched.runs[0], at(16, 0))
        assert sched.next_run(at(17, 0)) == at(23, 0)

    def test_one_failing_entry_does_not_stop_the_other(self, tmp_path):
        def job(run):
            if run.clock == "16:00":
                raise RuntimeError("fetch died")
            return "closes stamped"

        sched = DailySchedule(
            runs=parse_schedule("16:00, 23:00=close"), job=job, state_path=tmp_path / "s.json"
        )
        sched.run_once(sched.runs[0], at(16, 0))
        sched.run_once(sched.runs[1], at(23, 0))
        assert sched.state.for_label("16:00 full").last_status == "failed"
        assert sched.state.for_label("23:00 close").last_status == "ok"

    def test_status_lists_every_entry(self, tmp_path):
        sched = self.build(tmp_path)
        status = sched.status()
        assert [e["label"] for e in status["runs"]] == ["16:00 full", "23:00 close"]
        assert status["runs"][1]["steps"] == ["close"]

    def test_the_flattened_summary_is_the_most_recent_run(self, tmp_path):
        sched = self.build(tmp_path)
        sched.run_once(sched.runs[0], at(16, 0))
        sched.run_once(sched.runs[1], at(23, 0))
        assert sched.status()["last_label"] == "23:00 close"

    def test_state_from_the_single_job_era_is_carried_over(self, tmp_path):
        """Upgrading mustn't fire an extra full pass on the same day."""
        path = tmp_path / "scheduler.json"
        path.write_text(json.dumps({
            "last_run_date": "2026-08-11", "last_status": "ok", "runs": 4,
        }))
        sched = DailySchedule(runs=parse_schedule("16:00, 23:00=close"), job=lambda r: "x",
                              state_path=path)
        assert sched.state.for_label("16:00 full").runs == 4
        assert [r.clock for r in sched.due(at(17, 0))] == []  # the full pass already went
        assert [r.clock for r in sched.due(at(23, 30))] == ["23:00"]


class TestServeWiring:
    """`serve` has to arm the job from flags or the environment."""

    def build(self, monkeypatch, tmp_path, **env):
        import argparse

        from pmbot.cli import build_schedule
        from pmbot.config import Settings

        for key in ("PMBOT_DAILY_AT", "PMBOT_DAILY_SPORTS", "PMBOT_DAILY_LOG_BETS"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        settings = Settings()
        settings.data.db = str(tmp_path / "j.sqlite3")
        args = argparse.Namespace(daily=None, daily_sport=None, daily_log=False)
        return build_schedule(args, settings)

    def test_nothing_armed_without_a_time(self, monkeypatch, tmp_path):
        assert self.build(monkeypatch, tmp_path) is None

    def test_the_environment_arms_it(self, monkeypatch, tmp_path):
        sched = self.build(monkeypatch, tmp_path, PMBOT_DAILY_AT="16:30")
        assert sched is not None
        assert [r.clock for r in sched.runs] == ["16:30"]

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

    def test_a_list_arms_every_entry(self, monkeypatch, tmp_path):
        sched = self.build(monkeypatch, tmp_path, PMBOT_DAILY_AT="16:00, 23:00=close")
        assert [r.label for r in sched.runs] == ["16:00 full", "23:00 close"]

    def test_a_bad_step_is_rejected_at_startup(self, monkeypatch, tmp_path):
        with pytest.raises(ValueError, match="unknown schedule step"):
            self.build(monkeypatch, tmp_path, PMBOT_DAILY_AT="16:00=teleport")
