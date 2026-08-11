import json

import pytest

from pmbot.cli import main


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "cli.sqlite3")


def run(capsys, *args):
    """Every CLI test runs against the synthetic slate, never fetched data."""
    code = main(["--sample", *args])
    return code, capsys.readouterr().out


class TestInspection:
    def test_markets_lists_every_sport(self, capsys):
        code, out = run(capsys, "markets")
        assert code == 0
        for expected in ("NBA", "MLB", "NFL", "player_points", "pitcher_strikeouts"):
            assert expected in out

    def test_markets_can_be_narrowed(self, capsys):
        _, out = run(capsys, "markets", "--sport", "nfl")
        assert "player_pass_yds" in out and "player_points" not in out

    def test_config_dumps_effective_settings(self, capsys):
        code, out = run(capsys, "config")
        assert code == 0
        assert json.loads(out)["staking"]["kelly_fraction"] == 0.25

    def test_config_can_be_written_out(self, capsys, tmp_path):
        target = tmp_path / "written.json"
        code, out = run(capsys, "config", "--write", str(target))
        assert code == 0 and target.exists()
        assert "devig" in json.loads(target.read_text())


class TestScan:
    def test_scan_renders_a_slate(self, capsys, db):
        code, out = run(capsys, "--db", db, "scan", "--sport", "nba", "--bankroll", "5000", "--all")
        assert code == 0
        assert "NBA player props" in out
        assert "bankroll 5,000.00" in out
        assert "prop(s) scanned" in out
        assert ("BET" in out) or ("PASS" in out)

    def test_scan_explains_every_pass(self, capsys, db):
        _, out = run(capsys, "--db", db, "scan", "--sport", "nba", "--all")
        for block in out.split("\n\n"):
            if block.startswith("PASS"):
                assert "│" in block  # a reason line always follows

    def test_scan_json_is_machine_readable(self, capsys, db):
        code, out = run(capsys, "--db", db, "scan", "--sport", "mlb", "--all", "--json")
        payload = json.loads(out)
        assert code == 0 and payload
        assert {"action", "player", "market", "reason"} <= payload[0].keys()

    def test_scan_can_log_its_own_recommendations(self, capsys, db):
        _, out = run(capsys, "--db", db, "scan", "--sport", "nfl", "--bankroll", "5000", "--log")
        assert "logged to the journal" in out
        _, listing = run(capsys, "--db", db, "bets")
        assert "pending" in listing

    def test_verbose_adds_the_projection(self, capsys, db):
        _, out = run(capsys, "--db", db, "scan", "--sport", "nba", "--all", "-v")
        assert "projection" in out


class TestProject:
    def test_project_shows_both_models(self, capsys):
        code, out = run(
            capsys, "project", "--sport", "nba", "--player", "Anthony Edwards",
            "--market", "player_points", "--line", "27.5",
        )
        assert code == 0
        assert "bayes_rate" in out and "bootstrap_mc" in out
        assert "projection" in out and "gate (" in out

    def test_project_can_gate_against_a_real_price(self, capsys):
        _, out = run(
            capsys, "project", "--sport", "nba", "--player", "Anthony Edwards",
            "--market", "player_points", "--line", "27.5",
            "--market-prob", "0.42", "--odds", "165",
        )
        assert "breakeven" in out
        assert "AGREE" in out or "PASS" in out

    def test_project_gates_against_a_supplied_market_price(self, capsys):
        _, out = run(
            capsys, "project", "--sport", "nba", "--player", "Anthony Edwards",
            "--market", "player_points", "--line", "27.5", "--market-prob", "0.30",
        )
        assert "vs supplied fair over probability" in out

    def test_project_reports_a_missing_player_cleanly(self, capsys):
        code, out = run(
            capsys, "project", "--sport", "nba", "--player", "Nobody",
            "--market", "player_points", "--line", "20.5",
        )
        assert code == 1
        assert "no game logs" in out

    def test_an_unknown_market_is_an_error_not_a_traceback(self, capsys):
        code, _ = run(
            capsys, "project", "--sport", "nba", "--player", "Anthony Edwards",
            "--market", "player_dunks", "--line", "2.5",
        )
        assert code == 1


class TestJournalCommands:
    def bet(self, capsys, db, **kw):
        args = {
            "--sport": "nba", "--player": "Anthony Edwards", "--market": "player_points",
            "--side": "over", "--line": "25.5", "--book": "caesars",
            "--odds": "-110", "--stake": "50",
        }
        args.update(kw)
        flat = [x for pair in args.items() for x in pair]
        return run(capsys, "--db", db, "log", *flat)

    def test_the_full_lifecycle(self, capsys, db):
        run(capsys, "--db", db, "bankroll", "--deposit", "1000")
        code, out = self.bet(capsys, db)
        assert code == 0 and "logged #1" in out

        _, out = run(capsys, "--db", db, "bankroll")
        assert "pending stakes" in out and "50.00" in out

        code, out = run(capsys, "--db", db, "settle", "1", "--result", "31")
        assert code == 0 and "WON" in out

        code, out = run(capsys, "--db", db, "close", "1", "--over", "-130", "--under", "110")
        assert code == 0 and "CLV" in out

        code, out = run(capsys, "--db", db, "report")
        assert code == 0
        assert "BANKROLL" in out and "PERFORMANCE" in out and "ROI" in out

    def test_settling_needs_a_result_or_a_status(self, capsys, db):
        self.bet(capsys, db)
        code, out = run(capsys, "--db", db, "settle", "1")
        assert code == 2
        assert "--result" in out

    def test_bets_listing_and_filters(self, capsys, db):
        self.bet(capsys, db)
        self.bet(capsys, db, **{"--sport": "nfl", "--market": "player_pass_yds"})
        _, out = run(capsys, "--db", db, "bets")
        assert out.count("Anthony Edwards") == 2
        _, out = run(capsys, "--db", db, "bets", "--sport", "nfl")
        assert out.count("player_pass_yds") == 1

    def test_bets_json(self, capsys, db):
        self.bet(capsys, db)
        _, out = run(capsys, "--db", db, "bets", "--json")
        assert json.loads(out)[0]["player"] == "Anthony Edwards"

    def test_report_on_an_empty_journal_does_not_explode(self, capsys, db):
        code, out = run(capsys, "--db", db, "report")
        assert code == 0
        assert "0 settled" in out

    def test_report_groups_and_calibrates(self, capsys, db):
        run(capsys, "--db", db, "bankroll", "--deposit", "1000")
        for i in range(4):
            self.bet(capsys, db, **{"--model-prob": "0.6", "--fair-prob": "0.52"})
            run(capsys, "--db", db, "settle", str(i + 1), "--status", "won" if i % 2 else "lost")
        code, out = run(capsys, "--db", db, "report", "--group", "book")
        assert code == 0
        assert "BY BOOK" in out and "CALIBRATION" in out

    def test_report_json(self, capsys, db):
        run(capsys, "--db", db, "bankroll", "--deposit", "500")
        code, out = run(capsys, "--db", db, "report", "--json")
        payload = json.loads(out)
        assert code == 0
        assert payload["bankroll"]["balance"] == 500

    def test_bankroll_json(self, capsys, db):
        run(capsys, "--db", db, "bankroll", "--deposit", "250")
        _, out = run(capsys, "--db", db, "bankroll", "--withdraw", "100", "--json")
        assert json.loads(out)["balance"] == 150


class TestFetch:
    def test_sources_are_listed_with_a_default_each(self, capsys):
        code, out = run(capsys, "sources")
        assert code == 0
        for expected in ("NBA", "MLB", "NFL", "nflverse", "mlb-statsapi", "nba-stats"):
            assert expected in out

    def test_fetch_refuses_to_overwrite_the_shipped_fixture(self, capsys):
        """--sample plus fetch would replace synthetic logs with real ones."""
        code, out = run(capsys, "fetch", "--sport", "nfl", "--player", "Josh Allen")
        assert code == 2
        assert "refusing" in out and "data/gamelogs" in out

    def test_fetch_needs_someone_to_fetch(self, capsys, tmp_path):
        code, out = main(["fetch", "--sport", "nfl"]), capsys.readouterr().out
        assert code == 2
        assert "--player" in out and "--from-props" in out


class TestCron:
    def test_a_pass_runs_and_reports_each_step(self, capsys, db):
        code, out = run(capsys, "--db", db, "cron", "--sport", "nba", "--skip-fetch")
        assert code == 0
        assert "[close]" in out and "[scan]" in out
        assert "step(s) failed" in out

    def test_steps_can_be_skipped(self, capsys, db):
        _, out = run(capsys, "--db", db, "cron", "--sport", "nba",
                     "--skip-fetch", "--skip-scan", "--skip-close")
        assert "[scan]" not in out and "[close]" not in out

    def test_it_does_not_log_bets_unless_asked(self, capsys, db):
        """A journal of bets nobody placed destroys the only honest measure."""
        run(capsys, "--db", db, "cron", "--sport", "nba", "--skip-fetch")
        _, listing = run(capsys, "--db", db, "bets")
        assert "no bets recorded yet" in listing

    def test_log_opts_in(self, capsys, db):
        run(capsys, "--db", db, "bankroll", "--deposit", "5000")
        _, out = run(capsys, "--db", db, "cron", "--sport", "nba", "--skip-fetch", "--log")
        if "logged bet(s)" in out:
            _, listing = run(capsys, "--db", db, "bets")
            assert "pending" in listing

    def test_a_fetch_step_cannot_overwrite_the_fixture(self, capsys, db):
        """--sample plus a fetching pass used to replace synthetic logs."""
        import json
        from pathlib import Path

        target = Path("data/sample/gamelogs/nba/anthony-edwards.json")
        before = json.loads(target.read_text())
        run(capsys, "--db", db, "cron", "--sport", "nba", "--skip-scan", "--skip-close")
        assert json.loads(target.read_text())["source"] == before["source"] == "synthetic"

    def test_bankroll_defaults_to_the_journal_balance(self, capsys, db):
        run(capsys, "--db", db, "bankroll", "--deposit", "12345")
        _, out = run(capsys, "--db", db, "cron", "--sport", "nba", "--skip-fetch", "--skip-close")
        assert "12,345.00" in out


class TestErrors:
    def test_a_missing_subcommand_exits_nonzero(self, capsys):
        with pytest.raises(SystemExit):
            main([])

    def test_snapshot_without_an_api_key_says_so(self, capsys, tmp_path):
        code, out = run(
            capsys, "snapshot", "--sport", "nba", "--out", str(tmp_path / "snap.json")
        )
        assert code == 2
        assert "API key" in out
