"""The deployed surface: auth, routes, caching, and the dashboard's escaping."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from pmbot.cli import main
from pmbot.config import ModelSettings, Settings
from pmbot.server import ScanCache, make_server, render_dashboard
from pmbot.types import Signal

TOKEN = "test-token-please-ignore"


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.data.use_sample()
    s.data.db = str(tmp_path / "server.sqlite3")
    s.models = ModelSettings(mc_sims=3000)
    s.staking.bankroll = 5000
    return s


@pytest.fixture
def server(settings):
    srv = make_server(settings, host="127.0.0.1", port=0, token=TOKEN, scan_ttl=300.0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url, token=None, timeout=60):
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode()


def get_status(url, token=None):
    try:
        return get(url, token)[0]
    except urllib.error.HTTPError as exc:
        return exc.code


class TestAuth:
    def test_health_is_open_so_the_platform_can_probe_it(self, server):
        status, body = get(f"{server}/healthz")
        assert status == 200
        payload = json.loads(body)
        assert payload["status"] == "ok"
        assert payload["authenticated"] is True

    def test_everything_else_needs_a_token(self, server):
        for path in ("/", "/api/signals?sport=nba", "/api/overview", "/api/bets", "/api/config"):
            assert get_status(f"{server}{path}") == 401, path

    def test_a_wrong_token_is_rejected(self, server):
        assert get_status(f"{server}/api/overview", token="nope") == 401

    def test_a_bearer_token_works(self, server):
        assert get_status(f"{server}/api/overview", token=TOKEN) == 200

    def test_a_query_token_works_for_browsers(self, server):
        assert get_status(f"{server}/api/overview?token={TOKEN}") == 200

    def test_a_prefix_of_the_token_is_not_enough(self, server):
        assert get_status(f"{server}/api/overview", token=TOKEN[:-1]) == 401

    def test_anonymous_mode_opens_everything(self, settings):
        srv = make_server(settings, host="127.0.0.1", port=0, token=None)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            assert get_status(f"http://127.0.0.1:{srv.server_address[1]}/api/overview") == 200
        finally:
            srv.shutdown()
            srv.server_close()


class TestRoutes:
    def test_signals_are_served_as_json(self, server):
        status, body = get(f"{server}/api/signals?sport=nba&all=1", TOKEN)
        payload = json.loads(body)
        assert status == 200
        assert payload["sport"] == "nba"
        assert payload["count"] == len(payload["signals"])
        assert {"action", "player", "market", "reason"} <= payload["signals"][0].keys()

    def test_an_unknown_sport_is_an_error_not_a_crash(self, server):
        assert get_status(f"{server}/api/signals?sport=cricket", TOKEN) == 500

    def test_overview_carries_bankroll_and_performance(self, server):
        payload = json.loads(get(f"{server}/api/overview", TOKEN)[1])
        assert {"bankroll", "performance", "open_bets", "max_drawdown"} <= payload.keys()
        assert "balance" in payload["bankroll"]

    def test_bets_are_listable(self, server):
        assert json.loads(get(f"{server}/api/bets", TOKEN)[1]) == []

    def test_the_api_key_is_masked(self, server, settings):
        settings.api.api_key = "super-secret-odds-key"
        payload = json.loads(get(f"{server}/api/config", TOKEN)[1])
        assert payload["api"]["api_key"] == "***"
        assert "super-secret-odds-key" not in json.dumps(payload)

    def test_unknown_routes_404(self, server):
        assert get_status(f"{server}/admin", TOKEN) == 404

    def test_the_dashboard_renders_html(self, server):
        status, body = get(f"{server}/?token={TOKEN}&sport=nba", timeout=90)
        assert status == 200
        assert body.startswith("<!doctype html>")
        assert "pmbot" in body and "balance" in body

    def test_the_dashboard_is_not_indexable(self, server):
        _, body = get(f"{server}/?token={TOKEN}", timeout=90)
        assert 'name="robots" content="noindex' in body

    def test_head_requests_work_for_probes(self, server):
        request = urllib.request.Request(f"{server}/healthz", method="HEAD")
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.status == 200


class TestScheduleRoute:
    def test_it_says_so_when_nothing_is_armed(self, server):
        payload = json.loads(get(f"{server}/api/schedule", TOKEN)[1])
        assert payload["enabled"] is False

    def test_an_armed_job_is_visible(self, settings):
        """A scheduler you cannot observe is one you cannot trust."""
        from datetime import time as dtime

        from pmbot.scheduler import DailyScheduler

        srv = make_server(settings, host="127.0.0.1", port=0, token=TOKEN)
        srv.service.scheduler = DailyScheduler(
            job=lambda: "nba: 2 bet(s)", at=dtime(16, 0),
            state_path=settings.data.db + ".sched.json",
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            payload = json.loads(get(f"{base}/api/schedule", TOKEN)[1])
            assert payload["enabled"] is True
            assert payload["schedule_utc"] == "16:00"
            assert json.loads(get(f"{base}/healthz")[1])["daily_job"]["schedule_utc"] == "16:00"
        finally:
            srv.shutdown()
            srv.server_close()


class TestScanCache:
    def test_the_first_call_computes(self):
        cache = ScanCache(ttl=60)
        calls = []
        signals, age = cache.get_or_compute(("nba",), lambda: calls.append(1) or ["x"])
        assert signals == ["x"] and age == 0.0 and len(calls) == 1

    def test_a_second_call_reuses(self):
        """A slate is thousands of simulations; recomputing per request is a DoS."""
        cache = ScanCache(ttl=60)
        calls = []

        def compute():
            calls.append(1)
            return ["x"]

        cache.get_or_compute(("nba",), compute)
        signals, age = cache.get_or_compute(("nba",), compute)
        assert len(calls) == 1
        assert signals == ["x"] and age >= 0.0

    def test_an_expired_entry_recomputes(self):
        cache = ScanCache(ttl=-1)
        calls = []
        for _ in range(2):
            cache.get_or_compute(("nba",), lambda: calls.append(1) or ["x"])
        assert len(calls) == 2

    def test_different_keys_do_not_collide(self):
        cache = ScanCache(ttl=60)
        cache.get_or_compute(("nba", 1000), lambda: ["a"])
        signals, _ = cache.get_or_compute(("nba", 5000), lambda: ["b"])
        assert signals == ["b"]


class TestDashboardRendering:
    def overview(self):
        return {
            "bankroll": {"balance": 5000.0, "available": 4900.0, "pending_stake": 100.0,
                         "deposits": 5000.0, "realised_profit": 0.0},
            "performance": {"roi": 0.0, "wins": 0, "losses": 0, "clv_avg": None, "clv_n": 0},
            "open_bets": [],
        }

    def test_player_names_are_escaped(self):
        """Names come from a third-party feed and land in HTML."""
        nasty = Signal(
            action="PASS", sport="nba", event_id="e",
            commence_time=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            matchup="A @ B", player="<script>alert('xss')</script>",
            market="player_points", market_label="Points", reason="models disagree",
        )
        html = render_dashboard(
            sport="nba", signals=[nasty], age=0.0, overview=self.overview()
        )
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_the_reason_text_is_escaped_too(self):
        signal = Signal(
            action="PASS", sport="nba", event_id="e",
            commence_time=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            matchup="A @ B", player="Someone", market="player_points",
            market_label="Points", reason="<img src=x onerror=alert(1)>",
        )
        html = render_dashboard(sport="nba", signals=[signal], age=0.0, overview=self.overview())
        assert "<img src=x" not in html

    def test_the_daily_job_is_shown_in_the_footer(self):
        html = render_dashboard(
            sport="nba", signals=[], age=0.0, overview=self.overview(),
            schedule={"schedule_utc": "16:00", "last_status": "ok",
                      "last_finished_at": "2026-08-11T16:00:09+00:00",
                      "next_run": "2026-08-12T16:00:00+00:00",
                      "last_summary": "nba: 2 bet(s)"},
        )
        assert "Daily pass 16:00 UTC" in html and "nba: 2 bet(s)" in html

    def test_a_missing_daily_job_is_called_out(self):
        html = render_dashboard(sport="nba", signals=[], age=0.0, overview=self.overview())
        assert "No daily pass armed" in html

    def test_an_empty_board_says_so(self):
        html = render_dashboard(sport="nba", signals=[], age=0.0, overview=self.overview())
        assert "Nothing on the board" in html

    def test_a_scan_failure_is_shown_not_swallowed(self):
        html = render_dashboard(
            sport="nba", signals=[], age=0.0, overview=self.overview(),
            error="odds provider exploded",
        )
        assert "odds provider exploded" in html


class TestServeCommand:
    def test_it_refuses_to_start_without_auth(self, capsys, tmp_path, monkeypatch):
        """A hosted URL is public and this page shows the bankroll."""
        monkeypatch.delenv("PMBOT_API_TOKEN", raising=False)
        code = main(["--sample", "--db", str(tmp_path / "x.sqlite3"), "serve", "--port", "0"])
        assert code == 2
        assert "refusing to start without authentication" in capsys.readouterr().err

    def test_allow_anonymous_is_an_explicit_opt_in(self, tmp_path, monkeypatch):
        """It must be possible, just never by accident."""
        monkeypatch.delenv("PMBOT_API_TOKEN", raising=False)
        started = {}

        class FakeService:
            cache = ScanCache(ttl=1)
            scheduler = None

        class FakeServer:
            service = FakeService()

            def serve_forever(self):
                started["ran"] = True

            def server_close(self):
                started["closed"] = True

        # cmd_serve imports make_server at call time, so patching the module
        # attribute is enough.
        monkeypatch.setattr("pmbot.server.make_server", lambda *a, **k: FakeServer())
        code = main([
            "--sample", "--db", str(tmp_path / "x.sqlite3"),
            "serve", "--port", "0", "--allow-anonymous",
        ])
        assert code == 0
        assert started == {"ran": True, "closed": True}
