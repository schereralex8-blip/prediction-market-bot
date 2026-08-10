"""End-to-end scans over the checked-in sample slate.

The fixtures under ``data/`` are built so that a handful of props have one
book left stale by a few points of probability, and everything else is priced
efficiently. The bot should find the first group and refuse the second.
"""

from datetime import datetime, timedelta, timezone

import pytest

from pmbot.config import ModelSettings, Settings
from pmbot.pipeline import PropScanner
from pmbot.types import BookLine, PropMarket


@pytest.fixture
def settings():
    s = Settings()
    s.models = ModelSettings(mc_sims=6000)
    s.staking.bankroll = 10_000
    return s


@pytest.fixture
def scanner(settings):
    return PropScanner(settings)


def signals_for(scanner, sport):
    return scanner.scan(sport, bankroll=10_000, include_passes=True)


class TestScan:
    @pytest.mark.parametrize("sport", ["nba", "mlb", "nfl"])
    def test_every_sport_produces_signals(self, scanner, sport):
        signals = signals_for(scanner, sport)
        assert signals, f"no props scanned for {sport}"
        assert all(s.action in ("BET", "PASS") for s in signals)

    def test_bets_come_first_and_are_ranked_by_ev(self, scanner):
        signals = signals_for(scanner, "nba")
        bets = [s for s in signals if s.is_bet]
        assert [s.ev_per_unit for s in bets] == sorted(
            (s.ev_per_unit for s in bets), reverse=True
        )
        if bets:
            assert signals[0].is_bet

    def test_every_bet_clears_the_configured_floors(self, scanner, settings):
        for sport in ("nba", "mlb", "nfl"):
            for bet in scanner.scan(sport, bankroll=10_000):
                assert bet.ev_per_unit >= settings.staking.min_ev
                assert bet.edge >= settings.staking.min_price_edge
                assert bet.stake > 0
                assert bet.side in ("over", "under")
                assert bet.disagreement <= settings.gate.max_disagreement

    def test_the_planted_stale_price_is_found(self, scanner):
        """Josh Allen's passing yards are hung 6 points behind the market."""
        bets = {(s.player, s.market): s for s in scanner.scan("nfl", bankroll=10_000)}
        allen = bets.get(("Josh Allen", "player_pass_yds"))
        assert allen is not None, "the bot missed a book hanging a stale line"
        assert allen.book == "draftkings"
        assert allen.side == "over"
        assert allen.ev_per_unit > 0.05

    def test_a_pure_price_edge_is_taken_even_when_the_model_agrees_with_the_market(self, scanner):
        """The bet that a disagreement-based gate would wrongly throw away."""
        allen = next(
            s for s in scanner.scan("nfl", bankroll=10_000)
            if s.player == "Josh Allen"
        )
        assert abs(allen.market_edge) < 0.03  # the model is with the market
        assert allen.edge > 0.02  # but the price is not

    def test_passes_explain_themselves(self, scanner):
        passes = [s for s in signals_for(scanner, "nba") if not s.is_bet]
        assert passes
        for signal in passes:
            assert signal.reason
            assert signal.stake == 0.0

    def test_stakes_respect_the_per_bet_cap(self, scanner, settings):
        for bet in scanner.scan("nba", bankroll=10_000):
            assert bet.stake <= 10_000 * settings.staking.max_bet_fraction + 1e-9

    def test_total_slate_risk_respects_the_cap(self, settings):
        settings.staking.max_slate_fraction = 0.01
        bets = PropScanner(settings).scan("nba", bankroll=10_000)
        assert sum(b.stake for b in bets) <= 10_000 * 0.01 + 0.5

    def test_filtering_by_player_narrows_the_scan(self, scanner):
        signals = scanner.scan(
            "nba", players=["Anthony Edwards"], include_passes=True, bankroll=10_000
        )
        assert signals
        assert {s.player for s in signals} == {"Anthony Edwards"}

    def test_filtering_by_market_narrows_the_scan(self, scanner):
        signals = scanner.scan(
            "nba", markets=["player_points"], include_passes=True, bankroll=10_000
        )
        assert {s.market for s in signals} == {"player_points"}

    def test_passes_are_hidden_by_default(self, scanner):
        assert all(s.is_bet for s in scanner.scan("nba", bankroll=10_000))

    def test_a_bigger_bankroll_scales_the_stakes(self, scanner):
        small = {(s.player, s.market): s.stake for s in scanner.scan("nba", bankroll=5_000)}
        big = {(s.player, s.market): s.stake for s in scanner.scan("nba", bankroll=50_000)}
        assert small and big
        for key, stake in small.items():
            assert big[key] > stake

    def test_headlines_render(self, scanner):
        for signal in signals_for(scanner, "nba"):
            assert signal.headline()


class TestEdgeCases:
    def test_an_unmodelled_market_passes_gracefully(self, scanner):
        prop = PropMarket(
            sport="nba", event_id="x", commence_time=datetime.now(timezone.utc) + timedelta(hours=2),
            home_team="A", away_team="B", player="Anthony Edwards", market="player_dunks",
            books=(BookLine("pinnacle", 2.5, -110, -110),),
        )
        signal = scanner.evaluate(prop, 1000)
        assert not signal.is_bet
        assert "not modelled" in signal.reason

    def test_an_unknown_player_passes_gracefully(self, scanner):
        prop = PropMarket(
            sport="nba", event_id="x", commence_time=datetime.now(timezone.utc) + timedelta(hours=2),
            home_team="A", away_team="B", player="Nobody At All", market="player_points",
            books=(BookLine("pinnacle", 25.5, -110, -110),),
        )
        signal = scanner.evaluate(prop, 1000)
        assert not signal.is_bet
        assert "no game logs" in signal.reason

    def test_a_single_one_sided_quote_cannot_be_priced(self, scanner):
        prop = PropMarket(
            sport="nba", event_id="x", commence_time=datetime.now(timezone.utc) + timedelta(hours=2),
            home_team="A", away_team="B", player="Anthony Edwards", market="player_points",
            books=(BookLine("pinnacle", 25.5, -110, None),),
        )
        signal = scanner.evaluate(prop, 1000)
        assert not signal.is_bet

    def test_an_efficient_market_is_passed(self, scanner):
        """Five books at the same fair price: nothing to bet, by construction."""
        prop = PropMarket(
            sport="nba", event_id="x", commence_time=datetime.now(timezone.utc) + timedelta(hours=2),
            home_team="MIN", away_team="SAC", player="Anthony Edwards", market="player_points",
            team="MIN", opponent="SAC",
            books=tuple(
                BookLine(book, 25.5, -110, -110)
                for book in ("pinnacle", "draftkings", "fanduel", "betmgm", "caesars")
            ),
        )
        signal = scanner.evaluate(prop, 1000)
        assert not signal.is_bet

    def test_absurd_prices_are_skipped(self, scanner, settings):
        """A +900 lottery ticket is not a Kelly bet, whatever the model says."""
        settings.staking.max_american = 300
        prop = PropMarket(
            sport="nba", event_id="x", commence_time=datetime.now(timezone.utc) + timedelta(hours=2),
            home_team="MIN", away_team="SAC", player="Anthony Edwards", market="player_points",
            team="MIN", opponent="SAC",
            books=(
                BookLine("pinnacle", 40.5, 900, -1400),
                BookLine("fanduel", 40.5, 700, -1100),
                BookLine("betmgm", 40.5, 750, -1200),
            ),
        )
        signal = PropScanner(settings).evaluate(prop, 1000)
        assert not signal.is_bet

    def test_games_already_underway_are_skipped(self, scanner, monkeypatch):
        props = scanner.odds.player_props("nba", ["player_points"])
        monkeypatch.setattr(
            scanner.odds, "player_props",
            lambda *a, **k: [
                p.__class__(**{**p.__dict__, "commence_time": datetime.now(timezone.utc) - timedelta(hours=1)})
                for p in props
            ],
        )
        assert scanner.scan("nba", include_passes=True) == []
