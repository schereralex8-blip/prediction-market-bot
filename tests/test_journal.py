from datetime import datetime, timezone

import pytest

from pmbot.journal import (
    Journal,
    bankroll_curve,
    breakdown,
    calibration,
    grade,
    max_drawdown,
    summarise,
)
from pmbot.oddsmath import american_to_decimal
from pmbot.types import Signal


@pytest.fixture
def journal(tmp_path):
    with Journal(tmp_path / "test.sqlite3") as j:
        j.add_funds(1000, "deposit", "opening balance")
        yield j


def place(journal, **kw):
    defaults = dict(
        sport="nba", player="Anthony Edwards", market="player_points", side="over",
        line=25.5, book="caesars", american=-110, stake=50.0,
    )
    return journal.log_bet(**{**defaults, **kw})


class TestBankroll:
    def test_deposits_set_the_balance(self, journal):
        assert journal.balance() == pytest.approx(1000)
        assert journal.available() == pytest.approx(1000)

    def test_pending_stakes_are_still_yours_but_not_available(self, journal):
        place(journal, stake=100)
        assert journal.balance() == pytest.approx(1000)
        assert journal.available() == pytest.approx(900)
        assert journal.pending_stake() == pytest.approx(100)

    def test_withdrawals_are_signed_correctly_whichever_way_you_pass_them(self, journal):
        journal.add_funds(200, "withdrawal")
        assert journal.balance() == pytest.approx(800)
        journal.add_funds(-100, "withdrawal")
        assert journal.balance() == pytest.approx(700)

    def test_settled_profit_moves_the_balance(self, journal):
        bet = place(journal, stake=100, american=100)
        journal.settle(bet, "won")
        assert journal.balance() == pytest.approx(1100)
        assert journal.available() == pytest.approx(1100)

    def test_a_deposit_must_be_positive(self, journal):
        with pytest.raises(ValueError):
            journal.add_funds(-5, "deposit")

    def test_unknown_movement_types_are_rejected(self, journal):
        with pytest.raises(ValueError):
            journal.add_funds(5, "gift")


class TestLoggingBets:
    def test_a_bet_stores_its_reasoning(self, journal):
        bet_id = place(journal, model_prob=0.56, fair_prob=0.52, edge=0.04, ev=0.07)
        bet = journal.get(bet_id)
        assert bet.model_prob == pytest.approx(0.56)
        assert bet.edge == pytest.approx(0.04)
        assert bet.decimal_odds == pytest.approx(american_to_decimal(-110))
        assert bet.status == "pending"
        assert bet.stake_units == pytest.approx(5.0)  # 50 on a 1000 bankroll

    def test_a_signal_can_be_logged_directly(self, journal):
        signal = Signal(
            action="BET", sport="nfl", event_id="e1",
            commence_time=datetime.now(timezone.utc), matchup="A @ B",
            player="Josh Allen", market="player_pass_yds", market_label="Passing yards",
            reason="both models on over", side="over", line=236.5, book="draftkings",
            american=150, decimal=2.5, fair_prob=0.44, model_prob=0.46, edge=0.04,
            ev_per_unit=0.10, stake=80.0, model_probs={"a": 0.46, "b": 0.45},
        )
        bet = journal.get(journal.log_signal(signal))
        assert bet.player == "Josh Allen"
        assert bet.american == 150
        assert bet.model_probs == {"a": 0.46, "b": 0.45}

    def test_the_price_can_be_overridden_because_lines_move(self, journal):
        signal = Signal(
            action="BET", sport="nfl", event_id="e1", commence_time=datetime.now(timezone.utc),
            matchup="A @ B", player="Josh Allen", market="player_pass_yds",
            market_label="Passing yards", reason="r", side="over", line=236.5,
            book="draftkings", american=150, stake=80.0,
        )
        bet = journal.get(journal.log_signal(signal, stake=40, american=135))
        assert (bet.stake, bet.american) == (40, 135)

    def test_a_pass_cannot_be_logged_as_a_bet(self, journal):
        pass_signal = Signal(
            action="PASS", sport="nba", event_id="e", commence_time=datetime.now(timezone.utc),
            matchup="A @ B", player="X", market="player_points", market_label="Points",
            reason="models disagree",
        )
        with pytest.raises(ValueError, match="only BET signals"):
            journal.log_signal(pass_signal)

    def test_stakes_must_be_positive(self, journal):
        with pytest.raises(ValueError):
            place(journal, stake=0)

    def test_sides_are_validated(self, journal):
        with pytest.raises(ValueError):
            place(journal, side="sideways")


class TestSettlement:
    @pytest.mark.parametrize(
        "side,line,actual,expected",
        [
            ("over", 25.5, 31, "won"), ("over", 25.5, 20, "lost"),
            ("under", 25.5, 20, "won"), ("under", 25.5, 31, "lost"),
            ("over", 26.0, 26, "push"), ("under", 26.0, 26, "push"),
        ],
    )
    def test_grading_from_the_box_score(self, side, line, actual, expected):
        assert grade(side, line, actual) == expected

    def test_settling_computes_profit(self, journal):
        won = journal.settle(place(journal, stake=100, american=150), "won")
        assert won.profit == pytest.approx(150.0)
        lost = journal.settle(place(journal, stake=100, american=150), "lost")
        assert lost.profit == pytest.approx(-100.0)

    def test_a_push_returns_the_stake(self, journal):
        bet = journal.settle(place(journal, stake=100), "push")
        assert bet.profit == pytest.approx(0.0)
        assert journal.balance() == pytest.approx(1000)

    def test_settling_from_the_actual_result(self, journal):
        bet = journal.settle_with_result(place(journal, line=25.5, side="over"), 27)
        assert bet.status == "won"
        assert bet.result_value == 27

    def test_a_scratched_player_voids(self, journal):
        bet = journal.void(place(journal, stake=100), note="pitcher scratched")
        assert bet.status == "void"
        assert bet.profit == pytest.approx(0.0)
        assert journal.balance() == pytest.approx(1000)

    def test_unknown_status_is_rejected(self, journal):
        with pytest.raises(ValueError):
            journal.settle(place(journal), "sort of won")

    def test_settling_a_ghost_bet_fails_loudly(self, journal):
        with pytest.raises(KeyError):
            journal.settle(9999, "won")


class TestClosingLineValue:
    def test_clv_is_measured_against_the_devigged_close(self, journal):
        # Bet +140, closed -110/-110 (fair 50%): the price beat fair by 20%.
        bet_id = place(journal, american=140)
        bet = journal.record_closing(bet_id, over_american=-110, under_american=-110)
        assert bet.closing_fair_prob == pytest.approx(0.5)
        assert bet.clv_pct == pytest.approx(2.4 * 0.5 - 1)
        assert bet.clv_prob == pytest.approx(0.5 - 1 / 2.4)

    def test_a_price_that_got_worse_shows_negative_clv(self, journal):
        bet_id = place(journal, american=-140, side="over")
        bet = journal.record_closing(bet_id, over_american=-110, under_american=-110)
        assert bet.clv_pct < 0

    def test_vig_is_removed_before_comparing(self, journal):
        """A line that never moved is a losing bet, and CLV should say so.

        Measuring against the raw closing price would score this as break
        even and flatter every bet ever placed by roughly half the hold.
        """
        bet_id = place(journal, american=-110)
        bet = journal.record_closing(bet_id, over_american=-110, under_american=-110)
        assert bet.clv_pct == pytest.approx(-0.0455, abs=1e-3)  # paid the vig
        assert bet.clv_raw_pct == pytest.approx(0.0, abs=1e-9)  # price didn't move

    def test_closing_lines_can_be_matched_from_a_snapshot(self, journal, tmp_path):
        from pmbot.providers import LocalOddsProvider

        place(journal, player="Anthony Edwards", market="player_points", line=27.5, american=165)
        props = LocalOddsProvider("data/props").player_props("nba", ["player_points"])
        # Line the bet up with whatever the fixture actually quotes.
        target = next(p for p in props if p.player == "Anthony Edwards")
        journal.conn.execute("UPDATE bets SET line = ?", (target.books[0].line,))
        updated = journal.record_closing_from_props(props)
        assert updated
        assert journal.get(updated[0]).closing_fair_prob is not None


class TestMetrics:
    def build(self, journal):
        outcomes = [("won", 100, -110), ("lost", 100, -110), ("won", 50, 150),
                    ("lost", 50, 120), ("push", 75, -110), ("won", 100, -105)]
        for status, stake, price in outcomes:
            bet_id = place(journal, stake=stake, american=price, model_prob=0.56, ev=0.05)
            journal.settle(bet_id, status)
        return journal.bets()

    def test_summary_arithmetic(self, journal):
        bets = self.build(journal)
        perf = summarise(bets)
        assert perf.n_settled == 6
        assert (perf.wins, perf.losses, perf.pushes) == (3, 2, 1)
        assert perf.staked == pytest.approx(475.0)
        assert perf.win_rate == pytest.approx(3 / 5)
        assert perf.roi == pytest.approx(perf.profit / perf.staked)
        assert perf.breakeven_rate == pytest.approx(1 / perf.avg_decimal)

    def test_expected_profit_and_luck_are_reported(self, journal):
        perf = summarise(self.build(journal))
        assert perf.expected_profit == pytest.approx(0.05 * 475.0)
        assert perf.luck == pytest.approx(perf.profit - perf.expected_profit)

    def test_confidence_interval_brackets_the_point_estimate(self, journal):
        perf = summarise(self.build(journal))
        lo, hi = perf.roi_ci
        assert lo <= perf.roi <= hi
        assert perf.t_stat is not None

    def test_pending_bets_do_not_pollute_the_record(self, journal):
        self.build(journal)
        place(journal, stake=999)
        perf = summarise(journal.bets())
        assert perf.n_pending == 1
        assert perf.staked == pytest.approx(475.0)

    def test_voids_are_excluded_from_staked_and_roi(self, journal):
        journal.settle(place(journal, stake=500), "void")
        perf = summarise(journal.bets())
        assert perf.voids == 1
        assert perf.staked == 0.0

    def test_clv_summarises_across_bets(self, journal):
        for price in (140, -140):
            journal.record_closing(
                place(journal, american=price), over_american=-110, under_american=-110
            )
        perf = summarise(journal.bets())
        assert perf.clv_n == 2
        assert perf.beat_close_rate == pytest.approx(0.5)

    def test_breakdown_splits_by_key(self, journal):
        place(journal, sport="nba", book="caesars", stake=100)
        place(journal, sport="nfl", book="fanduel", stake=100)
        journal.settle(1, "won")
        journal.settle(2, "lost")
        rows = {p.label: p for p in breakdown(journal.bets(), "sport")}
        assert set(rows) == {"nba", "nfl"}
        assert rows["nba"].profit > 0 > rows["nfl"].profit

    def test_breakdown_rejects_unknown_keys(self, journal):
        with pytest.raises(ValueError):
            breakdown(journal.bets(), "astrological_sign")

    def test_bankroll_curve_and_drawdown(self, journal):
        for status in ("won", "lost", "lost", "won"):
            journal.settle(place(journal, stake=100, american=100), status)
        curve = bankroll_curve(journal.bets(), starting=1000)
        assert curve[-1].balance == pytest.approx(1000 + 100 - 100 - 100 + 100)
        assert max_drawdown(curve) == pytest.approx(200.0)

    def test_calibration_compares_predicted_with_actual(self, journal):
        for i in range(10):
            bet = place(journal, model_prob=0.65, american=100)
            journal.settle(bet, "won" if i < 5 else "lost")
        cal = calibration(journal.bets())
        assert cal.buckets
        label, n, predicted, actual = cal.buckets[0]
        assert (n, predicted, actual) == (10, pytest.approx(0.65), pytest.approx(0.5))
        assert cal.brier() == pytest.approx((0.65 - 0.5) ** 2)


class TestQueries:
    def test_filters_stack(self, journal):
        place(journal, sport="nba", market="player_points", book="caesars")
        place(journal, sport="nfl", market="player_pass_yds", book="fanduel")
        assert len(journal.bets(sport="nba")) == 1
        assert len(journal.bets(market="player_pass_yds")) == 1
        assert len(journal.bets(book="caesars")) == 1
        assert len(journal.bets()) == 2

    def test_pending_and_settled_are_separable(self, journal):
        journal.settle(place(journal), "won")
        place(journal)
        assert len(journal.pending()) == 1
        assert len(journal.bets(status="settled")) == 1

    def test_the_database_survives_reopening(self, tmp_path):
        path = tmp_path / "reopen.sqlite3"
        with Journal(path) as first:
            first.add_funds(500)
            place(first, stake=25)
        with Journal(path) as second:
            assert second.balance() == pytest.approx(500)
            assert len(second.pending()) == 1
