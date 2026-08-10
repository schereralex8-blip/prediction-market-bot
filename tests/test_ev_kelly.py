import pytest

from pmbot.ev import breakeven_prob, evaluate, expected_value
from pmbot.kelly import cap_total_exposure, full_kelly_fraction, growth_rate, size_bet
from pmbot.oddsmath import american_to_decimal


class TestExpectedValue:
    def test_a_fair_coin_at_even_money_is_zero_ev(self):
        assert expected_value(0.5, 0.5, 2.0) == pytest.approx(0.0)

    def test_edge_shows_up_as_positive_ev(self):
        assert expected_value(0.55, 0.45, 2.0) == pytest.approx(0.10)

    def test_standard_juice_needs_a_real_edge(self):
        assert expected_value(0.5, 0.5, american_to_decimal(-110)) < 0

    def test_breakeven_matches_the_price(self):
        assert breakeven_prob(american_to_decimal(-110)) == pytest.approx(0.5238, abs=1e-4)
        assert breakeven_prob(2.5) == pytest.approx(0.4)

    def test_pushes_lower_the_bar(self):
        # A quarter of the time the stake comes back, so less has to be won.
        assert breakeven_prob(2.0, p_push=0.25) == pytest.approx(0.375)

    def test_push_probability_must_be_a_probability(self):
        with pytest.raises(ValueError):
            breakeven_prob(2.0, p_push=1.0)


class TestEvaluate:
    def test_price_edge_drives_ev(self):
        result = evaluate(model_p_win=0.55, model_p_push=0.0, fair_prob=0.55, american_odds=110)
        # The model agrees with the market exactly, but the price is better
        # than the market's -- that is still a bet, and this is the case a
        # consensus-disagreement filter would wrongly throw away.
        assert result.market_edge == pytest.approx(0.0)
        assert result.price_edge > 0.0
        assert result.is_positive

    def test_ev_is_proportional_to_price_edge(self):
        result = evaluate(model_p_win=0.48, model_p_push=0.10, fair_prob=0.5, american_odds=150)
        expected = (1 - result.p_push) * result.decimal_odds * result.price_edge
        assert result.ev_per_unit == pytest.approx(expected)

    def test_a_bad_price_is_negative_ev(self):
        result = evaluate(model_p_win=0.5, model_p_push=0.0, fair_prob=0.5, american_odds=-140)
        assert not result.is_positive
        assert result.price_edge < 0

    def test_fair_american_is_the_break_even_price(self):
        result = evaluate(model_p_win=0.6, model_p_push=0.0, fair_prob=0.55, american_odds=100)
        neutral = evaluate(
            model_p_win=0.6, model_p_push=0.0, fair_prob=0.55, american_odds=result.fair_american
        )
        assert neutral.ev_per_unit == pytest.approx(0.0, abs=1e-3)

    def test_probabilities_are_validated(self):
        with pytest.raises(ValueError):
            evaluate(model_p_win=1.4, model_p_push=0.0, fair_prob=0.5, american_odds=100)


class TestKelly:
    def test_classic_formula(self):
        # p=0.6 at even money: f* = (0.6*1 - 0.4)/1 = 0.20
        assert full_kelly_fraction(0.6, 0.4, 2.0) == pytest.approx(0.2)

    def test_no_edge_means_no_bet(self):
        assert full_kelly_fraction(0.5, 0.5, 2.0) == pytest.approx(0.0)
        assert full_kelly_fraction(0.45, 0.55, 2.0) < 0

    @pytest.mark.parametrize(
        "p_win,p_push,decimal",
        [(0.55, 0.0, 2.0), (0.40, 0.15, 2.6), (0.62, 0.08, 1.75), (0.30, 0.30, 4.0)],
    )
    def test_closed_form_matches_a_brute_force_search(self, p_win, p_push, decimal):
        """The push-aware closed form must actually maximise log growth."""
        p_lose = 1 - p_win - p_push
        best = max(
            (i / 20000 for i in range(1, 19999)),
            key=lambda f: growth_rate(f, p_win, p_lose, decimal),
        )
        assert full_kelly_fraction(p_win, p_lose, decimal) == pytest.approx(best, abs=1e-3)

    def test_pushes_make_kelly_bet_bigger_than_treating_them_as_losses(self):
        with_push = full_kelly_fraction(0.5, 0.4, 2.0)
        as_losses = full_kelly_fraction(0.5, 0.5, 2.0)
        assert with_push > as_losses


class TestSizing:
    def test_fractional_kelly_scales_the_full_bet(self):
        stake = size_bet(
            p_win=0.6, p_push=0.0, decimal_odds=2.0, bankroll=10_000,
            kelly_fraction=0.25, max_bet_fraction=1.0, rounding=0.01,
        )
        assert stake.full_kelly == pytest.approx(0.2)
        assert stake.recommended_fraction == pytest.approx(0.05)
        assert stake.amount == pytest.approx(500.0)
        assert stake.units == pytest.approx(5.0)

    def test_the_cap_binds_on_big_edges(self):
        stake = size_bet(
            p_win=0.8, p_push=0.0, decimal_odds=2.0, bankroll=1000,
            kelly_fraction=0.5, max_bet_fraction=0.02,
        )
        assert stake.capped_by == "max_bet"
        assert stake.amount == pytest.approx(20.0)

    def test_no_edge_no_stake(self):
        stake = size_bet(p_win=0.45, p_push=0.0, decimal_odds=2.0, bankroll=1000)
        assert stake.amount == 0.0
        assert stake.recommended_fraction == 0.0

    def test_tiny_edges_round_away_rather_than_placing_a_silly_bet(self):
        stake = size_bet(
            p_win=0.5005, p_push=0.0, decimal_odds=2.0, bankroll=100,
            kelly_fraction=0.25, min_stake=1.0,
        )
        assert stake.capped_by == "min_stake"
        assert stake.amount == 0.0

    def test_stakes_round_down_not_up(self):
        stake = size_bet(
            p_win=0.6, p_push=0.0, decimal_odds=2.0, bankroll=1017,
            kelly_fraction=0.25, max_bet_fraction=1.0, rounding=5.0,
        )
        assert stake.amount % 5 == 0
        assert stake.amount <= 1017 * 0.05

    def test_growth_rate_is_positive_at_the_recommended_size(self):
        stake = size_bet(p_win=0.58, p_push=0.0, decimal_odds=2.0, bankroll=5000)
        assert stake.growth_rate > 0

    def test_bankroll_must_be_positive(self):
        with pytest.raises(ValueError):
            size_bet(p_win=0.6, p_push=0.0, decimal_odds=2.0, bankroll=0)

    def test_kelly_fraction_is_validated(self):
        with pytest.raises(ValueError):
            size_bet(p_win=0.6, p_push=0.0, decimal_odds=2.0, bankroll=100, kelly_fraction=1.5)


class TestExposureCap:
    def test_a_small_slate_is_left_alone(self):
        stakes = [10.0, 20.0]
        assert cap_total_exposure(stakes, 1000, 0.10) == stakes

    def test_a_big_slate_is_scaled_proportionally(self):
        scaled = cap_total_exposure([100.0, 200.0, 300.0], 1000, 0.10)
        assert sum(scaled) == pytest.approx(100.0)
        # Relative sizing survives the haircut.
        assert scaled[2] / scaled[0] == pytest.approx(3.0)

    def test_an_empty_slate_is_fine(self):
        assert cap_total_exposure([], 1000, 0.1) == []
