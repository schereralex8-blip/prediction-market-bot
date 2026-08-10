import math

import pytest

from pmbot.oddsmath import (
    DEVIG_METHODS,
    american_to_decimal,
    american_to_prob,
    decimal_to_american,
    devig,
    devig_two_way,
    format_american,
    hold_pct,
    overround,
    prob_to_american,
    prob_to_decimal,
)


class TestConversions:
    @pytest.mark.parametrize(
        "american,decimal",
        [(100, 2.0), (150, 2.5), (-110, 1 + 100 / 110), (-200, 1.5), (300, 4.0)],
    )
    def test_american_to_decimal(self, american, decimal):
        assert american_to_decimal(american) == pytest.approx(decimal)

    @pytest.mark.parametrize("american", [-500, -250, -110, -101, 105, 137, 900])
    def test_round_trip(self, american):
        assert decimal_to_american(american_to_decimal(american)) == pytest.approx(american)

    def test_even_money_is_a_coin_flip(self):
        assert american_to_prob(100) == pytest.approx(0.5)

    def test_favourites_imply_more_than_half(self):
        assert american_to_prob(-150) == pytest.approx(0.6)

    def test_prob_round_trip(self):
        for p in (0.05, 0.25, 0.5, 0.524, 0.9):
            assert 1 / prob_to_decimal(p) == pytest.approx(p)
            assert american_to_prob(prob_to_american(p)) == pytest.approx(p, abs=1e-3)

    def test_zero_is_not_a_price(self):
        with pytest.raises(ValueError):
            american_to_decimal(0)

    def test_decimal_must_beat_one(self):
        with pytest.raises(ValueError):
            decimal_to_american(0.95)

    def test_formatting_keeps_the_plus(self):
        assert format_american(150) == "+150"
        assert format_american(-110) == "-110"


class TestOverround:
    def test_standard_juice(self):
        assert overround([-110, -110]) == pytest.approx(1.0476, abs=1e-4)
        assert hold_pct([-110, -110]) == pytest.approx(0.04545, abs=1e-4)

    def test_a_fair_market_has_no_hold(self):
        assert hold_pct([100, 100]) == pytest.approx(0.0)


class TestDevig:
    @pytest.mark.parametrize("method", DEVIG_METHODS)
    def test_every_method_returns_a_probability_distribution(self, method):
        fair = devig([american_to_prob(-115), american_to_prob(-105)], method=method)
        assert sum(fair) == pytest.approx(1.0)
        assert all(0.0 < p < 1.0 for p in fair)

    @pytest.mark.parametrize("method", DEVIG_METHODS)
    def test_the_favourite_stays_the_favourite(self, method):
        fair = devig([american_to_prob(-200), american_to_prob(170)], method=method)
        assert fair[0] > fair[1]

    @pytest.mark.parametrize("method", DEVIG_METHODS)
    def test_a_balanced_market_devigs_to_a_coin_flip(self, method):
        fair = devig([american_to_prob(-110), american_to_prob(-110)], method=method)
        assert fair[0] == pytest.approx(0.5, abs=1e-9)

    @pytest.mark.parametrize("method", DEVIG_METHODS)
    def test_devigging_always_lowers_the_implied_probability(self, method):
        raw = [american_to_prob(-140), american_to_prob(115)]
        for r, f in zip(raw, devig(raw, method=method)):
            assert f < r

    def test_power_solves_its_defining_equation(self):
        raw = [american_to_prob(-250), american_to_prob(200)]
        fair = devig(raw, method="power")
        k = math.log(fair[0]) / math.log(raw[0])
        assert sum(r**k for r in raw) == pytest.approx(1.0, abs=1e-6)

    def test_methods_disagree_on_longshots_which_is_the_whole_point(self):
        raw = [american_to_prob(-400), american_to_prob(320)]
        mult = devig(raw, "multiplicative")[1]
        power = devig(raw, "power")[1]
        shin = devig(raw, "shin")[1]
        assert power != pytest.approx(mult, abs=1e-4)
        # Shin and power both take more vig off the longshot than multiplicative.
        assert shin < mult and power < mult

    def test_conservative_is_the_most_pessimistic(self):
        raw = [american_to_prob(-120), american_to_prob(100)]
        conservative = devig(raw, "conservative")[0]
        assert conservative > devig(raw, "multiplicative")[0]

    def test_three_way_markets_work(self):
        raw = [american_to_prob(x) for x in (150, 250, 180)]
        fair = devig(raw, "power")
        assert sum(fair) == pytest.approx(1.0)
        assert len(fair) == 3

    def test_no_vig_input_is_just_normalised(self):
        fair = devig([0.5, 0.5], "power")
        assert fair == pytest.approx([0.5, 0.5])

    def test_a_market_needs_two_sides(self):
        with pytest.raises(ValueError):
            devig([0.5], "power")

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="unknown devig method"):
            devig([0.55, 0.52], "vibes")  # type: ignore[arg-type]

    def test_two_way_helper_matches_the_general_case(self):
        over, under = devig_two_way(-130, 110, "power")
        assert over + under == pytest.approx(1.0)
        assert over > under

    def test_additive_falls_back_when_it_would_go_negative(self):
        # A huge favourite plus a longshot: additive would drive the longshot
        # below zero, so it must degrade gracefully instead of returning junk.
        fair = devig([american_to_prob(-5000), american_to_prob(1200)], "additive")
        assert all(p > 0 for p in fair)
        assert sum(fair) == pytest.approx(1.0)
