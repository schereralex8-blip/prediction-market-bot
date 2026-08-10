"""The two models, and the gate that makes them agree.

The bar these tests hold the models to: given game logs generated from a known
process, each model's projection must land near the truth, and its
probabilities must be calibrated enough to bet on.
"""

import random
from datetime import date, timedelta

import pytest

from pmbot.config import GateSettings, ModelSettings
from pmbot.markets import get_spec
from pmbot.models.consensus import evaluate_consensus
from pmbot.models.model_a import BayesRateModel
from pmbot.models.model_b import BootstrapModel
from pmbot.types import GameLog, MatchupContext, ModelOutput


def nba_logs(n=40, minutes=34.0, minutes_sd=3.0, pts_per_min=0.75, seed=1):
    """Points scale with minutes; both wobble. The truth is mean ~= 25.5."""
    rng = random.Random(seed)
    logs = []
    for i in range(n):
        mins = max(10.0, rng.gauss(minutes, minutes_sd))
        pts = max(0, round(rng.gauss(pts_per_min * mins, 0.9 * (pts_per_min * mins) ** 0.5 * 2)))
        logs.append(
            GameLog(
                date=date(2026, 1, 1) + timedelta(days=2 * i),
                player="Test Player",
                team="AAA",
                opponent="BBB",
                is_home=i % 2 == 0,
                stats={"minutes": round(mins, 1), "points": pts, "rebounds": rng.randint(2, 9)},
            )
        )
    return logs


def logs_mean(logs, spec):
    return sum(g.total(spec.components) for g in logs) / len(logs)


SPEC = get_spec("player_points")
REB = get_spec("player_rebounds")


@pytest.fixture
def models():
    settings = ModelSettings(mc_sims=6000)
    return BayesRateModel(settings), BootstrapModel(settings)


class TestProjectionAccuracy:
    """Both models must recover the mean of the process that made the logs."""

    def test_model_a_tracks_the_sample(self, models):
        logs = nba_logs()
        fit = models[0].fit(spec=SPEC, logs=logs)
        assert fit.ok
        assert fit.dist.mean == pytest.approx(logs_mean(logs, SPEC), rel=0.06)

    def test_model_b_tracks_the_sample(self, models):
        logs = nba_logs()
        fit = models[1].fit(spec=SPEC, logs=logs)
        assert fit.ok
        assert fit.dist.mean == pytest.approx(logs_mean(logs, SPEC), rel=0.06)

    def test_the_two_models_broadly_agree_on_clean_data(self, models):
        logs = nba_logs()
        a = models[0].predict(spec=SPEC, logs=logs, line=25.5)
        b = models[1].predict(spec=SPEC, logs=logs, line=25.5)
        assert abs(a.probs.p_over_nopush - b.probs.p_over_nopush) < 0.08

    def test_a_star_is_not_dragged_to_the_league_average(self, models):
        """The bug that makes a model bet unders forever."""
        elite = nba_logs(pts_per_min=1.0, minutes=36.0, seed=5)  # ~36 points a night
        for model in models:
            fit = model.fit(spec=SPEC, logs=elite)
            assert fit.dist.mean > 0.94 * logs_mean(elite, SPEC)

    def test_workload_variance_reaches_the_projection(self, models):
        """Minutes that swing wildly must widen the distribution."""
        steady = nba_logs(minutes_sd=1.0, seed=3)
        swingy = nba_logs(minutes_sd=8.0, seed=3)
        for model in models:
            assert model.fit(spec=SPEC, logs=swingy).dist.sd > model.fit(spec=SPEC, logs=steady).dist.sd


class TestModelBehaviour:
    def test_thin_samples_are_refused_rather_than_guessed(self, models):
        for model in models:
            fit = model.fit(spec=SPEC, logs=nba_logs(n=4))
            assert not fit.ok
            assert "usable games" in fit.reason

    def test_dnp_games_are_dropped(self, models):
        logs = nba_logs(n=20)
        logs.append(
            GameLog(date=date(2026, 6, 1), player="Test Player", team="AAA", opponent="BBB",
                    is_home=True, stats={"minutes": 0.0, "points": 0})
        )
        fit = models[0].fit(spec=SPEC, logs=logs)
        assert fit.n_games == 20
        assert fit.details["dnp_games_dropped"] == 1

    def test_a_soft_matchup_raises_the_projection(self, models):
        logs = nba_logs()
        soft = MatchupContext(opponent_factor=1.10)
        for model in models:
            base = model.fit(spec=SPEC, logs=logs).dist.mean
            assert model.fit(spec=SPEC, logs=logs, context=soft).dist.mean > base * 1.05

    def test_a_minutes_cap_lowers_the_projection(self, models):
        logs = nba_logs()
        capped = MatchupContext(workload_override=24.0)
        for model in models:
            base = model.fit(spec=SPEC, logs=logs).dist.mean
            assert model.fit(spec=SPEC, logs=logs, context=capped).dist.mean < base * 0.8

    def test_higher_lines_are_less_likely(self, models):
        logs = nba_logs()
        for model in models:
            probs = [
                model.predict(spec=SPEC, logs=logs, line=x).probs.p_over_nopush
                for x in (20.5, 25.5, 30.5)
            ]
            assert probs[0] > probs[1] > probs[2]

    def test_model_b_is_reproducible(self):
        """A simulation you can't reproduce is a number you can't audit."""
        logs = nba_logs()
        settings = ModelSettings(mc_sims=4000)
        first = BootstrapModel(settings).predict(spec=SPEC, logs=logs, line=25.5)
        second = BootstrapModel(settings).predict(spec=SPEC, logs=logs, line=25.5)
        assert first.probs.p_over == second.probs.p_over

    def test_count_markets_use_a_discrete_family(self, models):
        logs = nba_logs()
        out = models[0].predict(spec=REB, logs=logs, line=5.0)
        assert out.probs.p_push > 0  # a whole-number rebound line can push

    def test_models_report_their_workings(self, models):
        logs = nba_logs()
        details_a = models[0].fit(spec=SPEC, logs=logs).details
        details_b = models[1].fit(spec=SPEC, logs=logs).details
        assert {"rate", "career_rate", "workload_proj"} <= details_a.keys()
        assert {"target_workload", "sims", "effective_games"} <= details_b.keys()


def out(model, p_over, p_push=0.0):
    from pmbot.distributions import OutcomeProbs

    return ModelOutput(
        model=model, ok=True, n_games=30,
        probs=OutcomeProbs(p_over, 1 - p_over - p_push, p_push),
    )


class TestAgreementGate:
    settings = GateSettings()

    def gate(self, a, b, side="over", breakeven=0.5, fair=0.5, **kw):
        settings = GateSettings(**kw) if kw else self.settings
        return evaluate_consensus(a, b, side=side, breakeven=breakeven, fair_prob=fair, settings=settings)

    def test_both_models_ahead_of_the_price_is_a_bet(self):
        c = self.gate(out("a", 0.58), out("b", 0.57), breakeven=0.52, fair=0.55)
        assert c.agree and c.side == "over"

    def test_disagreement_is_a_pass(self):
        c = self.gate(out("a", 0.62), out("b", 0.50), breakeven=0.48)
        assert not c.agree
        assert "disagree" in c.reason

    def test_one_model_short_of_the_floor_is_a_pass(self):
        c = self.gate(out("a", 0.58), out("b", 0.525), breakeven=0.52)
        assert not c.agree
        assert "clears the price by only" in c.reason

    def test_wild_divergence_from_the_market_is_treated_as_bad_data(self):
        c = self.gate(out("a", 0.70), out("b", 0.69), breakeven=0.50, fair=0.45)
        assert not c.agree
        assert "stale data" in c.reason

    def test_missing_model_output_is_a_pass_not_a_crash(self):
        broken = ModelOutput(model="b", ok=False, reason="only 3 usable games")
        c = self.gate(out("a", 0.60), broken)
        assert not c.agree
        assert "insufficient data" in c.reason

    def test_consensus_is_shrunk_toward_the_market(self):
        c = self.gate(out("a", 0.60), out("b", 0.60), breakeven=0.50, fair=0.52, market_blend=0.25)
        assert c.p_win == pytest.approx(0.75 * 0.60 + 0.25 * 0.52)

    def test_min_takes_the_cautious_model(self):
        c = self.gate(out("a", 0.62), out("b", 0.58), breakeven=0.50, fair=0.58,
                      market_blend=0.0, combine="min")
        assert c.p_win == pytest.approx(0.58)

    def test_mean_splits_the_difference(self):
        c = self.gate(out("a", 0.62), out("b", 0.58), breakeven=0.50, fair=0.60,
                      market_blend=0.0, combine="mean")
        assert c.p_win == pytest.approx(0.60)

    def test_pushes_are_carried_through(self):
        c = self.gate(out("a", 0.55, p_push=0.10), out("b", 0.54, p_push=0.10),
                      breakeven=0.50, fair=0.55, market_blend=0.0)
        assert c.p_push == pytest.approx(0.10)
        # p_win is unconditional, so it must be discounted by the push mass.
        assert c.p_win == pytest.approx(0.9 * c.p_over_nopush)

    def test_unknown_combine_method_is_rejected(self):
        with pytest.raises(ValueError):
            self.gate(out("a", 0.60), out("b", 0.60), breakeven=0.5, fair=0.6, combine="vibes")
