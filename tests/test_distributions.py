import math

import pytest

from pmbot.distributions import (
    BetaBinomial,
    Binomial,
    Empirical,
    NegativeBinomial,
    Normal,
    Poisson,
    betabinom_from_moments,
    negbin_from_moments,
    outcome_probs,
)


def total_mass(dist, upto=200):
    return sum(dist.p_exactly(k) for k in range(upto))


class TestFamilies:
    def test_poisson_is_a_distribution(self):
        p = Poisson(6.0)
        assert total_mass(p) == pytest.approx(1.0, abs=1e-9)
        assert p.p_at_most(6) == pytest.approx(sum(p.p_exactly(k) for k in range(7)))
        assert p.sd == pytest.approx(math.sqrt(6.0))

    def test_negative_binomial_is_overdispersed(self):
        nb = NegativeBinomial(6.0, size=4.0)
        assert total_mass(nb) == pytest.approx(1.0, abs=1e-6)
        assert nb.sd > Poisson(6.0).sd
        assert nb.sd == pytest.approx(math.sqrt(6.0 + 36.0 / 4.0))

    def test_binomial_matches_its_moments(self):
        b = Binomial(10, 0.3)
        assert total_mass(b, 20) == pytest.approx(1.0, abs=1e-12)
        assert b.mean == pytest.approx(3.0)
        assert b.sd == pytest.approx(math.sqrt(10 * 0.3 * 0.7))

    def test_beta_binomial_widens_the_binomial(self):
        n, p = 10, 0.4
        bb = BetaBinomial(n, alpha=0.4 * 8, beta=0.6 * 8)
        assert total_mass(bb, 20) == pytest.approx(1.0, abs=1e-9)
        assert bb.mean == pytest.approx(n * p)
        assert bb.sd > Binomial(n, p).sd

    def test_normal_uses_a_continuity_correction(self):
        n = Normal(20.0, 5.0)
        # A whole number has real mass on the lattice, unlike a raw Gaussian.
        assert n.p_exactly(20) == pytest.approx(0.0797, abs=5e-3)
        assert n.p_at_most(19) + n.p_exactly(20) == pytest.approx(n.p_at_most(20))

    def test_empirical_tracks_its_samples(self):
        e = Empirical([1, 2, 2, 3, 10])
        assert e.mean == pytest.approx(3.6)
        assert e.p_exactly(2) == pytest.approx(0.4)
        assert e.p_at_most(2) == pytest.approx(0.6)
        assert e.quantile(0.5) == 2

    def test_empirical_needs_samples(self):
        with pytest.raises(ValueError):
            Empirical([])


class TestMomentFitting:
    def test_negbin_from_moments_matches_the_target(self):
        dist = negbin_from_moments(8.0, 14.0)
        assert dist.mean == pytest.approx(8.0)
        assert dist.sd**2 == pytest.approx(14.0, rel=1e-6)

    def test_underdispersed_counts_fall_back_to_poisson(self):
        # Variance below the mean has no negative binomial representation;
        # pretending otherwise produces absurd tail probabilities.
        assert isinstance(negbin_from_moments(8.0, 4.0), Poisson)

    def test_betabinom_from_moments_matches_the_target(self):
        dist = betabinom_from_moments(12, 6.0, 4.5)
        assert dist.mean == pytest.approx(6.0)
        assert dist.sd**2 == pytest.approx(4.5, rel=1e-6)

    def test_betabinom_degenerates_to_binomial_without_extra_variance(self):
        assert isinstance(betabinom_from_moments(10, 4.0, 10 * 0.4 * 0.6), Binomial)

    def test_betabinom_clamps_past_the_families_limit(self):
        dist = betabinom_from_moments(6, 3.0, 999.0)
        assert 0.999 < sum(dist.p_exactly(k) for k in range(7)) <= 1.0


class TestOutcomeProbs:
    def test_half_point_lines_cannot_push(self):
        probs = outcome_probs(Poisson(6.0), 6.5)
        assert probs.p_push == 0.0
        assert probs.p_over + probs.p_under == pytest.approx(1.0)

    def test_whole_number_lines_push_on_the_number(self):
        dist = Poisson(6.0)
        probs = outcome_probs(dist, 6.0)
        assert probs.p_push == pytest.approx(dist.p_exactly(6), abs=1e-9)
        assert probs.p_over == pytest.approx(1 - dist.p_at_most(6), abs=1e-9)

    def test_over_means_strictly_greater(self):
        dist = Binomial(4, 0.5)
        probs = outcome_probs(dist, 2.0)
        assert probs.p_over == pytest.approx(dist.p_exactly(3) + dist.p_exactly(4))
        assert probs.p_under == pytest.approx(dist.p_exactly(0) + dist.p_exactly(1))

    def test_no_push_normalisation_excludes_the_push(self):
        probs = outcome_probs(Poisson(6.0), 6.0)
        assert probs.p_over_nopush == pytest.approx(probs.p_over / (probs.p_over + probs.p_under))
        assert probs.p_over_nopush + probs.p_under_nopush == pytest.approx(1.0)

    def test_probabilities_always_sum_to_one(self):
        for line in (0.5, 3.0, 5.5, 12.0):
            probs = outcome_probs(NegativeBinomial(6.0, 3.0), line)
            assert probs.p_over + probs.p_under + probs.p_push == pytest.approx(1.0)

    def test_side_lookup_rejects_nonsense(self):
        probs = outcome_probs(Poisson(3.0), 2.5)
        assert probs.side("over") == probs.p_over
        with pytest.raises(ValueError):
            probs.side("middle")

    def test_a_line_far_below_the_mean_is_nearly_certain(self):
        probs = outcome_probs(Normal(250.0, 40.0), 100.5)
        assert probs.p_over > 0.999
