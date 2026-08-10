"""Outcome distributions for player props, on the integer lattice.

Props settle on integers: 7 rebounds, 84 receiving yards, 6 strikeouts. So
every distribution here exposes the same two lattice primitives --
``p_at_most(k)`` and ``p_exactly(k)`` -- and :func:`outcome_probs` turns those
into (over, under, push) for a given line. Continuous families get a
continuity correction so that a whole-number line reports a real push
probability instead of zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

_SQRT2 = math.sqrt(2.0)


class Distribution(Protocol):
    mean: float
    sd: float

    def p_at_most(self, k: int) -> float:
        """P(X <= k)."""

    def p_exactly(self, k: int) -> float:
        """P(X == k)."""


@dataclass(frozen=True)
class OutcomeProbs:
    p_over: float
    p_under: float
    p_push: float

    @property
    def p_over_nopush(self) -> float:
        live = self.p_over + self.p_under
        return self.p_over / live if live > 1e-12 else 0.5

    @property
    def p_under_nopush(self) -> float:
        return 1.0 - self.p_over_nopush

    def side(self, side: str) -> float:
        if side == "over":
            return self.p_over
        if side == "under":
            return self.p_under
        raise ValueError(f"side must be 'over' or 'under', got {side!r}")

    def side_nopush(self, side: str) -> float:
        return self.p_over_nopush if side == "over" else self.p_under_nopush


def outcome_probs(dist: Distribution, line: float) -> OutcomeProbs:
    """Split a distribution around a betting line.

    A whole-number line can push; a half-point line cannot. ``over`` always
    means strictly greater than the line.
    """
    if _is_integer(line):
        k = int(round(line))
        push = max(0.0, dist.p_exactly(k))
        under = max(0.0, dist.p_at_most(k) - push)
        over = max(0.0, 1.0 - under - push)
    else:
        k = math.floor(line)
        under = max(0.0, dist.p_at_most(k))
        over = max(0.0, 1.0 - under)
        push = 0.0
    total = over + under + push
    if total <= 0:  # pragma: no cover - defensive
        return OutcomeProbs(0.5, 0.5, 0.0)
    return OutcomeProbs(over / total, under / total, push / total)


def _is_integer(x: float, tol: float = 1e-9) -> bool:
    return abs(x - round(x)) < tol


# --------------------------------------------------------------------------
# families
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Normal:
    """Gaussian on the integer lattice (yards, points totals, fantasy scores)."""

    mean: float
    sd: float

    def _cdf(self, x: float) -> float:
        s = max(self.sd, 1e-9)
        return 0.5 * (1.0 + math.erf((x - self.mean) / (s * _SQRT2)))

    def p_at_most(self, k: int) -> float:
        return self._cdf(k + 0.5)  # continuity correction

    def p_exactly(self, k: int) -> float:
        return self._cdf(k + 0.5) - self._cdf(k - 0.5)


@dataclass(frozen=True)
class Poisson:
    """Counting stat where variance ~= mean (blocks, steals, home runs)."""

    mean: float

    @property
    def sd(self) -> float:
        return math.sqrt(max(self.mean, 1e-9))

    def p_exactly(self, k: int) -> float:
        if k < 0:
            return 0.0
        mu = max(self.mean, 1e-9)
        return math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1))

    def p_at_most(self, k: int) -> float:
        if k < 0:
            return 0.0
        return sum(self.p_exactly(i) for i in range(0, k + 1))


@dataclass(frozen=True)
class NegativeBinomial:
    """Counting stat that is over-dispersed, which in practice is all of them.

    Parameterised by mean and ``size`` (the dispersion r): var = mu + mu^2/r.
    Large r converges to Poisson.
    """

    mean: float
    size: float

    @property
    def sd(self) -> float:
        mu = max(self.mean, 1e-9)
        return math.sqrt(mu + mu * mu / max(self.size, 1e-9))

    def p_exactly(self, k: int) -> float:
        if k < 0:
            return 0.0
        mu = max(self.mean, 1e-9)
        r = max(self.size, 1e-6)
        log_p = (
            math.lgamma(k + r)
            - math.lgamma(r)
            - math.lgamma(k + 1)
            + r * math.log(r / (r + mu))
            + k * math.log(mu / (r + mu))
        )
        return math.exp(log_p)

    def p_at_most(self, k: int) -> float:
        if k < 0:
            return 0.0
        return sum(self.p_exactly(i) for i in range(0, k + 1))


@dataclass(frozen=True)
class Binomial:
    """n trials, p each (hits per plate appearance, TDs per red-zone look)."""

    n: int
    p: float

    @property
    def mean(self) -> float:
        return self.n * self.p

    @property
    def sd(self) -> float:
        return math.sqrt(max(self.n * self.p * (1.0 - self.p), 1e-12))

    def p_exactly(self, k: int) -> float:
        if k < 0 or k > self.n:
            return 0.0
        p = min(max(self.p, 1e-9), 1.0 - 1e-9)
        log_p = (
            math.lgamma(self.n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(self.n - k + 1)
            + k * math.log(p)
            + (self.n - k) * math.log(1.0 - p)
        )
        return math.exp(log_p)

    def p_at_most(self, k: int) -> float:
        if k < 0:
            return 0.0
        return sum(self.p_exactly(i) for i in range(0, min(k, self.n) + 1))


@dataclass(frozen=True)
class BetaBinomial:
    """Binomial with an uncertain success rate: p ~ Beta(a, b).

    Receptions on 7 targets is a binomial, but the catch rate is estimated
    from a few hundred targets, not known. Beta-binomial keeps the discrete
    n-trial shape while widening the tails by exactly the amount that
    estimation error justifies.
    """

    n: int
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.n * self.alpha / (self.alpha + self.beta)

    @property
    def sd(self) -> float:
        a, b, n = self.alpha, self.beta, self.n
        s = a + b
        var = n * a * b * (s + n) / (s * s * (s + 1.0))
        return math.sqrt(max(var, 1e-12))

    def p_exactly(self, k: int) -> float:
        if k < 0 or k > self.n:
            return 0.0
        a, b, n = self.alpha, self.beta, self.n
        log_p = (
            math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + math.lgamma(k + a) + math.lgamma(n - k + b) - math.lgamma(n + a + b)
            + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        )
        return math.exp(log_p)

    def p_at_most(self, k: int) -> float:
        if k < 0:
            return 0.0
        return sum(self.p_exactly(i) for i in range(0, min(k, self.n) + 1))


class Empirical:
    """Distribution defined by Monte Carlo draws, rounded to the lattice."""

    def __init__(self, samples: Sequence[float]) -> None:
        if not samples:
            raise ValueError("Empirical needs at least one sample")
        self._samples = sorted(int(round(s)) for s in samples)
        self._n = len(self._samples)
        self.mean = sum(self._samples) / self._n
        if self._n > 1:
            var = sum((s - self.mean) ** 2 for s in self._samples) / (self._n - 1)
        else:  # pragma: no cover - degenerate
            var = 0.0
        self.sd = math.sqrt(var)

    def p_at_most(self, k: int) -> float:
        import bisect

        return bisect.bisect_right(self._samples, k) / self._n

    def p_exactly(self, k: int) -> float:
        import bisect

        lo = bisect.bisect_left(self._samples, k)
        hi = bisect.bisect_right(self._samples, k)
        return (hi - lo) / self._n

    def quantile(self, q: float) -> float:
        idx = min(self._n - 1, max(0, int(q * self._n)))
        return float(self._samples[idx])


def betabinom_from_moments(n: int, mean: float, variance: float) -> Distribution:
    """Fit a beta-binomial to a target mean and variance on ``n`` trials.

    Var(BetaBinomial) = n*p*(1-p) * (s+n)/(s+1) where s is the beta
    concentration, so the over-dispersion ratio k = target/binomial pins s
    down exactly: s = (n - k) / (k - 1). At k <= 1 there's nothing to widen
    and it degenerates to a plain binomial; k >= n is past what the family can
    express, so the concentration is floored instead of going negative.
    """
    n = max(int(n), 1)
    p = min(max(mean / n, 1e-6), 1.0 - 1e-6)
    binomial_var = n * p * (1.0 - p)
    if binomial_var <= 1e-12:  # pragma: no cover - degenerate
        return Binomial(n, p)
    k = variance / binomial_var
    if k <= 1.0 + 1e-9:
        return Binomial(n, p)
    s = (n - k) / (k - 1.0) if k < n else 0.1
    s = max(s, 0.1)
    return BetaBinomial(n=n, alpha=p * s, beta=(1.0 - p) * s)


def negbin_from_moments(mean: float, variance: float, max_size: float = 500.0) -> Distribution:
    """Fit a negative binomial to a mean/variance pair, falling back to Poisson.

    Under-dispersed counts (variance below the mean) have no negative binomial
    representation, and pretending otherwise is how you end up with absurdly
    confident tail probabilities.
    """
    mu = max(mean, 1e-6)
    if variance <= mu * 1.02:
        return Poisson(mu)
    size = mu * mu / (variance - mu)
    return NegativeBinomial(mu, min(size, max_size))
