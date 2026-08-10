"""Kelly-criterion stake sizing.

Full Kelly maximises the long-run growth rate of the bankroll, and it is also
far too violent to actually bet: it assumes your probabilities are exactly
right. They aren't. So the default here is fractional Kelly with a hard cap on
any single bet and a cap on total simultaneous exposure.

With a push outcome the growth rate is

    g(f) = p_win * ln(1 + f*b) + p_lose * ln(1 - f) + p_push * ln(1)

and setting g'(f) = 0 gives a closed form:

    f* = (p_win * b - p_lose) / (b * (p_win + p_lose))

which collapses to the familiar (pb - q)/b when p_push = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Stake:
    full_kelly: float  # fraction of bankroll at full Kelly
    fraction_used: float  # e.g. 0.25 for quarter Kelly
    recommended_fraction: float  # after fraction + caps
    amount: float  # rounded currency amount
    capped_by: str | None  # "max_bet" | "min_stake" | None
    growth_rate: float  # expected log-growth per bet at the recommended size

    @property
    def units(self) -> float:
        """Stake expressed in 1%-of-bankroll units, the way most journals talk."""
        return 100.0 * self.recommended_fraction


def full_kelly_fraction(p_win: float, p_lose: float, decimal_odds: float) -> float:
    """Optimal fraction of bankroll at full Kelly. Negative means no bet."""
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")
    b = decimal_odds - 1.0
    live = p_win + p_lose
    if live <= 1e-12:
        return 0.0
    return (p_win * b - p_lose) / (b * live)


def growth_rate(f: float, p_win: float, p_lose: float, decimal_odds: float) -> float:
    """Expected log-growth of bankroll for staking fraction ``f``."""
    b = decimal_odds - 1.0
    if f <= 0.0:
        return 0.0
    if f >= 1.0:
        return float("-inf")
    return p_win * math.log(1.0 + f * b) + p_lose * math.log(1.0 - f)


def size_bet(
    *,
    p_win: float,
    p_push: float,
    decimal_odds: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_bet_fraction: float = 0.02,
    min_stake: float = 1.0,
    rounding: float = 1.0,
) -> Stake:
    """Size a single bet.

    ``kelly_fraction``   multiplier on full Kelly (0.25 = quarter Kelly).
    ``max_bet_fraction`` hard ceiling on any one bet as a share of bankroll.
    ``min_stake``        below this the bet isn't worth the ticket; returns 0.
    ``rounding``         round down to this increment (e.g. 1.0 or 0.5).
    """
    if bankroll <= 0:
        raise ValueError("bankroll must be positive")
    if not 0.0 < kelly_fraction <= 1.0:
        raise ValueError("kelly_fraction must be in (0, 1]")
    if not 0.0 < max_bet_fraction <= 1.0:
        raise ValueError("max_bet_fraction must be in (0, 1]")

    p_lose = max(0.0, 1.0 - p_win - p_push)
    full = full_kelly_fraction(p_win, p_lose, decimal_odds)
    if full <= 0.0:
        return Stake(full, kelly_fraction, 0.0, 0.0, None, 0.0)

    frac = full * kelly_fraction
    capped_by: str | None = None
    if frac > max_bet_fraction:
        frac = max_bet_fraction
        capped_by = "max_bet"

    amount = _round_down(bankroll * frac, rounding)
    if amount < min_stake:
        return Stake(full, kelly_fraction, 0.0, 0.0, "min_stake", 0.0)

    frac = amount / bankroll
    return Stake(
        full_kelly=full,
        fraction_used=kelly_fraction,
        recommended_fraction=frac,
        amount=amount,
        capped_by=capped_by,
        growth_rate=growth_rate(frac, p_win, p_lose, decimal_odds),
    )


def _round_down(x: float, increment: float) -> float:
    if increment <= 0:
        return x
    return math.floor(x / increment + 1e-9) * increment


def cap_total_exposure(
    stakes: list[float],
    bankroll: float,
    max_total_fraction: float = 0.10,
) -> list[float]:
    """Scale a slate of same-day bets down so total risk stays sane.

    Kelly sizes bets one at a time, which quietly assumes they settle one at a
    time. A twelve-leg Tuesday slate does not. Scaling the whole slate by a
    single factor preserves the relative sizing Kelly asked for.
    """
    total = sum(stakes)
    ceiling = bankroll * max_total_fraction
    if total <= ceiling or total <= 0:
        return list(stakes)
    scale = ceiling / total
    return [s * scale for s in stakes]
