"""Expected value: the gate between "I have an opinion" and "that's a bet".

A price is only a bet if the model's probability beats the *no-vig* fair
probability by enough to survive the model's own error bars. Everything here
is push-aware, because a whole-number prop line (25.0 points, 6.0 receptions)
refunds instead of losing, which is worth real basis points of EV.
"""

from __future__ import annotations

from dataclasses import dataclass

from .oddsmath import american_to_decimal, prob_to_american


@dataclass(frozen=True)
class EvResult:
    """Everything needed to judge one price on one side of one line.

    Two different "edges" live here and they answer different questions:

    ``price_edge``
        Model probability minus what the price needs. This is the one that
        pays: EV is proportional to it. A bet can have a big price edge while
        agreeing with the market completely -- that's an outlier book, and
        it's the most repeatable edge there is.
    ``market_edge``
        Model probability minus the market's de-vigged probability. This
        measures how far out on a limb the model is. Large values are a
        warning as often as an opportunity.
    """

    p_win: float
    p_lose: float
    p_push: float
    decimal_odds: float
    ev_per_unit: float  # expected profit per 1.0 staked
    price_edge: float
    market_edge: float
    model_prob_nopush: float
    fair_prob_nopush: float
    breakeven_prob: float  # p_win needed for EV = 0 at this price
    fair_american: float  # price at which this bet would be EV-neutral

    @property
    def ev_pct(self) -> float:
        return 100.0 * self.ev_per_unit

    @property
    def is_positive(self) -> bool:
        return self.ev_per_unit > 0.0


def expected_value(p_win: float, p_lose: float, decimal_odds: float) -> float:
    """Expected profit per unit staked. Pushes contribute exactly zero."""
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")
    return p_win * (decimal_odds - 1.0) - p_lose


def breakeven_prob(decimal_odds: float, p_push: float = 0.0) -> float:
    """Win probability required to break even, given the push probability.

    EV = 0  =>  p_win * (d - 1) = p_lose = (1 - p_push - p_win)
            =>  p_win = (1 - p_push) / d
    """
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")
    if not 0.0 <= p_push < 1.0:
        raise ValueError(f"push probability must be in [0, 1), got {p_push}")
    return (1.0 - p_push) / decimal_odds


def _normalise_no_push(p_win: float, p_push: float) -> float:
    live = 1.0 - p_push
    if live <= 1e-12:
        return 0.5
    return min(max(p_win / live, 0.0), 1.0)


def evaluate(
    *,
    model_p_win: float,
    model_p_push: float,
    fair_prob: float,
    american_odds: float,
) -> EvResult:
    """Score one bettable price.

    ``model_p_win`` / ``model_p_push`` come from the model's outcome
    distribution and are unconditional (they sum to <= 1 with p_lose).
    ``fair_prob`` is the market's de-vigged probability for the same side,
    conditional on no push -- which is how books quote two-way props, since a
    push voids the bet entirely.
    """
    if not 0.0 <= model_p_win <= 1.0:
        raise ValueError(f"model win probability out of range: {model_p_win}")
    if not 0.0 <= model_p_push < 1.0:
        raise ValueError(f"model push probability out of range: {model_p_push}")
    p_lose = max(0.0, 1.0 - model_p_win - model_p_push)
    decimal = american_to_decimal(american_odds)
    model_nopush = _normalise_no_push(model_p_win, model_p_push)
    ev = expected_value(model_p_win, p_lose, decimal)
    fair_nopush = min(max(fair_prob, 1e-9), 1.0 - 1e-9)
    # Excluding pushes, EV > 0 exactly when the model probability beats 1/d:
    #   EV = (1 - p_push) * d * (model_nopush - 1/d)
    price_needs = 1.0 / decimal
    neutral_prob = min(max(model_nopush, 1e-9), 1.0 - 1e-9)
    return EvResult(
        p_win=model_p_win,
        p_lose=p_lose,
        p_push=model_p_push,
        decimal_odds=decimal,
        ev_per_unit=ev,
        price_edge=model_nopush - price_needs,
        market_edge=model_nopush - fair_nopush,
        model_prob_nopush=model_nopush,
        fair_prob_nopush=fair_nopush,
        breakeven_prob=breakeven_prob(decimal, model_p_push),
        fair_american=prob_to_american(neutral_prob),
    )
