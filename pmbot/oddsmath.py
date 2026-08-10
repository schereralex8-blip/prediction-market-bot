"""Odds conversions and vig-removal (de-vigging).

Everything in the bot speaks three dialects of the same thing:

    american odds   -110, +145        what books display in the US
    decimal odds    1.909, 2.45       gross return per unit staked
    probability     0.524, 0.408      what we actually reason about

A book's posted prices imply probabilities that sum to more than 1. That
excess is the vig (juice/overround). Comparing a model probability against a
*vigged* implied probability is the single most common way people convince
themselves they have an edge when they don't, so every comparison in this
codebase goes through :func:`devig` first.
"""

from __future__ import annotations

import math
from typing import Iterable, Literal, Sequence

DevigMethod = Literal["multiplicative", "additive", "power", "shin", "conservative"]

DEVIG_METHODS: tuple[str, ...] = ("multiplicative", "additive", "power", "shin", "conservative")


# --------------------------------------------------------------------------
# conversions
# --------------------------------------------------------------------------
def american_to_decimal(american: float) -> float:
    """+150 -> 2.5, -120 -> 1.8333."""
    a = float(american)
    if a == 0:
        raise ValueError("american odds of 0 are not a price")
    if a > 0:
        return 1.0 + a / 100.0
    return 1.0 + 100.0 / abs(a)


def decimal_to_american(decimal: float) -> float:
    if decimal <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal}")
    if decimal >= 2.0:
        return round((decimal - 1.0) * 100.0)
    return round(-100.0 / (decimal - 1.0))


def decimal_to_prob(decimal: float) -> float:
    if decimal <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal}")
    return 1.0 / decimal


def american_to_prob(american: float) -> float:
    """Implied probability *including* vig. Never compare a model to this."""
    return decimal_to_prob(american_to_decimal(american))


def prob_to_decimal(prob: float) -> float:
    if not 0.0 < prob < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {prob}")
    return 1.0 / prob


def prob_to_american(prob: float) -> float:
    return decimal_to_american(prob_to_decimal(prob))


def format_american(american: float) -> str:
    a = int(round(american))
    return f"+{a}" if a > 0 else str(a)


# --------------------------------------------------------------------------
# vig / overround
# --------------------------------------------------------------------------
def overround(americans: Iterable[float]) -> float:
    """Sum of implied probabilities. 1.045 means a 4.5% overround."""
    return sum(american_to_prob(a) for a in americans)


def hold_pct(americans: Iterable[float]) -> float:
    """The book's theoretical hold, as a fraction of handle on a balanced book."""
    total = overround(americans)
    return (total - 1.0) / total


def devig(raw: Sequence[float], method: DevigMethod = "power") -> list[float]:
    """Turn vigged implied probabilities into fair probabilities summing to 1.

    ``raw`` is the vector of implied probabilities for a complete market
    (both sides of an over/under, or all N outcomes of an N-way market).

    Methods, roughly in order of how much structure they assume:

    ``multiplicative``
        Scale every outcome by the same factor. Simple, but it takes
        proportionally more vig out of longshots than favourites.
    ``additive``
        Subtract the same amount of probability from each outcome. The
        mirror-image bias of multiplicative.
    ``power``
        Find ``k`` with ``sum(q_i**k) == 1``. Handles the favourite-longshot
        bias better than either of the above and is the default for the
        two-way player-prop markets this bot trades.
    ``shin``
        Shin (1993): models the overround as the book protecting itself
        against a fraction ``z`` of insider money. Best on markets with a
        big favourite; solved numerically for ``z``.
    ``conservative``
        Assume *all* the vig sits on the side you want to bet:
        ``p_i = 1 - sum(q_j for j != i)``. Understates your edge on purpose.
        Useful as a sanity floor, not as a primary estimate.
    """
    q = [float(x) for x in raw]
    if len(q) < 2:
        raise ValueError("de-vigging needs a complete market (>= 2 outcomes)")
    if any(x <= 0.0 for x in q):
        raise ValueError("implied probabilities must be positive")
    total = sum(q)
    if total <= 1.0:
        # No vig to remove (or a genuine arb across books). Normalise and move on.
        return [x / total for x in q]

    if method == "multiplicative":
        return [x / total for x in q]

    if method == "additive":
        excess = (total - 1.0) / len(q)
        out = [x - excess for x in q]
        if any(x <= 0.0 for x in out):  # degenerate on lopsided markets
            return [x / total for x in q]
        return out

    if method == "power":
        return _power_devig(q)

    if method == "shin":
        return _shin_devig(q)

    if method == "conservative":
        out = [max(1.0 - (total - x), 1e-6) for x in q]
        s = sum(out)
        return [x / s for x in out]

    raise ValueError(f"unknown devig method {method!r}; expected one of {DEVIG_METHODS}")


def _power_devig(q: Sequence[float], tol: float = 1e-12, max_iter: int = 200) -> list[float]:
    """Solve sum(q_i ** k) == 1 for k, by bisection.

    sum(q_i ** k) is strictly decreasing in k for q_i in (0, 1), and equals
    sum(q) > 1 at k = 1, so the root is bracketed by [1, hi] for hi big enough.
    """
    lo, hi = 1.0, 2.0
    for _ in range(60):
        if sum(x**hi for x in q) < 1.0:
            break
        lo, hi = hi, hi * 2.0
    else:  # pragma: no cover - unreachable for sane inputs
        return [x / sum(q) for x in q]

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = sum(x**mid for x in q)
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    out = [x**k for x in q]
    s = sum(out)
    return [x / s for x in out]  # tidy up float drift


def _shin_devig(q: Sequence[float], tol: float = 1e-12, max_iter: int = 200) -> list[float]:
    """Shin's insider-trading model, solved for the insider fraction z."""
    total = sum(q)

    def pi_of(z: float) -> list[float]:
        if z >= 1.0 - 1e-9:
            return [x / total for x in q]
        denom = 2.0 * (1.0 - z)
        return [(math.sqrt(z * z + 4.0 * (1.0 - z) * x * x / total) - z) / denom for x in q]

    lo, hi = 0.0, 1.0 - 1e-9
    # sum(pi(z)) is decreasing in z; it is > 1 at z = 0 whenever there is vig.
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = sum(pi_of(mid))
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    out = pi_of(0.5 * (lo + hi))
    s = sum(out)
    if s <= 0:  # pragma: no cover - defensive
        return [x / total for x in q]
    return [x / s for x in out]


def devig_two_way(
    over_american: float,
    under_american: float,
    method: DevigMethod = "power",
) -> tuple[float, float]:
    """Fair (p_over, p_under) for a two-way price pair at a single book."""
    fair = devig(
        [american_to_prob(over_american), american_to_prob(under_american)],
        method=method,
    )
    return fair[0], fair[1]
