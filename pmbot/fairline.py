"""Turning a screen full of prices into one fair probability.

Three problems have to be solved before a model probability can be compared to
"the market":

1. Every book's price carries vig -- removed per book, per line, on the
   complete two-way pair (see :mod:`pmbot.oddsmath`).
2. Books quote *different lines*. A fair probability at 25.5 points says
   nothing directly about 26.5. Since log-odds are close to linear in the line
   over a couple of points, the consensus is fitted in logit space and
   evaluated at whichever line we're actually pricing.
3. The book you're betting is part of the market. If it's an outlier -- which
   is the entire reason it looks bettable -- including it in the consensus
   pulls the fair line toward the outlier and hides the edge. By default it is
   excluded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import DevigSettings
from .oddsmath import american_to_prob, devig_two_way, hold_pct
from .types import PropMarket


@dataclass(frozen=True)
class BookFair:
    book: str
    line: float
    p_over: float
    hold: float
    weight: float


@dataclass(frozen=True)
class FairLine:
    line: float
    p_over: float
    p_under: float
    n_books: int
    method: str
    interpolated: bool
    avg_hold: float
    books_used: tuple[str, ...] = ()
    slope: float | None = None  # d logit(p_over) / d line, for diagnostics
    warnings: tuple[str, ...] = field(default=())

    def prob(self, side: str) -> float:
        return self.p_over if side == "over" else self.p_under


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _expit(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def book_fair_probs(
    market: PropMarket,
    settings: DevigSettings,
    exclude_book: str | None = None,
) -> list[BookFair]:
    """De-vig every complete two-way quote in the market."""
    out: list[BookFair] = []
    for book in market.two_way_books:
        if exclude_book and book.book.lower() == exclude_book.lower():
            continue
        try:
            p_over, _ = devig_two_way(book.over_american, book.under_american, settings.method)
            hold = hold_pct([book.over_american, book.under_american])
        except (ValueError, ZeroDivisionError):
            continue
        weight = settings.sharp_weight if book.book.lower() in settings.sharp_books else 1.0
        out.append(BookFair(book.book, book.line, p_over, hold, weight))
    return out


def fair_prob_at(
    market: PropMarket,
    line: float,
    settings: DevigSettings,
    exclude_book: str | None = None,
) -> FairLine | None:
    """Consensus fair probability of the OVER at ``line``.

    Returns ``None`` when the market is too thin to price honestly -- which is
    a perfectly good answer and much better than a confident guess.
    """
    excluded = exclude_book if settings.exclude_own_book else None
    quotes = book_fair_probs(market, settings, exclude_book=excluded)
    warnings: list[str] = []

    if excluded and len(quotes) < settings.min_books:
        # Not enough independent books to exclude our own; fall back to
        # including it, but say so.
        fallback = book_fair_probs(market, settings, exclude_book=None)
        if len(fallback) >= settings.min_books or len(fallback) > len(quotes):
            quotes = fallback
            warnings.append(f"consensus includes {exclude_book} (too few other books)")

    if not quotes:
        return None

    # Drop books quoting a line too far away to say anything useful about ours.
    near = [q for q in quotes if abs(q.line - line) <= settings.max_line_gap]
    if not near:
        return None
    if len(near) < len(quotes):
        warnings.append(f"{len(quotes) - len(near)} book(s) dropped: line too far from {line:g}")

    if len(near) < settings.min_books:
        warnings.append(f"thin market: {len(near)} book(s)")

    # Proximity kernel: a quote two points away is evidence, but weak evidence.
    weights = [
        q.weight * math.exp(-abs(q.line - line) / max(settings.max_line_gap, 1e-9))
        for q in near
    ]
    distinct_lines = {round(q.line, 3) for q in near}
    avg_hold = sum(w * q.hold for w, q in zip(weights, near)) / sum(weights)

    if len(distinct_lines) == 1:
        only_line = next(iter(distinct_lines))
        if abs(only_line - line) > 1e-6:
            # Every book agrees on a line that isn't the one we're pricing, and
            # with a single x value there is no slope to extrapolate along.
            return None
        y = sum(w * _logit(q.p_over) for w, q in zip(weights, near)) / sum(weights)
        p_over = _expit(y)
        return FairLine(
            line=line,
            p_over=p_over,
            p_under=1.0 - p_over,
            n_books=len(near),
            method=settings.method,
            interpolated=False,
            avg_hold=avg_hold,
            books_used=tuple(q.book for q in near),
            warnings=tuple(warnings),
        )

    intercept, slope = _weighted_linear_fit(
        [q.line for q in near], [_logit(q.p_over) for q in near], weights
    )
    if slope >= 0:
        # Fair probability of the over must fall as the line rises. A positive
        # fitted slope is noise, not information -- price off the nearest line.
        nearest = min(near, key=lambda q: abs(q.line - line))
        if abs(nearest.line - line) > 1e-6:
            return None
        warnings.append("non-monotone book lines; used nearest line only")
        same = [q for q in near if abs(q.line - nearest.line) < 1e-6]
        w_same = [
            q.weight for q in same
        ]
        y = sum(w * _logit(q.p_over) for w, q in zip(w_same, same)) / sum(w_same)
        p_over = _expit(y)
        return FairLine(
            line, p_over, 1.0 - p_over, len(same), settings.method, False, avg_hold,
            tuple(q.book for q in same), None, tuple(warnings),
        )

    p_over = _expit(intercept + slope * line)
    interpolated = not any(abs(q.line - line) < 1e-6 for q in near)
    return FairLine(
        line=line,
        p_over=p_over,
        p_under=1.0 - p_over,
        n_books=len(near),
        method=settings.method,
        interpolated=interpolated,
        avg_hold=avg_hold,
        books_used=tuple(q.book for q in near),
        slope=slope,
        warnings=tuple(warnings),
    )


def _weighted_linear_fit(
    xs: list[float], ys: list[float], ws: list[float]
) -> tuple[float, float]:
    """Weighted least squares y = a + b*x."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    if sxx <= 1e-12:  # pragma: no cover - guarded by the caller
        return my, 0.0
    slope = sxy / sxx
    return my - slope * mx, slope


def best_price(market: PropMarket, side: str, line: float) -> tuple[str, float] | None:
    """Best available price for a side at an exact line, across books."""
    best: tuple[str, float] | None = None
    for book in market.books:
        if abs(book.line - line) > 1e-6:
            continue
        price = book.price(side)
        if price is None:
            continue
        if best is None or american_to_prob(price) < american_to_prob(best[1]):
            best = (book.book, price)
    return best
