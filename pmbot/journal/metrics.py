"""Performance analytics over the journal.

Three numbers matter, in increasing order of how quickly they tell you the
truth:

* **ROI** -- what actually happened. Buried in variance for hundreds of bets.
* **Win rate vs breakeven** -- the same information, normalised for price.
* **CLV** -- whether you beat the closing line. This one converges fastest.
  A bettor with positive CLV and negative ROI is unlucky. A bettor with
  negative CLV and positive ROI is about to find out why.

A ``t``-statistic and a bootstrap interval are attached to ROI so that a
40-bet sample stops being read as a verdict.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..oddsmath import decimal_to_american
from .repo import Bet


@dataclass
class Performance:
    label: str = "all"
    n_bets: int = 0
    n_pending: int = 0
    n_settled: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    voids: int = 0
    staked: float = 0.0
    profit: float = 0.0
    units_staked: float = 0.0
    units_profit: float = 0.0
    avg_stake: float = 0.0
    avg_decimal: float = 0.0
    avg_edge: float | None = None
    expected_profit: float | None = None
    clv_n: int = 0
    clv_avg: float | None = None  # mean price beaten vs no-vig close
    clv_prob_avg: float | None = None  # mean probability points beaten
    beat_close_rate: float | None = None
    roi_ci: tuple[float, float] | None = None
    t_stat: float | None = None

    @property
    def roi(self) -> float:
        return self.profit / self.staked if self.staked else 0.0

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided else 0.0

    @property
    def breakeven_rate(self) -> float:
        """Win rate needed at the average price just to stay level."""
        return 1.0 / self.avg_decimal if self.avg_decimal else 0.0

    @property
    def avg_american(self) -> float:
        return decimal_to_american(self.avg_decimal) if self.avg_decimal > 1.0 else 0.0

    @property
    def luck(self) -> float | None:
        """Profit minus what the model expected. Positive = ran good."""
        if self.expected_profit is None:
            return None
        return self.profit - self.expected_profit


def summarise(bets: Sequence[Bet], label: str = "all") -> Performance:
    perf = Performance(label=label, n_bets=len(bets))
    settled = [b for b in bets if b.is_settled]
    perf.n_pending = len(bets) - len(settled)
    perf.n_settled = len(settled)

    for bet in settled:
        perf.wins += bet.status in ("won", "half_won")
        perf.losses += bet.status in ("lost", "half_lost")
        perf.pushes += bet.status == "push"
        perf.voids += bet.status == "void"

    risked = [b for b in settled if b.status != "void"]
    perf.staked = sum(b.stake for b in risked)
    perf.profit = sum(b.profit or 0.0 for b in risked)
    perf.avg_stake = perf.staked / len(risked) if risked else 0.0
    if risked:
        perf.avg_decimal = sum(b.decimal_odds * b.stake for b in risked) / perf.staked

    perf.units_staked = sum(b.stake_units or 0.0 for b in risked)
    perf.units_profit = sum(
        100.0 * (b.profit or 0.0) / b.bankroll_before
        for b in risked
        if b.bankroll_before
    )

    edges = [b.edge for b in bets if b.edge is not None]
    perf.avg_edge = sum(edges) / len(edges) if edges else None
    evs = [(b.ev, b.stake) for b in risked if b.ev is not None]
    perf.expected_profit = sum(ev * stake for ev, stake in evs) if evs else None

    # --- CLV: computed over every bet with a closing line, settled or not ---
    clv = [(b.clv_pct, b.clv_prob) for b in bets if b.clv_pct is not None]
    perf.clv_n = len(clv)
    if clv:
        perf.clv_avg = sum(c for c, _ in clv) / len(clv)
        probs = [p for _, p in clv if p is not None]
        perf.clv_prob_avg = sum(probs) / len(probs) if probs else None
        perf.beat_close_rate = sum(1 for c, _ in clv if c > 0) / len(clv)

    returns = [(b.profit or 0.0) / b.stake for b in risked if b.stake]
    if len(returns) >= 2:
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        sd = math.sqrt(var)
        if sd > 0:
            perf.t_stat = mean / (sd / math.sqrt(len(returns)))
        perf.roi_ci = bootstrap_roi_ci(risked)
    return perf


def bootstrap_roi_ci(
    bets: Sequence[Bet],
    iterations: int = 2000,
    level: float = 0.95,
    seed: int = 7,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval for ROI.

    Betting returns are wildly non-normal (a pile of -1s and a few +0.9s), so
    resampling beats any closed-form interval here.
    """
    rows = [(b.profit or 0.0, b.stake) for b in bets if b.stake]
    if len(rows) < 2:
        return None
    rng = random.Random(seed)
    n = len(rows)
    samples = []
    for _ in range(iterations):
        picks = [rows[rng.randrange(n)] for _ in range(n)]
        staked = sum(s for _, s in picks)
        samples.append(sum(p for p, _ in picks) / staked if staked else 0.0)
    samples.sort()
    lo = samples[int((1 - level) / 2 * iterations)]
    hi = samples[min(iterations - 1, int((1 + level) / 2 * iterations))]
    return lo, hi


def breakdown(
    bets: Sequence[Bet],
    key: str = "sport",
    min_bets: int = 1,
) -> list[Performance]:
    """Split performance by sport, market, book, side, or tag."""
    getters: dict[str, Callable[[Bet], str]] = {
        "sport": lambda b: b.sport,
        "market": lambda b: b.market,
        "book": lambda b: b.book,
        "side": lambda b: b.side,
        "player": lambda b: b.player,
        "tag": lambda b: b.tags or "(untagged)",
        "month": lambda b: b.placed_at[:7],
    }
    if key not in getters:
        raise ValueError(f"cannot break down by {key!r}; try {', '.join(sorted(getters))}")
    get = getters[key]
    groups: dict[str, list[Bet]] = {}
    for bet in bets:
        groups.setdefault(get(bet), []).append(bet)
    out = [summarise(group, label=name) for name, group in groups.items() if len(group) >= min_bets]
    return sorted(out, key=lambda p: p.profit, reverse=True)


@dataclass
class CurvePoint:
    at: str
    balance: float
    bet_id: int | None = None
    profit: float = 0.0


def bankroll_curve(bets: Sequence[Bet], starting: float = 0.0) -> list[CurvePoint]:
    """Balance after each settled bet, in settlement order."""
    settled = sorted(
        (b for b in bets if b.is_settled and b.settled_at),
        key=lambda b: (b.settled_at or "", b.id),
    )
    balance = starting
    curve = [CurvePoint(at=settled[0].placed_at if settled else "", balance=balance)]
    for bet in settled:
        balance += bet.profit or 0.0
        curve.append(CurvePoint(at=bet.settled_at or "", balance=balance, bet_id=bet.id, profit=bet.profit or 0.0))
    return curve


def max_drawdown(curve: Sequence[CurvePoint]) -> float:
    """Largest peak-to-trough fall in the bankroll curve, in currency."""
    peak = float("-inf")
    worst = 0.0
    for point in curve:
        peak = max(peak, point.balance)
        worst = min(worst, point.balance - peak)
    return abs(worst)


@dataclass
class Calibration:
    """Did things happen as often as the model said they would?"""

    buckets: list[tuple[str, int, float, float]] = field(default_factory=list)  # label, n, predicted, actual

    def brier(self) -> float | None:
        total = sum(n for _, n, _, _ in self.buckets)
        if not total:
            return None
        return sum(n * (pred - act) ** 2 for _, n, pred, act in self.buckets) / total


def calibration(bets: Sequence[Bet], edges: Sequence[float] = (0.4, 0.5, 0.6, 0.7)) -> Calibration:
    """Bucket bets by model probability and compare predicted vs actual hit rate.

    The single most useful diagnostic in the journal: if the 60% bucket wins
    52% of the time, the edge was never there and the vig has been eating it.
    """
    graded = [
        b for b in bets
        if b.status in ("won", "lost") and b.model_prob is not None
    ]
    bounds = [0.0, *edges, 1.0]
    out = Calibration()
    for lo, hi in zip(bounds, bounds[1:]):
        group = [b for b in graded if lo <= (b.model_prob or 0) < hi]
        if not group:
            continue
        predicted = sum(b.model_prob or 0 for b in group) / len(group)
        actual = sum(1 for b in group if b.status == "won") / len(group)
        out.buckets.append((f"{lo:.0%}-{hi:.0%}", len(group), predicted, actual))
    return out
