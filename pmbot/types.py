"""Core domain objects shared by the odds side and the model side."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Sequence

from .distributions import OutcomeProbs

SIDES: tuple[str, str] = ("over", "under")


def other_side(side: str) -> str:
    if side not in SIDES:
        raise ValueError(f"side must be 'over' or 'under', got {side!r}")
    return "under" if side == "over" else "over"


# --------------------------------------------------------------------------
# market data
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BookLine:
    """One book's two-way price on one line of one prop."""

    book: str
    line: float
    over_american: float | None = None
    under_american: float | None = None
    last_update: datetime | None = None

    @property
    def is_two_way(self) -> bool:
        return self.over_american is not None and self.under_american is not None

    def price(self, side: str) -> float | None:
        return self.over_american if side == "over" else self.under_american


@dataclass(frozen=True)
class PropMarket:
    """All books' prices for one player, one stat, one game."""

    sport: str
    event_id: str
    commence_time: datetime
    home_team: str
    away_team: str
    player: str
    market: str
    books: tuple[BookLine, ...] = ()
    team: str | None = None
    opponent: str | None = None

    @property
    def is_home(self) -> bool | None:
        if self.team is None:
            return None
        return self.team == self.home_team

    @property
    def two_way_books(self) -> tuple[BookLine, ...]:
        return tuple(b for b in self.books if b.is_two_way)

    def lines(self) -> list[float]:
        return sorted({b.line for b in self.books})

    def describe(self) -> str:
        return f"{self.player} {self.market} ({self.away_team} @ {self.home_team})"


# --------------------------------------------------------------------------
# player data
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GameLog:
    """One player's box score from one game."""

    date: date
    player: str
    team: str
    opponent: str
    is_home: bool
    stats: dict[str, float] = field(default_factory=dict)

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.stats.get(key, default))

    def total(self, keys: Iterable[str]) -> float:
        return sum(self.get(k) for k in keys)


@dataclass(frozen=True)
class MatchupContext:
    """Everything about *this* game that isn't in the player's own history."""

    opponent_factor: float = 1.0  # >1 = opponent concedes more of this stat
    pace_factor: float = 1.0  # >1 = more possessions/plays than the player's norm
    home_factor: float = 1.0
    workload_override: float | None = None  # e.g. a known minutes/pitch-count cap
    notes: tuple[str, ...] = ()

    @property
    def combined(self) -> float:
        return self.opponent_factor * self.pace_factor * self.home_factor


# --------------------------------------------------------------------------
# model output
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelOutput:
    """What one model says about one player/market/line."""

    model: str
    ok: bool
    probs: OutcomeProbs | None = None
    mean: float = 0.0
    sd: float = 0.0
    n_games: int = 0
    reason: str | None = None  # populated when ok is False
    details: dict[str, Any] = field(default_factory=dict)

    def p_over_nopush(self) -> float:
        if not self.ok or self.probs is None:
            raise ValueError(f"model {self.model} produced no estimate: {self.reason}")
        return self.probs.p_over_nopush


@dataclass(frozen=True)
class Consensus:
    """The two-model agreement gate's verdict."""

    agree: bool
    side: str | None
    p_win: float  # consensus probability of the chosen side, incl. push mass
    p_push: float
    p_over_nopush: float
    reason: str
    model_probs: dict[str, float] = field(default_factory=dict)
    disagreement: float = 0.0


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Signal:
    """One line of the bot's answer: bet this, or pass and here's why."""

    action: str  # "BET" | "PASS"
    sport: str
    event_id: str
    commence_time: datetime
    matchup: str
    player: str
    market: str
    market_label: str
    reason: str
    side: str | None = None
    line: float | None = None
    book: str | None = None
    american: float | None = None
    decimal: float | None = None
    fair_prob: float | None = None
    fair_american: float | None = None
    model_prob: float | None = None
    p_push: float = 0.0
    edge: float | None = None  # price edge: model prob - breakeven at this price
    market_edge: float | None = None  # how far the model sits from the consensus
    ev_per_unit: float | None = None
    stake: float = 0.0
    stake_units: float = 0.0
    full_kelly: float = 0.0
    model_probs: dict[str, float] = field(default_factory=dict)
    disagreement: float = 0.0
    projection: float | None = None
    projection_sd: float | None = None
    n_books: int = 0
    devig_method: str = ""
    warnings: tuple[str, ...] = ()

    @property
    def is_bet(self) -> bool:
        return self.action == "BET"

    def headline(self) -> str:
        if not self.is_bet:
            return f"PASS  {self.player} {self.market_label} -- {self.reason}"
        from .oddsmath import format_american

        return (
            f"BET   {self.player} {self.market_label} {self.side.upper()} {self.line:g} "
            f"{format_american(self.american)} @ {self.book}  "
            f"edge {self.edge * 100:+.1f}%  EV {self.ev_per_unit * 100:+.1f}%  "
            f"stake {self.stake:,.2f} ({self.stake_units:.2f}u)"
        )


def sort_signals(signals: Sequence[Signal]) -> list[Signal]:
    """Best bets first, passes last."""
    return sorted(
        signals,
        key=lambda s: (0 if s.is_bet else 1, -(s.ev_per_unit or 0.0), s.player),
    )
