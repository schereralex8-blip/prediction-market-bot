"""Shared plumbing for prop models -- data extraction only, no inference.

The two models are meant to be independent estimators, so nothing that
constitutes a modelling *choice* lives here. This module only knows how to
pull numbers out of game logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from ..distributions import Distribution, outcome_probs
from ..markets import MarketSpec
from ..types import GameLog, MatchupContext, ModelOutput

EPS = 1e-9


@dataclass(frozen=True)
class Fit:
    """A model's view of a player/market, before any line is applied.

    Fitting is the expensive part and it does not depend on the line, so the
    scanner fits once and prices every line the books offer against it.
    """

    ok: bool
    dist: Distribution | None = None
    n_games: int = 0
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def at(self, model: str, line: float) -> ModelOutput:
        if not self.ok or self.dist is None:
            return ModelOutput(model=model, ok=False, n_games=self.n_games, reason=self.reason)
        return ModelOutput(
            model=model,
            ok=True,
            probs=outcome_probs(self.dist, line),
            mean=self.dist.mean,
            sd=self.dist.sd,
            n_games=self.n_games,
            details=self.details,
        )


class PropModel(Protocol):
    name: str

    def fit(
        self,
        *,
        spec: MarketSpec,
        logs: Sequence[GameLog],
        context: MatchupContext | None = None,
    ) -> Fit:
        """Estimate the outcome distribution for this player/market."""

    def predict(
        self,
        *,
        spec: MarketSpec,
        logs: Sequence[GameLog],
        line: float,
        context: MatchupContext | None = None,
    ) -> ModelOutput:
        """Probability of going over ``line``, plus the projection behind it."""


def recent_first(logs: Sequence[GameLog], max_games: int) -> list[GameLog]:
    ordered = sorted(logs, key=lambda g: g.date, reverse=True)
    return ordered[:max_games]


def stat_of(log: GameLog, spec: MarketSpec) -> float:
    return log.total(spec.components)


def workload_of(log: GameLog, spec: MarketSpec) -> float:
    if spec.workload is None:
        return 1.0
    return log.get(spec.workload)


def series(logs: Sequence[GameLog], spec: MarketSpec) -> list[tuple[float, float]]:
    """[(stat, workload)] for each game, most recent first."""
    return [(stat_of(g, spec), workload_of(g, spec)) for g in logs]


def decay_weights(n: int, half_life: float) -> list[float]:
    """Exponential recency weights, index 0 = most recent game."""
    hl = max(half_life, 0.5)
    return [0.5 ** (i / hl) for i in range(n)]


def insufficient(n_games: int, reason: str) -> Fit:
    return Fit(ok=False, n_games=n_games, reason=reason)
