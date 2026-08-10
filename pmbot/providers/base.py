"""Provider interfaces: where odds, game logs and matchup factors come from."""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol, Sequence

from ..types import GameLog, MatchupContext, PropMarket


class OddsProvider(Protocol):
    name: str

    def player_props(
        self,
        sport: str,
        markets: Sequence[str],
        event_ids: Sequence[str] | None = None,
    ) -> list[PropMarket]:
        """Every book's prices for the requested player-prop markets."""


class GameLogProvider(Protocol):
    name: str

    def logs_for(self, player: str, sport: str) -> list[GameLog]:
        """A player's game logs, most recent first is not required."""


class MatchupProvider(Protocol):
    def context_for(
        self,
        *,
        sport: str,
        market: str,
        opponent: str | None,
        is_home: bool | None,
        workload_override: float | None = None,
    ) -> MatchupContext:
        """Opponent / pace / venue adjustments for one prop."""


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """'Shai Gilgeous-Alexander' -> 'shai-gilgeous-alexander'.

    Accents are folded so that 'Nikola Jokić' and 'Nikola Jokic' -- which
    different feeds spell differently -- land on the same key.
    """
    folded = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in folded if not unicodedata.combining(c))
    return _SLUG_RE.sub("-", ascii_only.strip().lower()).strip("-")
