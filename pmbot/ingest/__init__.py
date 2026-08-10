"""Real game logs: one source per sport, plus the plumbing to store them.

    from pmbot.ingest import build_source, write_logs

    source = build_source("nfl")
    write_logs(source.fetch("Josh Allen", seasons=[2024, 2025]))

Sources are chosen per sport because no single free feed covers all three
well. Each one maps its provider's field names onto the box-score fields in
:mod:`pmbot.markets`, so the models never learn which feed they're reading.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import (
    AmbiguousPlayer,
    FetchedLogs,
    GameLogSource,
    PlayerRef,
    UnknownPlayer,
    name_key,
    normalise_name,
)
from .http import FetchError, HttpClient
from .mlb_statsapi import MlbStatsApiSource
from .nba_stats import NbaStatsSource
from .nflverse import NflverseSource
from .store import merge_games, path_for, summarise_file, write_logs

__all__ = [
    "build_source",
    "SOURCES",
    "default_seasons",
    "GameLogSource",
    "PlayerRef",
    "FetchedLogs",
    "HttpClient",
    "FetchError",
    "UnknownPlayer",
    "AmbiguousPlayer",
    "write_logs",
    "merge_games",
    "path_for",
    "summarise_file",
    "name_key",
    "normalise_name",
]

SOURCES: dict[str, dict[str, Callable[..., Any]]] = {
    "nba": {"nba-stats": NbaStatsSource},
    "mlb": {"mlb-statsapi": MlbStatsApiSource},
    "nfl": {"nflverse": NflverseSource},
}

DEFAULT_SOURCE: dict[str, str] = {
    "nba": "nba-stats",
    "mlb": "mlb-statsapi",
    "nfl": "nflverse",
}

SOURCE_NOTES: dict[str, str] = {
    "nba-stats": "stats.nba.com -- official and complete; rate limits hard and "
                 "often refuses cloud IPs",
    "mlb-statsapi": "statsapi.mlb.com -- official, keyless, hitting and pitching merged",
    "nflverse": "nflverse GitHub releases -- weekly player stats joined to the schedule",
}


def build_source(sport: str, name: str | None = None, client: HttpClient | None = None) -> GameLogSource:
    """Pick a game-log source for a sport."""
    key = sport.strip().lower()
    if key not in SOURCES:
        raise KeyError(f"no game-log sources for {sport!r}; expected one of {', '.join(SOURCES)}")
    chosen = (name or DEFAULT_SOURCE[key]).strip().lower()
    if chosen not in SOURCES[key]:
        raise KeyError(
            f"unknown {key} source {chosen!r}; available: {', '.join(SOURCES[key])}"
        )
    return SOURCES[key][chosen](client=client) if client else SOURCES[key][chosen]()


def default_seasons(sport: str, count: int = 2) -> list[int]:
    """The most recent ``count`` seasons, labelled the way that sport labels them.

    Two seasons by default: enough that an early-season slate still has a
    usable sample behind it, without dragging in years-old form.
    """
    key = sport.strip().lower()
    if key == "nfl":
        from .nflverse import default_season
    elif key == "mlb":
        from .mlb_statsapi import default_season
    elif key == "nba":
        from .nba_stats import default_season
    else:
        raise KeyError(f"unknown sport {sport!r}")
    latest = default_season()
    return list(range(latest - count + 1, latest + 1))
