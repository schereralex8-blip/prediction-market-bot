"""Data sources: odds feeds, player game logs, matchup adjustments."""

from __future__ import annotations

from ..config import Settings
from .base import GameLogProvider, MatchupProvider, OddsProvider, slugify
from .local import LocalGameLogProvider, LocalOddsProvider, MatchupBook

__all__ = [
    "OddsProvider",
    "GameLogProvider",
    "MatchupProvider",
    "LocalOddsProvider",
    "LocalGameLogProvider",
    "MatchupBook",
    "slugify",
    "build_odds_provider",
]


def build_odds_provider(settings: Settings) -> OddsProvider:
    """Pick the odds source named in the config."""
    provider = settings.api.provider.lower()
    if provider == "local":
        return LocalOddsProvider(settings.data.props)
    if provider in ("theoddsapi", "the-odds-api", "oddsapi"):
        from .theoddsapi import TheOddsApiProvider  # imported lazily: needs a key

        return TheOddsApiProvider(settings.api, settings.data.cache)
    raise ValueError(f"unknown odds provider {settings.api.provider!r}; expected local|theoddsapi")
