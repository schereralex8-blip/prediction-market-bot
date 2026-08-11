"""NFL game logs from nflverse.

nflverse publishes tidy, weekly player stats as CSV release assets on GitHub
-- the same data the R and Python nflverse packages read, maintained by the
community and refreshed through the season. One request per season, cached,
no API key, no scraping.

The weekly file carries season and week but not kickoff dates, so it's joined
to the schedule release to recover the real game date and whether the player
was at home. Both matter: the models weight by recency and adjust for venue.

    https://github.com/nflverse/nflverse-data
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Sequence

import logging

from .base import (
    FetchedLogs,
    PlayerRef,
    as_float,
    build_index,
    fuzzy_candidates,
    resolve_one,
)
from .http import FetchError, HttpClient

log = logging.getLogger("pmbot.ingest.nflverse")

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
WEEKLY_URL = BASE + "/player_stats/stats_player_week_{season}.csv"
SCHEDULE_URL = BASE + "/schedules/games.csv"

# nflverse column -> the box-score field pmbot's NFL markets are defined over.
FIELD_MAP: dict[str, str] = {
    "attempts": "pass_attempts",
    "completions": "completions",
    "passing_yards": "pass_yards",
    "passing_tds": "pass_tds",
    "carries": "rush_attempts",
    "rushing_yards": "rush_yards",
    "rushing_tds": "rush_tds",
    "targets": "targets",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_tds",
}


class NflverseSource:
    """NFL weekly player stats, straight from the nflverse releases."""

    name = "nflverse"
    sport = "nfl"

    def __init__(
        self,
        client: HttpClient | None = None,
        season_types: Sequence[str] = ("REG", "POST"),
    ) -> None:
        self.client = client or HttpClient()
        self.season_types = tuple(season_types)
        self._weekly: dict[int, list[dict[str, str]]] = {}
        self._schedule: dict[tuple[int, int, str], dict[str, str]] | None = None

    # ------------------------------------------------------------------
    def _rows(self, season: int) -> list[dict[str, str]]:
        """Rows for a season, or none if that season isn't published yet.

        Asking for next season before week 1 is normal -- the default is the
        last two -- and a 404 there must not sink the season that does exist.
        Letting it propagate meant one missing file skipped every player.
        """
        if season not in self._weekly:
            try:
                self._weekly[season] = list(self.client.get_csv(WEEKLY_URL.format(season=season)))
            except FetchError as exc:
                if "404" not in str(exc):
                    raise
                log.info("nflverse has no %s season file yet; skipping that season", season)
                self._weekly[season] = []
        return self._weekly[season]

    def _schedule_index(self) -> dict[tuple[int, int, str], dict[str, str]]:
        """(season, week, team) -> that team's game, for dates and venue."""
        if self._schedule is None:
            index: dict[tuple[int, int, str], dict[str, str]] = {}
            for row in self.client.get_csv(SCHEDULE_URL, ttl=86_400):
                try:
                    key = (int(row["season"]), int(row["week"]))
                except (KeyError, ValueError):
                    continue
                for team in (row.get("home_team"), row.get("away_team")):
                    if team:
                        index[(key[0], key[1], team)] = row
            self._schedule = index
        return self._schedule

    # ------------------------------------------------------------------
    def search(self, name: str, season: int | None = None) -> list[PlayerRef]:
        season = season or default_season()
        refs = [
            PlayerRef(
                source_id=row["player_id"],
                name=row.get("player_display_name") or row.get("player_name", ""),
                sport="nfl",
                team=row.get("team") or None,
                position=row.get("position") or None,
            )
            for row in self._unique_players(season)
        ]
        return fuzzy_candidates(name, build_index(refs))

    def _unique_players(self, season: int) -> list[dict[str, str]]:
        seen: dict[str, dict[str, str]] = {}
        for row in self._rows(season):
            seen.setdefault(row["player_id"], row)  # first week's team is good enough
        return list(seen.values())

    # ------------------------------------------------------------------
    def game_logs(self, ref: PlayerRef, seasons: Sequence[int]) -> list[dict[str, Any]]:
        schedule = self._schedule_index()
        games: list[dict[str, Any]] = []
        for season in seasons:
            for row in self._rows(season):
                if row["player_id"] != ref.source_id:
                    continue
                if self.season_types and row.get("season_type") not in self.season_types:
                    continue
                games.append(self._to_game(row, season, schedule))
        games.sort(key=lambda g: g["date"])
        return games

    def _to_game(
        self,
        row: dict[str, str],
        season: int,
        schedule: dict[tuple[int, int, str], dict[str, str]],
    ) -> dict[str, Any]:
        week = int(as_float(row.get("week"), 0))
        team = row.get("team", "")
        game = schedule.get((season, week, team))
        if game and game.get("gameday"):
            played = game["gameday"]
            is_home = game.get("home_team") == team
        else:
            # No schedule row (a very fresh week, usually). Fall back to a
            # date that preserves ordering, which is all the models need.
            played = str(week_to_date(season, week))
            is_home = True

        stats = {
            field: as_float(row.get(column))
            for column, field in FIELD_MAP.items()
        }
        return {
            "date": played,
            "season": season,
            "week": week,
            "team": team,
            "opponent": row.get("opponent_team", ""),
            "home": is_home,
            **stats,
        }

    # ------------------------------------------------------------------
    def fetch(self, name: str, seasons: Sequence[int], team: str | None = None) -> FetchedLogs:
        ref = resolve_one(name, self._find(name, seasons), team=team)
        return FetchedLogs(
            player=ref.name, sport="nfl", source=self.name, ref=ref,
            games=self.game_logs(ref, seasons), seasons=tuple(seasons),
        )

    def _find(self, name: str, seasons: Sequence[int]) -> list[PlayerRef]:
        """Search the newest season with data, falling back to older ones.

        The newest requested season may not be published yet, and an empty
        roster there would report every player as unknown. If *no* requested
        season has data, say that instead -- "unknown player" would send you
        hunting for a spelling mistake that isn't there.
        """
        saw_data = False
        for season in sorted(seasons, reverse=True):
            if not self._rows(season):
                continue
            saw_data = True
            found = self.search(name, season)
            if found:
                return found
        if not saw_data:
            raise FetchError(
                f"nflverse has no data for season(s) "
                f"{', '.join(str(s) for s in sorted(seasons))} -- not published yet. "
                f"Try an earlier season with --seasons."
            )
        return []


def week_to_date(season: int, week: int) -> date:
    """Approximate kickoff: NFL week 1 lands in the first full week of September."""
    return date(season, 9, 5) + timedelta(days=7 * max(week - 1, 0))


def default_season() -> int:
    """The NFL season currently in progress (it's named for the year it starts)."""
    today = date.today()
    return today.year if today.month >= 3 else today.year - 1
