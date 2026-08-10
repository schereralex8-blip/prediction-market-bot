"""NBA game logs from the sportsdataverse (hoopR) release store.

hoopR scrapes ESPN's NBA box scores daily and publishes one tidy CSV per
season as a GitHub release asset. That makes it the practical default for this
bot: keyless, one cached request per season, refreshed through the season, and
served from a host that ordinary networks can actually reach --
``stats.nba.com`` refuses plenty of them.

    https://github.com/sportsdataverse/sportsdataverse-data
    (release tag ``espn_nba_player_boxscores``, asset ``player_box_<year>.csv``)

One trap worth naming: hoopR labels a season by the year it **ends**, so the
2024-25 season is ``player_box_2025.csv``. Everything else in this codebase
follows the NBA's own convention of naming a season by the year it *starts*,
so the conversion happens here, in one place, rather than in every caller.
"""

from __future__ import annotations

from typing import Any, Sequence

from .base import (
    FetchedLogs,
    PlayerRef,
    as_float,
    build_index,
    fuzzy_candidates,
    resolve_one,
)
from .http import HttpClient
from .nba_stats import default_season

BOX_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
    "/espn_nba_player_boxscores/player_box_{year}.csv"
)

# ESPN's season_type codes, as they appear in the CSV.
SEASON_TYPES = {"1": "PRE", "2": "REG", "3": "POST", "4": "ALLSTAR"}

FIELD_MAP: dict[str, str] = {
    "minutes": "minutes",
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "three_point_field_goals_made": "threes",
    "blocks": "blocks",
    "steals": "steals",
    "turnovers": "turnovers",
    "offensive_rebounds": "offensive_rebounds",
    "defensive_rebounds": "defensive_rebounds",
    "field_goals_attempted": "field_goal_attempts",
    "free_throws_attempted": "free_throw_attempts",
}


class HoopRSource:
    """ESPN NBA player box scores, via the hoopR release store."""

    name = "hoopr"
    sport = "nba"

    def __init__(
        self,
        client: HttpClient | None = None,
        season_types: Sequence[str] = ("REG", "POST"),
    ) -> None:
        self.client = client or HttpClient()
        self.season_types = tuple(season_types)
        self._rows_by_year: dict[int, list[dict[str, str]]] = {}

    # ------------------------------------------------------------------
    def _rows(self, season: int) -> list[dict[str, str]]:
        """Rows for a season, keyed by the season's *starting* year."""
        year = file_year(season)
        if year not in self._rows_by_year:
            self._rows_by_year[year] = list(self.client.get_csv(BOX_URL.format(year=year)))
        return self._rows_by_year[year]

    # ------------------------------------------------------------------
    def search(self, name: str, season: int | None = None) -> list[PlayerRef]:
        season = default_season() if season is None else int(season)
        seen: dict[str, PlayerRef] = {}
        for row in self._rows(season):
            athlete = row.get("athlete_id")
            if athlete and athlete not in seen:
                seen[athlete] = PlayerRef(
                    source_id=athlete,
                    name=row.get("athlete_display_name", ""),
                    sport="nba",
                    team=row.get("team_abbreviation") or None,
                    position=row.get("athlete_position_abbreviation") or None,
                )
        return fuzzy_candidates(name, build_index(list(seen.values())))

    # ------------------------------------------------------------------
    def game_logs(self, ref: PlayerRef, seasons: Sequence[int]) -> list[dict[str, Any]]:
        games: list[dict[str, Any]] = []
        for season in seasons:
            for row in self._rows(int(season)):
                if row.get("athlete_id") != ref.source_id:
                    continue
                if not played(row):
                    continue
                if self.season_types and SEASON_TYPES.get(row.get("season_type", "")) not in self.season_types:
                    continue
                games.append(self._to_game(row, int(season)))
        games.sort(key=lambda g: g["date"])
        return games

    @staticmethod
    def _to_game(row: dict[str, str], season: int) -> dict[str, Any]:
        return {
            "date": (row.get("game_date") or "")[:10],
            "season": season,
            "team": row.get("team_abbreviation", ""),
            "opponent": row.get("opponent_team_abbreviation", ""),
            "home": row.get("home_away", "").lower() == "home",
            "started": row.get("starter", "").lower() == "true",
            **{field: as_float(row.get(column)) for column, field in FIELD_MAP.items()},
        }

    # ------------------------------------------------------------------
    def fetch(self, name: str, seasons: Sequence[int], team: str | None = None) -> FetchedLogs:
        ref = resolve_one(name, self.search(name, max(int(s) for s in seasons)), team=team)
        return FetchedLogs(
            player=ref.name, sport="nba", source=self.name, ref=ref,
            games=self.game_logs(ref, [int(s) for s in seasons]),
            seasons=tuple(int(s) for s in seasons),
        )


def played(row: dict[str, str]) -> bool:
    """Drop DNPs, and nothing else.

    A did-not-play row has blank stats, and letting one through tells the
    models the player is capable of scoring zero on any given night -- which
    tanks every projection they appear in.

    Note what is *not* checked: ESPN's ``active`` column is a roster flag, not
    an appearance flag, and it reads ``false`` on plenty of games the player
    started and finished. Filtering on it silently discarded 40% of Anthony
    Edwards' season, and the resulting logs looked perfectly reasonable.
    """
    if row.get("did_not_play", "").lower() == "true":
        return False
    return as_float(row.get("minutes")) > 0


def file_year(season: int) -> int:
    """2024 (the 2024-25 season) -> the 2025 file hoopR publishes it under."""
    year = int(season)
    return year + 1 if year < 2100 else year
