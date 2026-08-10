"""NBA game logs from stats.nba.com.

The NBA's own stats endpoints are free and complete, but they are fussy: they
reject requests that don't look like they came from nba.com, and they answer
in a column-oriented format (``headers`` plus ``rowSet``) rather than objects.
Both are handled here.

They also rate-limit aggressively and are known to refuse traffic from cloud
IP ranges. If this source times out or returns 403 from a server, that is the
endpoint's policy rather than a bug -- run it from a residential connection,
or point the fetcher at a mirror with ``--source``.

    https://stats.nba.com/stats/playergamelog?PlayerID=...&Season=2024-25
"""

from __future__ import annotations

from datetime import date, datetime
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

BASE = "https://stats.nba.com/stats"
PLAYERS_URL = BASE + "/commonallplayers?LeagueID=00&Season={season}&IsOnlyCurrentSeason=0"
GAMELOG_URL = BASE + "/playergamelog?PlayerID={player_id}&Season={season}&SeasonType={season_type}"

# Without these the endpoint returns nothing at all -- it is checking that the
# request looks like the nba.com front end talking to its own backend.
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

FIELD_MAP: dict[str, str] = {
    "MIN": "minutes",
    "PTS": "points",
    "REB": "rebounds",
    "AST": "assists",
    "FG3M": "threes",
    "BLK": "blocks",
    "STL": "steals",
    "TOV": "turnovers",
    "OREB": "offensive_rebounds",
    "DREB": "defensive_rebounds",
    "FGA": "field_goal_attempts",
    "FTA": "free_throw_attempts",
}


class NbaStatsSource:
    """Player game logs from the NBA's own stats endpoints."""

    name = "nba-stats"
    sport = "nba"

    def __init__(
        self,
        client: HttpClient | None = None,
        season_types: Sequence[str] = ("Regular Season", "Playoffs"),
    ) -> None:
        self.client = client or HttpClient(min_interval=1.5)  # this host is touchy
        self.season_types = tuple(season_types)

    # ------------------------------------------------------------------
    def search(self, name: str, season: int | str | None = None) -> list[PlayerRef]:
        payload = self.client.get_json(
            PLAYERS_URL.format(season=nba_season(season)), headers=HEADERS, ttl=86_400
        )
        refs = [
            PlayerRef(
                source_id=str(row["PERSON_ID"]),
                name=row.get("DISPLAY_FIRST_LAST", ""),
                sport="nba",
                team=row.get("TEAM_ABBREVIATION") or None,
            )
            for row in rows_to_dicts(payload)
        ]
        return fuzzy_candidates(name, build_index(refs))

    # ------------------------------------------------------------------
    def game_logs(self, ref: PlayerRef, seasons: Sequence[int | str]) -> list[dict[str, Any]]:
        games: list[dict[str, Any]] = []
        for season in seasons:
            for season_type in self.season_types:
                url = GAMELOG_URL.format(
                    player_id=ref.source_id,
                    season=nba_season(season),
                    season_type=season_type.replace(" ", "+"),
                )
                payload = self.client.get_json(url, headers=HEADERS)
                for row in rows_to_dicts(payload):
                    games.append(self._to_game(row, nba_season(season)))
        games.sort(key=lambda g: g["date"])
        return games

    @staticmethod
    def _to_game(row: dict[str, Any], season: str) -> dict[str, Any]:
        team, opponent, is_home = parse_matchup(str(row.get("MATCHUP", "")))
        return {
            "date": parse_game_date(str(row.get("GAME_DATE", ""))),
            "season": season,
            "team": team,
            "opponent": opponent,
            "home": is_home,
            **{field: as_float(row.get(key)) for key, field in FIELD_MAP.items()},
        }

    # ------------------------------------------------------------------
    def fetch(self, name: str, seasons: Sequence[int | str], team: str | None = None) -> FetchedLogs:
        ref = resolve_one(name, self.search(name, max(seasons, key=str)), team=team)
        return FetchedLogs(
            player=ref.name, sport="nba", source=self.name, ref=ref,
            games=self.game_logs(ref, seasons), seasons=tuple(int(str(s)[:4]) for s in seasons),
        )


# --------------------------------------------------------------------------
def rows_to_dicts(payload: dict[str, Any], result_set: int = 0) -> list[dict[str, Any]]:
    """Turn the NBA's column-oriented reply into ordinary dicts."""
    sets = payload.get("resultSets") or payload.get("resultSet") or []
    if isinstance(sets, dict):
        sets = [sets]
    if not sets or result_set >= len(sets):
        return []
    block = sets[result_set]
    headers = block.get("headers", [])
    return [dict(zip(headers, row)) for row in block.get("rowSet", [])]


def parse_matchup(matchup: str) -> tuple[str, str, bool]:
    """'MIN vs. SAC' -> home; 'MIN @ SAC' -> away."""
    if " vs. " in matchup:
        team, _, opponent = matchup.partition(" vs. ")
        return team.strip(), opponent.strip(), True
    if " @ " in matchup:
        team, _, opponent = matchup.partition(" @ ")
        return team.strip(), opponent.strip(), False
    return matchup.strip(), "", True


def parse_game_date(value: str) -> str:
    """'OCT 25, 2024' -> '2024-10-25'."""
    for fmt in ("%b %d, %Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return value.strip()[:10]


def nba_season(season: int | str | None = None) -> str:
    """Accept 2024, '2024' or '2024-25' and always return '2024-25'.

    NBA seasons are named for the year they start, which is a reliable source
    of off-by-one errors when everything else in the codebase uses the year a
    season ends.
    """
    if season is None:
        return nba_season(default_season())
    text = str(season)
    if "-" in text:
        return text
    year = int(text)
    return f"{year}-{str(year + 1)[-2:]}"


def default_season() -> int:
    """The season currently under way, by its starting year."""
    today = date.today()
    return today.year if today.month >= 10 else today.year - 1
