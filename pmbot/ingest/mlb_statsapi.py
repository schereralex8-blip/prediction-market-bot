"""MLB game logs from the official MLB Stats API.

``statsapi.mlb.com`` is free, keyless, and the same feed MLB's own products
read, which makes it the best source in any of the three sports. Hitting and
pitching are separate stat groups, so both are fetched and merged by date --
that way a two-way player ends up with one row per game carrying both halves
of their night rather than two half-rows that neither model can use.

    https://statsapi.mlb.com/api/v1/people/{id}/stats?stats=gameLog&group=hitting&season=2025
"""

from __future__ import annotations

from datetime import date
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

BASE = "https://statsapi.mlb.com/api/v1"
PLAYERS_URL = BASE + "/sports/1/players?season={season}"
TEAMS_URL = BASE + "/teams?sportId=1&season={season}"
GAMELOG_URL = BASE + "/people/{player_id}/stats?stats=gameLog&group={group}&season={season}"

HITTING_MAP: dict[str, str] = {
    "plateAppearances": "plate_appearances",
    "atBats": "at_bats",
    "hits": "hits",
    "totalBases": "total_bases",
    "homeRuns": "home_runs",
    "rbi": "rbis",
    "runs": "runs",
    "stolenBases": "stolen_bases",
    "strikeOuts": "batter_strikeouts",
    "baseOnBalls": "walks",
    "doubles": "doubles",
    "triples": "triples",
}

PITCHING_MAP: dict[str, str] = {
    "battersFaced": "batters_faced",
    "strikeOuts": "strikeouts",
    "hits": "hits_allowed",
    "earnedRuns": "earned_runs",
    "baseOnBalls": "walks_allowed",
    "homeRuns": "home_runs_allowed",
}


class MlbStatsApiSource:
    """Batter and pitcher game logs from MLB's own API."""

    name = "mlb-statsapi"
    sport = "mlb"

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()
        self._teams: dict[int, str] | None = None

    # ------------------------------------------------------------------
    def _team_abbreviations(self, season: int) -> dict[int, str]:
        if self._teams is None:
            payload = self.client.get_json(TEAMS_URL.format(season=season), ttl=604_800)
            self._teams = {
                int(t["id"]): t.get("abbreviation") or t.get("teamCode", "").upper()
                for t in payload.get("teams", [])
            }
        return self._teams

    # ------------------------------------------------------------------
    def search(self, name: str, season: int | None = None) -> list[PlayerRef]:
        season = season or default_season()
        payload = self.client.get_json(PLAYERS_URL.format(season=season), ttl=86_400)
        refs = [
            PlayerRef(
                source_id=str(person["id"]),
                name=person.get("fullName", ""),
                sport="mlb",
                team=(person.get("currentTeam") or {}).get("abbreviation")
                or self._team_abbreviations(season).get((person.get("currentTeam") or {}).get("id", -1)),
                position=(person.get("primaryPosition") or {}).get("abbreviation"),
            )
            for person in payload.get("people", [])
        ]
        return fuzzy_candidates(name, build_index(refs))

    # ------------------------------------------------------------------
    def game_logs(self, ref: PlayerRef, seasons: Sequence[int]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for season in seasons:
            teams = self._team_abbreviations(season)
            for group, mapping in (("hitting", HITTING_MAP), ("pitching", PITCHING_MAP)):
                for split in self._splits(ref.source_id, group, season):
                    game = self._to_game(split, mapping, teams, season)
                    key = f"{game['date']}#{split.get('game', {}).get('gamePk', '')}"
                    if key in merged:
                        merged[key].update(game)  # two-way player: same game, other half
                    else:
                        merged[key] = game
        return [merged[k] for k in sorted(merged)]

    def _splits(self, player_id: str, group: str, season: int) -> list[dict[str, Any]]:
        url = GAMELOG_URL.format(player_id=player_id, group=group, season=season)
        payload = self.client.get_json(url)
        out: list[dict[str, Any]] = []
        for block in payload.get("stats", []):
            out.extend(block.get("splits", []))
        return out

    @staticmethod
    def _to_game(
        split: dict[str, Any],
        mapping: dict[str, str],
        teams: dict[int, str],
        season: int,
    ) -> dict[str, Any]:
        stat = split.get("stat", {}) or {}
        opponent = split.get("opponent", {}) or {}
        game = {
            "date": split.get("date") or f"{season}-01-01",
            "season": season,
            "opponent": opponent.get("abbreviation")
            or teams.get(int(opponent.get("id", -1) or -1), "")
            or "",
            "home": bool(split.get("isHome", False)),
            **{field: as_float(stat.get(key)) for key, field in mapping.items()},
        }
        if "inningsPitched" in stat:
            game["outs"] = innings_to_outs(stat["inningsPitched"])
        return game

    # ------------------------------------------------------------------
    def fetch(self, name: str, seasons: Sequence[int], team: str | None = None) -> FetchedLogs:
        ref = resolve_one(name, self.search(name, max(seasons)), team=team)
        return FetchedLogs(
            player=ref.name, sport="mlb", source=self.name, ref=ref,
            games=self.game_logs(ref, seasons), seasons=tuple(seasons),
        )


def innings_to_outs(innings: Any) -> float:
    """'6.1' means six innings and one out -- 19 outs, not 6.1 of anything.

    Reading that decimal as a normal number is the classic baseball data bug
    and it quietly corrupts every outs-based projection.
    """
    text = str(innings).strip()
    if not text:
        return 0.0
    whole, _, fraction = text.partition(".")
    try:
        outs = int(float(whole or 0)) * 3
    except ValueError:
        return 0.0
    if fraction and fraction[0] in "12":
        outs += int(fraction[0])
    return float(outs)


def default_season() -> int:
    today = date.today()
    return today.year if today.month >= 3 else today.year - 1
