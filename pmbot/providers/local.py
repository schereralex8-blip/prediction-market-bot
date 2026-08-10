"""File-backed providers.

These let the whole pipeline run -- and be tested -- with no API key and no
network. The JSON shapes mirror what the live provider produces, so swapping
``provider: theoddsapi`` into the config changes nothing downstream.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..types import BookLine, GameLog, MatchupContext, PropMarket
from .base import slugify


def parse_ts(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    text = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class LocalOddsProvider:
    """Reads prop snapshots from ``data/props/<sport>*.json``.

    Expected shape::

        {"sport": "nba",
         "events": [{"id": "...", "commence_time": "...",
                     "home_team": "...", "away_team": "...",
                     "props": [{"player": "...", "team": "...",
                                "market": "player_points",
                                "books": [{"book": "pinnacle", "line": 25.5,
                                           "over": -112, "under": -108}]}]}]}
    """

    name = "local"

    def __init__(self, root: str | Path = "data/props", replay: bool = True) -> None:
        self.root = Path(root)
        self.replay = replay

    def _files(self, sport: str) -> list[Path]:
        if self.root.is_file():
            return [self.root]
        if not self.root.exists():
            return []
        return sorted(p for p in self.root.glob("*.json") if p.stem.split("_")[0] == sport)

    def player_props(
        self,
        sport: str,
        markets: Sequence[str],
        event_ids: Sequence[str] | None = None,
    ) -> list[PropMarket]:
        wanted = set(markets)
        keep_events = set(event_ids) if event_ids else None
        out: list[PropMarket] = []
        for path in self._files(sport):
            with path.open() as fh:
                payload = json.load(fh)
            out.extend(_parse_snapshot(payload, sport, wanted, keep_events))
        return _replay(out) if self.replay else out


def _replay(props: list[PropMarket]) -> list[PropMarket]:
    """Roll a stored slate forward so it reads as tonight's card.

    A snapshot on disk is frozen in time, and the scanner rightly ignores
    games that have already started. Shifting every tip-off by the same amount
    keeps the slate's internal ordering intact while letting a checked-in
    fixture stay usable a year later.
    """
    if not props:
        return props
    earliest = min(p.commence_time for p in props)
    target = datetime.now(timezone.utc) + timedelta(hours=3)
    if earliest >= datetime.now(timezone.utc):
        return props
    shift = target - earliest
    return [replace(p, commence_time=p.commence_time + shift) for p in props]


def _parse_snapshot(
    payload: dict[str, Any],
    sport: str,
    wanted: set[str],
    keep_events: set[str] | None,
) -> Iterable[PropMarket]:
    for event in payload.get("events", []):
        if keep_events and event.get("id") not in keep_events:
            continue
        home, away = event.get("home_team", ""), event.get("away_team", "")
        commence = parse_ts(event.get("commence_time"))
        for prop in event.get("props", []):
            market = prop.get("market")
            if wanted and market not in wanted:
                continue
            books = tuple(
                BookLine(
                    book=b["book"],
                    line=float(b["line"]),
                    over_american=_price(b.get("over")),
                    under_american=_price(b.get("under")),
                    last_update=parse_ts(b.get("last_update")) if b.get("last_update") else None,
                )
                for b in prop.get("books", [])
            )
            if not books:
                continue
            team = prop.get("team")
            opponent = prop.get("opponent")
            if opponent is None and team:
                opponent = away if team == home else home
            yield PropMarket(
                sport=sport,
                event_id=event.get("id", ""),
                commence_time=commence,
                home_team=home,
                away_team=away,
                player=prop["player"],
                market=market,
                books=books,
                team=team,
                opponent=opponent,
            )


def _price(value: Any) -> float | None:
    return None if value is None else float(value)


class LocalGameLogProvider:
    """Reads game logs from ``data/gamelogs/<sport>/<player-slug>.json``.

    Expected shape::

        {"player": "...", "team": "...",
         "games": [{"date": "2026-01-04", "opponent": "BOS", "home": true,
                    "minutes": 34.5, "points": 28, "rebounds": 7, ...}]}
    """

    name = "local"

    def __init__(self, root: str | Path = "data/gamelogs") -> None:
        self.root = Path(root)
        self._cache: dict[tuple[str, str], list[GameLog]] = {}

    def path_for(self, player: str, sport: str) -> Path:
        return self.root / sport / f"{slugify(player)}.json"

    def logs_for(self, player: str, sport: str) -> list[GameLog]:
        key = (sport, slugify(player))
        if key in self._cache:
            return self._cache[key]
        path = self.path_for(player, sport)
        if not path.exists():
            self._cache[key] = []
            return []
        with path.open() as fh:
            payload = json.load(fh)
        team = payload.get("team", "")
        logs = [
            GameLog(
                date=_parse_date(g["date"]),
                player=payload.get("player", player),
                team=g.get("team", team),
                opponent=g.get("opponent", ""),
                is_home=bool(g.get("home", True)),
                stats={k: float(v) for k, v in g.items() if isinstance(v, (int, float))},
            )
            for g in payload.get("games", [])
        ]
        self._cache[key] = logs
        return logs

    def known_players(self, sport: str) -> list[str]:
        folder = self.root / sport
        if not folder.exists():
            return []
        out = []
        for path in sorted(folder.glob("*.json")):
            with path.open() as fh:
                out.append(json.load(fh).get("player", path.stem))
        return out


def _parse_date(value: str) -> date:
    return date.fromisoformat(value[:10])


class MatchupBook:
    """Opponent, pace and venue multipliers, loaded from ``data/defense.json``.

    Shape::

        {"nba": {"home_factor": 1.012,
                 "pace": {"BOS": 1.03},
                 "markets": {"player_points": {"BOS": 0.96}}}}

    Anything missing defaults to 1.0, i.e. "no information, no adjustment".
    """

    def __init__(self, path: str | Path = "data/defense.json") -> None:
        self.path = Path(path)
        self._data: dict[str, Any] = {}
        if self.path.exists():
            with self.path.open() as fh:
                self._data = json.load(fh)

    def context_for(
        self,
        *,
        sport: str,
        market: str,
        opponent: str | None,
        is_home: bool | None,
        workload_override: float | None = None,
    ) -> MatchupContext:
        book = self._data.get(sport, {})
        notes: list[str] = []

        opp_factor = 1.0
        if opponent:
            opp_factor = float(book.get("markets", {}).get(market, {}).get(opponent, 1.0))
            if abs(opp_factor - 1.0) > 1e-9:
                notes.append(f"{opponent} defense x{opp_factor:.3f}")

        pace = float(book.get("pace", {}).get(opponent, 1.0)) if opponent else 1.0
        if abs(pace - 1.0) > 1e-9:
            notes.append(f"pace x{pace:.3f}")

        home_factor = 1.0
        if is_home is not None:
            hf = float(book.get("home_factor", 1.0))
            home_factor = hf if is_home else 1.0 / hf
            if abs(home_factor - 1.0) > 1e-9:
                notes.append("home" if is_home else "away")

        return MatchupContext(
            opponent_factor=opp_factor,
            pace_factor=pace,
            home_factor=home_factor,
            workload_override=workload_override,
            notes=tuple(notes),
        )
