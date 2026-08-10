"""Writing fetched game logs to disk, in the shape the models read.

Fetches merge rather than overwrite. Sources hand back one or two seasons at
a time, and a source that goes down shouldn't take your history with it -- so
existing games are kept and only games with the same date are replaced.
Provenance (which source, when) is recorded alongside, because six months
from now the first question about a weird projection is "where did these
numbers come from".
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..providers.base import slugify
from .base import FetchedLogs


def path_for(root: str | Path, sport: str, player: str) -> Path:
    return Path(root) / sport / f"{slugify(player)}.json"


def write_logs(
    fetched: FetchedLogs,
    root: str | Path = "data/gamelogs",
    merge: bool = True,
) -> Path:
    """Write (or merge into) ``<root>/<sport>/<player-slug>.json``."""
    target = path_for(root, fetched.sport, fetched.player)
    target.parent.mkdir(parents=True, exist_ok=True)

    games = list(fetched.games)
    existing: dict[str, Any] = {}
    if target.exists():
        existing = json.loads(target.read_text())
        previous = existing.get("source")
        if merge and previous != fetched.source:
            # Never blend feeds, and treat an unlabelled file as a different
            # feed. Two sources disagree about field definitions, date
            # conventions and which games even count, and the result is a file
            # that looks fine and models nothing. (Ask how I know.)
            merge = False
        if merge:
            games = merge_games(existing.get("games", []), games)

    payload = {
        "player": fetched.player,
        "team": _latest_team(games) or existing.get("team", ""),
        "sport": fetched.sport,
        "source": fetched.source,
        "source_id": fetched.ref.source_id,
        "position": fetched.ref.position or existing.get("position"),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "games": games,
    }
    target.write_text(json.dumps(payload, indent=1) + "\n")
    return target


def merge_games(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine two sets of games, newest data winning on a date collision."""
    by_date: dict[str, dict[str, Any]] = {str(g.get("date")): g for g in old}
    for game in new:
        by_date[str(game.get("date"))] = game
    return [by_date[key] for key in sorted(by_date)]


def _latest_team(games: list[dict[str, Any]]) -> str | None:
    for game in reversed(games):
        if game.get("team"):
            return str(game["team"])
    return None


def summarise_file(path: Path) -> str:
    """One line about a stored log file, for the fetch command's output."""
    payload = json.loads(path.read_text())
    games = payload.get("games", [])
    if not games:
        return f"{payload.get('player', path.stem)}: no games"
    return (
        f"{payload.get('player', path.stem)}: {len(games)} games "
        f"({games[0].get('date')} to {games[-1].get('date')}) via {payload.get('source', '?')}"
    )
