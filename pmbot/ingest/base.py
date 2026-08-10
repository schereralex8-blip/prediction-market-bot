"""What a real game-log source has to provide, and how names get matched.

Each source resolves a player name to that provider's own ID, then returns
game logs already mapped onto the box-score field names in
:mod:`pmbot.markets`. Everything downstream -- both models, the scanner --
sees the same shape whether the numbers came from the NFL, MLB or the NBA.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


@dataclass(frozen=True)
class PlayerRef:
    """A player as one provider knows them."""

    source_id: str
    name: str
    sport: str
    team: str | None = None
    position: str | None = None

    def describe(self) -> str:
        bits = [self.name]
        if self.team:
            bits.append(self.team)
        if self.position:
            bits.append(self.position)
        return " / ".join(bits) + f" [{self.source_id}]"


@dataclass
class FetchedLogs:
    """Game logs for one player, ready to be written to disk."""

    player: str
    sport: str
    source: str
    ref: PlayerRef
    games: list[dict[str, Any]] = field(default_factory=list)
    seasons: tuple[int, ...] = ()

    def __len__(self) -> int:
        return len(self.games)


class AmbiguousPlayer(LookupError):
    """More than one player matched, and guessing would be worse than asking."""

    def __init__(self, name: str, candidates: Sequence[PlayerRef]) -> None:
        self.candidates = list(candidates)
        listing = "; ".join(c.describe() for c in self.candidates[:8])
        super().__init__(f"{name!r} matches several players: {listing}")


class UnknownPlayer(LookupError):
    def __init__(self, name: str, near: Sequence[str] = ()) -> None:
        hint = f" Did you mean: {', '.join(near[:5])}?" if near else ""
        super().__init__(f"no player called {name!r} in this source.{hint}")


class GameLogSource(Protocol):
    name: str
    sport: str

    def search(self, name: str) -> list[PlayerRef]:
        """Every player in the source whose name plausibly matches."""

    def game_logs(self, ref: PlayerRef, seasons: Sequence[int]) -> list[dict[str, Any]]:
        """Box scores mapped onto pmbot's field names, oldest first."""


# --------------------------------------------------------------------------
# name matching
# --------------------------------------------------------------------------
_PUNCT = re.compile(r"[^a-z0-9 ]+")


def normalise_name(name: str) -> str:
    """Fold accents, drop punctuation, lowercase. 'Nikola Jokić' -> 'nikola jokic'."""
    folded = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in folded if not unicodedata.combining(c))
    return _PUNCT.sub(" ", ascii_only.lower()).strip()


def name_key(name: str) -> str:
    """A match key that ignores generational suffixes.

    Feeds disagree constantly about 'Bobby Witt Jr.' vs 'Bobby Witt': one
    lists the suffix, another doesn't, a third writes 'Jr' without the dot.
    """
    parts = [p for p in normalise_name(name).split() if p not in SUFFIXES]
    return " ".join(parts)


def resolve_one(
    name: str,
    candidates: Sequence[PlayerRef],
    team: str | None = None,
    position: str | None = None,
) -> PlayerRef:
    """Narrow a search down to exactly one player, or explain why it can't.

    Exact key matches win outright. If several survive, a supplied team or
    position breaks the tie; otherwise the caller is asked rather than being
    handed a coin flip -- picking the wrong Josh Allen silently produces a
    model that is confidently wrong.

    A *single* inexact candidate is accepted: callers pass in the output of
    :func:`fuzzy_candidates`, which has already applied a similarity cutoff,
    and that is what makes 'Patrick Mahommes' work. Several inexact
    candidates and no exact one is treated as not found, with suggestions.
    """
    if not candidates:
        raise UnknownPlayer(name)

    wanted = name_key(name)
    exact = [c for c in candidates if name_key(c.name) == wanted]
    pool = exact or list(candidates)

    if len(pool) > 1 and team:
        narrowed = [c for c in pool if (c.team or "").upper() == team.upper()]
        pool = narrowed or pool
    if len(pool) > 1 and position:
        narrowed = [c for c in pool if (c.position or "").upper() == position.upper()]
        pool = narrowed or pool

    if len(pool) == 1:
        return pool[0]
    if not exact:
        raise UnknownPlayer(name, near=[c.name for c in candidates[:5]])
    raise AmbiguousPlayer(name, pool)


def fuzzy_candidates(name: str, index: dict[str, list[PlayerRef]], limit: int = 8) -> list[PlayerRef]:
    """Look a name up by key, falling back to close spellings."""
    key = name_key(name)
    if key in index:
        return list(index[key])
    close = difflib.get_close_matches(key, list(index), n=limit, cutoff=0.85)
    return [ref for match in close for ref in index[match]]


def build_index(refs: Sequence[PlayerRef]) -> dict[str, list[PlayerRef]]:
    index: dict[str, list[PlayerRef]] = {}
    for ref in refs:
        index.setdefault(name_key(ref.name), []).append(ref)
    return index


def as_float(value: Any, default: float = 0.0) -> float:
    """Feeds are full of '', 'NA', None and '34:12'. None of those are numbers."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.upper() in ("NA", "N/A", "NULL", "-", "--"):
        return default
    if ":" in text:  # minutes as MM:SS
        minutes, _, seconds = text.partition(":")
        try:
            return float(minutes) + float(seconds) / 60.0
        except ValueError:
            return default
    try:
        return float(text)
    except ValueError:
        return default
