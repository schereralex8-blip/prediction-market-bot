"""Live odds via The Odds API (v4), using nothing but the standard library.

Player props live on the per-event endpoint, so pulling a slate is one call
for the event list plus one call per event. Those calls cost quota, so
responses are cached on disk with a short TTL and every response can be
dumped to a snapshot file -- which doubles as the closing-line record the
CLV tracker needs later.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..config import ApiSettings
from ..markets import sport_key
from ..types import BookLine, PropMarket
from .local import parse_ts


class OddsApiError(RuntimeError):
    pass


class TheOddsApiProvider:
    """Read-only client for the odds feed."""

    name = "theoddsapi"

    def __init__(self, settings: ApiSettings, cache_dir: str | Path = ".cache/odds") -> None:
        if not settings.api_key:
            raise OddsApiError(
                "no API key: set PMBOT_ODDS_API_KEY or api.api_key in pmbot.config.json"
            )
        self.settings = settings
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.requests_remaining: int | None = None
        self.requests_used: int | None = None

    # ------------------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any], ttl: float | None = None) -> Any:
        query = dict(params, apiKey=self.settings.api_key)
        url = f"{self.settings.base_url.rstrip('/')}/{path.lstrip('/')}?{urllib.parse.urlencode(query)}"
        ttl = self.settings.cache_ttl if ttl is None else ttl

        # Cache on everything except the key, so rotating keys still hits cache.
        cache_key = hashlib.sha256(url.replace(self.settings.api_key, "").encode()).hexdigest()[:24]
        cache_file = self.cache_dir / f"{cache_key}.json"
        if ttl > 0 and cache_file.exists() and time.time() - cache_file.stat().st_mtime < ttl:
            with cache_file.open() as fh:
                return json.load(fh)

        request = urllib.request.Request(url, headers={"User-Agent": "pmbot/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout) as response:
                body = response.read().decode()
                self.requests_remaining = _int_header(response, "x-requests-remaining")
                self.requests_used = _int_header(response, "x-requests-used")
        except urllib.error.HTTPError as exc:  # pragma: no cover - network
            detail = exc.read().decode(errors="replace")[:500]
            raise OddsApiError(f"HTTP {exc.code} from {path}: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network
            raise OddsApiError(f"could not reach the odds API: {exc.reason}") from exc

        data = json.loads(body)
        cache_file.write_text(body)
        return data

    # ------------------------------------------------------------------
    def events(self, sport: str, commence_before: datetime | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if commence_before:
            params["commenceTimeTo"] = _iso_z(commence_before)
        data = self._get(f"sports/{sport_key(sport)}/events", params, ttl=60.0)
        return list(data)

    def player_props(
        self,
        sport: str,
        markets: Sequence[str],
        event_ids: Sequence[str] | None = None,
    ) -> list[PropMarket]:
        if not markets:
            return []
        ids = list(event_ids) if event_ids else [e["id"] for e in self.events(sport)]
        out: list[PropMarket] = []
        for event_id in ids:
            payload = self._get(
                f"sports/{sport_key(sport)}/events/{event_id}/odds",
                {
                    "regions": self.settings.regions,
                    "markets": ",".join(markets),
                    "oddsFormat": self.settings.odds_format,
                },
            )
            out.extend(parse_event_odds(payload, sport))
        return out

    def snapshot(self, sport: str, markets: Sequence[str], path: str | Path) -> Path:
        """Write a raw snapshot in the local-provider format.

        Useful twice: replaying a slate offline, and recording closing lines
        right before lock so CLV can be computed after the fact.
        """
        props = self.player_props(sport, markets)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(to_snapshot(props, sport), indent=2))
        return target


def parse_event_odds(payload: dict[str, Any], sport: str) -> list[PropMarket]:
    """Turn one event's odds payload into PropMarket objects.

    The feed is outcome-oriented (one row per Over/Under per book); the bot is
    market-oriented (one object per player/stat with every book attached), so
    this regroups by (player, market) and pairs the two sides.
    """
    home = payload.get("home_team", "")
    away = payload.get("away_team", "")
    commence = parse_ts(payload.get("commence_time"))
    event_id = payload.get("id", "")

    # (player, market) -> (book, line) -> {"over": price, "under": price}
    grouped: dict[tuple[str, str], dict[tuple[str, float], dict[str, Any]]] = {}
    for book in payload.get("bookmakers", []):
        book_key = book.get("key", "")
        for market in book.get("markets", []):
            market_key = market.get("key", "")
            last_update = market.get("last_update") or book.get("last_update")
            for outcome in market.get("outcomes", []):
                player = outcome.get("description")
                point = outcome.get("point")
                side = str(outcome.get("name", "")).strip().lower()
                if not player or point is None or side not in ("over", "under"):
                    continue  # alternate/yes-no markets aren't two-way props
                slot = grouped.setdefault((player, market_key), {}).setdefault(
                    (book_key, float(point)), {"last_update": last_update}
                )
                slot[side] = float(outcome["price"])

    out: list[PropMarket] = []
    for (player, market_key), books in grouped.items():
        lines = tuple(
            BookLine(
                book=book_key,
                line=line,
                over_american=prices.get("over"),
                under_american=prices.get("under"),
                last_update=parse_ts(prices.get("last_update")) if prices.get("last_update") else None,
            )
            for (book_key, line), prices in sorted(books.items())
        )
        out.append(
            PropMarket(
                sport=sport,
                event_id=event_id,
                commence_time=commence,
                home_team=home,
                away_team=away,
                player=player,
                market=market_key,
                books=lines,
            )
        )
    return out


def to_snapshot(props: Sequence[PropMarket], sport: str) -> dict[str, Any]:
    events: dict[str, dict[str, Any]] = {}
    for prop in props:
        event = events.setdefault(
            prop.event_id,
            {
                "id": prop.event_id,
                "commence_time": _iso_z(prop.commence_time),
                "home_team": prop.home_team,
                "away_team": prop.away_team,
                "props": [],
            },
        )
        event["props"].append(
            {
                "player": prop.player,
                "team": prop.team,
                "market": prop.market,
                "books": [
                    {
                        "book": b.book,
                        "line": b.line,
                        "over": b.over_american,
                        "under": b.under_american,
                    }
                    for b in prop.books
                ],
            }
        )
    return {
        "sport": sport,
        "captured_at": _iso_z(datetime.now(timezone.utc)),
        "events": list(events.values()),
    }


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _int_header(response: Any, name: str) -> int | None:
    raw = response.headers.get(name)
    try:
        return int(raw) if raw is not None else None
    except ValueError:  # pragma: no cover - defensive
        return None
