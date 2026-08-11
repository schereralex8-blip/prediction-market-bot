"""A small read-only HTTP service: tonight's signals, the journal, the report.

Built on ``http.server`` so the deployment stays dependency-free. It is
deliberately **read-only** -- nothing here can place, settle or delete a bet.
Writes stay on the CLI, where they need a person.

Two things this gets right because the alternative is expensive:

* **Auth is on by default.** A platform URL is public, and this page shows
  your bankroll and your open positions. The server refuses to start without
  ``PMBOT_API_TOKEN`` unless you explicitly pass ``--allow-anonymous``.
* **Scans are cached.** A full slate is thousands of Monte Carlo simulations;
  recomputing it per request would let one refresh key peg a CPU. Results are
  memoised per (sport, bankroll) for ``scan_ttl`` seconds.
"""

from __future__ import annotations

import hmac
import json
import logging
import threading
import time
import urllib.parse
from dataclasses import asdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from . import __version__
from .config import Settings
from .journal import Journal, bankroll_curve, breakdown, max_drawdown, summarise
from .markets import SPORTS
from .oddsmath import format_american
from .pipeline import PropScanner
from .types import Signal

log = logging.getLogger("pmbot.server")


class ScanCache:
    """Memoise slate scans; they're far too expensive to run per request."""

    def __init__(self, ttl: float = 300.0) -> None:
        self.ttl = ttl
        self._entries: dict[tuple, tuple[float, list[Signal]]] = {}
        self._lock = threading.Lock()

    def get_or_compute(self, key: tuple, compute: Callable[[], list[Signal]]) -> tuple[list[Signal], float]:
        """Returns (signals, age_seconds)."""
        now = time.time()
        with self._lock:
            hit = self._entries.get(key)
            if hit and now - hit[0] < self.ttl:
                return hit[1], now - hit[0]
        signals = compute()  # computed outside the lock: slow, and not shared state
        with self._lock:
            self._entries[key] = (now, signals)
        return signals, 0.0

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class PmbotService:
    """Everything the handler needs, so the handler stays about HTTP."""

    def __init__(self, settings: Settings, token: str | None = None, scan_ttl: float = 300.0) -> None:
        self.settings = settings
        self.token = token
        self.cache = ScanCache(scan_ttl)
        self.started_at = datetime.now(timezone.utc)
        # Set by the CLI when a daily pass is armed. Exposed so a deployed
        # scheduler is observable -- otherwise you cannot tell a job that ran
        # quietly from one that never fired at all.
        self.scheduler: Any = None

    # ------------------------------------------------------------------
    def authorised(self, headers: Any, query: dict[str, list[str]]) -> bool:
        if not self.token:
            return True
        supplied = ""
        auth = headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            supplied = auth[7:]
        elif query.get("token"):
            supplied = query["token"][0]
        return hmac.compare_digest(supplied, self.token)

    # ------------------------------------------------------------------
    def signals(self, sport: str, bankroll: float | None, include_passes: bool) -> tuple[list[Signal], float]:
        roll = bankroll if bankroll is not None else self.settings.staking.bankroll
        key = (sport, round(roll, 2), include_passes)

        def compute() -> list[Signal]:
            scanner = PropScanner(self.settings)
            return scanner.scan(sport, bankroll=roll, include_passes=include_passes)

        return self.cache.get_or_compute(key, compute)

    def journal(self) -> Journal:
        return Journal(self.settings.data.db)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "odds_provider": self.settings.api.provider,
            "odds_key_configured": bool(self.settings.api.api_key),
            "authenticated": bool(self.token),
            "daily_job": self.schedule_status(),
        }

    def schedule_status(self) -> dict[str, Any] | None:
        return self.scheduler.status() if self.scheduler else None

    def overview(self) -> dict[str, Any]:
        with self.journal() as journal:
            bets = journal.bets()
            perf = summarise(bets)
            return {
                "bankroll": {
                    "balance": journal.balance(),
                    "deposits": journal.deposits(),
                    "realised_profit": journal.realised_profit(),
                    "pending_stake": journal.pending_stake(),
                    "available": journal.available(),
                },
                "performance": asdict(perf)
                | {"roi": perf.roi, "win_rate": perf.win_rate, "breakeven_rate": perf.breakeven_rate},
                "max_drawdown": max_drawdown(bankroll_curve(bets, starting=journal.deposits())),
                "by_market": [asdict(p) | {"roi": p.roi} for p in breakdown(bets, "market")],
                "open_bets": [b.to_dict() for b in journal.pending()],
            }


# ==========================================================================
# HTTP
# ==========================================================================
class Handler(BaseHTTPRequestHandler):
    service: PmbotService  # injected by make_server
    server_version = f"pmbot/{__version__}"
    sys_version = ""

    # -- plumbing ------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:
        # The default logs to stderr with no structure, and a token in a query
        # string would go straight into the platform's log drain with it.
        log.info("%s %s", self.command, urllib.parse.urlsplit(self.path).path)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, indent=2, default=_json_default).encode()
        self._send(status, body, "application/json; charset=utf-8")

    def html(self, markup: str, status: int = 200) -> None:
        self._send(status, markup.encode(), "text/html; charset=utf-8")

    # -- routing -------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802 - stdlib naming
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/healthz":  # unauthenticated on purpose: platform probes it
            return self.json(self.service.health())

        if not self.service.authorised(self.headers, query):
            return self.json({"error": "unauthorised"}, HTTPStatus.UNAUTHORIZED)

        try:
            if path == "/":
                return self.html(self._dashboard(query))
            if path == "/api/signals":
                return self.json(self._signals_payload(query))
            if path == "/api/overview":
                return self.json(self.service.overview())
            if path == "/api/bets":
                with self.service.journal() as journal:
                    status = (query.get("status") or [None])[0]
                    return self.json([b.to_dict() for b in journal.bets(status=status)])
            if path == "/api/schedule":
                status = self.service.schedule_status()
                if status is None:
                    return self.json({"enabled": False, "detail": "no daily pass armed"})
                return self.json({"enabled": True, **status})
            if path == "/api/config":
                shown = self.service.settings.to_dict()
                shown["api"]["api_key"] = "***" if shown["api"]["api_key"] else ""
                return self.json(shown)
        except Exception as exc:  # noqa: BLE001 - a 500 beats a dead worker
            log.exception("request failed: %s", path)
            return self.json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        self.json({"error": f"no route for {path}"}, HTTPStatus.NOT_FOUND)

    # -- payloads ------------------------------------------------------
    def _signals_payload(self, query: dict[str, list[str]]) -> dict[str, Any]:
        sport = (query.get("sport") or ["nba"])[0].lower()
        if sport not in SPORTS:
            raise ValueError(f"unknown sport {sport!r}; expected one of {', '.join(SPORTS)}")
        bankroll = float(query["bankroll"][0]) if query.get("bankroll") else None
        include_passes = (query.get("all") or ["0"])[0] in ("1", "true", "yes")
        signals, age = self.service.signals(sport, bankroll, include_passes)
        return {
            "sport": sport,
            "generated_seconds_ago": round(age, 1),
            "bankroll": bankroll if bankroll is not None else self.service.settings.staking.bankroll,
            "count": len(signals),
            "signals": [asdict(s) for s in signals],
        }

    def _dashboard(self, query: dict[str, list[str]]) -> str:
        sport = (query.get("sport") or ["nba"])[0].lower()
        if sport not in SPORTS:
            sport = "nba"
        token = (query.get("token") or [""])[0]
        try:
            signals, age = self.service.signals(sport, None, include_passes=True)
            error = None
        except Exception as exc:  # noqa: BLE001 - render the failure, don't 500 the page
            signals, age, error = [], 0.0, str(exc)
        return render_dashboard(
            sport=sport,
            signals=signals,
            age=age,
            overview=self.service.overview(),
            token=token,
            error=error,
            schedule=self.service.schedule_status(),
        )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


# ==========================================================================
# dashboard
# ==========================================================================
STYLE = """
:root {
  --bg: #fbfbfa; --panel: #fff; --ink: #1b1a19; --muted: #6b6a68;
  --line: #e6e4e1; --bet: #1a7f4b; --pass: #8a8885; --warn: #a8600a;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #17181a; --panel: #1f2124; --ink: #e9e8e6; --muted: #9b9a97;
    --line: #2e3033; --bet: #4ec98a; --pass: #7d7c79; --warn: #d99a4e;
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; background: var(--bg); color: var(--ink);
  font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 2px; letter-spacing: -0.01em; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
nav a { color: var(--muted); text-decoration: none; margin-right: 14px;
  padding-bottom: 3px; border-bottom: 2px solid transparent; }
nav a.on { color: var(--ink); border-bottom-color: var(--ink); }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px; margin: 18px 0 24px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }
.card .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
.card .v { font-size: 20px; font-variant-numeric: tabular-nums; margin-top: 3px; }
.sig { background: var(--panel); border: 1px solid var(--line); border-left-width: 3px;
  border-radius: 8px; padding: 12px 14px; margin-bottom: 8px; }
.sig.bet { border-left-color: var(--bet); }
.sig.pass { border-left-color: var(--line); }
.sig .top { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.tag { font-size: 11px; font-weight: 600; letter-spacing: 0.06em; }
.tag.bet { color: var(--bet); } .tag.pass { color: var(--pass); }
.who { font-weight: 600; }
.pick { font-variant-numeric: tabular-nums; }
.why { color: var(--muted); font-size: 13px; margin-top: 5px; }
.warn { color: var(--warn); font-size: 12px; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; color: var(--muted); font-weight: 500; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.05em; padding: 6px 8px; }
td { padding: 6px 8px; border-top: 1px solid var(--line); font-variant-numeric: tabular-nums; }
.empty { color: var(--muted); padding: 20px 0; }
.err { background: var(--panel); border: 1px solid var(--warn); border-radius: 8px;
  padding: 12px 14px; color: var(--warn); margin-bottom: 16px; }
footer { color: var(--muted); font-size: 12px; margin-top: 28px;
  border-top: 1px solid var(--line); padding-top: 12px; }
"""


def _esc(text: Any) -> str:
    from html import escape

    return escape(str(text), quote=True)


def _money(x: float) -> str:
    return f"{x:,.2f}"


def _pct(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "--"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def render_dashboard(
    *,
    sport: str,
    signals: list[Signal],
    age: float,
    overview: dict[str, Any],
    token: str = "",
    error: str | None = None,
    schedule: dict[str, Any] | None = None,
) -> str:
    bank = overview["bankroll"]
    perf = overview["performance"]
    suffix = f"?token={urllib.parse.quote(token)}&" if token else "?"

    nav = " ".join(
        f'<a class="{"on" if s == sport else ""}" href="{suffix}sport={s}">{s.upper()}</a>'
        for s in SPORTS
    )
    cards = "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div></div>'
        for k, v in [
            ("balance", _money(bank["balance"])),
            ("available", _money(bank["available"])),
            ("pending", _money(bank["pending_stake"])),
            ("roi", _pct(perf["roi"], signed=True)),
            ("record", f"{perf['wins']}-{perf['losses']}"),
            ("clv", _pct(perf["clv_avg"], signed=True) if perf["clv_n"] else "--"),
        ]
    )

    bets = [s for s in signals if s.is_bet]
    rows = "".join(_signal_html(s) for s in signals) or (
        '<div class="empty">Nothing on the board. With no odds API key configured the '
        "scanner has no prices to read.</div>"
    )

    open_bets = overview["open_bets"]
    if open_bets:
        body = "".join(
            "<tr>"
            f"<td>{_esc(b['placed_at'][:10])}</td><td>{_esc(b['player'])}</td>"
            f"<td>{_esc(b['market'])}</td><td>{_esc(b['side'])} {b['line']:g}</td>"
            f"<td>{_esc(format_american(b['american']))}</td><td>{_money(b['stake'])}</td>"
            f"<td>{_esc(b['book'])}</td></tr>"
            for b in open_bets
        )
        open_table = (
            "<h2 style='font-size:15px;margin:24px 0 8px'>Open bets</h2><table>"
            "<tr><th>placed</th><th>player</th><th>market</th><th>selection</th>"
            "<th>odds</th><th>stake</th><th>book</th></tr>" + body + "</table>"
        )
    else:
        open_table = ""

    err = f'<div class="err">{_esc(error)}</div>' if error else ""
    freshness = "just now" if age < 1 else f"{int(age)}s ago"

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>pmbot &middot; {_esc(sport.upper())}</title>
<style>{STYLE}</style>
</head><body><div class="wrap">
<h1>pmbot</h1>
<div class="sub">{len(bets)} bet(s) of {len(signals)} prop(s) &middot; scanned {freshness}</div>
<nav>{nav}</nav>
<div class="cards">{cards}</div>
{err}
{rows}
{open_table}
<footer>{_schedule_html(schedule)}Read-only. Place, settle and grade bets with the CLI &mdash; this page cannot.</footer>
</div></body></html>"""


def _schedule_html(schedule: dict[str, Any] | None) -> str:
    """Show the daily job's last outcome; a silent scheduler is a broken one."""
    if not schedule:
        return "No daily pass armed. &nbsp;&middot;&nbsp; "
    last = schedule.get("last_status")
    when = schedule.get("last_finished_at") or "never"
    state = {"ok": "ok", "failed": "FAILED", None: "not yet run"}.get(last, str(last))
    return (
        f"Daily pass {_esc(schedule.get('schedule_utc'))} UTC &middot; last {_esc(state)} "
        f"({_esc(when)}) &middot; next {_esc(schedule.get('next_run', '?'))}"
        f"{' &middot; ' + _esc(schedule['last_summary']) if schedule.get('last_summary') else ''}"
        " &nbsp;&middot;&nbsp; "
    )


def _signal_html(s: Signal) -> str:
    if not s.is_bet:
        return (
            '<div class="sig pass"><div class="top">'
            f'<span><span class="tag pass">PASS</span> &nbsp;<span class="who">{_esc(s.player)}</span>'
            f" &middot; {_esc(s.market_label)}</span></div>"
            f'<div class="why">{_esc(s.reason)}</div></div>'
        )
    warnings = "".join(f'<div class="warn">! {_esc(w)}</div>' for w in s.warnings)
    return (
        '<div class="sig bet"><div class="top">'
        f'<span><span class="tag bet">BET</span> &nbsp;<span class="who">{_esc(s.player)}</span>'
        f" &middot; {_esc(s.market_label)} "
        f'<span class="pick">{_esc(s.side.upper())} {s.line:g} '
        f"{_esc(format_american(s.american))} @ {_esc(s.book)}</span></span>"
        f"<span class='pick'>{_money(s.stake)} &middot; EV {_pct(s.ev_per_unit, signed=True)}</span>"
        "</div>"
        f'<div class="why">model {_pct(s.model_prob)} vs fair {_pct(s.fair_prob)} '
        f"from {s.n_books} book(s) &middot; edge {_pct(s.edge, signed=True)} &middot; {_esc(s.reason)}</div>"
        f"{warnings}</div>"
    )


# ==========================================================================
def make_server(
    settings: Settings,
    host: str = "0.0.0.0",
    port: int = 8080,
    token: str | None = None,
    scan_ttl: float = 300.0,
) -> ThreadingHTTPServer:
    service = PmbotService(settings, token=token, scan_ttl=scan_ttl)
    handler = type("BoundHandler", (Handler,), {"service": service})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    # Callers need the service too -- to attach a scheduler, or to drop the
    # scan cache once fresh game logs land.
    server.service = service  # type: ignore[attr-defined]
    return server
