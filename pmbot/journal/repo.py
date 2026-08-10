"""The betting journal: what you bet, why, what happened, and where you got
the number relative to the close.

The bankroll is derived, never stored as a mutable balance:

    balance   = deposits - withdrawals +/- adjustments + profit on settled bets
    available = balance - stake tied up in pending bets

so the books can't drift out of sync with the bet history.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..oddsmath import american_to_decimal, devig_two_way
from ..types import Signal
from .db import connect

SETTLED = ("won", "lost", "push", "void", "half_won", "half_lost")
GRADED = ("won", "lost", "half_won", "half_lost")  # pushes don't count as results


@dataclass
class Bet:
    id: int
    placed_at: str
    sport: str
    player: str
    market: str
    side: str
    line: float
    book: str
    american: float
    decimal_odds: float
    stake: float
    status: str
    event_id: str | None = None
    commence_time: str | None = None
    matchup: str | None = None
    stake_units: float | None = None
    bankroll_before: float | None = None
    model_prob: float | None = None
    fair_prob: float | None = None
    edge: float | None = None
    ev: float | None = None
    p_push: float = 0.0
    kelly_full: float | None = None
    model_probs: dict[str, float] | None = None
    settled_at: str | None = None
    result_value: float | None = None
    profit: float | None = None
    closing_line: float | None = None
    closing_american: float | None = None
    closing_fair_prob: float | None = None
    tags: str | None = None
    notes: str | None = None

    @property
    def is_settled(self) -> bool:
        return self.status in SETTLED

    @property
    def clv_pct(self) -> float | None:
        """Price beaten vs the **no-vig** closing line, as a fraction.

        This is the bet's expected value re-computed at closing probabilities,
        which is why it predicts long-run ROI better than results do over any
        sample you'll actually collect.

        Note the bar it sets: taking -110 on a market that closes -110/-110
        scores -4.5%, not zero. The line never moved your way, so you paid the
        hold and the bet was never +EV. A number above zero means you were
        still ahead of a vig-free market at the close -- which is the only
        version of "beat the close" that pays.
        """
        if self.closing_fair_prob is None or not 0 < self.closing_fair_prob < 1:
            return None
        return self.decimal_odds * self.closing_fair_prob - 1.0

    @property
    def clv_raw_pct(self) -> float | None:
        """Line movement on your side, vig included at both ends.

        The looser, more familiar number: did the price get shorter after you
        took it? Kept alongside :attr:`clv_pct` because it's what most books
        and trackers display, not because it's the better measure.
        """
        if self.closing_american is None:
            return None
        return self.decimal_odds / american_to_decimal(self.closing_american) - 1.0

    @property
    def clv_prob(self) -> float | None:
        """CLV in probability points: closing fair prob minus our implied."""
        if self.closing_fair_prob is None:
            return None
        return self.closing_fair_prob - 1.0 / self.decimal_odds

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["clv_pct"] = self.clv_pct
        data["clv_raw_pct"] = self.clv_raw_pct
        return data


class Journal:
    """CRUD plus settlement for bets and bankroll movements."""

    def __init__(self, path: str | Path = "pmbot.sqlite3") -> None:
        self.path = str(path)
        self.conn: sqlite3.Connection = connect(path)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # bankroll
    # ------------------------------------------------------------------
    def add_funds(self, amount: float, kind: str = "deposit", note: str | None = None) -> int:
        if kind not in ("deposit", "withdrawal", "adjustment"):
            raise ValueError(f"unknown bankroll event {kind!r}")
        if kind == "deposit" and amount <= 0:
            raise ValueError("a deposit must be positive")
        if kind == "withdrawal":
            amount = -abs(amount)
        cur = self.conn.execute(
            "INSERT INTO bankroll_events(ts, kind, amount, note) VALUES (?,?,?,?)",
            (_now(), kind, float(amount), note),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def deposits(self) -> float:
        row = self.conn.execute("SELECT COALESCE(SUM(amount), 0) AS total FROM bankroll_events").fetchone()
        return float(row["total"])

    def realised_profit(self) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(profit), 0) AS total FROM bets WHERE status != 'pending'"
        ).fetchone()
        return float(row["total"])

    def pending_stake(self) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(stake), 0) AS total FROM bets WHERE status = 'pending'"
        ).fetchone()
        return float(row["total"])

    def balance(self) -> float:
        """Deposits plus everything settled. Pending bets are still yours."""
        return self.deposits() + self.realised_profit()

    def available(self) -> float:
        """What's left to bet with once pending stakes are set aside."""
        return self.balance() - self.pending_stake()

    # ------------------------------------------------------------------
    # bets
    # ------------------------------------------------------------------
    def log_signal(
        self,
        signal: Signal,
        stake: float | None = None,
        american: float | None = None,
        bankroll: float | None = None,
        tags: str | None = None,
        notes: str | None = None,
    ) -> int:
        """Record a bet straight from a scanner signal.

        ``stake`` and ``american`` can be overridden for the very common case
        where the price moved (or the book limited you) between the scan and
        the click.
        """
        if not signal.is_bet:
            raise ValueError("only BET signals can be logged; this one was a PASS")
        price = american if american is not None else signal.american
        amount = stake if stake is not None else signal.stake
        assert price is not None and signal.line is not None and signal.side is not None
        return self.log_bet(
            sport=signal.sport,
            player=signal.player,
            market=signal.market,
            side=signal.side,
            line=signal.line,
            book=signal.book or "",
            american=price,
            stake=amount,
            event_id=signal.event_id,
            commence_time=signal.commence_time.isoformat(),
            matchup=signal.matchup,
            model_prob=signal.model_prob,
            fair_prob=signal.fair_prob,
            edge=signal.edge,
            ev=signal.ev_per_unit,
            p_push=signal.p_push,
            kelly_full=signal.full_kelly,
            model_probs=signal.model_probs,
            bankroll=bankroll,
            tags=tags,
            notes=notes,
        )

    def log_bet(
        self,
        *,
        sport: str,
        player: str,
        market: str,
        side: str,
        line: float,
        book: str,
        american: float,
        stake: float,
        event_id: str | None = None,
        commence_time: str | None = None,
        matchup: str | None = None,
        model_prob: float | None = None,
        fair_prob: float | None = None,
        edge: float | None = None,
        ev: float | None = None,
        p_push: float = 0.0,
        kelly_full: float | None = None,
        model_probs: dict[str, float] | None = None,
        bankroll: float | None = None,
        placed_at: str | None = None,
        tags: str | None = None,
        notes: str | None = None,
    ) -> int:
        if side not in ("over", "under"):
            raise ValueError(f"side must be 'over' or 'under', got {side!r}")
        if stake <= 0:
            raise ValueError("stake must be positive")
        decimal_odds = american_to_decimal(american)
        roll = self.balance() if bankroll is None else bankroll
        cur = self.conn.execute(
            """
            INSERT INTO bets (
                placed_at, sport, event_id, commence_time, matchup, player, market,
                side, line, book, american, decimal_odds, stake, stake_units,
                bankroll_before, model_prob, fair_prob, edge, ev, p_push,
                kelly_full, model_probs, tags, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                placed_at or _now(), sport, event_id, commence_time, matchup, player, market,
                side, float(line), book, float(american), decimal_odds, float(stake),
                (100.0 * stake / roll) if roll else None,
                roll, model_prob, fair_prob, edge, ev, p_push, kelly_full,
                json.dumps(model_probs) if model_probs else None, tags, notes,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    # ------------------------------------------------------------------
    def get(self, bet_id: int) -> Bet | None:
        row = self.conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
        return _to_bet(row) if row else None

    def bets(
        self,
        status: str | None = None,
        sport: str | None = None,
        market: str | None = None,
        book: str | None = None,
        since: str | None = None,
        tag: str | None = None,
        limit: int | None = None,
    ) -> list[Bet]:
        sql = "SELECT * FROM bets WHERE 1=1"
        params: list[Any] = []
        if status == "settled":
            sql += " AND status != 'pending'"
        elif status:
            sql += " AND status = ?"
            params.append(status)
        for column, value in (("sport", sport), ("market", market), ("book", book)):
            if value:
                sql += f" AND {column} = ?"
                params.append(value)
        if since:
            sql += " AND placed_at >= ?"
            params.append(since)
        if tag:
            sql += " AND tags LIKE ?"
            params.append(f"%{tag}%")
        sql += " ORDER BY placed_at ASC, id ASC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [_to_bet(r) for r in self.conn.execute(sql, params)]

    def pending(self) -> list[Bet]:
        return self.bets(status="pending")

    # ------------------------------------------------------------------
    # settlement
    # ------------------------------------------------------------------
    def settle(self, bet_id: int, status: str, result_value: float | None = None) -> Bet:
        if status not in SETTLED:
            raise ValueError(f"status must be one of {SETTLED}, got {status!r}")
        bet = self.get(bet_id)
        if bet is None:
            raise KeyError(f"no bet with id {bet_id}")
        profit = _profit_for(status, bet.stake, bet.decimal_odds)
        self.conn.execute(
            "UPDATE bets SET status=?, settled_at=?, result_value=?, profit=? WHERE id=?",
            (status, _now(), result_value, profit, bet_id),
        )
        self.conn.commit()
        return self.get(bet_id)  # type: ignore[return-value]

    def settle_with_result(self, bet_id: int, actual: float) -> Bet:
        """Grade a bet from what the player actually did."""
        bet = self.get(bet_id)
        if bet is None:
            raise KeyError(f"no bet with id {bet_id}")
        return self.settle(bet_id, grade(bet.side, bet.line, actual), result_value=actual)

    def void(self, bet_id: int, note: str | None = None) -> Bet:
        """DNP, scratched pitcher, cancelled game: stake back, no result."""
        if note:
            self.conn.execute("UPDATE bets SET notes = ? WHERE id = ?", (note, bet_id))
        return self.settle(bet_id, "void")

    # ------------------------------------------------------------------
    # closing lines
    # ------------------------------------------------------------------
    def record_closing(
        self,
        bet_id: int,
        *,
        over_american: float,
        under_american: float,
        line: float | None = None,
        devig_method: str = "power",
    ) -> Bet:
        """Store the de-vigged closing price for the side we bet.

        Both sides of the closing market are required on purpose: a raw
        closing price still has vig in it, and measuring CLV against a vigged
        number flatters every bet you've ever made by roughly half the hold.
        """
        bet = self.get(bet_id)
        if bet is None:
            raise KeyError(f"no bet with id {bet_id}")
        p_over, p_under = devig_two_way(over_american, under_american, devig_method)
        fair = p_over if bet.side == "over" else p_under
        ours = over_american if bet.side == "over" else under_american
        self.conn.execute(
            "UPDATE bets SET closing_line=?, closing_american=?, closing_fair_prob=? WHERE id=?",
            (line if line is not None else bet.line, ours, fair, bet_id),
        )
        self.conn.commit()
        return self.get(bet_id)  # type: ignore[return-value]

    def record_closing_from_props(
        self,
        props: Iterable[Any],
        devig_method: str = "power",
        books: Sequence[str] | None = None,
    ) -> list[int]:
        """Match a closing snapshot against pending bets and store CLV.

        Prefers the book the bet was placed at (that's the price you could
        actually have got at the close); falls back to any book quoting the
        same line.
        """
        updated: list[int] = []
        by_key: dict[tuple[str, str], list[Any]] = {}
        for prop in props:
            by_key.setdefault((prop.player.lower(), prop.market), []).append(prop)

        for bet in self.bets():
            if bet.closing_fair_prob is not None:
                continue
            for prop in by_key.get((bet.player.lower(), bet.market), []):
                quotes = [
                    b for b in prop.two_way_books
                    if abs(b.line - bet.line) < 1e-6 and (not books or b.book in books)
                ]
                if not quotes:
                    continue
                quote = next((q for q in quotes if q.book == bet.book), quotes[0])
                self.record_closing(
                    bet.id,
                    over_american=quote.over_american,
                    under_american=quote.under_american,
                    line=quote.line,
                    devig_method=devig_method,
                )
                updated.append(bet.id)
                break
        return updated


# --------------------------------------------------------------------------
def grade(side: str, line: float, actual: float) -> str:
    """over 25.5 with 27 scored -> 'won'; over 26.0 with 26 -> 'push'."""
    if abs(actual - line) < 1e-9:
        return "push"
    went_over = actual > line
    return "won" if (side == "over") == went_over else "lost"


def _profit_for(status: str, stake: float, decimal_odds: float) -> float:
    win = stake * (decimal_odds - 1.0)
    return {
        "won": win,
        "lost": -stake,
        "push": 0.0,
        "void": 0.0,
        "half_won": win / 2.0,
        "half_lost": -stake / 2.0,
    }[status]


def _to_bet(row: sqlite3.Row) -> Bet:
    data = dict(row)
    if data.get("model_probs"):
        try:
            data["model_probs"] = json.loads(data["model_probs"])
        except json.JSONDecodeError:  # pragma: no cover - defensive
            data["model_probs"] = None
    data["p_push"] = data.get("p_push") or 0.0
    return Bet(**data)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
