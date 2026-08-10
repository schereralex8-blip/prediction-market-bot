"""SQLite storage for the betting journal.

One file, no server, fully greppable with the sqlite3 CLI if you'd rather.
The schema records not just what was bet but *why* -- model probability, fair
probability, edge and EV at the time of the bet -- because a journal that
only stores results can tell you that you're losing but never why.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bankroll_events (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     TEXT    NOT NULL,
    kind   TEXT    NOT NULL CHECK (kind IN ('deposit', 'withdrawal', 'adjustment')),
    amount REAL    NOT NULL,
    note   TEXT
);

CREATE TABLE IF NOT EXISTS bets (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    placed_at        TEXT    NOT NULL,
    sport            TEXT    NOT NULL,
    event_id         TEXT,
    commence_time    TEXT,
    matchup          TEXT,
    player           TEXT    NOT NULL,
    market           TEXT    NOT NULL,
    side             TEXT    NOT NULL CHECK (side IN ('over', 'under')),
    line             REAL    NOT NULL,
    book             TEXT    NOT NULL,
    american         REAL    NOT NULL,
    decimal_odds     REAL    NOT NULL,
    stake            REAL    NOT NULL CHECK (stake > 0),
    stake_units      REAL,
    bankroll_before  REAL,
    -- the reasoning, frozen at placement time
    model_prob       REAL,
    fair_prob        REAL,
    edge             REAL,
    ev               REAL,
    p_push           REAL    DEFAULT 0,
    kelly_full       REAL,
    model_probs      TEXT,
    -- settlement
    status           TEXT    NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending','won','lost','push','void','half_won','half_lost')),
    settled_at       TEXT,
    result_value     REAL,
    profit           REAL,
    -- closing line, for CLV
    closing_line     REAL,
    closing_american REAL,
    closing_fair_prob REAL,
    tags             TEXT,
    notes            TEXT
);

CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status);
CREATE INDEX IF NOT EXISTS idx_bets_placed ON bets(placed_at);
CREATE INDEX IF NOT EXISTS idx_bets_event  ON bets(event_id);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the journal database."""
    target = Path(path)
    if target.parent and str(target.parent) not in ("", "."):
        target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
