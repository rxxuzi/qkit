"""SQLite persistence layer for qkit.

Stores market snapshots (fundamentals), watchlists, signal history,
and IV cache.  All data is keyed by symbol + date for daily granularity.

Usage::

    from qkit.data.store import Store

    db = Store()                     # default: data/market.db
    db.add_watchlist("AAPL")
    db.upsert_snapshot("AAPL", {...})
    rows = db.get_snapshots("AAPL", limit=30)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DB_DIR / "market.db"

# ── Schema ────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    symbol      TEXT PRIMARY KEY,
    added_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS snapshots (
    symbol          TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    last_price      REAL,
    open_price      REAL,
    high_price      REAL,
    low_price       REAL,
    prev_close      REAL,
    volume          INTEGER,
    turnover        REAL,
    market_cap      REAL,
    pe_ratio        REAL,
    pb_ratio        REAL,
    dividend_yield  REAL,
    eps             REAL,
    high_52w        REAL,
    low_52w         REAL,
    fetched_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    symbol      TEXT,
    signal      TEXT    NOT NULL,
    value       REAL,
    z_score     REAL,
    meta        TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_signals_date   ON signals(date);
CREATE INDEX IF NOT EXISTS idx_signals_source ON signals(source);

CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    level           TEXT    NOT NULL DEFAULT 'WARNING',
    message         TEXT    NOT NULL,
    value           REAL,
    threshold       REAL,
    acknowledged    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_alerts_date ON alerts(date);

CREATE TABLE IF NOT EXISTS iv_cache (
    symbol      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    expiry      TEXT    NOT NULL,
    strike      REAL    NOT NULL,
    option_type TEXT    NOT NULL,
    iv          REAL,
    bid         REAL,
    ask         REAL,
    volume      INTEGER,
    oi          INTEGER,
    spot        REAL,
    PRIMARY KEY (symbol, date, expiry, strike, option_type)
);
"""


# ── Dataclass for snapshot rows ───────────────────────────────────────────

@dataclass
class Snapshot:
    """A single daily market snapshot for a symbol."""

    symbol: str
    date: str
    last_price: Optional[float] = None
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    prev_close: Optional[float] = None
    volume: Optional[int] = None
    turnover: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    eps: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None


# ── Store class ───────────────────────────────────────────────────────────

class Store:
    """SQLite-backed persistence for qkit market data."""

    def __init__(self, db_path: str | Path | None = None):
        path = Path(db_path) if db_path else DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── Watchlist ─────────────────────────────────────────────────────

    def add_watchlist(self, symbol: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO watchlist (symbol) VALUES (?)",
            (symbol.upper(),),
        )
        self._conn.commit()

    def remove_watchlist(self, symbol: str) -> None:
        self._conn.execute(
            "DELETE FROM watchlist WHERE symbol = ?",
            (symbol.upper(),),
        )
        self._conn.commit()

    def get_watchlist(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT symbol FROM watchlist ORDER BY added_at"
        ).fetchall()
        return [r["symbol"] for r in rows]

    # ── Snapshots ─────────────────────────────────────────────────────

    def upsert_snapshot(self, snap: Snapshot) -> None:
        self._conn.execute(
            """INSERT INTO snapshots
               (symbol, date, last_price, open_price, high_price, low_price,
                prev_close, volume, turnover, market_cap, pe_ratio, pb_ratio,
                dividend_yield, eps, high_52w, low_52w)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(symbol, date) DO UPDATE SET
                last_price=excluded.last_price,
                open_price=excluded.open_price,
                high_price=excluded.high_price,
                low_price=excluded.low_price,
                prev_close=excluded.prev_close,
                volume=excluded.volume,
                turnover=excluded.turnover,
                market_cap=excluded.market_cap,
                pe_ratio=excluded.pe_ratio,
                pb_ratio=excluded.pb_ratio,
                dividend_yield=excluded.dividend_yield,
                eps=excluded.eps,
                high_52w=excluded.high_52w,
                low_52w=excluded.low_52w,
                fetched_at=datetime('now')
            """,
            (
                snap.symbol.upper(), snap.date,
                snap.last_price, snap.open_price, snap.high_price, snap.low_price,
                snap.prev_close, snap.volume, snap.turnover, snap.market_cap,
                snap.pe_ratio, snap.pb_ratio, snap.dividend_yield, snap.eps,
                snap.high_52w, snap.low_52w,
            ),
        )
        self._conn.commit()

    def upsert_snapshots(self, snaps: list[Snapshot]) -> None:
        for snap in snaps:
            self.upsert_snapshot(snap)

    def get_snapshots(
        self, symbol: str, limit: int = 30, since: str | None = None
    ) -> list[dict]:
        sql = "SELECT * FROM snapshots WHERE symbol = ?"
        params: list = [symbol.upper()]
        if since:
            sql += " AND date >= ?"
            params.append(since)
        sql += " ORDER BY date DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_latest_snapshot(self, symbol: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM snapshots WHERE symbol = ? ORDER BY date DESC LIMIT 1",
            (symbol.upper(),),
        ).fetchone()
        return dict(row) if row else None

    def get_all_latest_snapshots(self) -> list[dict]:
        """Get the most recent snapshot for every symbol in the watchlist."""
        rows = self._conn.execute(
            """SELECT s.* FROM snapshots s
               INNER JOIN (
                   SELECT symbol, MAX(date) as max_date
                   FROM snapshots GROUP BY symbol
               ) latest ON s.symbol = latest.symbol AND s.date = latest.max_date
               WHERE s.symbol IN (SELECT symbol FROM watchlist)
               ORDER BY s.symbol
            """
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Signals ───────────────────────────────────────────────────────

    def add_signal(
        self,
        source: str,
        signal: str,
        value: float | None = None,
        z_score: float | None = None,
        symbol: str | None = None,
        signal_date: str | None = None,
        meta: str | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO signals (date, source, symbol, signal, value, z_score, meta)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                signal_date or date.today().isoformat(),
                source, symbol, signal, value, z_score, meta,
            ),
        )
        self._conn.commit()

    def get_signals(
        self,
        source: str | None = None,
        since: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        sql = "SELECT * FROM signals WHERE 1=1"
        params: list = []
        if source:
            sql += " AND source = ?"
            params.append(source)
        if since:
            sql += " AND date >= ?"
            params.append(since)
        sql += " ORDER BY date DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_latest_signals(self) -> list[dict]:
        """Get the most recent signal from each source."""
        rows = self._conn.execute(
            """SELECT s.* FROM signals s
               INNER JOIN (
                   SELECT source, symbol, MAX(id) as max_id
                   FROM signals GROUP BY source, symbol
               ) latest ON s.id = latest.max_id
               ORDER BY s.date DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    # ── IV Cache ──────────────────────────────────────────────────────

    def cache_iv(
        self,
        symbol: str,
        cache_date: str,
        expiry: str,
        strike: float,
        option_type: str,
        iv: float | None,
        bid: float = 0,
        ask: float = 0,
        volume: int = 0,
        oi: int = 0,
        spot: float = 0,
    ) -> None:
        self._conn.execute(
            """INSERT INTO iv_cache
               (symbol, date, expiry, strike, option_type, iv, bid, ask, volume, oi, spot)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT DO UPDATE SET
                iv=excluded.iv, bid=excluded.bid, ask=excluded.ask,
                volume=excluded.volume, oi=excluded.oi, spot=excluded.spot
            """,
            (symbol.upper(), cache_date, expiry, strike, option_type,
             iv, bid, ask, volume, oi, spot),
        )
        self._conn.commit()

    def cache_iv_batch(self, rows: list[dict]) -> None:
        self._conn.executemany(
            """INSERT INTO iv_cache
               (symbol, date, expiry, strike, option_type, iv, bid, ask, volume, oi, spot)
               VALUES (:symbol, :date, :expiry, :strike, :option_type,
                       :iv, :bid, :ask, :volume, :oi, :spot)
               ON CONFLICT DO UPDATE SET
                iv=excluded.iv, bid=excluded.bid, ask=excluded.ask,
                volume=excluded.volume, oi=excluded.oi, spot=excluded.spot
            """,
            rows,
        )
        self._conn.commit()

    def get_cached_iv(
        self, symbol: str, cache_date: str | None = None
    ) -> list[dict]:
        if cache_date is None:
            cache_date = date.today().isoformat()
        rows = self._conn.execute(
            "SELECT * FROM iv_cache WHERE symbol = ? AND date = ? ORDER BY expiry, strike",
            (symbol.upper(), cache_date),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Alerts ─────────────────────────────────────────────────────────

    def add_alert(
        self,
        source: str,
        level: str,
        message: str,
        value: float | None = None,
        threshold: float | None = None,
        alert_date: str | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO alerts (date, source, level, message, value, threshold)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                alert_date or date.today().isoformat(),
                source, level, message, value, threshold,
            ),
        )
        self._conn.commit()

    def get_alerts(
        self,
        acknowledged: bool = False,
        limit: int = 50,
    ) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM alerts WHERE acknowledged = ? ORDER BY id DESC LIMIT ?",
            (1 if acknowledged else 0, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def acknowledge_alert(self, alert_id: int) -> None:
        self._conn.execute(
            "UPDATE alerts SET acknowledged = 1 WHERE id = ?",
            (alert_id,),
        )
        self._conn.commit()

    def cleanup_old_alerts(self, days: int = 7) -> int:
        """Delete alerts older than *days*. Returns count deleted."""
        cur = self._conn.execute(
            "DELETE FROM alerts WHERE date < date('now', ?)",
            (f"-{days} days",),
        )
        self._conn.commit()
        return cur.rowcount

    # ── Utilities ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return row counts for each table."""
        tables = ["watchlist", "snapshots", "signals", "alerts", "iv_cache"]
        result = {}
        for t in tables:
            row = self._conn.execute(f"SELECT COUNT(*) as n FROM {t}").fetchone()
            result[t] = row["n"]
        return result
