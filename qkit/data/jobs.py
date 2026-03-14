"""Daily batch jobs for data collection.

Run manually or via cron/task scheduler::

    python -m qkit.data.jobs              # fetch all watchlist snapshots
    python -m qkit.data.jobs --signals    # also update VRP/pairs signals

These jobs fetch data from moomoo (or yfinance fallback) and persist
to SQLite via ``Store``.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

from qkit.data.store import Store, Snapshot

log = logging.getLogger("qkit.jobs")


def fetch_snapshots(store: Store) -> int:
    """Fetch market snapshots for all watchlist symbols.

    Returns the number of snapshots stored.
    """
    symbols = store.get_watchlist()
    if not symbols:
        log.warning("Watchlist is empty — nothing to fetch")
        return 0

    from qkit.data import get_provider
    provider = get_provider()

    # Try batch fetch if moomoo
    if hasattr(provider, "get_snapshots_batch"):
        try:
            rows = provider.get_snapshots_batch(symbols)
            snaps = [Snapshot(**r) for r in rows]
            store.upsert_snapshots(snaps)
            log.info("Stored %d snapshots (batch)", len(snaps))
            return len(snaps)
        except Exception as e:
            log.warning("Batch fetch failed (%s), falling back to individual", e)

    # Individual fetch fallback
    count = 0
    today = date.today().isoformat()
    for sym in symbols:
        try:
            if hasattr(provider, "get_snapshot"):
                data = provider.get_snapshot(sym)
                store.upsert_snapshot(Snapshot(**data))
            else:
                # yfinance fallback — just store spot price
                spot = provider.get_spot_price(sym)
                store.upsert_snapshot(Snapshot(
                    symbol=sym, date=today, last_price=spot,
                ))
            count += 1
        except Exception as e:
            log.error("Failed to fetch %s: %s", sym, e)
    log.info("Stored %d snapshots (individual)", count)
    return count


def update_signals(store: Store) -> int:
    """Compute VRP and pairs signals and persist to store.

    Returns the number of signals stored.
    """
    import numpy as np
    from qkit.data import get_provider

    provider = get_provider()
    today = date.today().isoformat()
    count = 0

    # VRP signal
    try:
        from qkit.signals.vrp import compute_vrp
        spy_hist = provider.get_history("SPY", period="3y")
        spy_close = spy_hist["close"]
        spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()

        try:
            vix_hist = provider.get_history("^VIX", period="3y")
            vix_close = vix_hist["close"]
        except Exception:
            import yfinance as yf
            vix = yf.download("^VIX", period="3y", progress=False)
            vix_close = vix["Close"].squeeze()

        vrp = compute_vrp(spy_ret, vix_close)
        current = vrp.current()
        store.add_signal(
            source="VRP",
            signal=current["signal"],
            value=current["vrp"],
            z_score=current["z_score"],
            signal_date=today,
        )
        count += 1
        log.info("VRP signal: %s (z=%.2f)", current["signal"], current["z_score"])
    except Exception as e:
        log.error("VRP signal failed: %s", e)

    # Pairs signals
    try:
        from qkit.signals.pairs import screen_sector_pairs
        results = screen_sector_pairs(period="2y", significance=0.05)
        for r in results:
            if r["signal"] == "HOLD":
                continue
            store.add_signal(
                source="PAIRS",
                symbol=f"{r['asset_a']}/{r['asset_b']}",
                signal=r["signal"],
                value=float(r["current_z"]),
                z_score=float(r["current_z"]),
                signal_date=today,
                meta=f"p={r['pvalue']}, hl={r['half_life']}d",
            )
            count += 1
        log.info("Pairs: %d active signals", count - 1)
    except Exception as e:
        log.error("Pairs signal failed: %s", e)

    return count


def check_alerts(store: Store) -> int:
    """Check latest signals and generate alerts for threshold breaches.

    Returns the number of alerts created.
    """
    today = date.today().isoformat()
    count = 0

    # Cleanup old alerts
    store.cleanup_old_alerts(days=7)

    signals = store.get_latest_signals()
    for sig in signals:
        source = sig.get("source", "")
        z = sig.get("z_score")
        if z is None:
            continue

        if source == "VRP":
            if z > 1.5:
                store.add_alert(
                    source="VRP", level="WARNING",
                    message=f"VRP z-score at {z:.2f}σ — options overpriced, short vol opportunity",
                    value=z, threshold=1.5, alert_date=today,
                )
                count += 1
            elif z < -1.0:
                store.add_alert(
                    source="VRP", level="WARNING",
                    message=f"VRP z-score at {z:.2f}σ — risk underpriced, buy protection",
                    value=z, threshold=-1.0, alert_date=today,
                )
                count += 1

        elif source == "PAIRS":
            if abs(z) > 3.0:
                level = "CRITICAL" if abs(z) > 4.0 else "WARNING"
                symbol = sig.get("symbol", "")
                store.add_alert(
                    source="PAIRS", level=level,
                    message=f"{symbol} spread z={z:.2f} — {'stop loss zone' if abs(z) > 4.0 else 'extreme divergence'}",
                    value=z, threshold=3.0 if z > 0 else -3.0,
                    alert_date=today,
                )
                count += 1

    log.info("Alerts: %d new alerts", count)
    return count


def main():
    parser = argparse.ArgumentParser(description="qkit daily data jobs")
    parser.add_argument("--signals", action="store_true",
                        help="Also update VRP/pairs signals")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to SQLite database")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    with Store(args.db) as store:
        n = fetch_snapshots(store)
        print(f"Snapshots: {n} symbols updated")

        if args.signals:
            n = update_signals(store)
            print(f"Signals: {n} signals stored")
            a = check_alerts(store)
            print(f"Alerts: {a} new alerts")

        stats = store.stats()
        print(f"DB stats: {stats}")


if __name__ == "__main__":
    main()
