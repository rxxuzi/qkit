"""Web dashboard for qkit.

Run with ``qkit serve`` or ``python -m qkit.web.app``.
All charts are rendered as interactive plotly in the browser.
No static PNGs — everything is live.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from flask import Flask, render_template, jsonify, request, redirect, url_for

from qkit.pricing.bsm import BSM
from qkit.portfolio.risk import compute_all

app = Flask(__name__, template_folder="templates", static_folder="static")


# ── Pages ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/market/")
@app.route("/market/<symbol>")
def market(symbol="SPY"):
    return redirect(url_for("symbol_page", symbol=symbol.upper()))


@app.route("/pricer")
def pricer():
    return render_template("pricer.html")


@app.route("/greeks")
def greeks_page():
    return render_template("greeks.html")


@app.route("/risk")
def risk_page():
    return render_template("risk.html")


@app.route("/signals")
def signals_page():
    return render_template("signals.html")


@app.route("/symbol/<symbol>")
def symbol_page(symbol):
    return render_template("symbol.html", symbol=symbol.upper())


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/pairs")
def pairs_page():
    return render_template("pairs.html")


@app.route("/screener")
def screener_page():
    return render_template("screener.html")


@app.route("/options/<symbol>")
def options_page(symbol):
    return render_template("options.html", symbol=symbol.upper())


# ── API v1: Core ─────────────────────────────────────────────────────────

@app.route("/api/v1/bsm")
def api_bsm():
    try:
        S = float(request.args.get("S", 230))
        K = float(request.args.get("K", 235))
        T = float(request.args.get("T", 30)) / 365
        r = float(request.args.get("r", 4.3)) / 100
        sigma = float(request.args.get("sigma", 22)) / 100

        m = BSM(S=S, K=K, T=T, r=r, sigma=sigma)
        return jsonify({
            "call": round(m.call_price(), 4),
            "put": round(m.put_price(), 4),
            "call_greeks": {k: round(getattr(m, f"call_{k}")(), 6)
                           for k in ["delta", "gamma", "vega", "theta", "rho"]},
            "put_greeks": {k: round(getattr(m, f"put_{k}")(), 6)
                          for k in ["delta", "gamma", "vega", "theta", "rho"]},
            "parity": {k: v.item() if hasattr(v, 'item') else v
                       for k, v in m.verify_put_call_parity().items()},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/risk")
def api_risk():
    try:
        mu = float(request.args.get("mu", 0.05)) / 100
        sigma = float(request.args.get("sigma", 1.0)) / 100
        n = int(request.args.get("n", 500))
        conf = float(request.args.get("confidence", 95)) / 100

        returns = np.random.default_rng().normal(mu, sigma, n)
        report = compute_all(returns, confidence=conf)
        return jsonify(report.as_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── API v1: Charts (plotly JSON) ─────────────────────────────────────────

@app.route("/api/v1/chart/greeks")
def api_greeks_chart():
    from qkit.pipeline import greeks_heatmap_chart
    S = float(request.args.get("S", 230))
    sigma = float(request.args.get("sigma", 22)) / 100
    greek = request.args.get("greek", "delta")
    opt_type = request.args.get("type", "call")
    return jsonify(greeks_heatmap_chart(S, sigma, greek=greek, opt_type=opt_type))


@app.route("/api/v1/chart/greeks_dashboard")
def api_greeks_dashboard():
    from qkit.pipeline import greeks_dashboard_chart
    S = float(request.args.get("S", 230))
    sigma = float(request.args.get("sigma", 22)) / 100
    opt_type = request.args.get("type", "call")
    return jsonify(greeks_dashboard_chart(S, sigma, opt_type=opt_type))


@app.route("/api/v1/chart/garch")
def api_garch_chart():
    """GARCH analysis on real data."""
    from qkit.pipeline import garch_chart
    from qkit.data import get_provider
    symbol = request.args.get("symbol", "SPY")
    try:
        provider = get_provider()
        returns = provider.get_daily_returns(symbol, period="5y")
        result = garch_chart(returns, symbol)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/chart/vrp")
def api_vrp_chart():
    """VRP signal on real data."""
    from qkit.pipeline import vrp_chart
    from qkit.data import get_provider

    try:
        provider = get_provider()
        spy_hist = provider.get_history("SPY", period="3y")
        spy_close = spy_hist["close"]
        spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()

        # VIX: try provider first, fall back to yfinance for ^VIX
        try:
            vix_hist = provider.get_history("^VIX", period="3y")
            vix_close = vix_hist["close"]
        except Exception:
            import yfinance as yf
            vix = yf.download("^VIX", period="3y", progress=False)
            vix_close = vix["Close"].squeeze()

        result = vrp_chart(spy_ret, vix_close, spy_close)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/chart/iv_surface")
def api_iv_surface():
    """IV surface from real option chain."""
    from qkit.pipeline import iv_surface_chart, iv_smile_chart
    from qkit.pricing.iv import implied_vol, filter_chain
    from qkit.data import get_provider
    import pandas as pd

    symbol = request.args.get("symbol", "SPY")
    try:
        provider = get_provider()
        spot = provider.get_spot_price(symbol)
        chain = provider.get_option_chain(symbol)
        df = chain.to_dataframe()

        df = filter_chain(df, spot, min_volume=1, min_oi=1)
        if df.empty:
            return jsonify({"error": "No options data after filtering"})

        # compute expiry in years
        from datetime import datetime
        today = pd.Timestamp.now()
        df["expiry_years"] = df["expiry"].apply(
            lambda x: max((pd.Timestamp(x) - today).days, 1) / 365
        )

        # compute IV
        ivs = []
        for _, row in df.iterrows():
            if row["mid"] <= 0:
                ivs.append(np.nan)
                continue
            iv = implied_vol(row["mid"], spot, row["strike"],
                             row["expiry_years"], 0.043, row["type"])
            ivs.append(iv)
        df["iv"] = ivs
        df = df.dropna(subset=["iv"])
        df = df[(df["iv"] > 0.01) & (df["iv"] < 3.0)]

        surface = iv_surface_chart(df, spot, symbol)
        smile = iv_smile_chart(df, spot, symbol)

        return jsonify({"surface": surface, "smile": smile, "spot": spot,
                        "n_contracts": len(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/chart/payoff")
def api_payoff_chart():
    """Option payoff diagram."""
    from qkit.pipeline import payoff_chart
    try:
        S = float(request.args.get("S", 100))
        K = float(request.args.get("K", 105))
        premium = float(request.args.get("premium", 3))
        opt_type = request.args.get("type", "call")
        position = request.args.get("position", "long")
        return jsonify(payoff_chart(S, K, premium, opt_type, position))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/chart/payoff_strategy", methods=["GET", "POST"])
def api_payoff_strategy():
    """Multi-leg strategy payoff diagram."""
    from qkit.pipeline import payoff_strategy_chart
    try:
        if request.method == "POST":
            data = request.get_json()
            S = float(data.get("S", 100))
            legs = data.get("legs", [])
        else:
            S = float(request.args.get("S", 100))
            legs = json.loads(request.args.get("legs", "[]"))
        if not legs:
            return jsonify({"error": "No legs provided"}), 400
        return jsonify(payoff_strategy_chart(S, legs))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/chart/pairs")
def api_pairs_chart():
    """Pairs analysis chart with optional walk-forward backtest."""
    from qkit.pipeline import pairs_chart
    from qkit.data import get_provider

    sym_a = request.args.get("a", "KO").upper()
    sym_b = request.args.get("b", "PEP").upper()
    period = request.args.get("period", "2y")
    do_backtest = request.args.get("backtest", "false").lower() == "true"

    try:
        provider = get_provider()
        hist_a = provider.get_history(sym_a, period=period)["close"]
        hist_b = provider.get_history(sym_b, period=period)["close"]
        idx = hist_a.index.intersection(hist_b.index)
        if len(idx) < 60:
            return jsonify({"error": f"Insufficient data: {len(idx)} points"}), 400
        pa, pb = hist_a.loc[idx], hist_b.loc[idx]

        bt_results = None
        if do_backtest:
            from qkit.backtest import walk_forward
            bt_results = walk_forward(pa, pb, sym_a, sym_b)

        result = pairs_chart(pa, pb, sym_a, sym_b, bt_results)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/pairs/screen")
def api_pairs_screen():
    """Screen sector pairs for cointegration."""
    from qkit.signals.pairs import screen_sector_pairs

    period = request.args.get("period", "2y")
    sig = float(request.args.get("sig", 0.05))

    try:
        results = screen_sector_pairs(period=period, significance=sig)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── API v1: Store (symbol data, watchlist, signals) ──────────────────────

def _get_store():
    from qkit.data.store import Store
    return Store()


@app.route("/api/v1/symbol/<symbol>/snapshot")
def api_symbol_snapshot(symbol):
    """Latest snapshot for a symbol. Fetches live if Store is empty."""
    from qkit.data.store import Snapshot

    try:
        with _get_store() as store:
            snap = store.get_latest_snapshot(symbol)
            if snap is None:
                # Live fetch from provider (moomoo → yfinance fallback)
                from qkit.data import get_provider
                provider = get_provider()
                if hasattr(provider, "get_snapshot"):
                    data = provider.get_snapshot(symbol)
                    store.upsert_snapshot(Snapshot(**data))
                    snap = store.get_latest_snapshot(symbol)
            if snap is None:
                return jsonify({"error": "No snapshot data", "symbol": symbol.upper()})
        return jsonify(snap)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/symbol/<symbol>/signals")
def api_symbol_signals(symbol):
    """Signals related to a symbol from Store."""
    try:
        with _get_store() as store:
            all_signals = store.get_signals(limit=100)
        # Filter: VRP (no symbol), or signals matching this symbol
        sym = symbol.upper()
        filtered = [
            s for s in all_signals
            if s.get("symbol") is None or sym in (s.get("symbol") or "").upper()
        ]
        return jsonify(filtered[:20])
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/watchlist")
def api_watchlist():
    """Get current watchlist."""
    try:
        with _get_store() as store:
            symbols = store.get_watchlist()
        return jsonify(symbols)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/watchlist/<symbol>", methods=["POST"])
def api_watchlist_toggle(symbol):
    """Toggle a symbol in/out of watchlist."""
    try:
        sym = symbol.upper()
        with _get_store() as store:
            current = store.get_watchlist()
            if sym in current:
                store.remove_watchlist(sym)
                return jsonify({"symbol": sym, "action": "removed"})
            else:
                store.add_watchlist(sym)
                return jsonify({"symbol": sym, "action": "added"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── API v1: Options Chain ─────────────────────────────────────────────

@app.route("/api/v1/options/<symbol>/chain")
def api_options_chain(symbol):
    """Full option chain with IV for a symbol."""
    from qkit.data import get_provider

    try:
        provider = get_provider()
        chain = provider.get_option_chain(symbol)
        spot = chain.spot_price

        import math

        def _clean(v):
            """Convert NaN/Inf to None for JSON safety."""
            if v is None:
                return None
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v

        contracts = []
        for q in chain.quotes:
            contracts.append({
                "strike": q.strike,
                "expiry": q.expiry,
                "type": q.option_type,
                "bid": _clean(q.bid),
                "ask": _clean(q.ask),
                "last": _clean(q.last),
                "volume": q.volume,
                "oi": q.open_interest,
                "iv": _clean(q.implied_vol),
            })

        # Compute put/call ratio by OI
        total_call_oi = sum(q.open_interest for q in chain.calls())
        total_put_oi = sum(q.open_interest for q in chain.puts())
        pcr = total_put_oi / max(total_call_oi, 1)

        # Find max OI strike
        max_oi_strike = 0
        max_oi = 0
        for q in chain.quotes:
            if q.open_interest > max_oi:
                max_oi = q.open_interest
                max_oi_strike = q.strike

        return jsonify({
            "symbol": symbol.upper(),
            "spot": spot,
            "expiries": chain.expiries(),
            "chain": contracts,
            "total_contracts": len(contracts),
            "put_call_ratio": round(pcr, 3),
            "max_oi_strike": max_oi_strike,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── API v1: Screener ─────────────────────────────────────────────────

@app.route("/api/v1/screener")
def api_screener():
    """Screen watchlist symbols by fundamental filters."""
    try:
        with _get_store() as store:
            snaps = store.get_all_latest_snapshots()

        # Apply filters
        pe_min = request.args.get("pe_min", type=float)
        pe_max = request.args.get("pe_max", type=float)
        pb_min = request.args.get("pb_min", type=float)
        pb_max = request.args.get("pb_max", type=float)
        div_min = request.args.get("div_min", type=float)
        mcap_min = request.args.get("mcap_min", type=float)

        filtered = []
        for s in snaps:
            if pe_min is not None and (s.get("pe_ratio") is None or s["pe_ratio"] < pe_min):
                continue
            if pe_max is not None and (s.get("pe_ratio") is None or s["pe_ratio"] > pe_max):
                continue
            if pb_min is not None and (s.get("pb_ratio") is None or s["pb_ratio"] < pb_min):
                continue
            if pb_max is not None and (s.get("pb_ratio") is None or s["pb_ratio"] > pb_max):
                continue
            if div_min is not None and (s.get("dividend_yield") is None or s["dividend_yield"] < div_min):
                continue
            if mcap_min is not None and (s.get("market_cap") is None or s["market_cap"] < mcap_min):
                continue
            filtered.append(s)

        return jsonify(filtered)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── API v1: Alerts ───────────────────────────────────────────────────

@app.route("/api/v1/alerts")
def api_alerts():
    """Get unacknowledged alerts."""
    try:
        with _get_store() as store:
            alerts = store.get_alerts(acknowledged=False, limit=20)
        return jsonify(alerts)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/alerts/<int:alert_id>/ack", methods=["POST"])
def api_alert_ack(alert_id):
    """Acknowledge (dismiss) an alert."""
    try:
        with _get_store() as store:
            store.acknowledge_alert(alert_id)
        return jsonify({"id": alert_id, "acknowledged": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def run(host="127.0.0.1", port=5000, debug=True):
    print(f"\n  qkit dashboard")
    print(f"  http://{host}:{port}\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run()
