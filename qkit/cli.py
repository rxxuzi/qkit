"""Command-line interface for qkit.

After ``pip install -e .``, the ``qkit`` command becomes available.
Run ``qkit -h`` for usage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────

@contextmanager
def _catch_warnings():
    """Capture warnings and print them as yellow lines before output."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for w in caught:
        msg = str(w.message)
        if "unclosed" in msg:
            continue
        print(f"  \033[93mWARN\033[0m  {msg}")


def _warn(text: str) -> None:
    print(f"  \033[93mWARN\033[0m  {text}")


def _err(text: str) -> None:
    print(f"  \033[91mERR\033[0m   {text}")


def _fetch_spot(symbol: str, fallback: float = 230.0) -> float:
    """Try to fetch the spot price; return *fallback* on failure."""
    try:
        from qkit.data import get_provider
        with _catch_warnings():
            provider = get_provider()
            price = provider.get_spot_price(symbol)
        _info(f"{symbol} spot: ${price:.2f}")
        return price
    except Exception:
        _warn(f"Data unavailable, using S=${fallback:.2f}")
        return fallback


def _header(text: str) -> None:
    print(f"\n  \033[96m{text}\033[0m")


def _info(text: str) -> None:
    print(f"  {text}")


def _kv(key: str, value, indent: int = 4) -> None:
    pad = " " * indent
    print(f"{pad}\033[90m{key:<20}\033[0m {value}")


# ── Commands ─────────────────────────────────────────────────────────────

def cmd_demo(args):
    """BSM pricing demo."""
    from qkit.pricing.bsm import BSM

    S = args.spot or 230.0
    K = args.strike or 235.0
    T_days = args.days or 30
    r = args.rate or 0.043
    sigma = args.vol or 0.22

    model = BSM(S=S, K=K, T=T_days / 365, r=r, sigma=sigma)
    cg = model.call_greeks()
    pg = model.put_greeks()

    _header(f"BSM Pricing  S=${S:.0f}  K=${K:.0f}  T={T_days}d  vol={sigma:.0%}  r={r:.1%}")
    _kv("Call", f"\033[92m${model.call_price():.4f}\033[0m")
    _kv("Put", f"\033[91m${model.put_price():.4f}\033[0m")
    print()
    _info("  Call Greeks")
    _kv("Delta", f"{cg.delta:+.6f}", 6)
    _kv("Gamma", f"{cg.gamma:+.6f}", 6)
    _kv("Vega", f"{cg.vega:+.6f}", 6)
    _kv("Theta", f"{cg.theta:+.6f}", 6)
    _kv("Rho", f"{cg.rho:+.6f}", 6)
    print()
    _info("  Put Greeks")
    _kv("Delta", f"{pg.delta:+.6f}", 6)
    _kv("Gamma", f"{pg.gamma:+.6f}", 6)
    _kv("Vega", f"{pg.vega:+.6f}", 6)
    _kv("Theta", f"{pg.theta:+.6f}", 6)
    _kv("Rho", f"{pg.rho:+.6f}", 6)

    par = model.verify_put_call_parity()
    status = "\033[92mPASS\033[0m" if par["passed"] else "\033[91mFAIL\033[0m"
    print()
    _kv("Put-Call Parity", f"{status}  diff={par['difference']:.2e}")
    print()


def cmd_greeks(args):
    """Print Greeks table for a range of spot prices."""
    from qkit.pricing.bsm import BSM

    symbol = args.symbol or "DEMO"
    S = args.spot or _fetch_spot(symbol)
    K = args.strike or round(S)
    T = (args.days or 30) / 365
    sigma = args.vol or 0.22
    r = args.rate or 0.043

    _header(f"Greeks Scan  {symbol}  K=${K:.0f}  T={int(T*365)}d  vol={sigma:.0%}")

    print()
    fmt = "  {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}"
    print(fmt.format("Spot", "Call", "Put", "Delta", "Gamma", "Vega", "Theta"))
    print(f"  {'-' * 62}")

    for d in range(-8, 9):
        s = S + d
        m = BSM(S=s, K=K, T=T, r=r, sigma=sigma)
        cg = m.call_greeks()
        atm = ">" if abs(d) == 0 else " "
        delta_color = "\033[92m" if cg.delta > 0.5 else "\033[91m" if cg.delta < 0.3 else "\033[96m"
        theta_color = "\033[91m" if cg.theta < 0 else "\033[92m"
        print(f"{atm}" + fmt.format(
            f"${s:.0f}",
            f"\033[92m${m.call_price():.2f}\033[0m",
            f"\033[91m${m.put_price():.2f}\033[0m",
            f"{delta_color}{cg.delta:.4f}\033[0m",
            f"{cg.gamma:.6f}",
            f"{cg.vega:.4f}",
            f"{theta_color}{cg.theta:.4f}\033[0m",
        ))
    print()


def cmd_chain(args):
    """Fetch and display option chain."""
    from qkit.data import get_provider

    symbol = args.symbol
    with _catch_warnings():
        provider = get_provider(args.provider)

    _header(f"Option Chain  {symbol}")

    spot = provider.get_spot_price(symbol)
    _kv("Spot", f"${spot:.2f}")

    chain = provider.get_option_chain(symbol)
    _kv("Expiries", len(chain.expiries()))
    _kv("Contracts", len(chain.quotes))

    df = chain.to_dataframe()
    otm = df[
        ((df["type"] == "call") & (df["strike"] >= spot))
        | ((df["type"] == "put") & (df["strike"] < spot))
    ]

    for exp in chain.expiries()[:3]:
        subset = otm[otm["expiry"] == exp]
        if subset.empty:
            continue
        print(f"\n  \033[93m{exp}\033[0m  ({len(subset)} OTM)")
        cols = [c for c in ("strike", "type", "bid", "ask", "mid", "volume") if c in subset.columns]
        print(subset[cols].head(10).to_string(index=False))
    print()


def cmd_report(args):
    """Generate HTML/Markdown report -> out/reports/."""
    from qkit.pricing.bsm import BSM
    from qkit.pricing.greeks import Greeks
    from qkit.pricing.greeks_mpl import GreeksMpl
    from qkit.reports import ReportGenerator

    symbol = args.symbol or "DEMO"
    S = args.spot or _fetch_spot(symbol)
    K = round(S)
    T = (args.days or 30) / 365
    r = args.rate or 0.043
    sigma = args.vol or 0.22

    _header(f"Report  {symbol}")
    _info("Generating...")

    model = BSM(S=S, K=K, T=T, r=r, sigma=sigma)
    g = Greeks(S=S, r=r, sigma=sigma)
    gm = GreeksMpl(S=S, r=r, sigma=sigma)
    gm.save_all(prefix=symbol)

    rpt = ReportGenerator(title=f"{symbol} Options Analysis")
    rpt.add_section("BSM Summary", model.summary())
    rpt.add_figure("Call Delta", g.heatmap("delta", "call"))
    rpt.add_figure("Call Gamma", g.heatmap("gamma", "call"))
    rpt.add_figure("Dashboard", g.dashboard("call"))
    rpt.add_figure("Straddle Payoff", g.payoff_diagram([
        {"type": "call", "strike": K, "premium": model.call_price(), "qty": 1},
        {"type": "put", "strike": K, "premium": model.put_price(), "qty": 1},
    ]))

    out = Path("out/reports")
    out.mkdir(parents=True, exist_ok=True)
    rpt.save_html(str(out / f"{symbol}_level1.html"))
    rpt.save_markdown(str(out / f"{symbol}_level1.md"))

    _kv("HTML", f"out/reports/{symbol}_level1.html")
    _kv("Markdown", f"out/reports/{symbol}_level1.md")
    _kv("Charts", f"out/charts/{symbol}_*")
    print()


def cmd_test(args):
    """Run the test suite."""
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])


def cmd_serve(args):
    """Start the web dashboard."""
    from qkit.web import run
    run(host=args.host, port=args.port)


def cmd_regime(args):
    """HMM regime detection."""
    from qkit.data import get_provider
    from qkit.signals.regime import fit_regime

    symbol = args.symbol
    with _catch_warnings():
        provider = get_provider()
        returns = provider.get_daily_returns(symbol, period=args.period)

    _header(f"Regime Detection  {symbol}  ({args.states} states)")
    _info("Fitting HMM...")
    result = fit_regime(returns, n_states=args.states, n_fits=5)

    current = result.current_state()
    durations = result.state_durations()
    labels = result.state_labels

    label_color = "\033[92m" if current["label"] == "CALM" else "\033[91m"
    print()
    _kv("Current", f"{label_color}{current['label']}\033[0m  ({current['probability']:.1%})")
    _kv("Volatility", f"{current['volatility'] * 100:.1f}% ann.")

    print()
    _info("  Transition Matrix")
    header = "        " + "  ".join(f"{l:>8}" for l in labels)
    print(f"    {header}")
    for i, label in enumerate(labels):
        row = f"    {label:<8}"
        for j in range(result.n_states):
            row += f"  {result.transition_matrix[i, j]:8.3f}"
        print(row)

    print()
    _info("  Expected Duration")
    for label, dur in durations.items():
        _kv(label, f"{dur:.0f} days", 6)
    print()


def cmd_svi(args):
    """Calibrate SVI to IV smile."""
    from qkit.data import get_provider
    from qkit.pricing.iv import implied_vol, filter_chain
    from qkit.volatility.svi import calibrate_svi
    import pandas as pd
    import numpy as np

    symbol = args.symbol
    with _catch_warnings():
        provider = get_provider()
        spot = provider.get_spot_price(symbol)
        chain = provider.get_option_chain(symbol)

    _header(f"SVI Calibration  {symbol}")
    df = chain.to_dataframe()
    df = filter_chain(df, spot, min_volume=1, min_oi=1)

    if df.empty:
        _info("\033[91mNo options data after filtering\033[0m")
        return

    today = pd.Timestamp.now()
    df["expiry_years"] = df["expiry"].apply(
        lambda x: max((pd.Timestamp(x) - today).days, 1) / 365
    )

    expiry = df["expiry_years"].min()
    calls = df[(df["type"] == "call") & (df["expiry_years"] == expiry)].copy()

    ivs = []
    for _, row in calls.iterrows():
        iv = implied_vol(row["mid"], spot, row["strike"], expiry, 0.043, "call")
        ivs.append(iv)
    calls["iv"] = ivs
    calls = calls.dropna(subset=["iv"])
    calls = calls[(calls["iv"] > 0.01) & (calls["iv"] < 3.0)]

    if len(calls) < 5:
        _info("\033[91mInsufficient data for SVI fit\033[0m")
        return

    k = np.log(calls["strike"].values / spot)
    result = calibrate_svi(k, calls["iv"].values, T=expiry)

    _kv("Spot", f"${spot:.2f}")
    _kv("Expiry", f"{int(expiry * 365)}d ({len(calls)} contracts)")
    print()
    _info("  Parameters")
    _kv("a", f"{result.a:.6f}", 6)
    _kv("b", f"{result.b:.6f}", 6)
    _kv("rho", f"{result.rho:.4f}", 6)
    _kv("m", f"{result.m:.6f}", 6)
    _kv("sigma", f"{result.sigma:.6f}", 6)

    arb = result.is_arbitrage_free(k)
    status = "\033[92mPASS\033[0m" if arb["arbitrage_free"] else "\033[91mFAIL\033[0m"
    print()
    _kv("Arbitrage-free", status)
    print()


def cmd_jobs(args):
    """Run daily data batch jobs."""
    from qkit.data.jobs import main as jobs_main

    with _catch_warnings():
        # Pre-trigger provider to capture warnings
        from qkit.data import get_provider
        get_provider()

    _header("Daily Jobs")
    # Suppress duplicate warnings inside jobs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.argv = ["qkit-jobs"]
        if args.signals:
            sys.argv.append("--signals")
        jobs_main()


def cmd_risk(args):
    """VaR / CVaR risk analysis."""
    from qkit.data import get_provider
    from qkit.portfolio.risk import compute_all
    import numpy as np

    symbol = args.symbol.upper()
    conf = args.confidence / 100

    with _catch_warnings():
        provider = get_provider()
        returns = provider.get_daily_returns(symbol, period=args.period)

    _header(f"Risk  {symbol}  ({args.period}, {args.confidence}% conf)")

    report = compute_all(returns, confidence=conf)
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    ann_vol = sigma * np.sqrt(252) * 100

    _kv("Observations", len(returns))
    _kv("Ann. Volatility", f"{ann_vol:.1f}%")
    _kv("Daily Mean", f"{mu*100:.4f}%")
    _kv("Daily Std", f"{sigma*100:.4f}%")

    print()
    fmt = "  {:>16}  {:>12}  {:>12}"
    print(fmt.format("Method", "VaR", "CVaR"))
    print(f"  {'-' * 44}")

    def _row(name, var_val, cvar_val):
        var_str = f"\033[91m{var_val*100:.4f}%\033[0m"
        cvar_str = f"\033[91m{cvar_val*100:.4f}%\033[0m" if not np.isnan(cvar_val) else "-"
        print(fmt.format(name, var_str, cvar_str))

    _row("Parametric", report.var_parametric, report.cvar_parametric)
    _row("Historical", report.var_historical, report.cvar_historical)
    _row("Monte Carlo", report.var_montecarlo, report.cvar_montecarlo)

    # Cornish-Fisher has no CVaR
    var_cf = f"\033[91m{report.var_cornish_fisher*100:.4f}%\033[0m"
    print(fmt.format("Cornish-Fisher", var_cf, "-"))

    # Interpretation
    print()
    worst = max(report.var_parametric, report.var_historical,
                report.var_montecarlo, report.var_cornish_fisher)
    _kv("Worst-case VaR", f"\033[91m{worst*100:.4f}%\033[0m  (1-day, {args.confidence}%)")
    if args.notional:
        loss = worst * args.notional
        _kv("Dollar Loss", f"\033[91m${loss:,.0f}\033[0m  on ${args.notional:,.0f}")

    print()


def cmd_pair(args):
    """Pairs trading analysis between two symbols."""
    from qkit.data import get_provider
    from qkit.signals.pairs import analyze_pair, spread_zscore

    sym_a = args.a.upper()
    sym_b = args.b.upper()

    with _catch_warnings():
        provider = get_provider()
        hist_a = provider.get_history(sym_a, period=args.period)["close"]
        hist_b = provider.get_history(sym_b, period=args.period)["close"]

    idx = hist_a.index.intersection(hist_b.index)
    if len(idx) < 60:
        _err(f"Insufficient data: {len(idx)} overlapping points")
        return

    pa, pb = hist_a.loc[idx], hist_b.loc[idx]
    stats = analyze_pair(pa, pb, sym_a, sym_b)

    _header(f"Pairs  {sym_a} / {sym_b}")

    # Cointegration
    coint_color = "\033[92m" if stats.is_cointegrated else "\033[91m"
    _kv("Cointegrated", f"{coint_color}{'YES' if stats.is_cointegrated else 'NO'}\033[0m  (p={stats.coint_pvalue:.4f})")
    _kv("Hedge Ratio", f"{stats.beta:.4f}")
    _kv("Half-life", f"{stats.half_life:.1f} days")

    # OU parameters
    print()
    _info("  Ornstein-Uhlenbeck")
    _kv("theta", f"{stats.theta:.4f}", 6)
    _kv("mu", f"{stats.mu:.4f}", 6)
    _kv("sigma", f"{stats.sigma_ou:.4f}", 6)

    # Current signal
    z = stats.current_z
    sig = stats.signal()
    if sig == "SHORT_SPREAD":
        sig_color = "\033[91m"
    elif sig == "LONG_SPREAD":
        sig_color = "\033[92m"
    elif sig == "EXIT":
        sig_color = "\033[93m"
    else:
        sig_color = "\033[90m"

    print()
    _info("  Signal")
    _kv("z-score", f"{z:+.2f}")
    _kv("Signal", f"{sig_color}{sig}\033[0m")

    # Thresholds
    print()
    _info("  Thresholds")
    _kv("Entry", "|z| > 2.0", 6)
    _kv("Exit", "|z| < 0.5", 6)
    _kv("Stop", "|z| > 4.0", 6)

    # Kalman beta (if verbose)
    if args.verbose:
        try:
            from qkit.signals.pairs import analyze_pair_kalman
            kf = analyze_pair_kalman(pa, pb, sym_a, sym_b)

            print()
            _info("  Kalman Filter")
            _kv("Dynamic Beta", f"{kf['kalman_beta_current']:.4f}", 6)
            _kv("Beta Std", f"{kf['kalman_beta_std']:.4f}", 6)
            _kv("Kalman z", f"{kf['kalman_z']:+.2f}", 6)
            _kv("Kalman HL", f"{kf['kalman_half_life']:.1f} days", 6)
        except Exception:
            pass

        # Johansen
        try:
            import pandas as pd
            from qkit.signals.pairs import johansen_test
            prices = pd.DataFrame({sym_a: pa, sym_b: pb})
            joh = johansen_test(prices)

            print()
            _info("  Johansen Test")
            _kv("Coint Relations", f"{joh['n_coint']} (at 95%)", 6)
            for i, (ts, cv) in enumerate(zip(joh["trace_stat"], joh["crit_95"])):
                passed = "\033[92mPASS\033[0m" if ts > cv else "\033[91mFAIL\033[0m"
                _kv(f"r <= {i}", f"trace={ts:.2f}  crit={cv:.2f}  {passed}", 6)
        except Exception:
            pass

        # Regime-adjusted thresholds
        try:
            from qkit.signals.regime import fit_regime, regime_adjusted_thresholds
            with _catch_warnings():
                returns = provider.get_daily_returns(sym_a, period=args.period)
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                regime = fit_regime(returns, n_states=2, n_fits=3)
            thresholds = regime_adjusted_thresholds(regime)

            print()
            _info("  Regime-Adjusted Thresholds")
            label = thresholds["regime"]
            rc = "\033[92m" if label == "CALM" else "\033[91m"
            _kv("Regime", f"{rc}{label}\033[0m  ({thresholds['regime_prob']:.0%})", 6)
            _kv("Entry", f"|z| > {thresholds['entry']:.1f}", 6)
            _kv("Stop", f"|z| > {thresholds['stop']:.1f}", 6)
        except Exception:
            pass

    print()


def cmd_opt(args):
    """Options contract analysis."""
    from qkit.data import get_provider
    from qkit.data.contract import parse_contract, resolve_strike, find_expiry
    from qkit.pricing.bsm import BSM
    from qkit.pricing.iv import implied_vol
    from qkit.pricing import analysis
    import datetime
    import numpy as np

    # Parse contract spec
    spec = f"{args.symbol} {args.code}"
    try:
        contract = parse_contract(spec)
    except ValueError as exc:
        _err(str(exc))
        return

    symbol = contract.symbol
    r = args.rate or 0.043

    with _catch_warnings():
        provider = get_provider()
        spot = provider.get_spot_price(symbol)
        chain = provider.get_option_chain(symbol)

    # Resolve expiry in chain
    try:
        chain_expiry = find_expiry(contract, chain)
    except ValueError as exc:
        _err(str(exc))
        _info(f"  Available: {', '.join(chain.expiries()[:8])}")
        return

    # Time to expiry
    exp_date = datetime.date.fromisoformat(chain_expiry)
    today = datetime.date.today()
    days = (exp_date - today).days
    if days <= 0:
        _err(f"Expiry {chain_expiry} is in the past")
        return
    T = days / 365.0

    # Resolve strike
    if args.strike:
        contract.strike = args.strike
    strike = resolve_strike(contract, chain, spot)
    opt_type = contract.option_type

    _header(f"Options  {symbol}  {chain_expiry}  {opt_type.upper()}  K=${strike:.1f}")
    _kv("Spot", f"${spot:.2f}")
    _kv("Days", f"{days}")
    _kv("T", f"{T:.4f} yr")

    # ── Target quote ──────────────────────────────────────────
    exp_quotes = chain.by_expiry(chain_expiry)
    target = None
    for q in exp_quotes:
        if q.option_type == opt_type and abs(q.strike - strike) < 0.01:
            target = q
            break

    if target is None:
        # Find nearest
        same_type = [q for q in exp_quotes if q.option_type == opt_type]
        if same_type:
            same_type.sort(key=lambda q: abs(q.strike - strike))
            target = same_type[0]
            strike = target.strike
            _warn(f"Exact strike not found, using K=${strike:.1f}")

    if target is not None and target.mid > 0:
        # Use provider IV when available (skip expensive Brentq)
        iv_val = (target.implied_vol
                  if target.implied_vol and target.implied_vol > 0.001
                  else implied_vol(target.mid, spot, strike, T, r, opt_type,
                                   method="newton"))
        print()
        _info("  Quote")
        _kv("Bid", f"${target.bid:.2f}", 6)
        _kv("Ask", f"${target.ask:.2f}", 6)
        _kv("Mid", f"${target.mid:.2f}", 6)
        _kv("Volume", f"{target.volume:,}", 6)
        _kv("Open Int", f"{target.open_interest:,}", 6)
        if not np.isnan(iv_val):
            _kv("IV", f"{iv_val*100:.1f}%", 6)

        # Greeks
        if not np.isnan(iv_val) and iv_val > 0:
            m = BSM(S=spot, K=strike, T=T, r=r, sigma=iv_val)
            greeks = m.call_greeks() if opt_type == "call" else m.put_greeks()
            price = m.call_price() if opt_type == "call" else m.put_price()

            print()
            _info("  Greeks")
            _kv("BSM Price", f"${price:.4f}", 6)
            _kv("Delta", f"{greeks.delta:+.4f}", 6)
            _kv("Gamma", f"{greeks.gamma:+.6f}", 6)
            _kv("Vega", f"{greeks.vega:+.4f}", 6)
            _kv("Theta", f"{greeks.theta:+.4f}", 6)
            _kv("Rho", f"{greeks.rho:+.4f}", 6)

            # Probabilities
            p_itm = analysis.prob_itm(spot, strike, T, r, iv_val, opt_type)
            p_profit = analysis.prob_profit(spot, strike, T, r, iv_val,
                                            opt_type, target.mid)
            exp_ret = analysis.expected_return(spot, strike, T, r, iv_val,
                                              opt_type, target.mid)
            be = analysis.breakeven(strike, target.mid, opt_type)

            print()
            _info("  Probability")
            _kv("ITM", f"{p_itm*100:.1f}%", 6)
            _kv("Profit", f"{p_profit*100:.1f}%", 6)
            ret_color = "\033[92m" if exp_ret >= 0 else "\033[91m"
            _kv("Exp Return", f"{ret_color}{exp_ret*100:+.1f}%\033[0m", 6)
            _kv("Breakeven", f"${be:.2f}", 6)

    # ── GARCH vol ──────────────────────────────────────────────
    garch_vol = None
    try:
        with _catch_warnings():
            returns = provider.get_daily_returns(symbol, period="2y")
        from qkit.volatility.garch import fit as garch_fit
        pct_returns = returns * 100
        garch_result = garch_fit(pct_returns, model_type="garch")
        garch_vol = garch_result.forecast_vol(days) / 100
        print()
        _info("  GARCH Forecast")
        _kv(f"{days}d Vol", f"{garch_vol*100:.1f}%", 6)
        if target is not None and not np.isnan(iv_val) and iv_val > 0:
            diff = (iv_val - garch_vol) * 100
            label = "IV rich" if diff > 0 else "IV cheap"
            color = "\033[91m" if diff > 0 else "\033[92m"
            _kv("IV vs GARCH", f"{color}{diff:+.1f}pp  ({label})\033[0m", 6)
    except Exception:
        pass

    # ── Mispricing table ──────────────────────────────────────
    verbose = args.verbose
    moneyness = (0.0, 99.0) if verbose else (0.80, 1.20)
    type_quotes = [q for q in exp_quotes if q.option_type == opt_type]
    if type_quotes:
        mdf = analysis.mispricing_table(type_quotes, spot, r, T,
                                         garch_vol=garch_vol,
                                         moneyness=moneyness)
        if not mdf.empty and "mispricing_pct" in mdf.columns:
            # Sort by absolute mispricing
            mdf["abs_mis"] = mdf["mispricing_pct"].abs()
            mdf = mdf.sort_values("abs_mis", ascending=False)

            top_n = args.top or 5
            top = mdf.head(top_n)

            print()
            _info(f"  Mispricing Top {min(top_n, len(top))}  ({opt_type}s)")

            for _, row in top.iterrows():
                mis = row["mispricing_pct"]
                if mis > 2:
                    tag = "\033[91mRICH\033[0m"
                elif mis < -2:
                    tag = "\033[92mCHEAP\033[0m"
                else:
                    tag = "\033[90mFAIR\033[0m"

                iv_str = f"IV={row['iv']*100:.0f}%" if "iv" in row and not np.isnan(row.get("iv", np.nan)) else ""
                _kv(
                    f"K=${row['strike']:.0f}",
                    f"mid=${row['mid']:.2f}  {mis:+.1f}%  {tag}  {iv_str}",
                    6,
                )

    # ── ATM Term Structure ────────────────────────────────────
    max_exp = 0 if verbose else 10
    ts = analysis.atm_term_structure(chain, spot, r, max_expiries=max_exp)
    if not ts.empty and len(ts) >= 2:
        print()
        _info("  ATM Term Structure")
        for _, row in ts.iterrows():
            _kv(f"{row['days']:>3d}d", f"{row['atm_iv']*100:.1f}%", 6)

    # ── What-if P&L (-w) ─────────────────────────────────────
    if args.whatif and target is not None and not np.isnan(iv_val) and iv_val > 0:
        wif = analysis.whatif_table(spot, strike, T, r, iv_val,
                                    opt_type, target.mid)
        print()
        _info("  What-If P&L  (spot chg x vol chg)")
        # Header
        cols = wif.columns.tolist()
        hdr = "          " + "  ".join(f"{c:>8}" for c in cols)
        print(f"    {hdr}")
        print(f"    {'─' * (10 + 10 * len(cols))}")
        for idx, row_data in wif.iterrows():
            vals = []
            for v in row_data:
                if v >= 0:
                    vals.append(f"\033[92m{v:+8.2f}\033[0m")
                else:
                    vals.append(f"\033[91m{v:+8.2f}\033[0m")
            print(f"    {idx:>8}  " + "  ".join(vals))

    print()


def cmd_market(args):
    """Market overview for a symbol."""
    from qkit.data import get_provider
    from qkit.data.store import Store, Snapshot
    import math

    symbol = args.symbol.upper()
    verbose = args.verbose

    with _catch_warnings():
        provider = get_provider()

    # Always fetch live, persist to Store
    snap = None
    if hasattr(provider, "get_snapshot"):
        try:
            snap = provider.get_snapshot(symbol)
            try:
                with Store() as store:
                    store.upsert_snapshot(Snapshot(**snap))
            except Exception:
                pass
        except Exception:
            pass

    # Fallback to Store if live failed
    if snap is None:
        try:
            with Store() as store:
                snap = store.get_latest_snapshot(symbol)
        except Exception:
            pass

    # Last resort: just spot price
    if snap is None:
        try:
            spot = provider.get_spot_price(symbol)
            snap = {"symbol": symbol, "last_price": spot}
        except Exception:
            _err(f"No data available for {symbol}")
            return

    _header(symbol)

    # Price
    price = snap.get("last_price")
    prev = snap.get("prev_close")
    if price is not None:
        chg_str = ""
        if prev and prev > 0:
            chg = price - prev
            pct = chg / prev * 100
            sign = "+" if chg >= 0 else ""
            color = "\033[92m" if chg >= 0 else "\033[91m"
            chg_str = f"  {color}{sign}{chg:.2f} ({sign}{pct:.2f}%)\033[0m"
        _kv("Last", f"${price:.2f}{chg_str}")

    _kv("Volume", f"{snap.get('volume', 0):,.0f}" if snap.get("volume") else "-")

    mc = snap.get("market_cap")
    if mc:
        if mc >= 1e12:
            _kv("Market Cap", f"${mc/1e12:.2f}T")
        elif mc >= 1e9:
            _kv("Market Cap", f"${mc/1e9:.1f}B")
        else:
            _kv("Market Cap", f"${mc/1e6:.0f}M")

    # Fundamentals
    print()
    _info("  Fundamentals")
    _f = lambda v, d=2: f"{v:.{d}f}" if v is not None else "-"
    _kv("P/E", _f(snap.get("pe_ratio"), 1), 6)
    _kv("P/B", _f(snap.get("pb_ratio"), 2), 6)
    _kv("EPS", f"${_f(snap.get('eps'), 2)}" if snap.get("eps") is not None else "-", 6)
    dy = snap.get("dividend_yield")
    _kv("Div Yield", f"{dy*100:.2f}%" if dy else "-", 6)
    h52 = snap.get("high_52w")
    l52 = snap.get("low_52w")
    if h52 and l52:
        _kv("52W Range", f"${l52:.2f} - ${h52:.2f}", 6)

    if not verbose:
        print()
        return

    # ── Verbose: Resolve vol (IV → GARCH → default) ─────────
    import numpy as np
    spot = price or 230
    from qkit.pricing.bsm import BSM

    vol = 0.22  # default fallback
    vol_source = "default"
    atm_iv = None
    garch_vol = None

    # Try ATM IV from chain
    try:
        with _catch_warnings():
            chain = provider.get_option_chain(symbol)
        atm_calls = [q for q in chain.calls() if q.implied_vol and q.implied_vol > 0.01]
        if atm_calls:
            atm_calls.sort(key=lambda q: abs(q.strike - spot))
            atm_iv = atm_calls[0].implied_vol
            vol = atm_iv
            vol_source = "iv"
    except Exception:
        chain = None

    # GARCH forecast
    try:
        with _catch_warnings():
            returns = provider.get_daily_returns(symbol, period="5y")
        from qkit.volatility.garch import fit as garch_fit
        pct_returns = returns * 100
        garch_result = garch_fit(pct_returns, model_type="garch")
        garch_vol = garch_result.forecast_vol(30) / 100  # decimal
        if vol_source == "default":
            vol = garch_vol
            vol_source = "garch"
    except Exception:
        garch_result = None

    # ── Verbose: Greeks ───────────────────────────────────────
    vol_label = f"{vol_source}={vol*100:.1f}%"
    print()
    _info(f"  Greeks (ATM, 30d, {vol_label})")
    K = round(spot)
    m = BSM(S=spot, K=K, T=30/365, r=0.043, sigma=vol)
    cg = m.call_greeks()
    pg = m.put_greeks()
    fmt = "      {:>6}  {:>10}  {:>10}"
    print(fmt.format("", "\033[92mCall\033[0m", "\033[91mPut\033[0m"))
    print(fmt.format("Price", f"${m.call_price():.2f}", f"${m.put_price():.2f}"))
    print(fmt.format("Delta", f"{cg.delta:.4f}", f"{pg.delta:.4f}"))
    print(fmt.format("Gamma", f"{cg.gamma:.6f}", f"{pg.gamma:.6f}"))
    print(fmt.format("Vega", f"{cg.vega:.4f}", f"{pg.vega:.4f}"))
    print(fmt.format("Theta", f"{cg.theta:.4f}", f"{pg.theta:.4f}"))

    # ── Verbose: Chain summary ────────────────────────────────
    if chain is not None:
        n_calls = len(chain.calls())
        n_puts = len(chain.puts())
        n_exp = len(chain.expiries())
        total_call_oi = sum(q.open_interest for q in chain.calls())
        total_put_oi = sum(q.open_interest for q in chain.puts())
        pcr = total_put_oi / max(total_call_oi, 1)

        print()
        _info("  Option Chain")
        _kv("Expiries", n_exp, 6)
        _kv("Calls", f"{n_calls:,}", 6)
        _kv("Puts", f"{n_puts:,}", 6)
        _kv("P/C Ratio (OI)", f"{pcr:.2f}", 6)
        if atm_iv is not None:
            _kv("ATM IV", f"{atm_iv*100:.1f}%", 6)

    # ── Verbose: GARCH ────────────────────────────────────────
    if garch_result is not None:
        print()
        _info("  GARCH(1,1)")
        _kv("Persistence", f"{garch_result.persistence:.4f}", 6)
        _kv("Half-life", f"{garch_result.half_life:.0f} days", 6)
        _kv("30d Forecast", f"{garch_vol*100:.2f}%", 6)

        # IV vs GARCH divergence
        if atm_iv is not None and garch_vol is not None:
            diff_pct = (atm_iv - garch_vol) * 100
            sign = "+" if diff_pct >= 0 else ""
            label = "IV rich" if diff_pct > 0 else "IV cheap"
            color = "\033[91m" if diff_pct > 0 else "\033[92m"
            _kv("IV vs GARCH", f"{color}{sign}{diff_pct:.1f}%  ({label})\033[0m", 6)

    print()


# ── Parser ───────────────────────────────────────────────────────────────

EPILOG = """\
examples:
  qkit market AAPL
  qkit market AAPL -v
  qkit opt QQQ 260330P
  qkit opt SPY 260430C --strike 580 --top 10
  qkit opt QQQ 260404P --whatif
  qkit demo --spot 150 --strike 155 --days 45
  qkit greeks AAPL
  qkit greeks AAPL --strike 180 --days 14
  qkit chain SPY --provider yfinance
  qkit report AAPL
  qkit risk AAPL --notional 100000
  qkit pair AMD NVDA
  qkit pair KO PEP -v
  qkit regime SPY --period 5y
  qkit svi SPY
  qkit serve --port 8080
  qkit jobs --signals
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qkit",
        description="qkit - quantitative finance toolkit",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", title="commands")

    _f = argparse.RawDescriptionHelpFormatter

    # demo
    demo = sub.add_parser("demo", help="BSM pricing demo",
                          description="Black-Scholes-Merton pricing with Greeks and put-call parity check.",
                          formatter_class=_f)
    demo.add_argument("-s", "--spot", type=float, help="spot price (default: 230)")
    demo.add_argument("-k", "--strike", type=float, help="strike price (default: 235)")
    demo.add_argument("-d", "--days", type=int, help="days to expiry (default: 30)")
    demo.add_argument("-r", "--rate", type=float, help="risk-free rate (default: 0.043)")
    demo.add_argument("--vol", type=float, help="volatility (default: 0.22)")

    # greeks
    greeks = sub.add_parser("greeks", help="Greeks table for spot range",
                            description="Print a Greeks scan table across spot prices.\nFetches live spot if symbol is given.",
                            formatter_class=_f)
    greeks.add_argument("symbol", nargs="?", default="DEMO", help="ticker symbol (default: DEMO)")
    greeks.add_argument("-s", "--spot", type=float, help="override spot price")
    greeks.add_argument("-k", "--strike", type=float, help="strike price (default: ATM)")
    greeks.add_argument("-d", "--days", type=int, default=30, help="days to expiry (default: 30)")
    greeks.add_argument("--vol", type=float, default=0.22, help="volatility (default: 0.22)")
    greeks.add_argument("-r", "--rate", type=float, default=0.043, help="risk-free rate (default: 0.043)")

    # chain
    chain = sub.add_parser("chain", help="Fetch option chain",
                           description="Fetch and display OTM option chain for a symbol.\nShows first 3 expiries with bid/ask/volume.",
                           formatter_class=_f)
    chain.add_argument("symbol", help="ticker symbol")
    chain.add_argument("-p", "--provider", default=None, help="data provider (moomoo/yfinance)")

    # report
    report = sub.add_parser("report", help="Generate report -> out/reports/",
                            description="Generate a full HTML + Markdown analysis report.\nOutput saved to out/reports/ and out/charts/.",
                            formatter_class=_f)
    report.add_argument("symbol", nargs="?", default="DEMO", help="ticker symbol (default: DEMO)")
    report.add_argument("--spot", type=float, help="override spot price")
    report.add_argument("--days", type=int, default=30, help="days to expiry (default: 30)")
    report.add_argument("--vol", type=float, default=0.22, help="volatility (default: 0.22)")
    report.add_argument("--rate", type=float, default=0.043, help="risk-free rate (default: 0.043)")

    # test
    sub.add_parser("test", help="Run test suite",
                   description="Run the full pytest test suite.")

    # serve
    serve = sub.add_parser("serve", help="Start web dashboard",
                           description="Start the Flask web dashboard.\nOpen http://localhost:5000 in your browser.",
                           formatter_class=_f)
    serve.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    serve.add_argument("-p", "--port", type=int, default=5000, help="bind port (default: 5000)")

    # regime
    regime = sub.add_parser("regime", help="HMM regime detection",
                            description="Fit a Gaussian HMM to detect market regimes (CALM/STRESS).\nShows current state, transition matrix, and expected durations.",
                            formatter_class=_f)
    regime.add_argument("symbol", nargs="?", default="SPY", help="ticker symbol (default: SPY)")
    regime.add_argument("-p", "--period", default="3y", help="history period (default: 3y)")
    regime.add_argument("--states", type=int, default=2, help="number of hidden states (default: 2)")

    # opt
    opt = sub.add_parser("opt", help="Options contract analysis",
                         description="Analyse an option contract: Greeks, mispricing, IV, probabilities.\n"
                                     "Contract format: SYMBOL YYMMDD[C|P] (e.g. QQQ 260330P).\n"
                                     "Strike auto-resolves to ATM unless --strike is given.",
                         formatter_class=_f)
    opt.add_argument("symbol", help="ticker symbol (e.g. QQQ)")
    opt.add_argument("code", help="YYMMDD[C|P] contract code (e.g. 260330P)")
    opt.add_argument("-k", "--strike", type=float, default=None,
                     help="explicit strike price (default: ATM)")
    opt.add_argument("-r", "--rate", type=float, default=None,
                     help="risk-free rate (default: 0.043)")
    opt.add_argument("--top", type=int, default=5,
                     help="number of mispriced contracts to show (default: 5)")
    opt.add_argument("-v", "--verbose", action="store_true",
                     help="include deep OTM mispricing and all term structure expiries")
    opt.add_argument("-w", "--whatif", action="store_true",
                     help="show P&L what-if matrix (spot x vol changes)")

    # market
    market = sub.add_parser("market", help="Market overview for a symbol",
                            description="Show price, fundamentals, and optionally Greeks/chain/GARCH.\nUse -v for the full picture.",
                            formatter_class=_f)
    market.add_argument("symbol", help="ticker symbol")
    market.add_argument("-v", "--verbose", action="store_true",
                        help="include Greeks, chain summary, and GARCH")

    # risk
    risk = sub.add_parser("risk", help="VaR / CVaR risk analysis",
                          description="Compute VaR and CVaR using 4 methods on historical returns.\nParametric, Historical, Monte Carlo, and Cornish-Fisher.",
                          formatter_class=_f)
    risk.add_argument("symbol", help="ticker symbol")
    risk.add_argument("-p", "--period", default="1y", help="history period (default: 1y)")
    risk.add_argument("-c", "--confidence", type=float, default=95, help="confidence level %% (default: 95)")
    risk.add_argument("-n", "--notional", type=float, default=None, help="portfolio notional for dollar loss")

    # pair
    pair = sub.add_parser("pair", help="Pairs trading analysis",
                          description="Cointegration test, OU parameters, and z-score signal for a pair.\nUse -v for Kalman filter, Johansen test, and regime-adjusted thresholds.",
                          formatter_class=_f)
    pair.add_argument("a", help="first symbol")
    pair.add_argument("b", help="second symbol")
    pair.add_argument("-p", "--period", default="2y", help="history period (default: 2y)")
    pair.add_argument("-v", "--verbose", action="store_true",
                      help="include Kalman, Johansen, regime thresholds")

    # svi
    svi = sub.add_parser("svi", help="Calibrate SVI to IV smile",
                         description="Calibrate raw SVI parameters to the nearest-expiry IV smile.\nShows fitted params and arbitrage-free check.",
                         formatter_class=_f)
    svi.add_argument("symbol", nargs="?", default="SPY", help="ticker symbol (default: SPY)")

    # jobs
    jobs = sub.add_parser("jobs", help="Daily data batch jobs",
                          description="Fetch snapshots for all watchlist symbols.\nWith --signals, also compute VRP/pairs signals and generate alerts.",
                          formatter_class=_f)
    jobs.add_argument("-s", "--signals", action="store_true",
                      help="also update VRP/pairs signals + alerts")

    return p


BRIEF = """\
qkit - quantitative finance toolkit

commands:
  market     Market overview (price, fundamentals, -v for full)
  opt        Options contract analysis (Greeks, mispricing, IV, P&L)
  risk       VaR / CVaR (4 methods)
  pair       Pairs trading analysis (-v for Kalman/Johansen/regime)
  demo       BSM pricing demo
  greeks     Greeks table for spot range
  chain      Fetch option chain
  report     Generate report -> out/reports/
  serve      Start web dashboard
  regime     HMM regime detection
  svi        Calibrate SVI to IV smile
  jobs       Daily data batch jobs
  test       Run test suite

Run 'qkit -h' for detailed help and examples.
Run 'qkit <command> -h' for command-specific options.
"""


def main():
    """Entry point registered as ``qkit`` console script."""
    # No args or "help" -> show brief usage
    if len(sys.argv) == 1:
        print(BRIEF)
        return
    if sys.argv[1] == "help":
        sys.argv[1] = "-h"

    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "demo": cmd_demo,
        "greeks": cmd_greeks,
        "chain": cmd_chain,
        "report": cmd_report,
        "test": cmd_test,
        "serve": cmd_serve,
        "regime": cmd_regime,
        "opt": cmd_opt,
        "market": cmd_market,
        "risk": cmd_risk,
        "pair": cmd_pair,
        "svi": cmd_svi,
        "jobs": cmd_jobs,
    }

    func = dispatch.get(args.command)
    if func:
        func(args)
    else:
        parser.print_help()
