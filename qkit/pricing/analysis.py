"""Options analysis: mispricing, IV analytics, P&L simulation, probabilities.

This module provides the analytical functions behind ``qkit opt``.
All functions are pure (no I/O) and operate on data already fetched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from qkit.pricing.bsm import BSM, bsm_call_vec, bsm_put_vec
from qkit.pricing.iv import implied_vol
from qkit.data.provider import OptionChain, OptionQuote


# ── Mispricing ──────────────────────────────────────────────────────────


def mispricing_table(
    quotes: list[OptionQuote],
    spot: float,
    r: float,
    T: float,
    garch_vol: float | None = None,
    svi_params=None,
    moneyness: tuple[float, float] = (0.80, 1.20),
) -> pd.DataFrame:
    """Compute mispricing for a list of quotes at a single expiry.

    Parameters
    ----------
    quotes : list[OptionQuote]
        Quotes for a single expiry slice.
    spot : float
        Current underlying price.
    r : float
        Risk-free rate.
    T : float
        Time to expiry in years.
    garch_vol : float, optional
        GARCH forecast volatility (annualised, decimal).
    svi_params : SVIParams, optional
        Fitted SVI parameters for model IV comparison.
    moneyness : tuple
        (low, high) moneyness filter.  Only strikes within
        ``spot * low .. spot * high`` are analysed.

    Returns
    -------
    pd.DataFrame
        Columns: strike, type, bid, ask, mid, iv, bsm_price,
        mispricing_pct, [svi_iv, svi_dev], [garch_vol, hv_iv_ratio].
    """
    if not quotes or T <= 0:
        return pd.DataFrame()

    lo_k = spot * moneyness[0]
    hi_k = spot * moneyness[1]

    rows = []
    for q in quotes:
        if q.mid <= 0 or q.bid <= 0:
            continue
        # Moneyness filter — skip deep OTM/ITM to avoid noise and speed up
        if q.strike < lo_k or q.strike > hi_k:
            continue

        # Use provider IV when available (skip expensive Brentq)
        iv = q.implied_vol if q.implied_vol and q.implied_vol > 0.001 else None
        if iv is None:
            iv = implied_vol(q.mid, spot, q.strike, T, r, q.option_type,
                             method="newton")
        if np.isnan(iv) or iv <= 0:
            continue

        # BSM theoretical price using market IV
        m = BSM(S=spot, K=q.strike, T=T, r=r, sigma=iv)
        bsm_price = m.call_price() if q.option_type == "call" else m.put_price()

        row = {
            "strike": q.strike,
            "type": q.option_type,
            "bid": q.bid,
            "ask": q.ask,
            "mid": q.mid,
            "volume": q.volume,
            "open_interest": q.open_interest,
            "iv": iv,
            "bsm_price": bsm_price,
        }

        # GARCH-based mispricing: price with GARCH vol vs market mid
        if garch_vol is not None and garch_vol > 0:
            try:
                mg = BSM(S=spot, K=q.strike, T=T, r=r, sigma=garch_vol)
                garch_price = (mg.call_price() if q.option_type == "call"
                               else mg.put_price())
                row["garch_vol"] = garch_vol
                row["garch_price"] = garch_price
                if garch_price > 0:
                    row["mispricing_pct"] = (q.mid - garch_price) / garch_price * 100
                row["hv_iv_ratio"] = iv / garch_vol
            except (ValueError, ZeroDivisionError):
                pass

        # SVI model IV deviation
        if svi_params is not None:
            try:
                k = np.log(q.strike / spot)
                svi_iv = svi_params.implied_vol(k, T)
                row["svi_iv"] = svi_iv
                row["svi_dev"] = (iv - svi_iv) * 100  # in vol points
            except Exception:
                pass

        # If no GARCH, use simple BSM mispricing (market mid vs IV-repriced)
        if "mispricing_pct" not in row and bsm_price > 0:
            row["mispricing_pct"] = (q.mid - bsm_price) / bsm_price * 100

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("strike").reset_index(drop=True)
    return df


# ── IV Analytics ────────────────────────────────────────────────────────


def iv_rank(current_iv: float, iv_history: pd.Series) -> float:
    """IV Rank (52-week).

    ``(current - 52w_low) / (52w_high - 52w_low) * 100``

    Returns
    -------
    float
        Percentage 0-100, or nan if insufficient data.
    """
    if iv_history.empty or len(iv_history) < 5:
        return np.nan
    lo = iv_history.min()
    hi = iv_history.max()
    if hi <= lo:
        return 50.0
    return float((current_iv - lo) / (hi - lo) * 100)


def iv_percentile(current_iv: float, iv_history: pd.Series) -> float:
    """IV Percentile (52-week).

    Percentage of trading days where IV was below current IV.

    Returns
    -------
    float
        Percentage 0-100.
    """
    if iv_history.empty:
        return np.nan
    return float((iv_history < current_iv).sum() / len(iv_history) * 100)


def atm_term_structure(
    chain: OptionChain,
    spot: float,
    r: float,
    max_expiries: int = 10,
) -> pd.DataFrame:
    """ATM implied volatility by expiry.

    Parameters
    ----------
    max_expiries : int
        Maximum number of expiries to include (default 10).
        Limits computation time on chains with many expiries.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, days, atm_iv.
    """
    import datetime

    today = datetime.date.today()
    rows = []

    for exp_str in chain.expiries():
        if max_expiries and len(rows) >= max_expiries:
            break

        try:
            exp_date = datetime.date.fromisoformat(exp_str)
        except ValueError:
            continue

        days = (exp_date - today).days
        if days <= 0:
            continue
        T = days / 365.0

        # Find nearest ATM quote for this expiry
        exp_quotes = chain.by_expiry(exp_str)
        calls = [q for q in exp_quotes if q.option_type == "call" and q.mid > 0]
        if not calls:
            continue

        calls.sort(key=lambda q: abs(q.strike - spot))
        atm = calls[0]

        # Use provider IV when available
        iv_val = atm.implied_vol if atm.implied_vol and atm.implied_vol > 0.001 else None
        if iv_val is None:
            iv_val = implied_vol(atm.mid, spot, atm.strike, T, r, "call",
                                 method="newton")
        if not np.isnan(iv_val) and iv_val > 0:
            rows.append({
                "expiry": exp_str,
                "days": days,
                "atm_iv": iv_val,
            })

    return pd.DataFrame(rows)


def skew_25delta(
    quotes: list[OptionQuote],
    spot: float,
    r: float,
    T: float,
) -> float | None:
    """25-delta put-call skew.

    ``IV(25D Put) - IV(25D Call)``

    Returns
    -------
    float or None
        Skew in decimal vol, or None if insufficient data.
    """
    if T <= 0:
        return None

    puts = [q for q in quotes if q.option_type == "put" and q.mid > 0]
    calls = [q for q in quotes if q.option_type == "call" and q.mid > 0]

    def _find_25d(option_list: list[OptionQuote], opt_type: str) -> float | None:
        best = None
        best_dist = float("inf")
        for q in option_list:
            iv_val = implied_vol(q.mid, spot, q.strike, T, r, opt_type)
            if np.isnan(iv_val) or iv_val <= 0:
                continue
            try:
                m = BSM(S=spot, K=q.strike, T=T, r=r, sigma=iv_val)
                delta = m.call_delta() if opt_type == "call" else m.put_delta()
            except ValueError:
                continue
            dist = abs(abs(delta) - 0.25)
            if dist < best_dist:
                best_dist = dist
                best = iv_val
        return best

    iv_25p = _find_25d(puts, "put")
    iv_25c = _find_25d(calls, "call")

    if iv_25p is not None and iv_25c is not None:
        return iv_25p - iv_25c
    return None


# ── P&L Simulation ─────────────────────────────────────────────────────


def whatif_table(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    premium: float,
) -> pd.DataFrame:
    """Spot x Vol change P&L matrix.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Current BSM parameters.
    option_type : str
        ``"call"`` or ``"put"``.
    premium : float
        Entry premium paid.

    Returns
    -------
    pd.DataFrame
        Index = spot changes (%), columns = vol changes (pp).
    """
    spot_pcts = [-10, -5, -3, -1, 0, +1, +3, +5, +10]
    vol_pps = [-10, -5, -2, 0, +2, +5, +10]

    data = {}
    for vpp in vol_pps:
        col = []
        new_sigma = max(sigma + vpp / 100, 0.01)
        for spct in spot_pcts:
            new_S = S * (1 + spct / 100)
            try:
                m = BSM(S=new_S, K=K, T=T, r=r, sigma=new_sigma)
                price = m.call_price() if option_type == "call" else m.put_price()
            except ValueError:
                price = 0.0
            col.append(price - premium)
        data[f"{vpp:+d}pp"] = col

    df = pd.DataFrame(data, index=[f"{s:+d}%" for s in spot_pcts])
    df.index.name = "spot_chg"
    return df


def breakeven(K: float, premium: float, option_type: str) -> float:
    """Breakeven spot price at expiry.

    Returns
    -------
    float
        Spot at which P&L = 0 at expiry.
    """
    if option_type == "call":
        return K + premium
    return K - premium


def theta_decay_curve(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    steps: int = 20,
) -> pd.DataFrame:
    """Option value as time decays from T to 0.

    Returns
    -------
    pd.DataFrame
        Columns: days_left, value.
    """
    days_total = int(T * 365)
    time_points = np.linspace(T, 1 / 365, min(steps, days_total))

    rows = []
    for t in time_points:
        try:
            m = BSM(S=S, K=K, T=t, r=r, sigma=sigma)
            price = m.call_price() if option_type == "call" else m.put_price()
        except ValueError:
            price = 0.0
        rows.append({"days_left": int(t * 365), "value": price})

    return pd.DataFrame(rows)


# ── Probability ─────────────────────────────────────────────────────────


def prob_itm(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str,
) -> float:
    """Probability of finishing in-the-money (BSM risk-neutral).

    Call: N(d2), Put: N(-d2).
    """
    if T <= 0 or sigma <= 0:
        return np.nan
    sqrt_T = np.sqrt(T)
    d2 = (np.log(S / K) + (r - sigma**2 / 2) * T) / (sigma * sqrt_T)
    if option_type == "call":
        return float(norm.cdf(d2))
    return float(norm.cdf(-d2))


def prob_profit(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str, premium: float,
) -> float:
    """Probability of profit at expiry (long position).

    Call: P(S_T > K + premium), Put: P(S_T < K - premium).
    """
    if T <= 0 or sigma <= 0:
        return np.nan

    sqrt_T = np.sqrt(T)
    if option_type == "call":
        be = K + premium
        d2 = (np.log(S / be) + (r - sigma**2 / 2) * T) / (sigma * sqrt_T)
        return float(norm.cdf(d2))
    else:
        be = K - premium
        if be <= 0:
            return 0.0
        d2 = (np.log(S / be) + (r - sigma**2 / 2) * T) / (sigma * sqrt_T)
        return float(norm.cdf(-d2))


def expected_return(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str, premium: float,
) -> float:
    """Expected return = E[payoff] / premium - 1.

    Uses BSM theoretical price as E[discounted payoff], then
    undiscounts to get E[payoff].
    """
    if premium <= 0:
        return np.nan
    try:
        m = BSM(S=S, K=K, T=T, r=r, sigma=sigma)
    except ValueError:
        return np.nan
    bsm_price = m.call_price() if option_type == "call" else m.put_price()
    # BSM price = e^{-rT} * E^Q[payoff], so E^Q[payoff] = BSM * e^{rT}
    expected_payoff = bsm_price * np.exp(r * T)
    return float(expected_payoff / premium - 1)
