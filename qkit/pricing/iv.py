"""Implied volatility solver and surface construction.

Provides Brentq and Newton-Raphson IV solvers, grid computation
over an option chain, and 3-D surface visualisation helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from .bsm import BSM


def implied_vol(market_price: float, S: float, K: float, T: float,
                r: float, option_type: str = "call",
                method: str = "brentq") -> float:
    """Solve for the BSM implied volatility.

    Parameters
    ----------
    market_price : float
        Observed option price.
    S, K, T, r : float
        BSM parameters.
    option_type : str
        ``"call"`` or ``"put"``.
    method : str
        ``"brentq"`` (default, robust) or ``"newton"`` (fast).

    Returns
    -------
    float
        Implied volatility, or ``nan`` if no solution exists.
    """
    if method == "newton":
        return _iv_newton(market_price, S, K, T, r, option_type)
    return _iv_brentq(market_price, S, K, T, r, option_type)


def implied_vol_grid(chain_df: pd.DataFrame, S: float, r: float,
                     method: str = "brentq") -> np.ndarray:
    """Compute IV for every row in an option chain DataFrame.

    *chain_df* must contain columns ``strike``, ``expiry_years``,
    ``mid`` and ``type``.
    """
    prices = chain_df["mid"].values
    strikes = chain_df["strike"].values
    expiries = chain_df["expiry_years"].values
    types = chain_df["type"].values
    return np.array([
        implied_vol(p, S, k, t, r, ty, method)
        for p, k, t, ty in zip(prices, strikes, expiries, types)
    ])


def filter_chain(df: pd.DataFrame, S: float, min_volume: int = 5,
                 min_oi: int = 10, moneyness: tuple = (0.7, 1.3)
                 ) -> pd.DataFrame:
    """Apply standard data quality filters before IV computation.

    Keeps OTM options only, removes zero-bid quotes and extreme
    moneyness.  Based on Li et al. (2024) filtering criteria.
    """
    out = df.copy()
    out = out[out["volume"] >= min_volume]
    out = out[out["open_interest"] >= min_oi]
    out = out[out["bid"] > 0]

    otm_calls = (out["type"] == "call") & (out["strike"] >= S)
    otm_puts = (out["type"] == "put") & (out["strike"] < S)
    out = out[otm_calls | otm_puts]

    m = out["strike"] / S
    out = out[(m >= moneyness[0]) & (m <= moneyness[1])]
    return out


def _iv_brentq(price, S, K, T, r, opt_type, lo=1e-4, hi=5.0):
    def residual(sigma):
        m = BSM(S=S, K=K, T=T, r=r, sigma=sigma)
        return (m.call_price() if opt_type == "call" else m.put_price()) - price
    try:
        return brentq(residual, lo, hi, xtol=1e-10)
    except (ValueError, RuntimeError):
        return np.nan


def _iv_newton(price, S, K, T, r, opt_type, sigma0=0.3,
               max_iter=100, tol=1e-10):
    sigma = sigma0
    for _ in range(max_iter):
        m = BSM(S=S, K=K, T=T, r=r, sigma=sigma)
        if opt_type == "call":
            model_price, vega = m.call_price(), m.call_vega() * 100
        else:
            model_price, vega = m.put_price(), m.put_vega() * 100

        diff = model_price - price
        if abs(diff) < tol:
            return sigma
        if abs(vega) < 1e-14:
            return _iv_brentq(price, S, K, T, r, opt_type)

        sigma = max(1e-4, min(sigma - diff / vega, 5.0))
    return np.nan
