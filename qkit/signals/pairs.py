"""Statistical arbitrage via cointegration and Ornstein-Uhlenbeck modelling.

Provides pair selection through Engle-Granger cointegration testing,
OU parameter estimation, half-life calculation and z-score signal
generation for mean-reversion trading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


@dataclass
class PairStats:
    """Cointegration and OU statistics for a pair."""

    asset_a: str
    asset_b: str
    beta: float
    coint_pvalue: float
    theta: float
    mu: float
    sigma_ou: float
    half_life: float
    current_z: float

    @property
    def is_cointegrated(self) -> bool:
        return self.coint_pvalue < 0.05

    def signal(self) -> str:
        if self.current_z > 2.0:
            return "SHORT_SPREAD"
        if self.current_z < -2.0:
            return "LONG_SPREAD"
        if abs(self.current_z) < 0.5:
            return "EXIT"
        return "HOLD"


def find_cointegrated_pairs(prices: pd.DataFrame,
                            significance: float = 0.05
                            ) -> list[PairStats]:
    """Scan all pairs in a price DataFrame for cointegration.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns are ticker symbols, rows are dates, values are prices.
    significance : float
        p-value threshold for the Engle-Granger test.

    Returns
    -------
    list[PairStats]
        Pairs that pass the cointegration test, sorted by half-life.
    """
    tickers = prices.columns.tolist()
    results = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            stats = analyze_pair(prices[a], prices[b], a, b)
            if stats.coint_pvalue < significance:
                results.append(stats)

    return sorted(results, key=lambda s: s.half_life)


def analyze_pair(price_a: pd.Series, price_b: pd.Series,
              name_a: str = "A", name_b: str = "B") -> PairStats:
    """Run cointegration test and fit OU parameters for one pair."""
    _, pval, _ = coint(price_a, price_b)

    # OLS hedge ratio
    model = OLS(price_b, add_constant(price_a)).fit()
    beta = model.params.iloc[1] if hasattr(model.params, "iloc") else model.params[1]

    spread = price_b - beta * price_a
    theta, mu, sigma_ou = _fit_ou(spread)
    hl = np.log(2) / theta if theta > 0 else np.inf

    z = (spread.iloc[-1] - mu) / (sigma_ou / np.sqrt(2 * theta)) if theta > 0 else 0.0

    return PairStats(
        asset_a=name_a, asset_b=name_b, beta=beta,
        coint_pvalue=pval, theta=theta, mu=mu,
        sigma_ou=sigma_ou, half_life=hl, current_z=z,
    )


def spread_zscore(spread: pd.Series, theta: float, mu: float,
                  sigma_ou: float) -> pd.Series:
    """Compute the OU-normalised z-score of a spread series."""
    stationary_std = sigma_ou / np.sqrt(2 * theta) if theta > 0 else spread.std()
    return (spread - mu) / stationary_std


def backtest_pair(spread: pd.Series, theta: float, mu: float,
                  sigma_ou: float, entry_z: float = 2.0,
                  exit_z: float = 0.5, stop_z: float = 4.0
                  ) -> pd.DataFrame:
    """Simple z-score mean-reversion backtest on an OU spread.

    Returns a DataFrame with columns ``z``, ``position``, ``pnl``.
    """
    z = spread_zscore(spread, theta, mu, sigma_ou)
    pos = pd.Series(0.0, index=spread.index)
    pnl = pd.Series(0.0, index=spread.index)

    position = 0.0
    entry_price = 0.0

    for i in range(1, len(z)):
        zval = z.iloc[i]

        if position == 0:
            if zval > entry_z:
                position = -1.0
                entry_price = spread.iloc[i]
            elif zval < -entry_z:
                position = 1.0
                entry_price = spread.iloc[i]
        else:
            if abs(zval) < exit_z or abs(zval) > stop_z:
                pnl.iloc[i] = position * (spread.iloc[i] - entry_price)
                position = 0.0
                entry_price = 0.0

        pos.iloc[i] = position

    return pd.DataFrame({"z": z, "position": pos, "pnl": pnl})


SECTOR_PAIRS: dict[str, list[tuple[str, str]]] = {
    "Consumer Staples": [("KO", "PEP"), ("PG", "CL"), ("WMT", "COST")],
    "Financials": [("JPM", "BAC"), ("GS", "MS"), ("C", "WFC")],
    "Technology": [("GOOGL", "META"), ("MSFT", "AAPL"), ("CRM", "NOW")],
    "Energy": [("XOM", "CVX"), ("COP", "EOG"), ("SLB", "HAL")],
    "Healthcare": [("JNJ", "PFE"), ("UNH", "CI"), ("ABT", "MDT")],
}


def screen_sector_pairs(
    period: str = "2y",
    significance: float = 0.05,
    min_half_life: float = 5.0,
    max_half_life: float = 120.0,
) -> list[dict]:
    """Screen all SECTOR_PAIRS for cointegration.

    Uses the configured data provider (moomoo or yfinance).
    Returns a list of dicts with pair info for candidates that pass
    the cointegration test and half-life filter.
    """
    from qkit.data import get_provider
    provider = get_provider()

    results: list[dict] = []
    for sector, pairs in SECTOR_PAIRS.items():
        for sym_a, sym_b in pairs:
            try:
                hist_a = provider.get_history(sym_a, period=period)["close"]
                hist_b = provider.get_history(sym_b, period=period)["close"]
                idx = hist_a.index.intersection(hist_b.index)
                if len(idx) < 100:
                    continue
                stats = analyze_pair(hist_a.loc[idx], hist_b.loc[idx],
                                     sym_a, sym_b)
                if (stats.coint_pvalue < significance
                        and min_half_life <= stats.half_life <= max_half_life):
                    results.append({
                        "sector": sector,
                        "asset_a": sym_a,
                        "asset_b": sym_b,
                        "pvalue": round(stats.coint_pvalue, 4),
                        "half_life": round(stats.half_life, 1),
                        "beta": round(stats.beta, 4),
                        "current_z": round(stats.current_z, 2),
                        "signal": stats.signal(),
                    })
            except Exception:
                continue
    return sorted(results, key=lambda r: r["half_life"])


def johansen_test(
    prices: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict:
    """Johansen cointegration test for multiple time series.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns are ticker symbols, values are prices.
    det_order : int
        Deterministic trend order (-1=none, 0=constant, 1=trend).
    k_ar_diff : int
        Number of lagged differences in the VAR model.

    Returns
    -------
    dict
        Keys: ``trace_stat``, ``crit_90``, ``crit_95``, ``crit_99``,
        ``n_coint`` (number of cointegrating relations at 95%),
        ``eigenvectors`` (cointegrating vectors),
        ``eigenvalues``.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    result = coint_johansen(prices.dropna(), det_order, k_ar_diff)

    # Count cointegrating relations at 95% critical value
    n_coint = int(np.sum(result.lr1 > result.cvt[:, 1]))

    return {
        "trace_stat": result.lr1.tolist(),
        "crit_90": result.cvt[:, 0].tolist(),
        "crit_95": result.cvt[:, 1].tolist(),
        "crit_99": result.cvt[:, 2].tolist(),
        "n_coint": n_coint,
        "eigenvectors": result.evec.tolist(),
        "eigenvalues": result.eig.tolist(),
        "symbols": prices.columns.tolist(),
    }


def analyze_pair_kalman(
    price_a: pd.Series,
    price_b: pd.Series,
    name_a: str = "A",
    name_b: str = "B",
    delta: float = 1e-4,
) -> dict:
    """Analyze a pair using Kalman filter for dynamic hedge ratio.

    Returns Engle-Granger stats plus Kalman-filtered beta time series.

    Parameters
    ----------
    delta : float
        State noise scaling (how fast beta adapts).
    """
    from .spectral import kalman_regression

    # Standard static analysis
    static = analyze_pair(price_a, price_b, name_a, name_b)

    # Kalman dynamic beta
    kf = kalman_regression(price_b, price_a, delta=delta)
    dynamic_beta = kf.states[:, 1]  # beta column
    dynamic_alpha = kf.states[:, 0]

    # Dynamic spread using Kalman beta
    idx = price_a.index.intersection(price_b.index)
    spread_kalman = price_b.loc[idx].values - dynamic_beta * price_a.loc[idx].values

    # OU fit on Kalman spread
    spread_series = pd.Series(spread_kalman, index=idx)
    theta_k, mu_k, sigma_k = _fit_ou(spread_series)
    hl_k = np.log(2) / theta_k if theta_k > 0 else np.inf

    z_k = (spread_kalman[-1] - mu_k) / (sigma_k / np.sqrt(2 * theta_k)) if theta_k > 0 else 0.0

    return {
        "static": static,
        "kalman_beta_current": float(dynamic_beta[-1]),
        "kalman_beta_mean": float(np.mean(dynamic_beta)),
        "kalman_beta_std": float(np.std(dynamic_beta)),
        "kalman_half_life": float(hl_k),
        "kalman_z": float(z_k),
        "kalman_beta_series": pd.Series(dynamic_beta, index=idx, name="beta"),
        "kalman_spread": spread_series,
        "kalman_ll": kf.log_likelihood,
    }


def _fit_ou(spread: pd.Series) -> tuple[float, float, float]:
    """Estimate OU parameters (theta, mu, sigma) from a spread series.

    Uses AR(1) regression: Z_t = c + phi * Z_{t-1} + eps.
    """
    y = spread.values
    n = len(y)
    if n < 10:
        return 0.0, float(np.mean(y)), float(np.std(y))

    Y = y[1:]
    X = add_constant(y[:-1])
    model = OLS(Y, X).fit()
    c, phi = model.params[0], model.params[1]

    dt = 1.0 / 252
    if phi >= 1 or phi <= 0:
        return 0.0, float(np.mean(y)), float(np.std(y))

    theta = -np.log(phi) / dt
    mu = c / (1 - phi)
    sigma_ou = np.std(model.resid) * np.sqrt(-2 * np.log(phi) / (dt * (1 - phi ** 2)))

    return float(theta), float(mu), float(sigma_ou)
