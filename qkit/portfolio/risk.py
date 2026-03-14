"""Portfolio risk metrics: VaR, CVaR, and stress testing.

Implements parametric, historical, Monte Carlo and Cornish-Fisher
methods for Value at Risk and Conditional Value at Risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class RiskReport:
    """Container for computed risk metrics."""

    var_parametric: float
    var_historical: float
    var_montecarlo: float
    var_cornish_fisher: float
    cvar_parametric: float
    cvar_historical: float
    cvar_montecarlo: float
    confidence: float

    def as_dict(self) -> dict:
        return {k: round(v, 6) for k, v in self.__dict__.items()}

    def as_dataframe(self) -> pd.DataFrame:
        data = {
            "Parametric": [self.var_parametric, self.cvar_parametric],
            "Historical": [self.var_historical, self.cvar_historical],
            "Monte Carlo": [self.var_montecarlo, self.cvar_montecarlo],
            "Cornish-Fisher": [self.var_cornish_fisher, np.nan],
        }
        return pd.DataFrame(data, index=["VaR", "CVaR"])


def var_parametric(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Parametric (variance-covariance) VaR assuming normal returns."""
    mu, sigma = np.mean(returns), np.std(returns, ddof=1)
    return -(mu + norm.ppf(1 - confidence) * sigma)


def cvar_parametric(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Parametric CVaR (Expected Shortfall) under normality."""
    mu, sigma = np.mean(returns), np.std(returns, ddof=1)
    alpha = 1 - confidence
    return -(mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha)


def var_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Non-parametric historical VaR."""
    return -np.percentile(returns, 100 * (1 - confidence))


def cvar_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Non-parametric historical CVaR."""
    threshold = -var_historical(returns, confidence)
    tail = returns[returns <= threshold]
    return -tail.mean() if len(tail) > 0 else var_historical(returns, confidence)


def var_montecarlo(returns: np.ndarray, confidence: float = 0.95,
                   n_sims: int = 100_000) -> float:
    """Monte Carlo VaR by resampling from fitted normal."""
    mu, sigma = np.mean(returns), np.std(returns, ddof=1)
    sims = np.random.normal(mu, sigma, n_sims)
    return -np.percentile(sims, 100 * (1 - confidence))


def cvar_montecarlo(returns: np.ndarray, confidence: float = 0.95,
                    n_sims: int = 100_000) -> float:
    """Monte Carlo CVaR."""
    mu, sigma = np.mean(returns), np.std(returns, ddof=1)
    sims = np.random.normal(mu, sigma, n_sims)
    threshold = np.percentile(sims, 100 * (1 - confidence))
    return -sims[sims <= threshold].mean()


def var_cornish_fisher(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Cornish-Fisher expansion VaR adjusting for skew and kurtosis."""
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    s = _skewness(returns)
    k = _kurtosis(returns)
    z = norm.ppf(confidence)

    z_cf = (z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * (k - 3) / 24
            - (2 * z**3 - 5 * z) * s**2 / 36)
    return -(mu - z_cf * sigma)


def compute_all(returns: pd.Series | np.ndarray,
                confidence: float = 0.95) -> RiskReport:
    """Compute all risk metrics in one call."""
    r = np.asarray(returns)
    return RiskReport(
        var_parametric=var_parametric(r, confidence),
        var_historical=var_historical(r, confidence),
        var_montecarlo=var_montecarlo(r, confidence),
        var_cornish_fisher=var_cornish_fisher(r, confidence),
        cvar_parametric=cvar_parametric(r, confidence),
        cvar_historical=cvar_historical(r, confidence),
        cvar_montecarlo=cvar_montecarlo(r, confidence),
        confidence=confidence,
    )


def stress_test(portfolio_value: float, scenarios: dict[str, float]
                ) -> pd.DataFrame:
    """Apply shock scenarios and compute P&L.

    *scenarios* maps a label to a return shock (e.g. ``{"Crash": -0.20}``).
    """
    rows = []
    for label, shock in scenarios.items():
        pnl = portfolio_value * shock
        rows.append({"scenario": label, "shock": shock,
                     "pnl": pnl, "new_value": portfolio_value + pnl})
    return pd.DataFrame(rows)


def _skewness(x):
    m = np.mean(x)
    return np.mean((x - m) ** 3) / np.std(x, ddof=1) ** 3


def _kurtosis(x):
    m = np.mean(x)
    return np.mean((x - m) ** 4) / np.std(x, ddof=1) ** 4
