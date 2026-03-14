"""Variance Risk Premium signal generator.

Computes the VRP as IV² - RV, standardises it as a z-score and
generates trading signals following Bollerslev et al. (2009).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VRPSignal:
    """Daily VRP time series and derived signals."""

    vrp: pd.Series
    z_score: pd.Series
    rv: pd.Series
    iv_sq: pd.Series

    def current(self) -> dict:
        return {
            "vrp": float(self.vrp.iloc[-1]),
            "z_score": float(self.z_score.iloc[-1]),
            "rv": float(self.rv.iloc[-1]),
            "iv_sq": float(self.iv_sq.iloc[-1]),
            "signal": self.interpret(float(self.z_score.iloc[-1])),
        }

    @staticmethod
    def interpret(z: float) -> str:
        if z > 1.5:
            return "HIGH_VRP: options overpriced, short vol"
        if z < -1.0:
            return "LOW_VRP: risk underpriced, buy protection"
        return "NEUTRAL"


def compute_vrp(spy_returns: pd.Series, vix: pd.Series,
                rv_window: int = 22, z_window: int = 252) -> VRPSignal:
    """Compute the Variance Risk Premium from SPY returns and VIX.

    Parameters
    ----------
    spy_returns : pd.Series
        Daily log returns of SPY.
    vix : pd.Series
        VIX closing values (annualised vol in %).
    rv_window : int
        Rolling window for realised variance (default 22 trading days).
    z_window : int
        Lookback for z-score standardisation (default 252 days).
    """
    rv = (spy_returns ** 2).rolling(rv_window).sum()
    iv_sq = (vix / 100) ** 2 * (rv_window / 252)

    # align indices
    common = rv.index.intersection(iv_sq.index)
    rv, iv_sq = rv.loc[common], iv_sq.loc[common]

    vrp = iv_sq - rv
    z = (vrp - vrp.rolling(z_window).mean()) / vrp.rolling(z_window).std()

    return VRPSignal(vrp=vrp, z_score=z, rv=rv, iv_sq=iv_sq)


def predictive_regression(vrp: pd.Series, forward_returns: pd.Series,
                          horizon: int = 22) -> dict:
    """Run Bollerslev et al. (2009) predictive regression.

    ``R_{t+h} = alpha + beta * VRP_t + epsilon``

    Returns slope, t-stat, and R-squared.
    """
    y = forward_returns.rolling(horizon).sum().shift(-horizon).dropna()
    x = vrp.reindex(y.index).dropna()
    common = x.index.intersection(y.index)
    x, y = x.loc[common].values, y.loc[common].values

    if len(x) < 30:
        return {"beta": np.nan, "t_stat": np.nan, "r_squared": np.nan}

    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    sse = np.sum(residuals ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - sse / sst

    se = np.sqrt(sse / (len(y) - 2) * np.diag(np.linalg.inv(X.T @ X)))
    t_stat = beta[1] / se[1]

    return {"alpha": beta[0], "beta": beta[1],
            "t_stat": t_stat, "r_squared": r2, "n_obs": len(x)}
