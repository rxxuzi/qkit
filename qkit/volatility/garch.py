"""GARCH volatility modelling, forecasting and risk-neutral transformation.

Wraps the ``arch`` library for fitting GARCH(1,1), GJR-GARCH and
EGARCH, then provides volatility forecasting and Duan (1995) LRNVR
transformation for option pricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from arch import arch_model


@dataclass
class GARCHResult:
    """Container for fitted GARCH model outputs."""

    model_type: str
    params: dict
    log_likelihood: float
    aic: float
    bic: float
    persistence: float
    unconditional_var: float
    conditional_var: pd.Series
    residuals: pd.Series

    @property
    def unconditional_vol(self) -> float:
        return np.sqrt(self.unconditional_var)

    @property
    def half_life(self) -> float:
        if self.persistence >= 1:
            return np.inf
        return np.log(2) / np.log(1 / self.persistence)

    def forecast_variance(self, horizon: int = 30) -> np.ndarray:
        """Analytic h-step ahead conditional variance forecast.

        Uses the closed-form recursion for GARCH(1,1):
        ``E[h_{t+h}] = h_bar + (alpha+beta)^(h-1) * (h_{t+1} - h_bar)``
        """
        h_bar = self.unconditional_var
        h1 = float(self.conditional_var.iloc[-1])
        p = self.persistence
        return np.array([h_bar + p**h * (h1 - h_bar)
                         for h in range(horizon)])

    def forecast_vol(self, horizon: int = 30, annualise: bool = True) -> float:
        """Average annualised volatility over *horizon* days."""
        fv = self.forecast_variance(horizon)
        mean_var = np.mean(fv)
        if annualise:
            return np.sqrt(mean_var * 252)
        return np.sqrt(mean_var)


def fit(returns: pd.Series, model_type: str = "garch",
        p: int = 1, q: int = 1, o: int = 0,
        dist: str = "normal") -> GARCHResult:
    """Fit a GARCH-family model to a return series.

    Parameters
    ----------
    returns : pd.Series
        Percentage-scale daily returns (multiply by 100 first).
    model_type : str
        ``"garch"``, ``"gjr"`` or ``"egarch"``.
    p, q, o : int
        Lag orders.  For GJR set *o=1*.
    dist : str
        Innovation distribution (``"normal"``, ``"t"``, ``"skewt"``).

    Returns
    -------
    GARCHResult
    """
    vol = "EGARCH" if model_type == "egarch" else "Garch"
    if model_type == "gjr":
        o = max(o, 1)

    am = arch_model(returns, vol=vol, p=p, o=o, q=q,
                    mean="Constant", dist=dist)
    res = am.fit(disp="off")

    params = dict(res.params)
    cond_var = res.conditional_volatility ** 2

    omega = params.get("omega", 0)
    alpha = params.get("alpha[1]", 0)
    beta = params.get("beta[1]", 0)
    gamma = params.get("gamma[1]", 0)

    if model_type == "egarch":
        persistence = abs(beta)
        uvar = float(np.mean(cond_var))
    else:
        persistence = alpha + beta + gamma / 2
        denom = 1 - persistence
        uvar = omega / denom if denom > 0 else float(np.mean(cond_var))

    return GARCHResult(
        model_type=model_type,
        params=params,
        log_likelihood=float(res.loglikelihood),
        aic=float(res.aic),
        bic=float(res.bic),
        persistence=persistence,
        unconditional_var=uvar,
        conditional_var=cond_var,
        residuals=res.resid,
    )


def compare(returns: pd.Series, models: Optional[list[str]] = None
            ) -> pd.DataFrame:
    """Fit several GARCH variants and return a comparison table."""
    if models is None:
        models = ["garch", "gjr", "egarch"]

    rows = []
    for name in models:
        r = fit(returns, model_type=name)
        rows.append({
            "model": name.upper(),
            "log_lik": r.log_likelihood,
            "aic": r.aic,
            "bic": r.bic,
            "persistence": r.persistence,
            "unconditional_vol": r.unconditional_vol,
            "half_life": r.half_life,
        })
    return pd.DataFrame(rows).set_index("model")


def lrnvr_transform(params: dict, risk_premium: float = 0.0) -> dict:
    """Duan (1995) LRNVR: transform P-measure to Q-measure parameters.

    Under LRNVR the conditional variance process is identical under P
    and Q, but the asymmetry parameter theta becomes theta + lambda.
    """
    q_params = params.copy()
    theta = params.get("gamma[1]", params.get("theta", 0))
    q_params["theta_star"] = theta + risk_premium
    return q_params
