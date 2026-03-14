"""SVI and SSVI implied volatility parametrisation.

Implements the Stochastic Volatility Inspired (SVI) model of
Gatheral (2004) in raw and natural forms, plus the Surface SVI (SSVI)
extension of Gatheral & Jacquier (2014).

References
----------
- Gatheral, J. (2004). "A parsimonious arbitrage-free implied
  volatility parameterization with application to the valuation
  of volatility derivatives."
- Gatheral, J. & Jacquier, A. (2014). "Arbitrage-free SVI
  volatility surfaces."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, differential_evolution


@dataclass
class SVIParams:
    """Raw SVI parameters for a single smile slice.

    The raw SVI total implied variance is:
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

    where k = log(K/F) is log-moneyness.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """Compute total implied variance w(k) = sigma_BS^2 * T."""
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (
            self.rho * (k - self.m)
            + np.sqrt((k - self.m) ** 2 + self.sigma ** 2)
        )

    def implied_vol(self, k: np.ndarray, T: float) -> np.ndarray:
        """BSM implied volatility from total variance."""
        w = self.total_variance(k)
        w = np.maximum(w, 0)
        return np.sqrt(w / T)

    def is_arbitrage_free(self, k_grid: np.ndarray | None = None) -> dict:
        """Check Gatheral-Jacquier butterfly arbitrage-free conditions.

        Conditions:
        1. w(k) >= 0 for all k  (non-negative total variance)
        2. 1 + k * w'(k) / (2 * w(k)) > 0  (no butterfly arbitrage)
        3. b >= 0 and a + b * sigma * sqrt(1 - rho^2) >= 0
        """
        if k_grid is None:
            k_grid = np.linspace(-2, 2, 500)

        w = self.total_variance(k_grid)
        nonneg = bool(np.all(w >= -1e-10))

        # Condition 3: parameter constraints
        param_ok = self.b >= 0 and (
            self.a + self.b * self.sigma * np.sqrt(1 - self.rho ** 2) >= -1e-10
        )

        # Condition 2: check g(k) = 1 + k * w'/(2w) > 0
        dk = k_grid[1] - k_grid[0]
        wp = np.gradient(w, dk)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = 1 + k_grid * wp / (2 * np.maximum(w, 1e-15))
        butterfly_ok = bool(np.all(g > -1e-10))

        return {
            "nonneg_variance": nonneg,
            "param_constraint": param_ok,
            "no_butterfly_arb": butterfly_ok,
            "arbitrage_free": nonneg and param_ok and butterfly_ok,
        }


def calibrate_svi(
    k: np.ndarray,
    iv: np.ndarray,
    T: float,
    method: str = "de",
) -> SVIParams:
    """Calibrate raw SVI to a single smile slice.

    Parameters
    ----------
    k : array
        Log-moneyness (log(K/F)).
    iv : array
        Market implied volatilities.
    T : float
        Time to expiry in years.
    method : str
        ``"de"`` for Differential Evolution (global, robust),
        ``"nelder-mead"`` for local simplex.

    Returns
    -------
    SVIParams
    """
    k = np.asarray(k, dtype=float)
    w_market = np.asarray(iv, dtype=float) ** 2 * T  # total variance

    def objective(params):
        a, b, rho, m, sigma = params
        w_model = a + b * (
            rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2)
        )
        return np.sum((w_model - w_market) ** 2)

    # Bounds: a free, b>=0, -1<rho<1, m free, sigma>0
    bounds = [
        (-0.5, 0.5),   # a
        (0.0, 1.0),    # b
        (-0.99, 0.99), # rho
        (-1.0, 1.0),   # m
        (0.01, 2.0),   # sigma
    ]

    if method == "de":
        result = differential_evolution(objective, bounds, seed=42,
                                        maxiter=500, tol=1e-12)
    else:
        x0 = [0.04, 0.1, -0.5, 0.0, 0.2]
        result = minimize(objective, x0, method="Nelder-Mead",
                          options={"maxiter": 5000, "xatol": 1e-12})

    a, b, rho, m, sigma = result.x
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


# ── SSVI Surface ─────────────────────────────────────────────────────────

@dataclass
class SSVIParams:
    """SSVI surface parameters (Gatheral-Jacquier 2014).

    The SSVI total variance surface is:
        w(k, theta_t) = (theta_t / 2) * (1 + rho * phi(theta_t) * k
            + sqrt((phi(theta_t) * k + rho)^2 + (1 - rho^2)))

    where theta_t = ATM total variance at expiry t, and
    phi(theta) is a mixing function.

    Uses the power-law mixing function:
        phi(theta) = eta / (theta^gamma * (1 + theta)^(1 - gamma))
    """

    rho: float
    eta: float
    gamma: float

    def phi(self, theta: float) -> float:
        """Power-law mixing function."""
        return self.eta / (theta ** self.gamma * (1 + theta) ** (1 - self.gamma))

    def total_variance(self, k: np.ndarray, theta_t: float) -> np.ndarray:
        """SSVI total variance for a given ATM variance slice."""
        k = np.asarray(k, dtype=float)
        p = self.phi(theta_t)
        return (theta_t / 2) * (
            1 + self.rho * p * k
            + np.sqrt((p * k + self.rho) ** 2 + (1 - self.rho ** 2))
        )

    def implied_vol(self, k: np.ndarray, theta_t: float, T: float) -> np.ndarray:
        """BSM implied vol from SSVI."""
        w = self.total_variance(k, theta_t)
        w = np.maximum(w, 0)
        return np.sqrt(w / T)

    def is_arbitrage_free(self) -> bool:
        """Check sufficient conditions for no butterfly arbitrage.

        Gatheral-Jacquier (2014) Theorem 4.2:
            eta * (1 + |rho|) <= 2
        and 0 < gamma <= 1.
        """
        return (
            self.eta * (1 + abs(self.rho)) <= 2
            and 0 < self.gamma <= 1
        )


def calibrate_ssvi(
    slices: list[dict],
) -> SSVIParams:
    """Calibrate SSVI to multiple smile slices.

    Parameters
    ----------
    slices : list of dict
        Each dict has keys: ``k`` (log-moneyness array),
        ``iv`` (implied vol array), ``T`` (time to expiry),
        ``theta`` (ATM total variance = atm_iv^2 * T).

    Returns
    -------
    SSVIParams
    """
    def objective(params):
        rho, eta, gamma = params
        ssvi = SSVIParams(rho=rho, eta=eta, gamma=gamma)
        total_err = 0.0
        for s in slices:
            k = np.asarray(s["k"], dtype=float)
            w_market = np.asarray(s["iv"], dtype=float) ** 2 * s["T"]
            w_model = ssvi.total_variance(k, s["theta"])
            total_err += np.sum((w_model - w_market) ** 2)
        return total_err

    bounds = [
        (-0.99, 0.99),  # rho
        (0.01, 4.0),    # eta
        (0.01, 1.0),    # gamma
    ]

    result = differential_evolution(objective, bounds, seed=42,
                                    maxiter=500, tol=1e-12)
    rho, eta, gamma = result.x
    return SSVIParams(rho=rho, eta=eta, gamma=gamma)
