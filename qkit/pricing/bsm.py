"""Black-Scholes-Merton model implemented from scratch.

Only numpy and scipy.stats are used.  No third-party pricing libraries.

Example::

    from qkit.pricing.bsm import BSM

    m = BSM(S=150, K=155, T=30/365, r=0.043, sigma=0.25)
    print(m.call_price())
    print(m.call_greeks())
    m.verify_put_call_parity()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class GreeksResult:
    """Container for a single set of option Greeks.

    Attributes
    ----------
    delta : float
        Price sensitivity to the underlying (per $1 move).
    gamma : float
        Rate of change of delta (second-order).
    vega : float
        Sensitivity to implied volatility (per 1 pp move).
    theta : float
        Time decay per calendar day.
    rho : float
        Sensitivity to the risk-free rate (per 1 pp move).
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def __repr__(self) -> str:
        return (
            f"Greeks(delta={self.delta:+.6f}, gamma={self.gamma:+.6f}, "
            f"vega={self.vega:+.6f}, theta={self.theta:+.6f}, "
            f"rho={self.rho:+.6f})"
        )


class BSM:
    """European option pricer using the Black-Scholes-Merton model.

    Parameters
    ----------
    S : float
        Spot price of the underlying asset.
    K : float
        Strike (exercise) price.
    T : float
        Time to expiry in years (e.g. 30 days = 30/365).
    r : float
        Annualised risk-free rate (e.g. 0.043 for 4.3 %).
    sigma : float
        Annualised volatility (e.g. 0.25 for 25 %).
    """

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if S <= 0 or K <= 0:
            raise ValueError("S and K must be positive")

        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

        self._sqrt_T = np.sqrt(T)
        self._d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * self._sqrt_T)
        self._d2 = self._d1 - sigma * self._sqrt_T

        self._discount = np.exp(-r * T)
        self._Nd1 = norm.cdf(self._d1)
        self._Nd2 = norm.cdf(self._d2)
        self._Nnd1 = norm.cdf(-self._d1)
        self._Nnd2 = norm.cdf(-self._d2)
        self._nd1 = norm.pdf(self._d1)

    # -- prices ---------------------------------------------------------------

    def call_price(self) -> float:
        """European call price: ``C = S N(d1) - K e^{-rT} N(d2)``."""
        return self.S * self._Nd1 - self.K * self._discount * self._Nd2

    def put_price(self) -> float:
        """European put price: ``P = K e^{-rT} N(-d2) - S N(-d1)``."""
        return self.K * self._discount * self._Nnd2 - self.S * self._Nnd1

    # -- call Greeks ----------------------------------------------------------

    def call_delta(self) -> float:
        """dC/dS = N(d1)."""
        return self._Nd1

    def call_gamma(self) -> float:
        """d²C/dS² = n(d1) / (S sigma sqrt(T)).  Same for put."""
        return self._nd1 / (self.S * self.sigma * self._sqrt_T)

    def call_vega(self) -> float:
        """dC/dsigma, scaled to a 1 pp vol move (divided by 100)."""
        return self.S * self._nd1 * self._sqrt_T / 100

    def call_theta(self) -> float:
        """dC/dT, expressed per calendar day (divided by 365)."""
        term1 = -(self.S * self._nd1 * self.sigma) / (2 * self._sqrt_T)
        term2 = -self.r * self.K * self._discount * self._Nd2
        return (term1 + term2) / 365

    def call_rho(self) -> float:
        """dC/dr, scaled to a 1 pp rate move (divided by 100)."""
        return self.K * self.T * self._discount * self._Nd2 / 100

    # -- put Greeks -----------------------------------------------------------

    def put_delta(self) -> float:
        """dP/dS = N(d1) - 1."""
        return self._Nd1 - 1

    def put_gamma(self) -> float:
        """Same as call gamma."""
        return self.call_gamma()

    def put_vega(self) -> float:
        """Same as call vega."""
        return self.call_vega()

    def put_theta(self) -> float:
        """dP/dT, expressed per calendar day."""
        term1 = -(self.S * self._nd1 * self.sigma) / (2 * self._sqrt_T)
        term2 = self.r * self.K * self._discount * self._Nnd2
        return (term1 + term2) / 365

    def put_rho(self) -> float:
        """dP/dr, scaled to a 1 pp rate move."""
        return -self.K * self.T * self._discount * self._Nnd2 / 100

    # -- aggregate accessors --------------------------------------------------

    def call_greeks(self) -> GreeksResult:
        """Return all Greeks for the call in a single object."""
        return GreeksResult(
            delta=self.call_delta(), gamma=self.call_gamma(),
            vega=self.call_vega(), theta=self.call_theta(),
            rho=self.call_rho(),
        )

    def put_greeks(self) -> GreeksResult:
        """Return all Greeks for the put in a single object."""
        return GreeksResult(
            delta=self.put_delta(), gamma=self.put_gamma(),
            vega=self.put_vega(), theta=self.put_theta(),
            rho=self.put_rho(),
        )

    # -- verification ---------------------------------------------------------

    def verify_put_call_parity(self, tol: float = 1e-8) -> dict:
        """Check put-call parity: ``C - P == S - K e^{-rT}``.

        Returns a dict with ``"passed"`` set to *True* when the absolute
        difference is below *tol*.
        """
        lhs = self.call_price() - self.put_price()
        rhs = self.S - self.K * self._discount
        diff = abs(lhs - rhs)
        return {
            "call_price": self.call_price(),
            "put_price": self.put_price(),
            "C_minus_P": lhs,
            "S_minus_PV_K": rhs,
            "difference": diff,
            "tolerance": tol,
            "passed": diff < tol,
        }

    # -- display --------------------------------------------------------------

    def summary(self) -> str:
        """Return a multi-line text summary of prices, Greeks and parity."""
        C = self.call_price()
        P = self.put_price()
        cg = self.call_greeks()
        pg = self.put_greeks()
        par = self.verify_put_call_parity()

        return "\n".join([
            "Black-Scholes-Merton Model",
            f"  S={self.S:.2f}  K={self.K:.2f}  T={self.T:.4f}yr "
            f"({self.T * 365:.0f}d)  r={self.r:.2%}  sigma={self.sigma:.2%}",
            f"  d1={self._d1:.6f}  d2={self._d2:.6f}",
            "",
            f"  Call = ${C:.4f}   Put = ${P:.4f}",
            "",
            "  Call Greeks:",
            f"    Delta={cg.delta:+.6f}  Gamma={cg.gamma:+.6f}  "
            f"Vega={cg.vega:+.6f}",
            f"    Theta={cg.theta:+.6f}  Rho={cg.rho:+.6f}",
            "",
            "  Put Greeks:",
            f"    Delta={pg.delta:+.6f}  Gamma={pg.gamma:+.6f}  "
            f"Vega={pg.vega:+.6f}",
            f"    Theta={pg.theta:+.6f}  Rho={pg.rho:+.6f}",
            "",
            "  Put-Call Parity:",
            f"    C-P={par['C_minus_P']:.8f}  S-Ke^(-rT)={par['S_minus_PV_K']:.8f}"
            f"  diff={par['difference']:.2e}  {'PASS' if par['passed'] else 'FAIL'}",
        ])


def bsm_call_vec(S: np.ndarray, K: np.ndarray, T: np.ndarray,
                 r: float, sigma: np.ndarray) -> np.ndarray:
    """Vectorised BSM call price for numpy array inputs."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bsm_put_vec(S: np.ndarray, K: np.ndarray, T: np.ndarray,
                r: float, sigma: np.ndarray) -> np.ndarray:
    """Vectorised BSM put price for numpy array inputs."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


if __name__ == "__main__":
    m = BSM(S=230.0, K=235.0, T=30 / 365, r=0.043, sigma=0.22)
    print(m.summary())
