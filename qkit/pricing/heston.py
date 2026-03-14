"""Heston stochastic volatility model and Carr-Madan FFT pricer.

Implements the Heston (1993) characteristic function in a numerically
stable formulation that avoids both the ``u = 0`` singularity and
the ``exp(-d tau)`` overflow (Albrecher et al. 2006 "Little Trap").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.fft import fft

from .bsm import BSM


def characteristic_fn(u, S0: float, r: float, tau: float, V0: float,
                      kappa: float, theta: float, xi: float,
                      rho: float) -> np.ndarray:
    """Heston log-price characteristic function.

    Uses a reformulation that expresses *C* and *D* without computing
    the ratio ``g`` directly, eliminating the ``0/0`` at ``u = 0``
    and using only ``exp(d tau)`` (with ``Re(d) >= 0``) to avoid
    overflow.
    """
    u = np.asarray(u, dtype=complex)
    i = 1j

    d = np.sqrt((rho * xi * i * u - kappa) ** 2
                + xi ** 2 * (i * u + u ** 2))

    # A = kappa - rho*xi*i*u,  B1 = A + d,  B2 = A - d
    A = kappa - rho * xi * i * u
    B1 = A + d     # = kappa - rho*xi*iu + d
    B2 = A - d     # = kappa - rho*xi*iu - d

    # exp(d*tau) with Re(d) >= 0 (numpy sqrt convention) is always safe
    exp_dt = np.exp(d * tau)

    # Numerically stable form avoiding exp(-d*tau):
    #   C = kappa*theta/xi^2 * [B1*tau - 2*ln((B1*exp_dt - B2) / (2d))]
    #   D = B2*B1/xi^2 * (exp_dt - 1) / (B1*exp_dt - B2)
    #
    # At u=0: B2=0, so D=0 and C=kappa*theta/xi^2*[2d*tau - 2*ln(exp_dt)] = 0.
    denom = B1 * exp_dt - B2
    safe_d = np.where(np.abs(d) > 1e-30, d, 1e-30 + 0j)

    C = (kappa * theta / xi ** 2) * (
        B1 * tau - 2.0 * np.log(denom / (2.0 * safe_d))
    )
    D = (B2 * B1 / xi ** 2) * (exp_dt - 1.0) / (denom + 1e-300)

    log_S_fwd = np.log(S0) + r * tau
    with np.errstate(over="ignore"):
        return np.exp(C + D * V0 + i * u * log_S_fwd)


def call_price_quad(S0: float, K: float, r: float, tau: float,
                    V0: float, kappa: float, theta: float,
                    xi: float, rho: float, upper: float = 200
                    ) -> float:
    """European call via Gil-Pelaez inversion (numerical quadrature)."""
    ln_K = np.log(K)

    def integrand_p1(u):
        if u < 1e-10:
            return 0.0
        phi = characteristic_fn(complex(1, u), S0, r, tau, V0,
                                kappa, theta, xi, rho)
        val = phi * np.exp(-1j * u * ln_K) / (1j * u)
        rv = np.real(val)
        return rv if np.isfinite(rv) else 0.0

    def integrand_p2(u):
        if u < 1e-10:
            return 0.0
        phi = characteristic_fn(complex(0, u), S0, r, tau, V0,
                                kappa, theta, xi, rho)
        val = phi * np.exp(-1j * u * ln_K) / (1j * u)
        rv = np.real(val)
        return rv if np.isfinite(rv) else 0.0

    P1 = 0.5 + (1 / np.pi) * quad(integrand_p1, 0, upper, limit=500)[0]
    P2 = 0.5 + (1 / np.pi) * quad(integrand_p2, 0, upper, limit=500)[0]

    return float(max(S0 * P1 - K * np.exp(-r * tau) * P2, 0.0))


def call_prices_fft(S0: float, r: float, tau: float,
                    V0: float, kappa: float, theta: float,
                    xi: float, rho: float,
                    N: int = 4096, alpha: float = 1.5,
                    eta: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """European call prices for a grid of strikes via Carr-Madan FFT.

    Returns ``(strikes, call_prices)`` arrays of length *N*.
    """
    lam = 2 * np.pi / (N * eta)
    b = N * lam / 2

    j = np.arange(N)
    u = j * eta
    k = -b + j * lam

    w = eta * (3 + (-1.0) ** j) / 3
    w[0] = eta / 3

    cf = characteristic_fn(u - (alpha + 1) * 1j, S0, r, tau, V0,
                           kappa, theta, xi, rho)
    denom = (alpha + 1j * u) * (alpha + 1 + 1j * u)
    psi = np.exp(-r * tau) * cf / denom

    x = np.exp(1j * b * u) * psi * w
    fft_result = np.real(fft(x))

    strikes = np.exp(k)
    prices = np.exp(-alpha * k) / np.pi * fft_result
    return strikes, prices


def call_price_fft(S0: float, K: float, r: float, tau: float,
                   V0: float, kappa: float, theta: float,
                   xi: float, rho: float, **kwargs) -> float:
    """Single-strike call price via FFT interpolation."""
    strikes, prices = call_prices_fft(S0, r, tau, V0, kappa, theta,
                                      xi, rho, **kwargs)
    return float(np.interp(K, strikes, prices))


# ── Heston Calibration ───────────────────────────────────────────────────

@dataclass
class HestonCalibResult:
    """Container for Heston calibration output."""

    V0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    rmse: float
    n_contracts: int

    @property
    def params(self) -> dict:
        return dict(V0=self.V0, kappa=self.kappa, theta=self.theta,
                    xi=self.xi, rho=self.rho)

    def model_iv(self, S0: float, r: float, strikes: np.ndarray,
                 taus: np.ndarray) -> np.ndarray:
        """Compute model implied vols for given strikes and expiries."""
        from .iv import implied_vol as bsm_iv
        ivs = np.empty(len(strikes))
        for i, (K, tau) in enumerate(zip(strikes, taus)):
            price = call_price_fft(S0, K, r, tau, **self.params)
            price = max(price, 1e-10)
            iv = bsm_iv(price, S0, K, tau, r, "call")
            ivs[i] = iv
        return ivs


def calibrate_heston(
    S0: float,
    r: float,
    strikes: np.ndarray,
    taus: np.ndarray,
    market_ivs: np.ndarray,
    method: str = "de+lm",
) -> HestonCalibResult:
    """Calibrate Heston model to market IV surface.

    Parameters
    ----------
    S0 : float
        Spot price.
    r : float
        Risk-free rate.
    strikes : array
        Option strike prices.
    taus : array
        Time to expiry in years for each option.
    market_ivs : array
        Market implied volatilities.
    method : str
        ``"de+lm"`` (2-stage: Differential Evolution then L-BFGS-B),
        ``"de"`` (global only), or ``"lm"`` (local only).

    Returns
    -------
    HestonCalibResult
    """
    from scipy.optimize import minimize, differential_evolution
    from .iv import implied_vol as bsm_iv

    strikes = np.asarray(strikes, dtype=float)
    taus = np.asarray(taus, dtype=float)
    market_ivs = np.asarray(market_ivs, dtype=float)

    # Filter out NaN IVs
    valid = np.isfinite(market_ivs) & (market_ivs > 0.01) & (market_ivs < 3.0)
    strikes = strikes[valid]
    taus = taus[valid]
    market_ivs = market_ivs[valid]

    if len(strikes) < 3:
        raise ValueError(f"Need at least 3 valid contracts, got {len(strikes)}")

    market_prices = np.array([
        BSM(S=S0, K=K, T=tau, r=r, sigma=iv).call_price()
        for K, tau, iv in zip(strikes, taus, market_ivs)
    ])

    # Group strikes by expiry for batched FFT pricing
    unique_taus = np.unique(taus)
    tau_groups = {t: np.where(taus == t)[0] for t in unique_taus}

    def objective(params):
        V0, kappa, theta, xi, rho = params
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model_prices = np.empty(len(strikes))
                for t, indices in tau_groups.items():
                    grid_strikes, grid_prices = call_prices_fft(
                        S0, r, t, V0, kappa, theta, xi, rho)
                    model_prices[indices] = np.interp(
                        strikes[indices], grid_strikes, grid_prices)
                if not np.all(np.isfinite(model_prices)):
                    return 1e10
                return float(np.mean((model_prices - market_prices) ** 2))
            except Exception:
                return 1e10

    # Bounds: V0>0, kappa>0, theta>0, xi>0, -1<rho<1
    bounds = [
        (0.001, 1.0),   # V0
        (0.1, 20.0),    # kappa
        (0.001, 1.0),   # theta
        (0.01, 2.0),    # xi
        (-0.99, 0.99),  # rho
    ]

    # Stage 1: Global search
    if method in ("de", "de+lm"):
        de_result = differential_evolution(
            objective, bounds, seed=42, maxiter=200,
            tol=1e-10, polish=False,
        )
        x0 = de_result.x
    else:
        x0 = [0.04, 2.0, 0.04, 0.3, -0.7]

    # Stage 2: Local refinement
    if method in ("lm", "de+lm"):
        local_result = minimize(
            objective, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-14},
        )
        x_final = local_result.x
    else:
        x_final = x0

    V0, kappa, theta, xi, rho = x_final
    rmse = float(np.sqrt(objective(x_final)))

    return HestonCalibResult(
        V0=V0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        rmse=rmse, n_contracts=len(strikes),
    )
