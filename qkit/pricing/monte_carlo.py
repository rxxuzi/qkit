"""Monte Carlo simulation for option pricing.

Supports GBM and Heston dynamics with antithetic variates,
control variates (BSM) and full truncation discretisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bsm import BSM


@dataclass
class MCResult:
    """Container for Monte Carlo pricing output."""

    price: float
    std_error: float
    n_paths: int
    ci_lower: float
    ci_upper: float

    def __repr__(self) -> str:
        return (f"MCResult(price={self.price:.4f}, se={self.std_error:.4f}, "
                f"95%CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}])")


def price_gbm(S0: float, K: float, r: float, sigma: float, tau: float,
              n_paths: int = 100_000, antithetic: bool = True,
              control_variate: bool = False) -> MCResult:
    """European call price under GBM via Monte Carlo.

    Parameters
    ----------
    antithetic : bool
        Use antithetic variates (halves variance at no extra cost).
    control_variate : bool
        Use BSM analytical price as a control variate.
    """
    Z = np.random.standard_normal(n_paths)

    def simulate(z):
        ST = S0 * np.exp((r - 0.5 * sigma ** 2) * tau + sigma * np.sqrt(tau) * z)
        return np.exp(-r * tau) * np.maximum(ST - K, 0)

    payoffs = simulate(Z)

    if antithetic:
        payoffs_neg = simulate(-Z)
        payoffs = 0.5 * (payoffs + payoffs_neg)

    if control_variate:
        bsm_exact = BSM(S=S0, K=K, T=tau, r=r, sigma=sigma).call_price()
        bsm_mc = payoffs.copy()
        beta = np.cov(payoffs, bsm_mc)[0, 1] / np.var(bsm_mc)
        payoffs = payoffs - beta * (bsm_mc - bsm_exact)

    return _build_result(payoffs, n_paths)


def price_heston(S0: float, K: float, r: float, tau: float,
                 V0: float, kappa: float, theta: float,
                 xi: float, rho: float,
                 n_paths: int = 100_000, n_steps: int = 252,
                 antithetic: bool = True) -> MCResult:
    """European call price under Heston via Monte Carlo.

    Uses full truncation discretisation (Lord et al. 2010).
    """
    dt = tau / n_steps
    sqrt_dt = np.sqrt(dt)

    S = np.full(n_paths, float(S0))
    V = np.full(n_paths, float(V0))

    if antithetic:
        S2 = np.full(n_paths, float(S0))
        V2 = np.full(n_paths, float(V0))

    rho_comp = np.sqrt(1 - rho ** 2)

    for _ in range(n_steps):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = rho * Z1 + rho_comp * np.random.standard_normal(n_paths)

        Vp = np.maximum(V, 0)
        sqrt_Vp = np.sqrt(Vp)
        sqrt_Vp_dt = sqrt_Vp * sqrt_dt
        S = S * np.exp((r - 0.5 * Vp) * dt + sqrt_Vp_dt * Z1)
        V = V + kappa * (theta - Vp) * dt + xi * sqrt_Vp_dt * Z2

        if antithetic:
            Vp2 = np.maximum(V2, 0)
            sqrt_Vp2_dt = np.sqrt(Vp2) * sqrt_dt
            S2 = S2 * np.exp((r - 0.5 * Vp2) * dt + sqrt_Vp2_dt * (-Z1))
            V2 = V2 + kappa * (theta - Vp2) * dt + xi * sqrt_Vp2_dt * (-Z2)

    payoffs = np.exp(-r * tau) * np.maximum(S - K, 0)

    if antithetic:
        payoffs2 = np.exp(-r * tau) * np.maximum(S2 - K, 0)
        payoffs = 0.5 * (payoffs + payoffs2)

    return _build_result(payoffs, n_paths)


def price_asian(S0: float, K: float, r: float, sigma: float, tau: float,
                n_paths: int = 100_000, n_steps: int = 252) -> MCResult:
    """Arithmetic average Asian call option via Monte Carlo."""
    dt = tau / n_steps
    sqrt_dt = np.sqrt(dt)
    drift = (r - 0.5 * sigma ** 2) * dt
    vol_sqrt_dt = sigma * sqrt_dt
    paths = np.zeros((n_paths, n_steps))
    S = np.full(n_paths, float(S0))

    for i in range(n_steps):
        Z = np.random.standard_normal(n_paths)
        S = S * np.exp(drift + vol_sqrt_dt * Z)
        paths[:, i] = S

    avg = paths.mean(axis=1)
    payoffs = np.exp(-r * tau) * np.maximum(avg - K, 0)
    return _build_result(payoffs, n_paths)


def price_barrier(S0: float, K: float, B: float, r: float, sigma: float,
                  tau: float, barrier_type: str = "down-and-out",
                  n_paths: int = 100_000, n_steps: int = 252) -> MCResult:
    """Barrier option (down-and-out call) via Monte Carlo.

    Applies Broadie-Glasserman-Kou (1997) continuity correction.
    """
    dt = tau / n_steps
    sqrt_dt = np.sqrt(dt)
    beta = 0.5826
    B_adj = B * np.exp(beta * sigma * sqrt_dt)
    drift = (r - 0.5 * sigma ** 2) * dt
    vol_sqrt_dt = sigma * sqrt_dt

    S = np.full(n_paths, float(S0))
    alive = np.ones(n_paths, dtype=bool)

    for _ in range(n_steps):
        Z = np.random.standard_normal(n_paths)
        S = S * np.exp(drift + vol_sqrt_dt * Z)

        if "down" in barrier_type:
            alive &= (S > B_adj)
        else:
            alive &= (S < B_adj)

    payoffs = np.exp(-r * tau) * np.maximum(S - K, 0) * alive

    if "in" in barrier_type:
        vanilla = np.exp(-r * tau) * np.maximum(S - K, 0)
        payoffs = vanilla - payoffs

    return _build_result(payoffs, n_paths)


def price_heston_qe(S0: float, K: float, r: float, tau: float,
                    V0: float, kappa: float, theta: float,
                    xi: float, rho: float,
                    n_paths: int = 100_000, n_steps: int = 252,
                    antithetic: bool = True) -> MCResult:
    """European call under Heston via Andersen (2008) QE scheme.

    The Quadratic Exponential (QE) scheme avoids negative variances
    without simple truncation.  Uses a critical threshold psi_c = 1.5
    to switch between the exponential and quadratic approximations.
    """
    dt = tau / n_steps
    sqrt_dt = np.sqrt(dt)
    psi_c = 1.5

    exp_kdt = np.exp(-kappa * dt)
    k0 = -rho * kappa * theta * dt / xi
    k1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = 0.5 * dt * (1 - rho ** 2)

    S = np.full(n_paths, float(S0))
    V = np.full(n_paths, float(V0))

    if antithetic:
        S2 = np.full(n_paths, float(S0))
        V2 = np.full(n_paths, float(V0))

    for _ in range(n_steps):
        Uv = np.random.uniform(size=n_paths)
        Z1 = np.random.standard_normal(n_paths)

        V_next = _qe_step(V, kappa, theta, xi, dt, exp_kdt, psi_c, Uv)
        log_S = np.log(S) + k0 + k1 * V + k2 * V_next + np.sqrt(k3 * (V + V_next)) * Z1
        S = np.exp(log_S + r * dt - 0.5 * (V + V_next) * 0.5 * dt)

        # Simpler log-Euler for S using QE variance
        S = np.exp(np.log(S))  # ensure positive
        V = V_next

        if antithetic:
            V2_next = _qe_step(V2, kappa, theta, xi, dt, exp_kdt, psi_c, 1 - Uv)
            log_S2 = np.log(S2) + k0 + k1 * V2 + k2 * V2_next + np.sqrt(k3 * (V2 + V2_next)) * (-Z1)
            S2 = np.exp(log_S2 + r * dt - 0.5 * (V2 + V2_next) * 0.5 * dt)
            S2 = np.exp(np.log(S2))
            V2 = V2_next

    payoffs = np.exp(-r * tau) * np.maximum(S - K, 0)
    if antithetic:
        payoffs2 = np.exp(-r * tau) * np.maximum(S2 - K, 0)
        payoffs = 0.5 * (payoffs + payoffs2)

    return _build_result(payoffs, n_paths)


def _qe_step(V, kappa, theta, xi, dt, exp_kdt, psi_c, U):
    """Single QE variance step (Andersen 2008)."""
    m = theta + (V - theta) * exp_kdt
    s2 = (V * xi ** 2 * exp_kdt / kappa * (1 - exp_kdt)
          + theta * xi ** 2 / (2 * kappa) * (1 - exp_kdt) ** 2)
    s2 = np.maximum(s2, 0)
    psi = s2 / np.maximum(m ** 2, 1e-15)

    V_next = np.empty_like(V)

    # Quadratic regime (psi <= psi_c)
    quad_mask = psi <= psi_c
    if np.any(quad_mask):
        b2 = 2 / np.maximum(psi[quad_mask], 1e-15) - 1 + np.sqrt(
            2 / np.maximum(psi[quad_mask], 1e-15)
        ) * np.sqrt(np.maximum(2 / np.maximum(psi[quad_mask], 1e-15) - 1, 0))
        a_q = m[quad_mask] / (1 + b2)
        from scipy.stats import norm as norm_dist
        Zv = norm_dist.ppf(np.clip(U[quad_mask], 1e-10, 1 - 1e-10))
        V_next[quad_mask] = a_q * (np.sqrt(b2) + Zv) ** 2

    # Exponential regime (psi > psi_c)
    exp_mask = ~quad_mask
    if np.any(exp_mask):
        p = (psi[exp_mask] - 1) / (psi[exp_mask] + 1)
        beta = (1 - p) / np.maximum(m[exp_mask], 1e-15)
        V_next[exp_mask] = np.where(
            U[exp_mask] <= p, 0.0,
            np.log((1 - p) / np.maximum(1 - U[exp_mask], 1e-15)) / beta
        )

    return np.maximum(V_next, 0)


def price_gbm_sobol(S0: float, K: float, r: float, sigma: float, tau: float,
                    n_paths: int = 2 ** 14) -> MCResult:
    """European call under GBM using Sobol quasi-random sequences.

    Quasi-Monte Carlo generally converges faster (O(1/N) vs O(1/sqrt(N)))
    than pseudo-random MC for smooth integrands.
    """
    from scipy.stats.qmc import Sobol

    # n_paths must be power of 2 for Sobol
    m = int(np.ceil(np.log2(max(n_paths, 4))))
    n = 2 ** m

    sampler = Sobol(d=1, scramble=True, seed=42)
    U = sampler.random(n).flatten()
    from scipy.stats import norm as norm_dist
    Z = norm_dist.ppf(np.clip(U, 1e-10, 1 - 1e-10))

    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * tau + sigma * np.sqrt(tau) * Z)
    payoffs = np.exp(-r * tau) * np.maximum(ST - K, 0)

    return _build_result(payoffs, n)


def cos_price(S0: float, K: float, r: float, sigma: float, tau: float,
              N: int = 256) -> float:
    """European call via COS method (Fang & Oosterlee 2008).

    Uses the characteristic function of GBM log-returns and a cosine
    expansion to recover option prices with spectral convergence.

    The COS method represents the density via cosine expansion on [a, b]:
        f(y) ≈ sum_k Re[phi(k*pi/(b-a)) * exp(-ik*a*pi/(b-a))] * cos(k*pi*(y-a)/(b-a))

    where y = ln(S_T/K) and phi is the characteristic function of y.
    """
    # Cumulants of y = ln(S_T/S_0)
    c1 = (r - 0.5 * sigma ** 2) * tau
    c2 = sigma ** 2 * tau
    L = 10
    a = c1 - L * np.sqrt(c2)
    b = c1 + L * np.sqrt(c2)

    x = np.log(S0 / K)  # ln(S_0/K)
    ba = b - a

    k = np.arange(N)
    omega = k * np.pi / ba

    # Characteristic function of y = ln(S_T/S_0):
    #   phi_y(u) = exp(i*u*c1 - 0.5*c2*u^2)
    # Then phi of ln(S_T/K) = exp(i*u*x) * phi_y(u)
    cf = np.exp(1j * omega * x) * np.exp(1j * omega * c1 - 0.5 * c2 * omega ** 2)

    # Call payoff coefficients: V_k = 2/(b-a) * K * (chi_k - psi_k)
    # where chi/psi are integrals over [0, b] (call payoff region)
    chi = _cos_chi_vec(omega, a, 0, b)
    psi = _cos_psi_vec(omega, a, 0, b)

    V_k = 2.0 / ba * K * (chi - psi)

    # Sum with half-first-term rule
    terms = np.real(cf * np.exp(-1j * omega * a)) * V_k
    terms[0] *= 0.5

    price = np.exp(-r * tau) * np.sum(terms)
    return max(float(price), 0.0)


def _cos_chi_vec(omega, a, c, d):
    """Vectorised chi coefficient: integral of e^y cos(omega*(y-a)) dy on [c,d]."""
    w = np.asarray(omega, dtype=float)
    singular = np.abs(w) < 1e-15

    exp_d, exp_c = np.exp(d), np.exp(c)
    denom = 1 + w ** 2

    chi = (exp_d * (w * np.sin(w * (d - a)) + np.cos(w * (d - a)))
           - exp_c * (w * np.sin(w * (c - a)) + np.cos(w * (c - a)))) / denom

    if np.any(singular):
        chi[singular] = exp_d - exp_c
    return chi


def _cos_psi_vec(omega, a, c, d):
    """Vectorised psi coefficient: integral of cos(omega*(y-a)) dy on [c,d]."""
    w = np.asarray(omega, dtype=float)
    singular = np.abs(w) < 1e-15

    # Use safe divisor to avoid divide-by-zero, then fix singular entries
    w_safe = np.where(singular, 1.0, w)
    psi = (np.sin(w_safe * (d - a)) - np.sin(w_safe * (c - a))) / w_safe

    if np.any(singular):
        psi[singular] = d - c
    return psi


def _build_result(payoffs: np.ndarray, n_paths: int) -> MCResult:
    mean = float(np.mean(payoffs))
    se = float(np.std(payoffs, ddof=1) / np.sqrt(n_paths))
    return MCResult(
        price=mean, std_error=se, n_paths=n_paths,
        ci_lower=mean - 1.96 * se, ci_upper=mean + 1.96 * se,
    )
