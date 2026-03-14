"""Spectral analysis and Kalman filtering for financial time series.

Provides FFT-based power spectral density estimation, wavelet
decomposition, and a linear Kalman filter for dynamic parameter
estimation (e.g., time-varying hedge ratios).

References
----------
- Granger, C.W.J. (1966). "The typical spectral shape of an
  economic variable."
- Hamilton, J.D. (1994). "Time Series Analysis", Ch. 13.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SpectralResult:
    """Container for spectral analysis output."""

    frequencies: np.ndarray
    power: np.ndarray
    dominant_period: float
    dominant_frequency: float

    def top_periods(self, n: int = 5) -> list[dict]:
        """Return top-n dominant periods sorted by power."""
        # Skip DC component (index 0)
        idx = np.argsort(self.power[1:])[::-1][:n] + 1
        results = []
        for i in idx:
            if self.frequencies[i] > 0:
                results.append({
                    "frequency": float(self.frequencies[i]),
                    "period": float(1.0 / self.frequencies[i]),
                    "power": float(self.power[i]),
                })
        return results


def power_spectrum(
    series: pd.Series,
    detrend: bool = True,
    window: str = "hann",
) -> SpectralResult:
    """Compute power spectral density via FFT.

    Parameters
    ----------
    series : pd.Series
        Input time series (e.g., log returns, spreads).
    detrend : bool
        Remove linear trend before FFT.
    window : str
        Window function (``"hann"``, ``"hamming"``, ``"blackman"``, ``"none"``).

    Returns
    -------
    SpectralResult
    """
    y = series.dropna().values.astype(float)
    n = len(y)

    if detrend:
        t = np.arange(n)
        coeffs = np.polyfit(t, y, 1)
        y = y - np.polyval(coeffs, t)

    # Apply window
    if window == "hann":
        w = np.hanning(n)
    elif window == "hamming":
        w = np.hamming(n)
    elif window == "blackman":
        w = np.blackman(n)
    else:
        w = np.ones(n)

    y_windowed = y * w

    # FFT
    fft_vals = np.fft.rfft(y_windowed)
    power = np.abs(fft_vals) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=1.0)  # cycles per sample (trading day)

    # Dominant frequency (skip DC)
    if len(power) > 1:
        dom_idx = np.argmax(power[1:]) + 1
        dom_freq = freqs[dom_idx]
        dom_period = 1.0 / dom_freq if dom_freq > 0 else np.inf
    else:
        dom_freq = 0.0
        dom_period = np.inf

    return SpectralResult(
        frequencies=freqs,
        power=power,
        dominant_period=dom_period,
        dominant_frequency=dom_freq,
    )


@dataclass
class WaveletResult:
    """Container for wavelet decomposition output."""

    levels: int
    detail_coeffs: list[np.ndarray]
    approx_coeffs: np.ndarray
    detail_energy: list[float]
    total_energy: float

    def energy_by_scale(self) -> list[dict]:
        """Return energy contribution by wavelet scale."""
        results = []
        for i, e in enumerate(self.detail_energy):
            # Scale i corresponds to periods 2^(i+1) trading days
            period = 2 ** (i + 1)
            results.append({
                "level": i + 1,
                "period_days": period,
                "energy": e,
                "pct_total": e / max(self.total_energy, 1e-15) * 100,
            })
        return results


def wavelet_decompose(
    series: pd.Series,
    wavelet: str = "db4",
    levels: int | None = None,
) -> WaveletResult:
    """Multi-resolution wavelet decomposition.

    Parameters
    ----------
    series : pd.Series
        Input time series.
    wavelet : str
        Wavelet type (``"db4"``, ``"haar"``, ``"sym4"``).
    levels : int or None
        Number of decomposition levels. Auto-determined if None.

    Returns
    -------
    WaveletResult
    """
    import pywt

    y = series.dropna().values.astype(float)

    if levels is None:
        levels = min(pywt.dwt_max_level(len(y), wavelet), 8)

    coeffs = pywt.wavedec(y, wavelet, level=levels)
    approx = coeffs[0]
    details = coeffs[1:]

    detail_energy = [float(np.sum(d ** 2)) for d in details]
    total_energy = float(np.sum(y ** 2))

    return WaveletResult(
        levels=levels,
        detail_coeffs=details,
        approx_coeffs=approx,
        detail_energy=detail_energy,
        total_energy=total_energy,
    )


@dataclass
class KalmanResult:
    """Container for Kalman filter output."""

    states: np.ndarray
    covariances: np.ndarray
    log_likelihood: float
    dates: pd.DatetimeIndex | None

    def state_series(self, name: str = "beta") -> pd.Series:
        """Return filtered state as a pandas Series.

        If *name* is ``"beta"`` returns column 1 (slope),
        if ``"alpha"`` returns column 0 (intercept).
        """
        col = 1 if name == "beta" else 0
        vals = self.states[:, col] if self.states.ndim == 2 else self.states
        s = pd.Series(vals, name=name)
        if self.dates is not None:
            s.index = self.dates
        return s


def kalman_regression(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    Ve: float = 1e-3,
) -> KalmanResult:
    """Kalman filter for time-varying linear regression (dynamic beta).

    Estimates y_t = alpha_t + beta_t * x_t with random-walk state dynamics.

    Parameters
    ----------
    y : pd.Series
        Dependent variable (e.g., asset B prices).
    x : pd.Series
        Independent variable (e.g., asset A prices).
    delta : float
        State noise scaling (controls how fast beta changes).
    Ve : float
        Observation noise variance.

    Returns
    -------
    KalmanResult
    """
    idx = y.index.intersection(x.index)
    y_vals = y.loc[idx].values
    x_vals = x.loc[idx].values
    n = len(y_vals)
    dates = idx

    # State: [alpha, beta]
    # Observation: y_t = [1, x_t] * [alpha, beta]'
    n_states = 2
    beta = np.zeros((n, n_states))
    P = np.zeros((n, n_states, n_states))
    e = np.zeros(n)

    # Initialise
    beta[0] = [0.0, 0.0]
    P[0] = np.eye(n_states) * 1.0
    Vw = delta / (1 - delta) * np.eye(n_states)

    ll = 0.0

    for t in range(1, n):
        # Predict
        beta_pred = beta[t - 1]
        P_pred = P[t - 1] + Vw

        # Observe
        F = np.array([1.0, x_vals[t]])
        y_pred = F @ beta_pred
        e[t] = y_vals[t] - y_pred

        # Innovation variance
        S = F @ P_pred @ F + Ve

        # Kalman gain
        K = P_pred @ F / S

        # Update
        beta[t] = beta_pred + K * e[t]
        P[t] = P_pred - np.outer(K, K) * S

        # Log-likelihood
        ll += -0.5 * (np.log(2 * np.pi * S) + e[t] ** 2 / S)

    return KalmanResult(
        states=beta,
        covariances=P,
        log_likelihood=ll,
        dates=dates,
    )
