"""Hidden Markov Model regime detection for market states.

Fits a 2-state Gaussian HMM to return or volatility series to
identify bull/bear (or low-vol/high-vol) regimes.

Uses ``hmmlearn`` for the core HMM fitting.

References
----------
- Hamilton, J.D. (1989). "A new approach to the economic analysis
  of nonstationary time series and the business cycle."
- Ang, A. & Bekaert, G. (2002). "Regime switches in interest rates."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegimeResult:
    """Container for fitted HMM regime model."""

    n_states: int
    means: np.ndarray
    variances: np.ndarray
    transition_matrix: np.ndarray
    state_sequence: np.ndarray
    state_probs: np.ndarray
    log_likelihood: float
    aic: float
    dates: pd.DatetimeIndex | None

    @property
    def state_labels(self) -> list[str]:
        """Label states by volatility: lower vol = 'CALM', higher = 'STRESS'."""
        order = np.argsort(self.variances)
        labels = [""] * self.n_states
        names = ["CALM", "STRESS"] if self.n_states == 2 else [
            f"STATE_{i}" for i in range(self.n_states)
        ]
        for i, idx in enumerate(order):
            labels[idx] = names[i]
        return labels

    def current_state(self) -> dict:
        """Return the current (last) regime state."""
        idx = int(self.state_sequence[-1])
        labels = self.state_labels
        probs = self.state_probs[-1]
        return {
            "state": idx,
            "label": labels[idx],
            "probability": float(probs[idx]),
            "mean": float(self.means[idx]),
            "volatility": float(np.sqrt(self.variances[idx])),
        }

    def state_durations(self) -> dict[str, float]:
        """Average duration in each state (in periods)."""
        labels = self.state_labels
        durations = {}
        for i in range(self.n_states):
            # Expected duration = 1 / (1 - p_ii)
            p_stay = self.transition_matrix[i, i]
            durations[labels[i]] = 1.0 / max(1 - p_stay, 1e-10)
        return durations

    def to_series(self) -> pd.Series:
        """Return state sequence as a labeled pandas Series."""
        labels = self.state_labels
        mapped = [labels[s] for s in self.state_sequence]
        if self.dates is not None:
            return pd.Series(mapped, index=self.dates, name="regime")
        return pd.Series(mapped, name="regime")

    def prob_series(self) -> pd.DataFrame:
        """Return state probabilities as a DataFrame."""
        labels = self.state_labels
        df = pd.DataFrame(self.state_probs, columns=labels)
        if self.dates is not None:
            df.index = self.dates
        return df


def fit_regime(
    returns: pd.Series,
    n_states: int = 2,
    n_iter: int = 200,
    n_fits: int = 10,
) -> RegimeResult:
    """Fit a Gaussian HMM to a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns (or any 1-D time series).
    n_states : int
        Number of hidden states (default 2: calm/stress).
    n_iter : int
        Maximum EM iterations per fit.
    n_fits : int
        Number of random restarts (best log-likelihood wins).

    Returns
    -------
    RegimeResult
    """
    from hmmlearn.hmm import GaussianHMM

    X = returns.dropna().values.reshape(-1, 1)
    dates = returns.dropna().index if hasattr(returns, "index") else None

    best_model = None
    best_score = -np.inf

    for seed in range(n_fits):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=seed,
        )
        try:
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("HMM fitting failed on all random restarts")

    states = best_model.predict(X)
    probs = best_model.predict_proba(X)

    means = best_model.means_.flatten()
    variances = best_model.covars_.flatten()
    transmat = best_model.transmat_

    n_params = n_states ** 2 + 2 * n_states - 1  # transition + mean + var
    aic = -2 * best_score + 2 * n_params

    return RegimeResult(
        n_states=n_states,
        means=means,
        variances=variances,
        transition_matrix=transmat,
        state_sequence=states,
        state_probs=probs,
        log_likelihood=best_score,
        aic=aic,
        dates=dates,
    )


def regime_adjusted_thresholds(
    regime_result: RegimeResult,
    base_entry: float = 2.0,
    base_exit: float = 0.5,
    base_stop: float = 4.0,
) -> dict:
    """Compute regime-dependent z-score thresholds.

    In STRESS regime, widen thresholds (higher entry, wider stop).
    In CALM regime, tighten thresholds (lower entry, tighter stop).

    Returns
    -------
    dict
        Keys: ``entry``, ``exit``, ``stop``, ``regime``.
    """
    current = regime_result.current_state()
    label = current["label"]

    if label == "STRESS":
        # Widen thresholds in high-vol regime
        scale = 1.5
    else:
        # Tighter thresholds in calm regime
        scale = 0.8

    return {
        "entry": base_entry * scale,
        "exit": base_exit,
        "stop": base_stop * scale,
        "regime": label,
        "regime_prob": current["probability"],
    }
