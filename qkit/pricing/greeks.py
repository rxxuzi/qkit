"""Greeks grid computation and interactive visualisation with plotly.

Computes all five first-order Greeks over a strike * expiry grid and
offers heatmaps, 3-D surfaces, a six-panel dashboard and a payoff
diagram for arbitrary multi-leg strategies.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.stats import norm

from .bsm import BSM

GREEK_LABELS = {
    "delta": ("Delta", "Price change per $1 underlying move"),
    "gamma": ("Gamma", "Rate of change of delta"),
    "vega":  ("Vega",  "Price change per 1 pp vol move"),
    "theta": ("Theta", "Time decay per calendar day"),
    "rho":   ("Rho",   "Price change per 1 pp rate move"),
}


class Greeks:
    """Pre-compute a Greeks grid and expose visualisation helpers.

    Parameters
    ----------
    S, r, sigma : float
        BSM parameters shared across the grid.
    strike_range : tuple[float, float]
        Strike bounds expressed as fractions of *S*.
    strike_steps, expiry_steps : int
        Grid resolution.
    expiry_range : tuple[int, int]
        Expiry bounds in calendar days.
    """

    def __init__(self, S: float, r: float = 0.043, sigma: float = 0.22,
                 strike_range: tuple[float, float] = (0.80, 1.20),
                 strike_steps: int = 50,
                 expiry_range: tuple[int, int] = (7, 180),
                 expiry_steps: int = 50):
        self.S = S
        self.r = r
        self.sigma = sigma
        self.strikes = np.linspace(S * strike_range[0], S * strike_range[1],
                                   strike_steps)
        self.expiry_days = np.linspace(expiry_range[0], expiry_range[1],
                                      expiry_steps)
        self.expiry_years = self.expiry_days / 365.0

        self._grids: dict[str, dict[str, np.ndarray]] = {}
        self._compute()

    def _compute(self):
        S = self.S
        r = self.r
        sigma = self.sigma

        # 2-D grids: rows = expiry, cols = strike
        K = self.strikes[np.newaxis, :]          # (1, nk)
        T = self.expiry_years[:, np.newaxis]     # (nt, 1)

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)
        Nnd1 = norm.cdf(-d1)
        Nnd2 = norm.cdf(-d2)
        nd1 = norm.pdf(d1)
        discount = np.exp(-r * T)

        # Shared Greeks (same for call and put)
        gamma = nd1 / (S * sigma * sqrt_T)
        vega = S * nd1 * sqrt_T / 100
        theta_term1 = -(S * nd1 * sigma) / (2 * sqrt_T)

        # Call
        call_price = S * Nd1 - K * discount * Nd2
        call_delta = Nd1
        call_theta = (theta_term1 - r * K * discount * Nd2) / 365
        call_rho = K * T * discount * Nd2 / 100

        self._grids["call"] = {
            "delta": call_delta, "gamma": gamma, "vega": vega,
            "theta": call_theta, "rho": call_rho, "price": call_price,
        }

        # Put
        put_price = K * discount * Nnd2 - S * Nnd1
        put_delta = Nd1 - 1
        put_theta = (theta_term1 + r * K * discount * Nnd2) / 365
        put_rho = -K * T * discount * Nnd2 / 100

        self._grids["put"] = {
            "delta": put_delta, "gamma": gamma, "vega": vega,
            "theta": put_theta, "rho": put_rho, "price": put_price,
        }

    def get_grid(self, greek: str, option_type: str = "call") -> np.ndarray:
        """Return the pre-computed 2-D array for *greek*."""
        return self._grids[option_type][greek]

    def heatmap(self, greek: str = "delta", option_type: str = "call",
                colorscale: str = "RdBu_r") -> go.Figure:
        """Single-Greek heatmap (plotly)."""
        Z = self._grids[option_type][greek]
        label = GREEK_LABELS.get(greek, (greek.title(), ""))[0]

        fig = go.Figure(data=go.Heatmap(
            z=Z, x=np.round(self.strikes, 1),
            y=np.round(self.expiry_days).astype(int),
            colorscale=colorscale, colorbar=dict(title=label),
        ))
        fig.update_layout(
            title=f"{option_type.title()} {label}  S=${self.S:.1f}",
            xaxis_title="Strike ($)", yaxis_title="Days to Expiry",
            width=900, height=600,
        )
        fig.add_vline(x=self.S, line_dash="dash", line_color="gray",
                      annotation_text="ATM")
        return fig

    def surface_3d(self, greek: str = "delta", option_type: str = "call",
                   colorscale: str = "Viridis") -> go.Figure:
        """3-D surface plot (plotly)."""
        Z = self._grids[option_type][greek]
        label = GREEK_LABELS.get(greek, (greek.title(), ""))[0]

        fig = go.Figure(data=go.Surface(
            z=Z, x=self.strikes, y=self.expiry_days,
            colorscale=colorscale, colorbar=dict(title=label),
        ))
        fig.update_layout(
            title=f"{option_type.title()} {label} Surface",
            scene=dict(xaxis_title="Strike ($)",
                       yaxis_title="Days", zaxis_title=label),
            width=900, height=700,
        )
        return fig

    def dashboard(self, option_type: str = "call") -> go.Figure:
        """Six-panel dashboard with all Greeks and price."""
        names = list(GREEK_LABELS) + ["price"]
        fig = make_subplots(rows=2, cols=3,
                            subplot_titles=[GREEK_LABELS.get(n, (n,))[0]
                                            for n in names])
        for idx, name in enumerate(names):
            r, c = divmod(idx, 3)
            fig.add_trace(go.Heatmap(
                z=self._grids[option_type][name],
                x=np.round(self.strikes, 1),
                y=np.round(self.expiry_days).astype(int),
                colorscale="RdBu_r" if name != "price" else "Viridis",
                showscale=False,
            ), row=r + 1, col=c + 1)
        fig.update_layout(
            title=f"{option_type.title()} Greeks Dashboard  S=${self.S:.1f}",
            height=800, width=1200,
        )
        return fig

    def payoff_diagram(self, positions: list[dict]) -> go.Figure:
        """Multi-leg payoff diagram at expiry.

        Each element of *positions* is a dict with keys ``type``
        (``"call"``/``"put"``), ``strike``, ``premium`` and ``qty``
        (+1 = long, -1 = short).
        """
        S_range = np.linspace(self.S * 0.7, self.S * 1.3, 300)
        total = np.zeros_like(S_range)
        fig = go.Figure()

        for pos in positions:
            K, prem, qty = pos["strike"], pos["premium"], pos["qty"]
            if pos["type"] == "call":
                pnl = (np.maximum(S_range - K, 0) - prem) * qty
            else:
                pnl = (np.maximum(K - S_range, 0) - prem) * qty
            total += pnl
            label = f"{'Long' if qty > 0 else 'Short'} {pos['type'].title()} K={K}"
            fig.add_trace(go.Scatter(x=S_range, y=pnl, mode="lines",
                                     name=label, line=dict(dash="dot", width=1),
                                     opacity=0.5))

        fig.add_trace(go.Scatter(x=S_range, y=total, mode="lines",
                                 name="Combined P&L",
                                 line=dict(color="white", width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=self.S, line_dash="dash", line_color="yellow",
                      annotation_text=f"Spot ${self.S:.1f}")
        fig.update_layout(title="Payoff at Expiry",
                          xaxis_title="Stock Price ($)",
                          yaxis_title="P&L ($)",
                          template="plotly_dark", width=900, height=500)
        return fig
