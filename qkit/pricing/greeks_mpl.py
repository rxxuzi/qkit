"""Static Greeks charts via matplotlib, saved as PNG to ``out/charts/``.

Parallel to :mod:`qkit.pricing.greeks` (plotly/interactive), this module
produces publication-quality static images suitable for reports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

from .bsm import BSM

_OUT = Path(__file__).resolve().parent.parent.parent / "out" / "charts"
_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#0f1117",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#ccc",
    "text.color": "#ccc",
    "xtick.color": "#999",
    "ytick.color": "#999",
    "grid.color": "#222",
    "grid.alpha": 0.5,
    "font.size": 11,
    "figure.dpi": 150,
})

_GREEKS = {
    "delta": {"cmap": "RdBu_r", "label": "Delta"},
    "gamma": {"cmap": "YlOrRd", "label": "Gamma"},
    "vega": {"cmap": "PuBuGn", "label": "Vega (per 1 pp)"},
    "theta": {"cmap": "RdPu", "label": "Theta (per day)"},
    "rho": {"cmap": "BrBG", "label": "Rho (per 1 pp)"},
    "price": {"cmap": "viridis", "label": "Price ($)"},
}


class GreeksMpl:
    """Matplotlib-based Greeks heatmap generator.

    Parameters
    ----------
    S, r, sigma : float
        BSM parameters.
    strike_range : tuple[float, float]
        Strike bounds as fractions of *S*.
    strike_steps, expiry_steps : int
        Grid resolution.
    expiry_range : tuple[int, int]
        Expiry bounds in calendar days.
    """

    def __init__(self, S: float, r: float = 0.043, sigma: float = 0.22,
                 strike_range: tuple[float, float] = (0.80, 1.20),
                 strike_steps: int = 60,
                 expiry_range: tuple[int, int] = (7, 180),
                 expiry_steps: int = 60):
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
        nk, nt = len(self.strikes), len(self.expiry_years)
        for opt in ("call", "put"):
            g = {k: np.zeros((nt, nk)) for k in _GREEKS}
            for i, T in enumerate(self.expiry_years):
                for j, K in enumerate(self.strikes):
                    m = BSM(S=self.S, K=K, T=T, r=self.r, sigma=self.sigma)
                    gr = m.call_greeks() if opt == "call" else m.put_greeks()
                    g["delta"][i, j] = gr.delta
                    g["gamma"][i, j] = gr.gamma
                    g["vega"][i, j] = gr.vega
                    g["theta"][i, j] = gr.theta
                    g["rho"][i, j] = gr.rho
                    g["price"][i, j] = (m.call_price() if opt == "call"
                                        else m.put_price())
            self._grids[opt] = g

    def heatmap(self, greek: str = "delta", option_type: str = "call",
                save: bool = True, prefix: str = "") -> plt.Figure:
        """Render a single-Greek heatmap and optionally save to disk."""
        Z = self._grids[option_type][greek]
        meta = _GREEKS[greek]

        fig, ax = plt.subplots(figsize=(10, 6))
        norm = (TwoSlopeNorm(vmin=Z.min(), vcenter=0, vmax=Z.max())
                if greek == "delta" and Z.min() < 0 < Z.max() else None)

        im = ax.pcolormesh(self.strikes, self.expiry_days, Z,
                           cmap=meta["cmap"], norm=norm, shading="auto")
        fig.colorbar(im, ax=ax, pad=0.02).set_label(meta["label"])

        ax.axvline(self.S, color="#ffcc00", ls="--", lw=1, alpha=0.7)
        ax.set_xlabel("Strike ($)")
        ax.set_ylabel("Days to Expiry")
        ax.set_title(f"{option_type.title()} {meta['label']}  "
                     f"S=${self.S:.0f}  sigma={self.sigma:.0%}")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
        fig.tight_layout()

        if save:
            path = _OUT / f"{prefix + '_' if prefix else ''}{option_type}_{greek}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  -> {path}")
            plt.close(fig)
        return fig

    def dashboard(self, option_type: str = "call", save: bool = True,
                  prefix: str = "") -> plt.Figure:
        """Six-panel dashboard with all Greeks and price."""
        names = list(_GREEKS)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for idx, (name, ax) in enumerate(zip(names, axes.flatten())):
            Z = self._grids[option_type][name]
            meta = _GREEKS[name]
            norm = (TwoSlopeNorm(vmin=Z.min(), vcenter=0, vmax=Z.max())
                    if name == "delta" and Z.min() < 0 < Z.max() else None)
            im = ax.pcolormesh(self.strikes, self.expiry_days, Z,
                               cmap=meta["cmap"], norm=norm, shading="auto")
            fig.colorbar(im, ax=ax, pad=0.02)
            ax.axvline(self.S, color="#ffcc00", ls="--", lw=0.8, alpha=0.6)
            ax.set_title(meta["label"], fontweight="bold")
            ax.set_xlabel("Strike ($)", fontsize=9)
            ax.set_ylabel("Days", fontsize=9)

        fig.suptitle(f"{option_type.title()} Greeks  S=${self.S:.0f}  "
                     f"sigma={self.sigma:.0%}", fontsize=15, fontweight="bold")
        fig.tight_layout()

        if save:
            path = _OUT / f"{prefix + '_' if prefix else ''}{option_type}_dashboard.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  -> {path}")
            plt.close(fig)
        return fig

    def payoff(self, positions: list[dict], save: bool = True,
               prefix: str = "") -> plt.Figure:
        """Multi-leg payoff diagram at expiry.

        *positions* is a list of dicts with keys ``type``, ``strike``,
        ``premium`` and ``qty``.
        """
        S_range = np.linspace(self.S * 0.7, self.S * 1.3, 300)
        total = np.zeros_like(S_range)
        fig, ax = plt.subplots(figsize=(10, 5))

        for pos in positions:
            K, prem, qty = pos["strike"], pos["premium"], pos["qty"]
            if pos["type"] == "call":
                pnl = (np.maximum(S_range - K, 0) - prem) * qty
            else:
                pnl = (np.maximum(K - S_range, 0) - prem) * qty
            total += pnl
            label = f"{'Long' if qty > 0 else 'Short'} {pos['type'].title()} K={K}"
            ax.plot(S_range, pnl, ls="--", lw=1, alpha=0.4, label=label)

        ax.plot(S_range, total, color="#00d2ff", lw=2.5, label="Combined P&L")
        ax.axhline(0, color="#555", lw=0.8)
        ax.axvline(self.S, color="#ffcc00", ls="--", lw=1, alpha=0.6)
        ax.fill_between(S_range, total, 0, where=total > 0,
                        color="#00d2ff", alpha=0.08)
        ax.fill_between(S_range, total, 0, where=total < 0,
                        color="#ff4444", alpha=0.08)
        ax.set_xlabel("Stock Price at Expiry ($)")
        ax.set_ylabel("P&L ($)")
        ax.set_title("Payoff at Expiry", fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save:
            path = _OUT / f"{prefix + '_' if prefix else ''}payoff.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  -> {path}")
            plt.close(fig)
        return fig

    def save_all(self, prefix: str = ""):
        """Write all heatmaps, dashboards and a sample payoff to disk."""
        print(f"\n  Generating charts  S=${self.S:.0f}  sigma={self.sigma:.0%}")

        for opt in ("call", "put"):
            print(f"\n  [{opt.upper()}]")
            for greek in _GREEKS:
                self.heatmap(greek, opt, prefix=prefix)
            self.dashboard(opt, prefix=prefix)

        m = BSM(S=self.S, K=self.S, T=30 / 365, r=self.r, sigma=self.sigma)
        print("\n  [PAYOFF]")
        self.payoff([
            {"type": "call", "strike": self.S, "premium": m.call_price(), "qty": 1},
            {"type": "put", "strike": self.S, "premium": m.put_price(), "qty": 1},
        ], prefix=prefix)
        print()


if __name__ == "__main__":
    GreeksMpl(S=230.0, r=0.043, sigma=0.22).save_all(prefix="AAPL")
