"""Portfolio Greeks aggregation and hedging calculations.

Aggregates Greeks across multiple option positions and computes
the hedge ratios required for delta-neutral, delta-gamma-neutral,
and delta-gamma-vega-neutral portfolios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qkit.pricing.bsm import BSM


@dataclass
class Position:
    """A single option or stock position."""

    instrument: str          # "call", "put", or "stock"
    qty: int                 # positive = long, negative = short
    S: float = 0.0
    K: float = 0.0
    T: float = 0.0
    r: float = 0.043
    sigma: float = 0.22
    multiplier: int = 100    # option contract multiplier

    def greeks(self) -> dict:
        if self.instrument == "stock":
            return {"delta": self.qty, "gamma": 0.0, "vega": 0.0,
                    "theta": 0.0, "rho": 0.0}
        m = BSM(S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma)
        g = m.call_greeks() if self.instrument == "call" else m.put_greeks()
        n = self.qty * self.multiplier
        return {"delta": n * g.delta, "gamma": n * g.gamma,
                "vega": n * g.vega, "theta": n * g.theta, "rho": n * g.rho}


@dataclass
class Portfolio:
    """A collection of positions with aggregate Greeks."""

    positions: list[Position] = field(default_factory=list)

    def add(self, pos: Position):
        self.positions.append(pos)

    def aggregate_greeks(self) -> dict:
        totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0,
                  "theta": 0.0, "rho": 0.0}
        for pos in self.positions:
            g = pos.greeks()
            for k in totals:
                totals[k] += g[k]
        return totals

    def delta_hedge_shares(self) -> float:
        """Number of shares to buy/sell to achieve delta neutrality."""
        return -self.aggregate_greeks()["delta"]

    def delta_gamma_hedge(self, hedge_instrument: Position) -> dict:
        """Compute quantities for delta-gamma neutrality.

        Returns the number of *hedge_instrument* contracts and
        the residual stock hedge.
        """
        pg = self.aggregate_greeks()
        hg = hedge_instrument.greeks()

        # gamma of one contract of the hedge instrument
        gamma_per = hg["gamma"] / (hedge_instrument.qty
                                    * hedge_instrument.multiplier)
        if abs(gamma_per) < 1e-14:
            raise ValueError("Hedge instrument has zero gamma")

        n_hedge = -pg["gamma"] / gamma_per
        residual_delta = pg["delta"] + n_hedge * (
            hg["delta"] / (hedge_instrument.qty * hedge_instrument.multiplier)
        )
        n_stock = -residual_delta

        return {"hedge_contracts": n_hedge, "stock_shares": n_stock}

    def summary(self) -> str:
        g = self.aggregate_greeks()
        lines = ["Portfolio Greeks:"]
        for k, v in g.items():
            lines.append(f"  {k:>6} = {v:+.4f}")
        lines.append(f"  Delta hedge = {self.delta_hedge_shares():+.1f} shares")
        return "\n".join(lines)
