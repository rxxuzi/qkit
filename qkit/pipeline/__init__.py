"""Phase 2 data analysis pipeline.

Fetches real market data and generates interactive plotly charts
returned as JSON for web rendering. No static PNGs.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def iv_surface_chart(chain_df: pd.DataFrame, spot: float,
                     symbol: str = "SPY") -> dict:
    """Build a 3-D IV surface from an option chain DataFrame.

    *chain_df* must have columns: strike, expiry_years, iv, type.
    Returns plotly figure as JSON dict.
    """
    calls = chain_df[chain_df["type"] == "call"].copy()
    if calls.empty:
        return _empty_chart("No call data available")

    pivoted = calls.pivot_table(index="expiry_years", columns="strike",
                                values="iv", aggfunc="mean")
    pivoted = pivoted.dropna(axis=1, how="all").dropna(axis=0, how="all")

    if pivoted.empty:
        return _empty_chart("Insufficient data for IV surface")

    fig = go.Figure(data=go.Surface(
        z=pivoted.values * 100,
        x=pivoted.columns.values,
        y=(pivoted.index.values * 365).astype(int),
        colorscale="Plasma",
        colorbar=dict(title=dict(text="IV (%)", side="right")),
    ))
    fig.update_layout(
        title=f"{symbol} IV Surface",
        scene=dict(xaxis_title="Strike ($)", yaxis_title="Days to Expiry",
                   zaxis_title="IV (%)",
                   bgcolor="rgba(6,8,13,1)",
                   xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                   yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                   zaxis=dict(gridcolor="rgba(255,255,255,0.05)")),
        **_dark_layout(),
    )
    fig.add_trace(go.Scatter3d(
        x=[spot], y=[30], z=[pivoted.values.mean() * 100],
        mode="markers", marker=dict(size=6, color="#22d3ee"),
        name="ATM", showlegend=True,
    ))
    return json.loads(fig.to_json())


def iv_smile_chart(chain_df: pd.DataFrame, spot: float,
                   symbol: str = "SPY") -> dict:
    """IV smile slices for each expiry."""
    fig = go.Figure()
    calls = chain_df[chain_df["type"] == "call"].copy()
    if calls.empty:
        return _empty_chart("No data")

    expiries = sorted(calls["expiry_years"].unique())
    colors = ["#22d3ee", "#a78bfa", "#10b981", "#f59e0b", "#ef4444",
              "#3b82f6", "#ec4899"]

    for i, exp in enumerate(expiries[:7]):
        subset = calls[calls["expiry_years"] == exp].sort_values("strike")
        days = int(exp * 365)
        fig.add_trace(go.Scatter(
            x=subset["strike"], y=subset["iv"] * 100,
            mode="lines+markers", name=f"{days}d",
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4),
        ))

    fig.add_vline(x=spot, line_dash="dash", line_color="#22d3ee",
                  annotation_text="ATM")
    fig.update_layout(
        title=f"{symbol} IV Smile",
        xaxis_title="Strike ($)", yaxis_title="IV (%)",
        **_dark_layout(),
    )
    return json.loads(fig.to_json())


def garch_chart(returns: pd.Series, symbol: str = "SPY") -> dict:
    """Fit GARCH(1,1) and plot conditional volatility vs price."""
    from qkit.volatility.garch import fit

    pct_returns = returns * 100
    result = fit(pct_returns, model_type="garch")

    cond_vol = result.conditional_var.apply(np.sqrt) / 100 * np.sqrt(252)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4],
                        vertical_spacing=0.06,
                        subplot_titles=(f"{symbol} Returns", "GARCH Conditional Vol (ann.)"))

    fig.add_trace(go.Scatter(
        x=returns.index, y=returns.values * 100,
        mode="lines", name="Daily Return %",
        line=dict(color="rgba(34,211,238,0.5)", width=1),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=cond_vol.index, y=cond_vol.values * 100,
        mode="lines", name="GARCH Vol %",
        line=dict(color="#a78bfa", width=2),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.08)",
    ), row=2, col=1)

    fig.update_layout(
        **_dark_layout(),
        height=500,
    )
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Vol (%)", row=2, col=1)

    info = {
        "persistence": round(result.persistence, 4),
        "half_life": round(result.half_life, 1),
        "unconditional_vol": round(result.unconditional_vol / 100 * np.sqrt(252) * 100, 2),
        "log_likelihood": round(result.log_likelihood, 2),
        "forecast_30d_vol": round(result.forecast_vol(30), 2),
    }
    return {"chart": json.loads(fig.to_json()), "info": info}


def vrp_chart(spy_returns: pd.Series, vix: pd.Series,
              spy_price: pd.Series = None) -> dict:
    """VRP time series with z-score signals."""
    from qkit.signals.vrp import compute_vrp

    sig = compute_vrp(spy_returns, vix, rv_window=22, z_window=252)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.35, 0.35, 0.30],
                        vertical_spacing=0.05,
                        subplot_titles=("SPY Price" if spy_price is not None else "Returns",
                                        "VRP (IV² − RV)", "VRP z-score"))

    if spy_price is not None:
        fig.add_trace(go.Scatter(
            x=spy_price.index, y=spy_price.values,
            mode="lines", name="SPY",
            line=dict(color="#22d3ee", width=1.5),
        ), row=1, col=1)
    else:
        cum = (1 + spy_returns).cumprod()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            mode="lines", name="Cumulative Return",
            line=dict(color="#22d3ee", width=1.5),
        ), row=1, col=1)

    vrp = sig.vrp.dropna()
    fig.add_trace(go.Scatter(
        x=vrp.index, y=vrp.values,
        mode="lines", name="VRP",
        line=dict(color="#a78bfa", width=1.5),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.06)",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#5e6b80", row=2, col=1)

    z = sig.z_score.dropna()
    fig.add_trace(go.Scatter(
        x=z.index, y=z.values,
        mode="lines", name="z-score",
        line=dict(color="#10b981", width=1.5),
    ), row=3, col=1)
    fig.add_hline(y=1.5, line_dash="dash", line_color="#ef4444",
                  annotation_text="Short Vol", row=3, col=1)
    fig.add_hline(y=-1.0, line_dash="dash", line_color="#10b981",
                  annotation_text="Buy Protection", row=3, col=1)

    # Highlight signal zones
    high_vrp = z[z > 1.5]
    low_vrp = z[z < -1.0]
    if not high_vrp.empty:
        fig.add_trace(go.Scatter(
            x=high_vrp.index, y=high_vrp.values,
            mode="markers", name="HIGH VRP",
            marker=dict(color="#ef4444", size=5, symbol="triangle-down"),
        ), row=3, col=1)
    if not low_vrp.empty:
        fig.add_trace(go.Scatter(
            x=low_vrp.index, y=low_vrp.values,
            mode="markers", name="LOW VRP",
            marker=dict(color="#10b981", size=5, symbol="triangle-up"),
        ), row=3, col=1)

    fig.update_layout(**_dark_layout(), height=700)

    current = sig.current()
    return {"chart": json.loads(fig.to_json()), "current": current}


def greeks_heatmap_chart(S: float, sigma: float = 0.22, r: float = 0.043,
                         greek: str = "delta", opt_type: str = "call") -> dict:
    """Interactive Greeks heatmap via plotly (replaces static PNG)."""
    from qkit.pricing.bsm import BSM

    strikes = np.linspace(S * 0.80, S * 1.20, 50)
    expiry_days = np.linspace(7, 180, 50)
    Z = np.zeros((len(expiry_days), len(strikes)))

    for i, days in enumerate(expiry_days):
        T = days / 365
        for j, K in enumerate(strikes):
            m = BSM(S=S, K=K, T=T, r=r, sigma=sigma)
            g = m.call_greeks() if opt_type == "call" else m.put_greeks()
            Z[i, j] = getattr(g, greek)

    fig = go.Figure(data=go.Heatmap(
        z=Z, x=np.round(strikes, 1), y=np.round(expiry_days).astype(int),
        colorscale="RdBu_r" if greek == "delta" else "Viridis",
        colorbar=dict(title=greek.title()),
    ))
    fig.add_vline(x=S, line_dash="dash", line_color="#22d3ee",
                  annotation_text="ATM")
    fig.update_layout(
        title=f"{opt_type.title()} {greek.title()}  S=${S:.0f}  σ={sigma:.0%}",
        xaxis_title="Strike ($)", yaxis_title="Days to Expiry",
        **_dark_layout(),
    )
    return json.loads(fig.to_json())


def greeks_dashboard_chart(S: float, sigma: float = 0.22, r: float = 0.043,
                           opt_type: str = "call") -> dict:
    """Six-panel Greeks dashboard (interactive plotly)."""
    from qkit.pricing.bsm import BSM

    greek_names = ["delta", "gamma", "vega", "theta", "rho"]
    strikes = np.linspace(S * 0.80, S * 1.20, 40)
    expiry_days = np.linspace(7, 180, 40)
    grids = {}

    for gn in greek_names:
        Z = np.zeros((len(expiry_days), len(strikes)))
        for i, days in enumerate(expiry_days):
            T = days / 365
            for j, K in enumerate(strikes):
                m = BSM(S=S, K=K, T=T, r=r, sigma=sigma)
                g = m.call_greeks() if opt_type == "call" else m.put_greeks()
                Z[i, j] = getattr(g, gn)
        grids[gn] = Z

    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=[g.title() for g in greek_names] + ["Price"],
                        vertical_spacing=0.12, horizontal_spacing=0.06)

    cmaps = ["RdBu_r", "YlOrRd", "PuBuGn", "RdPu", "BrBG", "Viridis"]
    for idx, gn in enumerate(greek_names):
        r_pos, c_pos = divmod(idx, 3)
        fig.add_trace(go.Heatmap(
            z=grids[gn], x=np.round(strikes, 1),
            y=np.round(expiry_days).astype(int),
            colorscale=cmaps[idx], showscale=False,
        ), row=r_pos + 1, col=c_pos + 1)

    fig.update_layout(
        title=f"{opt_type.title()} Greeks Dashboard  S=${S:.0f}",
        **_dark_layout(), height=650,
    )
    return json.loads(fig.to_json())


def payoff_chart(S: float, K: float, premium: float,
                 opt_type: str = "call",
                 position: str = "long") -> dict:
    """Option payoff diagram at expiry.

    Parameters
    ----------
    S : float
        Current spot price (used to set price range).
    K : float
        Strike price.
    premium : float
        Option premium paid (or received).
    opt_type : str
        ``"call"`` or ``"put"``.
    position : str
        ``"long"`` (buy) or ``"short"`` (sell).

    Returns
    -------
    dict
        Plotly figure JSON.
    """
    prices = np.linspace(S * 0.5, S * 1.5, 200)

    if opt_type == "call":
        intrinsic = np.maximum(prices - K, 0)
    else:
        intrinsic = np.maximum(K - prices, 0)

    if position == "long":
        pnl = intrinsic - premium
    else:
        pnl = premium - intrinsic

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices, y=pnl, mode="lines",
        name=f"{position.title()} {opt_type.title()}",
        line=dict(color="#22d3ee", width=2.5),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#5e6b80")
    fig.add_vline(x=K, line_dash="dot", line_color="#f59e0b",
                  annotation_text=f"K={K:.0f}")
    fig.add_vline(x=S, line_dash="dot", line_color="#a78bfa",
                  annotation_text=f"S={S:.0f}")

    # Shade profit / loss regions
    profit_mask = pnl >= 0
    loss_mask = pnl < 0
    fig.add_trace(go.Scatter(
        x=prices[profit_mask], y=pnl[profit_mask],
        fill="tozeroy", fillcolor="rgba(16,185,129,0.1)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=prices[loss_mask], y=pnl[loss_mask],
        fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # Break-even annotation
    if opt_type == "call":
        be = K + premium if position == "long" else K + premium
    else:
        be = K - premium if position == "long" else K - premium
    if S * 0.5 <= be <= S * 1.5:
        fig.add_vline(x=be, line_dash="dash", line_color="#10b981",
                      annotation_text=f"BE={be:.1f}")

    direction = "Long" if position == "long" else "Short"
    fig.update_layout(
        title=f"{direction} {opt_type.title()} Payoff  K={K:.0f}  Premium={premium:.2f}",
        xaxis_title="Underlying Price at Expiry ($)",
        yaxis_title="P&L ($)",
        **_dark_layout(),
        height=400,
    )
    return json.loads(fig.to_json())


def payoff_strategy_chart(S: float, legs: list[dict]) -> dict:
    """Multi-leg option strategy payoff diagram.

    Parameters
    ----------
    S : float
        Current spot price.
    legs : list[dict]
        Each dict: ``{"K": float, "premium": float, "type": "call"|"put",
        "position": "long"|"short", "qty": int}``.

    Returns
    -------
    dict
        Plotly figure JSON.
    """
    prices = np.linspace(S * 0.5, S * 1.5, 200)
    total_pnl = np.zeros_like(prices)

    colors = ["#22d3ee", "#a78bfa", "#10b981", "#f59e0b", "#ef4444", "#3b82f6"]
    fig = go.Figure()

    for i, leg in enumerate(legs):
        K = leg["K"]
        prem = leg["premium"]
        qty = leg.get("qty", 1)
        otype = leg.get("type", "call")
        pos = leg.get("position", "long")

        if otype == "call":
            intrinsic = np.maximum(prices - K, 0)
        else:
            intrinsic = np.maximum(K - prices, 0)

        if pos == "long":
            leg_pnl = (intrinsic - prem) * qty
        else:
            leg_pnl = (prem - intrinsic) * qty

        total_pnl += leg_pnl
        direction = "L" if pos == "long" else "S"
        fig.add_trace(go.Scatter(
            x=prices, y=leg_pnl, mode="lines",
            name=f"{direction} {qty}x {otype[0].upper()} K={K:.0f}",
            line=dict(color=colors[i % len(colors)], width=1, dash="dot"),
            opacity=0.5,
        ))

    fig.add_trace(go.Scatter(
        x=prices, y=total_pnl, mode="lines",
        name="Total P&L",
        line=dict(color="#22d3ee", width=3),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#5e6b80")
    fig.add_vline(x=S, line_dash="dot", line_color="#a78bfa",
                  annotation_text=f"S={S:.0f}")

    # Shade profit / loss
    profit_mask = total_pnl >= 0
    loss_mask = total_pnl < 0
    fig.add_trace(go.Scatter(
        x=prices[profit_mask], y=total_pnl[profit_mask],
        fill="tozeroy", fillcolor="rgba(16,185,129,0.08)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=prices[loss_mask], y=total_pnl[loss_mask],
        fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    max_profit = float(total_pnl.max())
    max_loss = float(total_pnl.min())
    fig.update_layout(
        title=f"Strategy Payoff  ({len(legs)} legs)  Max P&L: {max_profit:.2f} / {max_loss:.2f}",
        xaxis_title="Underlying Price at Expiry ($)",
        yaxis_title="P&L ($)",
        **_dark_layout(),
        height=450,
    )

    info = {
        "max_profit": round(max_profit, 2),
        "max_loss": round(max_loss, 2),
        "breakevens": [],
    }
    # Find break-even points
    for j in range(len(total_pnl) - 1):
        if total_pnl[j] * total_pnl[j + 1] < 0:
            be = prices[j] + (prices[j + 1] - prices[j]) * (-total_pnl[j]) / (total_pnl[j + 1] - total_pnl[j])
            info["breakevens"].append(round(float(be), 2))

    return {"chart": json.loads(fig.to_json()), "info": info}


def pairs_chart(price_a: pd.Series, price_b: pd.Series,
                name_a: str = "A", name_b: str = "B",
                backtest_results: list | None = None) -> dict:
    """Build multi-panel pairs trading chart.

    Panels: normalised prices, spread + OU mean, z-score + thresholds,
    and (optionally) cumulative PnL from walk-forward backtest.

    Returns
    -------
    dict
        ``{"chart": plotly_json, "stats": dict, "backtest_summary": dict | None}``
    """
    from qkit.signals.pairs import analyze_pair, spread_zscore

    stats = analyze_pair(price_a, price_b, name_a, name_b)
    spread = price_b - stats.beta * price_a

    n_rows = 4 if backtest_results else 3
    row_heights = [0.30, 0.25, 0.25, 0.20] if backtest_results else [0.35, 0.30, 0.35]
    titles = [f"Normalised Prices ({name_a} vs {name_b})",
              "Spread + OU Mean", "z-score"]
    if backtest_results:
        titles.append("Cumulative PnL (walk-forward)")

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        row_heights=row_heights,
                        vertical_spacing=0.05,
                        subplot_titles=titles)

    # Panel 1: normalised prices (rebased to 100)
    norm_a = price_a / price_a.iloc[0] * 100
    norm_b = price_b / price_b.iloc[0] * 100
    fig.add_trace(go.Scatter(x=norm_a.index, y=norm_a.values,
                             mode="lines", name=name_a,
                             line=dict(color="#22d3ee", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=norm_b.index, y=norm_b.values,
                             mode="lines", name=name_b,
                             line=dict(color="#a78bfa", width=1.5)), row=1, col=1)

    # Panel 2: spread + OU mean
    fig.add_trace(go.Scatter(x=spread.index, y=spread.values,
                             mode="lines", name="Spread",
                             line=dict(color="#10b981", width=1.5)), row=2, col=1)
    fig.add_hline(y=stats.mu, line_dash="dash", line_color="#f59e0b",
                  annotation_text="μ (OU mean)", row=2, col=1)

    # Panel 3: z-score + thresholds
    z = spread_zscore(spread, stats.theta, stats.mu, stats.sigma_ou)
    fig.add_trace(go.Scatter(x=z.index, y=z.values,
                             mode="lines", name="z-score",
                             line=dict(color="#3b82f6", width=1.5)), row=3, col=1)
    for level, color, label in [(2.0, "#ef4444", "Entry"),
                                (-2.0, "#ef4444", None),
                                (0.5, "#10b981", "Exit"),
                                (-0.5, "#10b981", None),
                                (4.0, "#f59e0b", "Stop"),
                                (-4.0, "#f59e0b", None)]:
        fig.add_hline(y=level, line_dash="dot", line_color=color,
                      annotation_text=label, row=3, col=1)

    # Signal markers on z-score
    short_mask = z > 2.0
    long_mask = z < -2.0
    if short_mask.any():
        fig.add_trace(go.Scatter(
            x=z[short_mask].index, y=z[short_mask].values,
            mode="markers", name="SHORT_SPREAD",
            marker=dict(color="#ef4444", size=5, symbol="triangle-down"),
            showlegend=True), row=3, col=1)
    if long_mask.any():
        fig.add_trace(go.Scatter(
            x=z[long_mask].index, y=z[long_mask].values,
            mode="markers", name="LONG_SPREAD",
            marker=dict(color="#10b981", size=5, symbol="triangle-up"),
            showlegend=True), row=3, col=1)

    # Panel 4: cumulative PnL (if backtest results provided)
    bt_summary = None
    if backtest_results:
        from qkit.backtest.engine import walk_forward_summary
        pnl_pieces = []
        for r in backtest_results:
            pnl_pieces.append(r.backtest_df["pnl"])
        if pnl_pieces:
            all_pnl = pd.concat(pnl_pieces)
            cum_pnl = all_pnl.cumsum()
            fig.add_trace(go.Scatter(
                x=cum_pnl.index, y=cum_pnl.values,
                mode="lines", name="Cum. PnL",
                line=dict(color="#22d3ee", width=2),
                fill="tozeroy", fillcolor="rgba(34,211,238,0.06)",
            ), row=4, col=1)
        bt_summary = walk_forward_summary(backtest_results)

    fig.update_layout(**_dark_layout(), height=250 * n_rows)

    stats_dict = {
        "asset_a": stats.asset_a, "asset_b": stats.asset_b,
        "pvalue": float(round(stats.coint_pvalue, 4)),
        "beta": float(round(stats.beta, 4)),
        "half_life": float(round(stats.half_life, 1)),
        "theta": float(round(stats.theta, 4)),
        "mu": float(round(stats.mu, 4)),
        "current_z": float(round(stats.current_z, 2)),
        "signal": stats.signal(),
        "is_cointegrated": bool(stats.is_cointegrated),
    }

    return {
        "chart": json.loads(fig.to_json()),
        "stats": stats_dict,
        "backtest_summary": bt_summary,
    }


def _dark_layout():
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0a0a0a",
        font=dict(family="JetBrains Mono, Consolas, monospace", color="#d4d4d4"),
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )


def _empty_chart(msg: str) -> dict:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=16, color="#5e6b80"))
    fig.update_layout(**_dark_layout())
    return json.loads(fig.to_json())
