"""Walk-forward backtest engine for pairs trading.

Slides a train/test window across price data, re-fitting OU parameters
each fold and running ``backtest_pair()`` on the out-of-sample segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from qkit.signals.pairs import analyze_pair, backtest_pair, PairStats


@dataclass
class BacktestResult:
    """Metrics for a single walk-forward fold."""

    pair: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    gross_pnl: float
    net_pnl: float
    cost_drag: float
    backtest_df: pd.DataFrame = field(repr=False)
    pair_stats: PairStats = field(repr=False)


def compute_metrics(pnl_series: pd.Series,
                    cost_per_trade: float = 0.001) -> dict:
    """Derive performance metrics from a per-trade PnL series.

    Parameters
    ----------
    pnl_series : pd.Series
        Individual trade PnLs (non-zero entries from ``backtest_pair``).
    cost_per_trade : float
        Round-trip cost as fraction of spread notional.

    Returns
    -------
    dict
        Keys: annual_return, sharpe_ratio, max_drawdown, total_trades,
        win_rate, gross_pnl, net_pnl, cost_drag.
    """
    trades = pnl_series[pnl_series != 0]
    n_trades = len(trades)

    if n_trades == 0:
        return dict(
            annual_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
            total_trades=0, win_rate=0.0, gross_pnl=0.0,
            net_pnl=0.0, cost_drag=0.0,
        )

    gross_pnl = float(trades.sum())
    cost_drag = n_trades * cost_per_trade
    net_pnl = gross_pnl - cost_drag

    cum = pnl_series.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = float(drawdown.min())

    n_days = len(pnl_series)
    daily_ret = pnl_series.mean()
    daily_std = pnl_series.std()
    annual_return = float(daily_ret * 252)
    sharpe = float(daily_ret / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0

    win_rate = float((trades > 0).sum() / n_trades)

    return dict(
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        total_trades=n_trades,
        win_rate=win_rate,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        cost_drag=cost_drag,
    )


def walk_forward(price_a: pd.Series, price_b: pd.Series,
                 name_a: str = "A", name_b: str = "B",
                 train_days: int = 252, test_days: int = 63,
                 cost_per_trade: float = 0.001,
                 entry_z: float = 2.0, exit_z: float = 0.5,
                 stop_z: float = 4.0) -> list[BacktestResult]:
    """Run walk-forward backtest with sliding train/test windows.

    Parameters
    ----------
    price_a, price_b : pd.Series
        Price series for asset A and B (aligned index).
    name_a, name_b : str
        Ticker labels.
    train_days : int
        Training window length (default 252 = ~1 year).
    test_days : int
        Test window length (default 63 = ~1 quarter).
    cost_per_trade : float
        Round-trip cost fraction.
    entry_z, exit_z, stop_z : float
        Z-score thresholds forwarded to ``backtest_pair()``.

    Returns
    -------
    list[BacktestResult]
        One result per fold.
    """
    n = len(price_a)
    min_len = train_days + test_days
    if n < min_len:
        return []

    results: list[BacktestResult] = []
    start = 0

    while start + min_len <= n:
        train_end = start + train_days
        test_end = min(train_end + test_days, n)

        train_a = price_a.iloc[start:train_end]
        train_b = price_b.iloc[start:train_end]
        test_a = price_a.iloc[train_end:test_end]
        test_b = price_b.iloc[train_end:test_end]

        # Fit on training window
        stats = analyze_pair(train_a, train_b, name_a, name_b)

        # Backtest on test window
        test_spread = test_b - stats.beta * test_a
        bt_df = backtest_pair(test_spread, stats.theta, stats.mu,
                              stats.sigma_ou, entry_z, exit_z, stop_z)

        metrics = compute_metrics(bt_df["pnl"], cost_per_trade)

        results.append(BacktestResult(
            pair=f"{name_a}/{name_b}",
            train_start=str(train_a.index[0].date()) if hasattr(train_a.index[0], 'date') else str(train_a.index[0]),
            train_end=str(train_a.index[-1].date()) if hasattr(train_a.index[-1], 'date') else str(train_a.index[-1]),
            test_start=str(test_a.index[0].date()) if hasattr(test_a.index[0], 'date') else str(test_a.index[0]),
            test_end=str(test_a.index[-1].date()) if hasattr(test_a.index[-1], 'date') else str(test_a.index[-1]),
            backtest_df=bt_df,
            pair_stats=stats,
            **metrics,
        ))

        start += test_days

    return results


def walk_forward_summary(results: list[BacktestResult]) -> dict:
    """Aggregate metrics across all walk-forward folds.

    Returns
    -------
    dict
        Keys: n_folds, avg_sharpe, avg_annual_return, worst_drawdown,
        total_trades, avg_win_rate, total_net_pnl,
        cointegrated_folds, cointegrated_ratio.
    """
    if not results:
        return dict(
            n_folds=0, avg_sharpe=0.0, avg_annual_return=0.0,
            worst_drawdown=0.0, total_trades=0, avg_win_rate=0.0,
            total_net_pnl=0.0, cointegrated_folds=0,
            cointegrated_ratio=0.0,
        )

    sharpes = [r.sharpe_ratio for r in results]
    returns = [r.annual_return for r in results]
    drawdowns = [r.max_drawdown for r in results]
    win_rates = [r.win_rate for r in results]
    coint_count = sum(1 for r in results if r.pair_stats.is_cointegrated)

    return dict(
        n_folds=len(results),
        avg_sharpe=float(np.mean(sharpes)),
        avg_annual_return=float(np.mean(returns)),
        worst_drawdown=float(min(drawdowns)),
        total_trades=sum(r.total_trades for r in results),
        avg_win_rate=float(np.mean(win_rates)),
        total_net_pnl=sum(r.net_pnl for r in results),
        cointegrated_folds=coint_count,
        cointegrated_ratio=coint_count / len(results),
    )
