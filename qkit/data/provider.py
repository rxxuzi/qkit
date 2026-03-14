"""Abstract data provider with factory function.

The :func:`get_provider` factory returns either a moomoo or yfinance
backend depending on the ``DATA_PROVIDER`` environment variable.
Application code only needs::

    from qkit.data import get_provider

    p = get_provider()
    spot = p.get_spot_price("AAPL")
    chain = p.get_option_chain("AAPL")
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from qkit import config


@dataclass
class OptionQuote:
    """Market data for a single option contract."""

    strike: float
    expiry: str
    option_type: str
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_vol: Optional[float] = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass
class OptionChain:
    """Full option chain for a single underlying."""

    symbol: str
    spot_price: float
    quotes: list[OptionQuote]

    def calls(self) -> list[OptionQuote]:
        return [q for q in self.quotes if q.option_type == "call"]

    def puts(self) -> list[OptionQuote]:
        return [q for q in self.quotes if q.option_type == "put"]

    def expiries(self) -> list[str]:
        return sorted({q.expiry for q in self.quotes})

    def by_expiry(self, expiry: str) -> list[OptionQuote]:
        return [q for q in self.quotes if q.expiry == expiry]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "strike": q.strike, "expiry": q.expiry, "type": q.option_type,
                "bid": q.bid, "ask": q.ask, "mid": q.mid, "last": q.last,
                "volume": q.volume, "open_interest": q.open_interest,
                "implied_vol": q.implied_vol,
            }
            for q in self.quotes
        ]
        return pd.DataFrame(rows)


class DataProvider(abc.ABC):
    """Interface that every data backend must implement."""

    @abc.abstractmethod
    def get_spot_price(self, symbol: str) -> float: ...

    @abc.abstractmethod
    def get_history(self, symbol: str, period: str = "1y",
                    interval: str = "1d") -> pd.DataFrame: ...

    @abc.abstractmethod
    def get_option_chain(self, symbol: str) -> OptionChain: ...

    def get_daily_returns(self, symbol: str, period: str = "1y") -> pd.Series:
        """Compute log returns from daily close prices."""
        hist = self.get_history(symbol, period=period, interval="1d")
        return np.log(hist["close"] / hist["close"].shift(1)).dropna()


class _FallbackProvider(DataProvider):
    """Wraps a primary provider and falls back to yfinance on errors."""

    def __init__(self, primary: DataProvider, fallback: DataProvider):
        self._primary = primary
        self._fallback = fallback

    def _try(self, method: str, *args, **kwargs):
        try:
            return getattr(self._primary, method)(*args, **kwargs)
        except Exception:
            import warnings
            warnings.warn(
                f"moomoo {method}() failed, falling back to yfinance",
                stacklevel=3,
            )
            return getattr(self._fallback, method)(*args, **kwargs)

    def get_spot_price(self, symbol: str) -> float:
        return self._try("get_spot_price", symbol)

    def get_history(self, symbol: str, period: str = "1y",
                    interval: str = "1d") -> pd.DataFrame:
        return self._try("get_history", symbol, period=period, interval=interval)

    def get_option_chain(self, symbol: str) -> OptionChain:
        return self._try("get_option_chain", symbol)

    def __getattr__(self, name):
        primary_attr = getattr(self._primary, name, None)
        fallback_attr = getattr(self._fallback, name, None)
        if callable(primary_attr):
            def wrapper(*args, **kwargs):
                try:
                    return primary_attr(*args, **kwargs)
                except Exception:
                    if callable(fallback_attr):
                        return fallback_attr(*args, **kwargs)
                    raise
            return wrapper
        if primary_attr is not None:
            return primary_attr
        if fallback_attr is not None:
            return fallback_attr
        raise AttributeError(name)


def get_provider(provider_name: Optional[str] = None) -> DataProvider:
    """Return a data provider instance based on configuration.

    When *moomoo* is configured but OpenD is unreachable, automatically
    falls back to yfinance.  When moomoo is reachable but an API call
    fails (e.g. market closed), individual calls also fall back.
    """
    name = (provider_name or config.DATA_PROVIDER).lower()
    if name == "moomoo":
        from .moomoo_client import MoomooProvider
        from .yfinance_client import YFinanceProvider
        provider = MoomooProvider()
        fallback = YFinanceProvider()
        # Quick connectivity check — full fallback if OpenD is down
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect((provider._host, provider._port))
            s.close()
            return _FallbackProvider(provider, fallback)
        except (OSError, ConnectionRefusedError):
            import warnings
            warnings.warn(
                f"moomoo OpenD not reachable at {provider._host}:{provider._port}, "
                "falling back to yfinance",
                stacklevel=2,
            )
            return fallback
    if name == "yfinance":
        from .yfinance_client import YFinanceProvider
        return YFinanceProvider()
    raise ValueError(f"Unknown provider: {name!r}")
