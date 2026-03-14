"""moomoo (Futu OpenD) data provider for real-time market data.

Requires the OpenD gateway running locally.  See
https://openapi.moomoo.com/ for setup instructions.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from moomoo import OpenQuoteContext

from qkit import config
from .provider import DataProvider, OptionChain, OptionQuote

_PERIOD_DAYS = {
    "1mo": 30, "3mo": 90, "6mo": 180,
    "1y": 365, "2y": 730, "5y": 1825,
}


class MoomooProvider(DataProvider):
    """Fetch data through the moomoo OpenD gateway."""

    def __init__(self):
        self._host = config.MOOMOO_HOST
        self._port = config.MOOMOO_PORT

    def _ctx(self) -> OpenQuoteContext:
        return OpenQuoteContext(host=self._host, port=self._port)

    def get_spot_price(self, symbol: str) -> float:
        sym = _normalise(symbol)
        ctx = self._ctx()
        try:
            ret, data = ctx.get_market_snapshot([sym])
            if ret != 0:
                raise RuntimeError(f"moomoo snapshot error: {data}")
            return float(data.iloc[0]["last_price"])
        finally:
            ctx.close()

    def get_snapshot(self, symbol: str) -> dict:
        """Fetch full market snapshot including fundamentals.

        Returns a dict with keys matching the ``Snapshot`` dataclass fields
        in ``qkit.data.store``.
        """
        sym = _normalise(symbol)
        ctx = self._ctx()
        try:
            ret, data = ctx.get_market_snapshot([sym])
            if ret != 0:
                raise RuntimeError(f"moomoo snapshot error: {data}")
            row = data.iloc[0]

            def _f(key: str) -> float | None:
                v = row.get(key)
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return None
                return float(v)

            def _i(key: str) -> int | None:
                v = row.get(key)
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return None
                return int(v)

            return {
                "symbol": symbol.upper(),
                "date": dt.datetime.now().strftime("%Y-%m-%d"),
                "last_price": _f("last_price"),
                "open_price": _f("open_price"),
                "high_price": _f("high_price"),
                "low_price": _f("low_price"),
                "prev_close": _f("prev_close_price"),
                "volume": _i("volume"),
                "turnover": _f("turnover"),
                "market_cap": _f("market_val"),
                "pe_ratio": _f("pe_ratio"),
                "pb_ratio": _f("pb_ratio"),
                "dividend_yield": _f("dividend_ratio_ttm"),
                "eps": _f("eps"),
                "high_52w": _f("high_price_52w") or _f("52w_high"),
                "low_52w": _f("low_price_52w") or _f("52w_low"),
            }
        finally:
            ctx.close()

    def get_snapshots_batch(self, symbols: list[str]) -> list[dict]:
        """Fetch snapshots for multiple symbols in one API call."""
        syms = [_normalise(s) for s in symbols]
        ctx = self._ctx()
        try:
            ret, data = ctx.get_market_snapshot(syms)
            if ret != 0:
                raise RuntimeError(f"moomoo snapshot error: {data}")

            results = []
            today = dt.datetime.now().strftime("%Y-%m-%d")
            for _, row in data.iterrows():
                def _f(key: str) -> float | None:
                    v = row.get(key)
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return None
                    return float(v)

                def _i(key: str) -> int | None:
                    v = row.get(key)
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return None
                    return int(v)

                # Strip market prefix for storage
                code = str(row.get("code", ""))
                sym = code.split(".")[-1] if "." in code else code

                results.append({
                    "symbol": sym.upper(),
                    "date": today,
                    "last_price": _f("last_price"),
                    "open_price": _f("open_price"),
                    "high_price": _f("high_price"),
                    "low_price": _f("low_price"),
                    "prev_close": _f("prev_close_price"),
                    "volume": _i("volume"),
                    "turnover": _f("turnover"),
                    "market_cap": _f("market_val"),
                    "pe_ratio": _f("pe_ratio"),
                    "pb_ratio": _f("pb_ratio"),
                    "dividend_yield": _f("dividend_ratio_ttm"),
                    "eps": _f("eps"),
                    "high_52w": _f("high_price_52w") or _f("52w_high"),
                    "low_52w": _f("low_price_52w") or _f("52w_low"),
                })
            return results
        finally:
            ctx.close()

    def get_history(self, symbol: str, period: str = "1y",
                    interval: str = "1d") -> pd.DataFrame:
        sym = _normalise(symbol)
        ctx = self._ctx()
        try:
            end = dt.datetime.now().strftime("%Y-%m-%d")
            days = _PERIOD_DAYS.get(period, 365)
            start = (dt.datetime.now() - dt.timedelta(days=days)).strftime("%Y-%m-%d")

            ret, data, _ = ctx.request_history_kline(
                sym, start=start, end=end, ktype="K_DAY", max_count=5000,
            )
            if ret != 0:
                raise RuntimeError(f"moomoo history error: {data}")

            data.index = pd.to_datetime(data["time_key"])
            return data[["open", "high", "low", "close", "volume"]]
        finally:
            ctx.close()

    def get_option_chain(self, symbol: str) -> OptionChain:
        sym = _normalise(symbol)
        ctx = self._ctx()
        try:
            spot = self.get_spot_price(symbol)

            ret, expiries = ctx.get_option_expiration_date(sym)
            if ret != 0:
                raise RuntimeError(f"moomoo expiry error: {expiries}")

            quotes: list[OptionQuote] = []
            for _, row in expiries.iterrows():
                exp = str(row["strike_time"])[:10]
                ret2, chain = ctx.get_option_chain(sym, exp, option_type="ALL")
                if ret2 != 0:
                    continue
                for _, opt in chain.iterrows():
                    otype = ("call" if "CALL" in str(opt.get("option_type", "")).upper()
                             else "put")
                    iv = opt.get("implied_volatility")
                    quotes.append(OptionQuote(
                        strike=float(opt.get("strike_price", 0)),
                        expiry=exp, option_type=otype,
                        bid=float(opt.get("bid_price", 0) or 0),
                        ask=float(opt.get("ask_price", 0) or 0),
                        last=float(opt.get("last_price", 0) or 0),
                        volume=int(opt.get("volume", 0) or 0),
                        open_interest=int(opt.get("open_interest", 0) or 0),
                        implied_vol=float(iv) if pd.notna(iv) else None,
                    ))

            return OptionChain(symbol=symbol, spot_price=spot, quotes=quotes)
        finally:
            ctx.close()


def _normalise(symbol: str) -> str:
    """Ensure *symbol* has a market prefix (e.g. ``AAPL`` -> ``US.AAPL``)."""
    if "." not in symbol:
        return f"US.{symbol}"
    return symbol
