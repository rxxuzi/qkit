"""yfinance data provider (free, delayed, prototyping)."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from .provider import DataProvider, OptionChain, OptionQuote


class YFinanceProvider(DataProvider):
    """Fetch market data via the yfinance library."""

    def get_spot_price(self, symbol: str) -> float:
        return float(yf.Ticker(symbol).fast_info["lastPrice"])

    def get_history(self, symbol: str, period: str = "1y",
                    interval: str = "1d") -> pd.DataFrame:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        # Flatten MultiIndex columns (yfinance ≥ 0.2.40 returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c
                          for c in df.columns]
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]

    def get_option_chain(self, symbol: str) -> OptionChain:
        ticker = yf.Ticker(symbol)
        spot = self.get_spot_price(symbol)
        quotes: list[OptionQuote] = []

        for expiry in ticker.options:
            chain = ticker.option_chain(expiry)
            for side, opt_type in ((chain.calls, "call"), (chain.puts, "put")):
                # Flatten MultiIndex columns if yfinance returns them
                if isinstance(side.columns, pd.MultiIndex):
                    side = side.copy()
                    side.columns = [c[0] if isinstance(c, tuple) else c
                                    for c in side.columns]
                for _, row in side.iterrows():
                    iv = row.get("impliedVolatility")
                    vol = row.get("volume", 0)
                    oi = row.get("openInterest", 0)
                    quotes.append(OptionQuote(
                        strike=float(row["strike"]),
                        expiry=expiry,
                        option_type=opt_type,
                        bid=float(row.get("bid", 0)),
                        ask=float(row.get("ask", 0)),
                        last=float(row.get("lastPrice", 0)),
                        volume=int(vol) if pd.notna(vol) else 0,
                        open_interest=int(oi) if pd.notna(oi) else 0,
                        implied_vol=float(iv) if pd.notna(iv) else None,
                    ))

        return OptionChain(symbol=symbol, spot_price=spot, quotes=quotes)

    def get_snapshot(self, symbol: str) -> dict:
        """Fetch a fundamentals snapshot via yfinance ticker.info."""
        from datetime import date as _date

        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        def _f(key):
            v = info.get(key)
            return float(v) if v is not None else None

        def _i(key):
            v = info.get(key)
            return int(v) if v is not None else None

        return {
            "symbol": symbol.upper(),
            "date": _date.today().isoformat(),
            "last_price": _f("currentPrice") or _f("regularMarketPrice"),
            "open_price": _f("open") or _f("regularMarketOpen"),
            "high_price": _f("dayHigh") or _f("regularMarketDayHigh"),
            "low_price": _f("dayLow") or _f("regularMarketDayLow"),
            "prev_close": _f("previousClose") or _f("regularMarketPreviousClose"),
            "volume": _i("volume") or _i("regularMarketVolume"),
            "turnover": None,
            "market_cap": _f("marketCap"),
            "pe_ratio": _f("trailingPE"),
            "pb_ratio": _f("priceToBook"),
            "dividend_yield": _f("trailingAnnualDividendYield") or _f("yield"),
            "eps": _f("trailingEps"),
            "high_52w": _f("fiftyTwoWeekHigh"),
            "low_52w": _f("fiftyTwoWeekLow"),
        }
