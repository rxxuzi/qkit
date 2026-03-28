"""Contract notation parser for options.

Parses shorthand like ``"QQQ 260330P"`` into a structured :class:`Contract`
dataclass.  The format is ``SYMBOL YYMMDD[C|P][STRIKE]``.

Examples::

    >>> parse_contract("QQQ 260330P")
    Contract(symbol='QQQ', expiry=datetime.date(2026, 3, 30), option_type='put', strike=None)

    >>> parse_contract("SPY 260430C450")
    Contract(symbol='SPY', expiry=datetime.date(2026, 4, 30), option_type='call', strike=450.0)
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from typing import Optional

from qkit.data.provider import OptionChain


@dataclass
class Contract:
    """Parsed option contract specification."""

    symbol: str
    expiry: datetime.date
    option_type: str        # "call" or "put"
    strike: Optional[float] = None  # None → resolve from chain (ATM)

    @property
    def expiry_str(self) -> str:
        """Expiry as ``YYYY-MM-DD`` string for matching chain data."""
        return self.expiry.isoformat()


_SPEC_RE = re.compile(
    r"^(\d{6})"           # YYMMDD
    r"([CPcp])"            # C or P
    r"(\d+(?:\.\d+)?)?$"  # optional strike
)


def parse_contract(spec: str) -> Contract:
    """Parse a contract specification string.

    Parameters
    ----------
    spec : str
        One of:
        - ``"QQQ 260330P"``       — symbol + YYMMDD + C/P
        - ``"SPY 260430C450"``    — with explicit strike
        - ``"AAPL 260530P217.5"`` — decimal strike

    Returns
    -------
    Contract

    Raises
    ------
    ValueError
        If *spec* cannot be parsed.
    """
    parts = spec.strip().split()
    if len(parts) != 2:
        raise ValueError(
            f"Expected 'SYMBOL YYMMDDC|P[STRIKE]', got {spec!r}"
        )

    symbol = parts[0].upper()
    tail = parts[1].upper()

    m = _SPEC_RE.match(tail)
    if not m:
        raise ValueError(
            f"Cannot parse contract code {tail!r}. "
            "Expected format: YYMMDD[C|P][STRIKE]"
        )

    date_str, cp, strike_str = m.groups()

    # Parse YYMMDD
    try:
        yy = int(date_str[:2])
        mm = int(date_str[2:4])
        dd = int(date_str[4:6])
        year = 2000 + yy
        expiry = datetime.date(year, mm, dd)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid date in {tail!r}: {exc}") from exc

    option_type = "call" if cp == "C" else "put"
    strike = float(strike_str) if strike_str else None

    return Contract(
        symbol=symbol,
        expiry=expiry,
        option_type=option_type,
        strike=strike,
    )


def resolve_strike(contract: Contract, chain: OptionChain,
                   spot: float) -> float:
    """Resolve the strike price for a contract.

    If ``contract.strike`` is set, return it directly.  Otherwise find
    the nearest ATM strike in the chain for the contract's expiry and
    option type.

    Returns
    -------
    float
        The resolved strike price.

    Raises
    ------
    ValueError
        If no matching quotes are found.
    """
    if contract.strike is not None:
        return contract.strike

    # Find quotes matching expiry and type
    target_exp = contract.expiry_str
    candidates = [
        q for q in chain.quotes
        if q.expiry == target_exp and q.option_type == contract.option_type
    ]

    if not candidates:
        # Try prefix match (chain might have "2026-03-30" vs "2026-3-30")
        candidates = [
            q for q in chain.quotes
            if q.option_type == contract.option_type
            and _expiry_matches(q.expiry, target_exp)
        ]

    if not candidates:
        raise ValueError(
            f"No {contract.option_type} quotes found for "
            f"{contract.symbol} expiry {target_exp}"
        )

    # Nearest to spot
    candidates.sort(key=lambda q: abs(q.strike - spot))
    return candidates[0].strike


def find_expiry(contract: Contract, chain: OptionChain) -> str:
    """Find the best matching expiry string in the chain.

    The chain expiry format may differ from ``contract.expiry_str``
    (e.g. ``"2026-03-28"`` vs ``"2026-3-28"``).  This returns the
    chain's own expiry string.

    Raises
    ------
    ValueError
        If no matching expiry is found.
    """
    target = contract.expiry_str
    for exp in chain.expiries():
        if _expiry_matches(exp, target):
            return exp

    # If exact date not found, find nearest
    available = chain.expiries()
    if not available:
        raise ValueError(f"No expiries in chain for {contract.symbol}")

    target_date = contract.expiry
    best = min(available, key=lambda e: abs(
        (datetime.date.fromisoformat(e) - target_date).days
    ))
    diff = abs((datetime.date.fromisoformat(best) - target_date).days)
    if diff > 7:
        raise ValueError(
            f"No expiry near {target} for {contract.symbol}. "
            f"Available: {', '.join(available[:5])}"
        )
    return best


def _expiry_matches(chain_expiry: str, target: str) -> bool:
    """Fuzzy date match between chain expiry and target."""
    try:
        d1 = datetime.date.fromisoformat(chain_expiry)
        d2 = datetime.date.fromisoformat(target)
        return d1 == d2
    except ValueError:
        return chain_expiry == target
