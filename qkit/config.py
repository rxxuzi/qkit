"""Global configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()

DATA_PROVIDER = os.getenv("DATA_PROVIDER", "yfinance")

MOOMOO_HOST = os.getenv("MOOMOO_HOST", "127.0.0.1")
MOOMOO_PORT = int(os.getenv("MOOMOO_PORT", "11111"))

# US 10-year Treasury yield as of 2026-03. Update periodically.
RISK_FREE_RATE = 0.043

TRADING_DAYS_PER_YEAR = 252
