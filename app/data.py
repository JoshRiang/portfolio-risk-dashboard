"""Market data loader.

Wraps yfinance with a tiny in-memory cache so repeated requests for the same
ticker don't hammer the upstream API. Caches survive the lifetime of the
process — restart clears them.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import yfinance as yf


_CACHE: Dict[str, pd.Series] = {}
_CACHE_MAX_AGE_DAYS = 7  # stale-ish but fine for demo; yfinance limits pulls


def get_returns(ticker: str, period: str = "1y") -> pd.Series:
    """Return daily simple returns for a single ticker.

    Falls back to an empty Series if yfinance can't find the ticker.
    """
    ticker = ticker.upper().strip()
    cache_key = f"{ticker}:{period}"

    if cache_key in _CACHE:
        cached = _CACHE[cache_key]
        # sanity-check it's not a stale empty frame
        if not cached.empty:
            return cached

    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    except Exception:
        return pd.Series(dtype=float)

    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)

    closes = df["Close"].astype(float).dropna()
    returns = closes.pct_change().dropna()
    _CACHE[cache_key] = returns
    return returns


def get_prices(ticker: str, period: str = "1y") -> pd.Series:
    """Return adjusted close prices for a single ticker."""
    ticker = ticker.upper().strip()
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    except Exception:
        return pd.Series(dtype=float)
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    return df["Close"].astype(float).dropna()


def clear_cache() -> None:
    """Clear the in-memory price/return cache."""
    _CACHE.clear()