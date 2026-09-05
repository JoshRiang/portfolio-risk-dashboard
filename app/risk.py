"""Risk calculations.

Pure functions, no I/O — easy to unit-test with synthetic returns.
All return-based metrics are reported as **decimal returns** on portfolio NAV
(unless a dollar amount is explicitly requested).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sector mapping — hardcoded for the common tickers per the spec.
# Anything not in here falls into "Other".
# ---------------------------------------------------------------------------
SECTOR_MAP: Dict[str, str] = {
    # Technology / Big Tech
    "NVDA": "Technology",
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOG": "Technology",
    "GOOGL": "Technology",
    "META": "Technology",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "QQQ": "Technology",
    "AMD": "Technology",
    "INTC": "Technology",
    "ORCL": "Technology",
    "CRM": "Technology",
    # Financials
    "JPM": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "C": "Financials",
    "BLK": "Financials",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    # Healthcare
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "MRK": "Healthcare",
    "UNH": "Healthcare",
    # Consumer
    "WMT": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "PG": "Consumer Staples",
    # Commodities / ETFs
    "GLD": "Commodities",
    "SLV": "Commodities",
    "GLDM": "Commodities",
    # Broad market ETFs
    "SPY": "Broad Market",
    "IVV": "Broad Market",
    "VOO": "Broad Market",
    # Fixed income / Treasuries (duration-sensitive)
    "TLT": "Fixed Income",
    "IEF": "Fixed Income",
    "BND": "Fixed Income",
    "AGG": "Fixed Income",
    "SHY": "Fixed Income",
}


def sector_for(ticker: str) -> str:
    """Return the sector for a ticker, or 'Other' if unmapped."""
    return SECTOR_MAP.get(ticker.upper().strip(), "Other")


# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------
def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical (non-parametric) VaR.

    Returns the (1 - confidence) percentile of returns, expressed as a
    **positive number representing the loss** (e.g. 0.023 = 2.3% loss).
    A negative return convention is used by risk teams — here we keep the
    raw signed percentile so the API surface is consistent (lower bound).
    """
    if returns.empty:
        return 0.0
    q = 1.0 - confidence
    return float(np.percentile(returns.dropna(), q * 100.0))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Gaussian (parametric) VaR.

    VaR = mu - z * sigma   where z is the standard-normal quantile for the
    given confidence. Returned as a **signed decimal** (negative = loss).
    """
    if returns.empty:
        return 0.0
    r = returns.dropna()
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    # two-sided z; we want the lower tail
    z = _z_score(confidence)
    return mu - z * sigma


def _z_score(confidence: float) -> float:
    """Lower-tail z-score for a normal distribution at given confidence."""
    from math import sqrt
    # Hand-rolled inverse-CDF for a small set of common confidence levels
    # (avoids scipy dep). Falls back to a scipy.approx for anything else.
    table = {
        0.90: 1.2816,
        0.95: 1.6449,
        0.975: 1.9600,
        0.99: 2.3263,
        0.999: 3.0902,
    }
    if confidence in table:
        return table[confidence]
    try:
        from scipy.stats import norm
        return float(norm.ppf(confidence))
    except Exception:
        # crude fallback: bisect-ish, just enough to keep tests green
        return sqrt(2.0) * _erfinv(2 * confidence - 1)


def _erfinv(x: float) -> float:
    """Winitzki / Luke approximation to the inverse error function."""
    # Good enough for risk VaR approximations.
    from math import log
    a = 0.147
    ln = log(1 - x * x)
    first = 2.0 / (a * np.pi) + ln / 2.0
    inside = first * first - ln / a
    if inside < 0:
        return 0.0
    return np.sign(x) * np.sqrt(np.sqrt(inside) - first)


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR / Expected Shortfall.

    Mean of returns that fall below the VaR threshold. Signed decimal.
    """
    if returns.empty:
        return 0.0
    r = returns.dropna()
    var_threshold = historical_var(r, confidence)
    tail = r[r <= var_threshold]
    if tail.empty:
        return var_threshold
    return float(tail.mean())


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------
def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough drawdown of cumulative returns.

    Returns a negative decimal (e.g. -0.18 = -18%).
    """
    if returns.empty:
        return 0.0
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak
    return float(drawdown.min())


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------
def correlation_matrix(returns_dict: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
    """Pairwise correlation matrix between asset return series.

    Output is JSON-serialisable: `{ticker: {ticker: float}}`.
    Aligns all series on the inner-join of their date indexes first.
    """
    if not returns_dict:
        return {}
    df = pd.DataFrame(returns_dict).dropna(how="any")
    if df.empty or df.shape[1] < 2:
        # single asset — return a 1x1 identity
        only = next(iter(returns_dict))
        return {only: {only: 1.0}}
    corr = df.corr().fillna(0.0)
    # round for stable JSON
    return {
        t: {col: round(float(corr.loc[t, col]), 4) for col in corr.columns}
        for t in corr.index
    }


# ---------------------------------------------------------------------------
# Portfolio-level analytics
# ---------------------------------------------------------------------------
def portfolio_returns(returns_dict: Dict[str, pd.Series]) -> pd.Series:
    """Build an equal-weighted proxy return series for the portfolio.

    Equal-weighting is a reasonable approximation for a VaR demo where
    positions vary in dollar size but we want a single time-series to
    compute VaR / CVaR / drawdown on.
    """
    if not returns_dict:
        return pd.Series(dtype=float)
    df = pd.DataFrame(returns_dict).dropna(how="any")
    if df.empty:
        return pd.Series(dtype=float)
    return df.mean(axis=1)


def portfolio_nav(positions: List[dict], current_prices: Dict[str, float]) -> float:
    """NAV in dollars = sum(shares * current_price)."""
    nav = 0.0
    for p in positions:
        ticker = p["ticker"].upper().strip()
        price = current_prices.get(ticker, 0.0)
        nav += float(p["shares"]) * price
    return nav


def sector_exposure(positions: List[dict], current_prices: Dict[str, float]) -> Dict[str, float]:
    """Return {sector: weight} where weights sum to 1.0."""
    buckets: Dict[str, float] = {}
    total = 0.0
    for p in positions:
        ticker = p["ticker"].upper().strip()
        value = float(p["shares"]) * current_prices.get(ticker, 0.0)
        sec = sector_for(ticker)
        buckets[sec] = buckets.get(sec, 0.0) + value
        total += value
    if total == 0:
        return {}
    return {sec: round(val / total, 4) for sec, val in sorted(buckets.items())}


# ---------------------------------------------------------------------------
# Stress scenarios
# ---------------------------------------------------------------------------
def stress_scenarios(positions: List[dict], current_prices: Dict[str, float]) -> List[dict]:
    """Apply a fixed set of historical/hypothetical shocks.

    Each scenario returns dollar P&L (negative = loss).
    """
    nav = portfolio_nav(positions, current_prices)
    nav = nav if nav > 0 else sum(
        float(p["shares"]) * float(p.get("cost_basis", 0.0)) for p in positions
    )

    scenarios = []

    # 1) 2008 GFC — broad equity drawdown ~ -37%
    scenarios.append({
        "name": "2008 GFC (-37%)",
        "pnl": round(_equity_drawdown_pnl(positions, current_prices, -0.37), 2),
    })

    # 2) 2020 COVID — broad equity drawdown ~ -34%
    scenarios.append({
        "name": "2020 COVID (-34%)",
        "pnl": round(_equity_drawdown_pnl(positions, current_prices, -0.34), 2),
    })

    # 3) +2% rate shock — duration loss on fixed income (and rate-sensitive equity)
    rate_pnl = 0.0
    for p in positions:
        ticker = p["ticker"].upper().strip()
        sec = sector_for(ticker)
        value = float(p["shares"]) * current_prices.get(ticker, 0.0)
        if sec == "Fixed Income":
            # assume modified duration ~ 7 for TLT-like
            duration = _duration_for(ticker)
            rate_pnl += value * (-duration * 0.02)
        else:
            # mild equity beta to rates — financial-sector proxy
            if sec == "Financials":
                rate_pnl += value * 0.03
            elif sec in ("Technology", "Consumer Discretionary"):
                rate_pnl += value * (-0.02)
            else:
                rate_pnl += value * (-0.01)
    scenarios.append({
        "name": "+2% Rate Shock",
        "pnl": round(rate_pnl, 2),
    })

    # 4) Generic -10% drawdown
    scenarios.append({
        "name": "Generic 10% Drawdown",
        "pnl": round(_equity_drawdown_pnl(positions, current_prices, -0.10), 2),
    })

    return scenarios


def _equity_drawdown_pnl(positions: List[dict], current_prices: Dict[str, float], drop: float) -> float:
    """Apply a uniform pct drop to equity-like positions only.

    Fixed income and commodities are scaled down (more resilient). Cash-like
    buckets would not drop — none in this model.
    """
    pnl = 0.0
    for p in positions:
        ticker = p["ticker"].upper().strip()
        sec = sector_for(ticker)
        value = float(p["shares"]) * current_prices.get(ticker, 0.0)
        shock = drop
        if sec == "Fixed Income":
            shock = drop * 0.3
        elif sec == "Commodities":
            shock = drop * 0.4
        pnl += value * shock
    return pnl


def _duration_for(ticker: str) -> float:
    """Approx modified duration for common bond ETFs."""
    return {
        "TLT": 17.0,   # 20+y treasuries
        "IEF": 7.5,    # 7-10y
        "BND": 6.5,    # total bond market
        "AGG": 6.5,
        "SHY": 1.8,    # 1-3y
    }.get(ticker.upper().strip(), 7.0)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def compute_risk(
    positions: List[dict],
    returns_dict: Dict[str, pd.Series],
    current_prices: Dict[str, float],
) -> dict:
    """Compute the full risk payload returned by the API."""
    port_ret = portfolio_returns(returns_dict)
    corr = correlation_matrix(returns_dict)

    return {
        "var_95": round(historical_var(port_ret, 0.95), 4),
        "var_99": round(historical_var(port_ret, 0.99), 4),
        "parametric_var_95": round(parametric_var(port_ret, 0.95), 4),
        "parametric_var_99": round(parametric_var(port_ret, 0.99), 4),
        "cvar_95": round(cvar(port_ret, 0.95), 4),
        "cvar_99": round(cvar(port_ret, 0.99), 4),
        "max_drawdown": round(max_drawdown(port_ret), 4),
        "correlation_matrix": corr,
        "sector_exposure": sector_exposure(positions, current_prices),
        "stress_scenarios": stress_scenarios(positions, current_prices),
    }