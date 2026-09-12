"""Tests for the risk math — all synthetic, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.risk import (
    correlation_matrix,
    cvar,
    historical_var,
    max_drawdown,
    parametric_var,
    portfolio_nav,
    portfolio_returns,
    sector_exposure,
    sector_for,
    stress_scenarios,
)


# ---------------------------------------------------------------------------
# Fixtures: deterministic synthetic returns
# ---------------------------------------------------------------------------
@pytest.fixture
def normal_returns() -> pd.Series:
    """~252 days of N(0.001, 0.02) returns — a happy, mean-positive market."""

# Maintenance: last reviewed 2026-09-12 (daily improvement cycle)
    rng = np.random.default_rng(seed=42)
    return pd.Series(rng.normal(loc=0.001, scale=0.02, size=252))


@pytest.fixture
def bearish_returns() -> pd.Series:
    """Strongly negative drift — useful for tail tests."""
    rng = np.random.default_rng(seed=7)
    return pd.Series(rng.normal(loc=-0.005, scale=0.04, size=252))


@pytest.fixture
def two_assets() -> dict:
    """Two correlated assets."""
    rng = np.random.default_rng(seed=11)
    n = 252
    a = rng.normal(0, 0.02, n)
    b = a * 0.6 + rng.normal(0, 0.015, n)  # correlated
    return {
        "AAA": pd.Series(a),
        "BBB": pd.Series(b),
    }


@pytest.fixture
def positions() -> list:
    return [
        {"ticker": "NVDA", "shares": 50, "cost_basis": 400},
        {"ticker": "AAPL", "shares": 100, "cost_basis": 170},
        {"ticker": "TLT",  "shares": 200, "cost_basis": 95},
        {"ticker": "JPM",  "shares": 75, "cost_basis": 150},
        {"ticker": "XYZ_UNMAPPED", "shares": 10, "cost_basis": 50},
    ]


@pytest.fixture
def prices() -> dict:
    return {"NVDA": 480.0, "AAPL": 195.0, "TLT": 92.0, "JPM": 175.0, "XYZ_UNMAPPED": 50.0}


# ---------------------------------------------------------------------------
# historical_var
# ---------------------------------------------------------------------------
def test_historical_var_95_is_negative_in_loss_regime(bearish_returns):
    var = historical_var(bearish_returns, 0.95)
    assert var < 0
    # tail should be substantially negative for a -0.5%/day drift, 4% vol
    assert var < -0.03


def test_historical_var_99_more_extreme_than_95(bearish_returns):
    assert historical_var(bearish_returns, 0.99) < historical_var(bearish_returns, 0.95)


def test_historical_var_empty_series():
    assert historical_var(pd.Series(dtype=float), 0.95) == 0.0


def test_historical_var_known_percentile():
    # Build a known series: 5th percentile should be -0.05
    s = pd.Series(np.linspace(-0.05, 0.05, 100))
    var = historical_var(s, 0.95)
    assert pytest.approx(var, abs=0.01) == -0.05


# ---------------------------------------------------------------------------
# parametric_var
# ---------------------------------------------------------------------------
def test_parametric_var_matches_formula(normal_returns):
    r = normal_returns.dropna()
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    expected = mu - 1.6449 * sigma
    assert pytest.approx(parametric_var(normal_returns, 0.95), rel=1e-4) == expected


def test_parametric_var_higher_confidence_more_extreme(bearish_returns):
    assert parametric_var(bearish_returns, 0.99) < parametric_var(bearish_returns, 0.95)


# ---------------------------------------------------------------------------
# CVaR
# ---------------------------------------------------------------------------
def test_cvar_worse_than_var(bearish_returns):
    """CVaR (expected shortfall) should always be at least as bad as VaR."""
    var = historical_var(bearish_returns, 0.95)
    es  = cvar(bearish_returns, 0.95)
    assert es <= var


def test_cvar_in_tail(bearish_returns):
    """CVaR should lie somewhere in the left tail."""
    es = cvar(bearish_returns, 0.95)
    r = bearish_returns.dropna()
    assert es >= float(r.min())
    assert es <= historical_var(bearish_returns, 0.95)


def test_cvar_empty():
    assert cvar(pd.Series(dtype=float), 0.95) == 0.0


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------
def test_max_drawdown_negative_and_reasonable(normal_returns):
    mdd = max_drawdown(normal_returns)
    assert mdd <= 0.0
    assert mdd > -1.0  # not a total wipeout


def test_max_drawdown_synthetic_perfect_bull():
    # monotonic up → drawdown is 0
    s = pd.Series([0.01] * 50)
    assert max_drawdown(s) == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_synthetic_known():
    # up 100%, then down 50% → peak 2.0, trough 1.0 → -50%
    s = pd.Series([1.0, -0.5])
    assert max_drawdown(s) == pytest.approx(-0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# correlation_matrix
# ---------------------------------------------------------------------------
def test_correlation_diagonal_is_one(two_assets):
    m = correlation_matrix(two_assets)
    for t in ("AAA", "BBB"):
        assert m[t][t] == pytest.approx(1.0, abs=1e-6)


def test_correlation_offdiag_positive_when_correlated(two_assets):
    m = correlation_matrix(two_assets)
    assert m["AAA"]["BBB"] > 0.5
    assert m["BBB"]["AAA"] == pytest.approx(m["AAA"]["BBB"], abs=1e-6)


def test_correlation_empty():
    assert correlation_matrix({}) == {}


def test_correlation_single_asset_identity():
    s = pd.Series(np.random.default_rng(0).normal(0, 0.02, 100))
    m = correlation_matrix({"X": s})
    assert m == {"X": {"X": 1.0}}


# ---------------------------------------------------------------------------
# portfolio_returns
# ---------------------------------------------------------------------------
def test_portfolio_returns_equal_weight(two_assets):
    pr = portfolio_returns(two_assets)
    # mean of AAA + BBB for each row
    expected = (two_assets["AAA"] + two_assets["BBB"]) / 2
    pd.testing.assert_series_equal(pr.reset_index(drop=True), expected.reset_index(drop=True))


def test_portfolio_returns_empty():
    assert portfolio_returns({}).empty


# ---------------------------------------------------------------------------
# Sector mapping
# ---------------------------------------------------------------------------
def test_sector_for_known():
    assert sector_for("NVDA") == "Technology"
    assert sector_for("JPM") == "Financials"
    assert sector_for("TLT") == "Fixed Income"
    assert sector_for("GLD") == "Commodities"


def test_sector_for_unknown_falls_back_to_other():
    assert sector_for("ZZZZ") == "Other"
    assert sector_for("xyz_unmapped") == "Other"  # case-insensitive


# ---------------------------------------------------------------------------
# NAV / sector exposure
# ---------------------------------------------------------------------------
def test_portfolio_nav(positions, prices):
    nav = portfolio_nav(positions, prices)
    expected = 50*480 + 100*195 + 200*92 + 75*175 + 10*50
    assert nav == pytest.approx(expected, rel=1e-6)


def test_sector_exposure_sums_to_one(positions, prices):
    se = sector_exposure(positions, prices)
    assert pytest.approx(sum(se.values()), abs=1e-6) == 1.0


def test_sector_exposure_includes_other_bucket(positions, prices):
    se = sector_exposure(positions, prices)
    assert "Other" in se  # XYZ_UNMAPPED


def test_sector_exposure_all_fixed_income():
    se = sector_exposure(
        [{"ticker": "TLT", "shares": 10, "cost_basis": 100}],
        {"TLT": 100.0},
    )
    assert se == {"Fixed Income": 1.0}


# ---------------------------------------------------------------------------
# Stress scenarios
# ---------------------------------------------------------------------------
def test_stress_scenarios_returns_four(positions, prices):
    ss = stress_scenarios(positions, prices)
    assert len(ss) == 4
    for s in ss:
        assert "name" in s and "pnl" in s


def test_stress_gfc_more_severe_than_covid(positions, prices):
    ss = stress_scenarios(positions, prices)
    by_name = {s["name"]: s["pnl"] for s in ss}
    # GFC deeper drop → bigger loss than COVID
    assert by_name["2008 GFC (-37%)"] < by_name["2020 COVID (-34%)"]


def test_stress_gfc_is_a_loss(positions, prices):
    ss = stress_scenarios(positions, prices)
    by_name = {s["name"]: s["pnl"] for s in ss}
    assert by_name["2008 GFC (-37%)"] < 0
    assert by_name["2020 COVID (-34%)"] < 0
    assert by_name["Generic 10% Drawdown"] < 0


def test_stress_rate_shock_penalises_fixed_income():
    # A pure-TLT portfolio should lose on a +2% rate shock
    ss = stress_scenarios(
        [{"ticker": "TLT", "shares": 100, "cost_basis": 100}],
        {"TLT": 100.0},
    )
    by_name = {s["name"]: s["pnl"] for s in ss}
    assert by_name["+2% Rate Shock"] < 0


def test_stress_handles_missing_prices_falls_back_to_cost_basis():
    positions = [
        {"ticker": "TLT", "shares": 100, "cost_basis": 100},
        {"ticker": "NVDA", "shares": 10, "cost_basis": 400},
    ]
    # empty prices — nav falls back to cost basis
    ss = stress_scenarios(positions, {})
    assert all(isinstance(s["pnl"], (int, float)) for s in ss)