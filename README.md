# Portfolio Risk Dashboard

A lightweight, self-contained portfolio risk analytics tool. Submit a portfolio (tickers, share counts, cost basis) and get back historical Value-at-Risk, parametric VaR, Conditional VaR (expected shortfall), maximum drawdown, a correlation matrix, sector exposure, and P&L under several stress scenarios.

## Features

- **Historical VaR (95% / 99%)** — 5th / 1st percentile of trailing 1y daily returns
- **Parametric VaR (95% / 99%)** — Gaussian assumption: `mean - z * std`
- **Conditional VaR / Expected Shortfall** — mean return below the VaR threshold
- **Maximum Drawdown** — worst peak-to-trough loss over the period
- **Correlation Matrix** — pairwise correlations between portfolio assets
- **Sector Exposure** — % of portfolio NAV by sector (hardcoded ticker map)
- **Stress Scenarios** — P&L under 2008 GFC, 2020 COVID, +2% rate shock, 10% drawdown

## Tech Stack

- **FastAPI** for the HTTP layer
- **yfinance** for market data (1y of daily closes)
- **pandas / numpy** for the math
- **Jinja2** for the single-page UI
- **pytest** for unit tests

## Quick Start

```bash
cd portfolio-risk-dashboard
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000>, paste a JSON portfolio like:

```json
{
  "positions": [
    {"ticker": "NVDA", "shares": 50,  "cost_basis": 400},
    {"ticker": "AAPL", "shares": 100, "cost_basis": 170},
    {"ticker": "TLT",  "shares": 200, "cost_basis": 95}
  ]
}
```

…and hit **Compute Risk**.

## API

### `POST /api/risk`

**Request body:**

```json
{
  "positions": [
    {"ticker": "NVDA", "shares": 50, "cost_basis": 400}
  ]
}
```

**Response:**

```json
{
  "var_95": -0.0234,
  "var_99": -0.0412,
  "cvar_95": -0.0311,
  "cvar_99": -0.0488,
  "max_drawdown": -0.1834,
  "correlation_matrix": {"NVDA": {"NVDA": 1.0, ...}, ...},
  "sector_exposure": {"Technology": 0.62, "Other": 0.38},
  "stress_scenarios": [
    {"name": "2008 GFC",            "pnl": -12345.67},
    {"name": "2020 COVID",          "pnl": -11345.67},
    {"name": "+2% Rate Shock",      "pnl":  -1200.00},
    {"name": "Generic 10% Drawdown","pnl":  -9876.54}
  ]
}
```

VaR / CVaR / drawdown are returned as **decimal returns on portfolio NAV** (e.g. `-0.0234` = 2.34% loss). Stress scenarios return **dollar P&L**.

### `GET /`

Serves the minimal web UI (positions table + VaR gauge + correlation heatmap).

## Tests

```bash
pytest tests/
```

Tests use synthetic returns — no network required.

## Sector Map

The following tickers are mapped to sectors. Anything not in the map is bucketed as `"Other"`.

| Ticker | Sector |
|--------|--------|
| NVDA, AAPL, MSFT, GOOG, META, AMZN, TSLA, QQQ | Technology |
| JPM, BAC, WFC, GS, MS | Financials |
| XOM, CVX, COP | Energy |
| GLD, SLV, GLDM | Commodities |
| SPY, IVV, VOO | Index / Broad Market |
| TLT, IEF, BND, AGG | Fixed Income / Treasuries |

## Notes & Limitations

- yfinance pulls can be slow / rate-limited — allow 5–15 s for an unfamiliar portfolio.
- VaR/CVaR computed on **equal-weight proxy returns** per asset; for a real risk engine, weight the historical returns by portfolio weights and recompute.
- Parametric VaR assumes normality — fat tails will underestimate tail risk.