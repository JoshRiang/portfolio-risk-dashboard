# Portfolio Risk Dashboard

Portfolio risk analytics: VaR, CVaR, drawdown, correlation, sector exposure, stress scenarios.

FastAPI app that computes the full risk payload for a submitted portfolio. Historical VaR (non-parametric percentile) and parametric VaR (Gaussian) at 95% / 99%, Conditional VaR / expected shortfall, max drawdown, pairwise correlation matrix, sector exposure (hardcoded map: Technology, Financials, Energy, Healthcare, Consumer Staples, Commodities, Broad Market, Fixed Income, Other), and four stress scenarios (2008 GFC −37%, 2020 COVID −34%, +2% rate shock, generic −10% drawdown). Market data via yfinance with a 7-day in-memory cache. Pure risk math lives in `app/risk.py` — no I/O, easy to unit-test with synthetic returns.

```bash
git clone https://github.com/JoshRiang/portfolio-risk-dashboard.git
cd portfolio-risk-dashboard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# Start the API
uvicorn app.main:app --reload

# Open the UI at http://localhost:8000
```

```python
from app.risk import (
    historical_var, parametric_var, cvar, max_drawdown,
    correlation_matrix, portfolio_returns, portfolio_nav,
    sector_exposure, stress_scenarios, sector_for,
)
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
rets = pd.Series(rng.normal(0.001, 0.02, 252))

print(f'VaR 95%:      {historical_var(rets, 0.95):.2%}')
print(f'CVaR 95%:     {cvar(rets, 0.95):.2%}')
print(f'Param VaR:    {parametric_var(rets, 0.95):.2%}')
print(f'Max drawdown: {max_drawdown(rets):.2%}')

two = {
    'AAA': pd.Series(rng.normal(0, 0.02, 252)),
    'BBB': pd.Series(rng.normal(0, 0.02, 252) * 0.6),
}
print(f'Corr: {correlation_matrix(two)["AAA"]["BBB"]:.3f}')

positions = [
    {'ticker': 'NVDA', 'shares': 50,  'cost_basis': 400},
    {'ticker': 'AAPL', 'shares': 100, 'cost_basis': 170},
    {'ticker': 'TLT',  'shares': 200, 'cost_basis': 95},
    {'ticker': 'JPM',  'shares': 75,  'cost_basis': 150},
]
prices = {'NVDA': 480.0, 'AAPL': 195.0, 'TLT': 92.0, 'JPM': 175.0}

print(f'NAV:            ${portfolio_nav(positions, prices):,.0f}')
print(f'Sector weights: {sector_exposure(positions, prices)}')

scenarios = stress_scenarios(positions, prices)
for s in scenarios:
    print(f'  {s["name"]}: {s["pnl"]:+,.2f}')
```

Project structure:

```
portfolio-risk-dashboard/
├── app/
│   ├── __init__.py         # version 0.1.0
│   ├── main.py             # FastAPI: / (HTML UI), /api/risk POST, mounts /static
│   ├── data.py             # get_prices, get_returns (yfinance + 7-day in-memory cache), clear_cache
│   ├── risk.py             # VaR/CVaR/drawdown/correlation/sector/stress/compute_risk
│   ├── static/style.css
│   └── templates/index.html
├── tests/test_risk.py      # VaR percentiles, CVaR <= VaR, drawdown known cases,
│                            # correlation identity, portfolio_nav, sector sums to 1, stress scenarios
├── requirements.txt        # fastapi, uvicorn, pandas, numpy, yfinance, jinja2, python-multipart
├── LICENSE
└── README.md
```

Testing:

```bash
pytest -v
```

All synthetic — no network required.

License: MIT — see [LICENSE](LICENSE).

Author: Joshua Riangkamang — [github.com/JoshRiang](https://github.com/JoshRiang)
