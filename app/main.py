"""FastAPI app — Portfolio Risk Dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import risk
from .data import get_prices, get_returns

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Portfolio Risk Dashboard",
    version="0.1.0",
    description="Compute VaR, CVaR, drawdown, correlation, sector exposure, and stress P&L.",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Position(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    shares: float = Field(..., gt=0)
    cost_basis: float = Field(..., gt=0)


class PortfolioRequest(BaseModel):
    positions: List[Position]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the minimal web UI."""
    template = (BASE_DIR / "templates" / "index.html").read_text()
    return HTMLResponse(content=template)


@app.post("/api/risk")
def api_risk(req: PortfolioRequest) -> dict:
    """Compute the full risk payload for a submitted portfolio."""
    if not req.positions:
        raise HTTPException(status_code=400, detail="At least one position required.")

    positions = [p.model_dump() for p in req.positions]
    tickers = [p["ticker"].upper().strip() for p in positions]

    # Pull data
    returns_dict = {t: get_returns(t, period="1y") for t in tickers}
    current_prices = {t: _latest_price(get_prices(t, period="1y")) for t in tickers}

    # Drop tickers we couldn't resolve — but keep going if at least one is good
    missing = [t for t, r in returns_dict.items() if r.empty]
    if len(missing) == len(tickers):
        raise HTTPException(
            status_code=502,
            detail=f"Could not load any of: {', '.join(tickers)}. "
                   "Check ticker symbols and try again.",
        )

    good_returns = {t: r for t, r in returns_dict.items() if not r.empty}
    # Restrict positions/price dicts to good tickers for nav calculations
    good_set = set(good_returns)
    positions_filtered = [p for p in positions if p["ticker"].upper().strip() in good_set]
    current_prices_filtered = {t: p for t, p in current_prices.items() if t in good_set}

    payload = risk.compute_risk(
        positions=positions_filtered,
        returns_dict=good_returns,
        current_prices=current_prices_filtered,
    )
    payload["warnings"] = [f"No data for {m}" for m in missing] if missing else []
    payload["tickers"] = list(good_set)
    return payload


def _latest_price(prices) -> float:
    if prices is None or len(prices) == 0:
        return 0.0
    return float(prices.iloc[-1])