"""
Dashboard API and UI: start/stop trading, live state, P&L, trades, events.
Run: uvicorn monitoring.dashboard:app --reload --host 0.0.0.0
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from monitoring.engine import TradingEngine

_engine = TradingEngine()
_template_path = Path(__file__).parent / "templates" / "dashboard.html"


def get_engine() -> TradingEngine:
    return _engine


def _html() -> str:
    return _template_path.read_text(encoding="utf-8")


app = FastAPI(title="Betfair Arb Dashboard")


@app.get("/", response_class=HTMLResponse)
def index():
    return _html()


@app.post("/api/start")
def api_start():
    return get_engine().start()


@app.post("/api/stop")
def api_stop():
    return get_engine().stop()


@app.get("/api/state")
def api_state():
    return get_engine().get_state()
