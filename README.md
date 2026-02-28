# Betfair Arbitrage Engine

Automated detection and execution of arbitrage opportunities on Betfair Exchange (Spain). Paper trading first, then live with small capital.

See [brief.md](brief.md) for strategy and architecture, [CLAUDE.md](CLAUDE.md) for development rules. [docs/BETFAIR_REFERENCES.md](docs/BETFAIR_REFERENCES.md) summarizes Betfair’s official [Sample Code](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687537/Sample+Code) and related links.

---

## Local AI (Qwen)

We use **Qwen via Ollama** (e.g. `qwen25-32b`) for subagents that don’t need full coding power: Tier 1 tasks (scaffolding, docstrings, tests from spec, simple wrappers, UI) and straightforward Tier 2 work. Full-power models (e.g. Claude) are reserved for Tier 3 (financial-critical code) and complex integration. See [CLAUDE.md](CLAUDE.md) for tier rules and the “Prefer Qwen” guidance. Ensure Ollama is running and the model is available (e.g. `ollama list`) when dispatching Tier 1 agents.

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env        # then edit .env with your credentials
pytest tests/ -v
```

---

## While you wait for your API key

1. **Confirm Betfair account**
   - Log in at [betfair.es](https://betfair.es), KYC and funding done.

2. **Request streaming**
   - Once you have your app key, email **automation@betfair.com** to activate **live streaming** on that key (required for real-time prices).

3. **Telegram alerts (optional)**
   - Create a bot with [@BotFather](https://t.me/BotFather), get `TELEGRAM_BOT_TOKEN`.
   - Start a chat with the bot, get your `TELEGRAM_CHAT_ID` (e.g. via [api.telegram.org](https://api.telegram.org)).
   - Add both to `.env` so circuit breaker and failures can alert you.

4. **Run tests**
   - `PAPER_TRADING=true pytest tests/ -v` — all should pass without an API key.

5. **Check readiness**
   - When your key arrives, run: `python scripts/check_env.py` to confirm `.env` is ready before starting the stream.

6. **Backtest data (optional)**
   - You can buy [Betfair Historical Data](https://historicdata.betfair.com) later to run `strategy/backtester.py` on real price history.

---

## Application keys (Betfair)

Each account has two keys ([Betfair: How do I get started?](https://support.developer.betfair.com/hc/en-us/articles/115003864651-How-do-I-get-started)):

- **Delayed key (active)** — For development and testing. Use this in `.env` as `BF_APP_KEY` to test locally with real data (REST and, if enabled, stream). Data may be delayed.
- **Live key (inactive until approved)** — For real betting. Requires application and a one-off activation fee. After approval, switch `BF_APP_KEY` in `.env` to the Live key for production.

## Test arbitrage trading (no login)

Run the full pipeline with synthetic data (scanner → risk manager → paper executor):

```bash
python scripts/run_arbitrage_test.py
```

You should see one market with an arb executed (paper) and one with no opportunity. Confirms the trading loop works.

## Test with real data locally

1. In `.env`, set `BF_APP_KEY` to your **Delayed** application key.
2. **Certificate login (recommended):**  
   One-time setup: run `python scripts/setup_betfair_cert.py` — it creates the cert if missing and prints the exact link to upload it. Then upload `certs/client-2048.crt` at [Betfair Spain → My Security → Automated Betting Program Access](https://myaccount.betfair.es/accountdetails/mysecurity?showAPI=1), and set `BF_CERTS_PATH=./certs` in `.env`. See [certs/README.md](certs/README.md).
3. Run: `python scripts/test_real_data.py`  
   This logs in (via cert if `BF_CERTS_PATH` points at a folder with `.crt` and `.key`, else interactive), fetches football markets (ES), gets prices via REST, runs the scanner, and prints opportunities or "no arb" per market.

## Dashboard (UI)

Web UI to start/stop trading and watch live activity, P&L, and trades. Run from **project root** so `.env` is found (the script sets cwd and loads `.env` automatically):

```bash
cd /path/to/Arbitrage
python scripts/run_dashboard.py
```

Or: `uvicorn monitoring.dashboard:app --reload --host 0.0.0.0` (run from project root).

Open **http://127.0.0.1:8000**. Use **Start trading** to run the engine (login → watchlist → poll prices → scan → paper execute). The page shows balance, daily P&L, markets watched, a live feed of scans/opportunities/trades, and a trades table. State polls every 2 seconds.

If the dashboard still shows **no markets**, run `python scripts/seed_market_ids.py` to fetch market IDs and write `MARKET_IDS=id1,id2,...` to `.env`. Alternatively run `python scripts/list_markets.py` to inspect what the API returns and set `MARKET_IDS` manually.

## When going live

1. Set `BF_APP_KEY` in `.env` to your **Live** application key (after it’s activated).
2. Run `python scripts/check_env.py` to verify credentials and config.
3. Keep `PAPER_TRADING=true` and run the engine in paper mode first.
4. Only after 2+ weeks of paper trading and meeting the gate in [CLAUDE.md](CLAUDE.md) consider setting `PAPER_TRADING=false`.

---

## Project layout

- `config.py` — env-based config (no secrets in code)
- `data/` — Betfair stream, market catalogue, price cache
- `core/` — scanner, commission, stake calculator, risk manager
- `execution/` — paper/live executor, order monitor
- `strategy/` — market selector, liquidity filter, backtester
- `monitoring/` — dashboard, Telegram alerting, trade logger
- `main.py` — paper loop entry (set `MARKET_IDS` or use stream + market_selector)
- **Dashboard** — run `uvicorn monitoring.dashboard:app` then open the UI to start/stop trading and view P&L
