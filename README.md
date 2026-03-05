# Betfair Arbitrage Engine

Automated detection and execution of arbitrage opportunities on Betfair Exchange (Spain). Paper trading first, then live with small capital.

See [brief.md](brief.md) for strategy and architecture, [CLAUDE.md](CLAUDE.md) for development rules. [docs/BETFAIR_REFERENCES.md](docs/BETFAIR_REFERENCES.md) summarizes Betfair’s official [Sample Code](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687537/Sample+Code) and related links.

---

## Local AI (optional)

The trading runtime runs without any local LLM/Ollama dependencies.

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

The dashboard also shows:
- **System Status** (explicit PAPER/LIVE mode + feed/prediction/architect/risk health).
- **Model Artifacts** (scoring/fill model present + last modified).
- **Performance Gates** (latest baseline vs ML report, pass/fail and lift metrics).
- **Opportunity Funnel** and decision split (EXECUTE/DEFER/SKIP).

### Stop/Resume behavior (state persistence)

Stopping and restarting the engine keeps learning/state by default:
- Paper account balance + recent trade log tail are restored from `PAPER_STATE_PATH`.
- Prediction model weights are saved in `PREDICTION_MODEL_DIR`.
- Prediction account state (balance, counters, open positions) is restored from `PREDICTION_STATE_DIR`.
- Candidate logs and CLV logs are append-only on disk.

Defaults:
- `PAPER_STATE_PATH=data/state/paper_executor_state.json`
- `PAPER_TRADES_LOG_PATH=data/state/paper_trades.jsonl`
- `PREDICTION_STATE_DIR=data/prediction/state`

If the dashboard still shows **no markets**, run `python scripts/seed_market_ids.py` to fetch market IDs and write `MARKET_IDS=id1,id2,...` to `.env`. Alternatively run `python scripts/list_markets.py` to inspect what the API returns and set `MARKET_IDS` manually.

## When going live

1. Set `BF_APP_KEY` in `.env` to your **Live** application key (after it’s activated).
2. Run `python scripts/check_env.py` to verify credentials and config.
3. Keep `PAPER_TRADING=true` and run the engine in paper mode first.
4. Only after 2+ weeks of paper trading and meeting the gate in [CLAUDE.md](CLAUDE.md) consider setting `PAPER_TRADING=false`.

## Troubleshooting

To check/fix macOS deps: ./scripts/ensure_libomp.sh

### XGBoost / Contrarian training: `Library not loaded: libomp.dylib` (macOS)

The funding contrarian strategy and Strategy Orchestrator use XGBoost, which needs the OpenMP runtime on macOS. If you see:

```text
XGBoostError: Library not loaded: @rpath/libomp.dylib
Mac OSX users: Run `brew install libomp` to install OpenMP runtime.
```

install the OpenMP library with Homebrew (in a terminal where `brew` is available):

```bash
brew install libomp
```

Then restart the Funding engine or dashboard. No Python or repo code changes are required.

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

## Quant scoring and data capture

- `strategy/features.py` builds deterministic microstructure features per opportunity.
- `strategy/model_inference.py` scores each opportunity (`EXECUTE` / `DEFER` / `SKIP`) with model fallback.
- `data/candidate_logger.py` writes all scan outcomes to `data/candidates/*.jsonl` for supervised training.
- Dashboard exposes an opportunity funnel (scanned, scored, deferred, executed) and scoring telemetry.

### Gate validation (baseline vs ML)

Run:

```bash
python scripts/validate_performance_gates.py --input-dir data/candidates --output-dir data/reports/performance_gates --min-samples 200
```

This writes:
- `data/reports/performance_gates/gate_report_<timestamp>.json`
- `data/reports/performance_gates/latest.json`

And evaluates gates:
- Precision lift >= 25%
- Profit/trade lift >= 15%
- Max drawdown not increased

## Multi-agent delivery

See [docs/MULTI_AGENT_DELIVERY.md](docs/MULTI_AGENT_DELIVERY.md) for tier routing (cheap vs high-capability agents), lane ownership, and merge protocol.

## Predictive model track (optional, paper-mode)

This is separate from arbitrage and can be tested independently.
For the live parallel comparison league in dashboard, see [docs/PREDICTION_MODELS.md](docs/PREDICTION_MODELS.md).
For the meta-controller that tunes model parameters, see [docs/LEARNING_ARCHITECT.md](docs/LEARNING_ARCHITECT.md).

1. Generate a synthetic dataset (for tooling validation):
```bash
python scripts/generate_synthetic_prediction_data.py --output data/prediction/training.csv
```
2. Train the residual predictive model:
```bash
python scripts/train_predictive_model.py --input data/prediction/training.csv --output models/predictive_model_v1.json
```
3. Backtest the saved model:
```bash
python scripts/backtest_predictive_model.py --model models/predictive_model_v1.json --input data/prediction/training.csv
```

CSV schema expected by training/backtest scripts:
`timestamp,base_prob,odds,label,spread_mean,imbalance,depth_total_eur,price_velocity,short_volatility,time_to_start_sec,in_play`
