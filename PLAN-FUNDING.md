# Crypto Funding Rate Arbitrage Module — Implementation Plan

> **Purpose**: This document is a prompting guide for Claude Code. Add it to the repo root. Reference it when building the funding module: `@PLAN-FUNDING.md implement phase 1`.

## Relationship to Existing Repo

This is a **new vertical** within the existing `betfair-arb-engine/` project. It reuses:
- `core/commission.py` pattern → `funding/core/fee_calculator.py`
- `execution/paper_executor.py` pattern → `funding/execution/paper_executor.py`
- `execution/executor.py` routing pattern (paper vs live via env flag)
- `monitoring/` (dashboard, alerting, logger) — shared across verticals
- `config.py` — extended with funding-specific constants
- Same tech stack: Python 3.12, `asyncio`, `decimal.Decimal`, Redis, PostgreSQL

It does NOT reuse: `betfairlightweight`, `flumine`, or anything Betfair-specific.

---

## How Perpetual Funding Rates Work

Perpetual futures contracts have no expiry date. To keep their price anchored to the spot index price, exchanges use a **funding rate** mechanism — a periodic payment between long and short position holders.

### Core Mechanics

- **Funding settlement**: Every **8 hours** on Binance (00:00, 08:00, 16:00 UTC)
- **Positive funding rate**: Longs pay shorts. This is the norm in bull markets (more demand to go long)
- **Negative funding rate**: Shorts pay longs. Occurs in bearish or hedging-heavy periods
- **Payment formula**: `Funding Payment = Position Notional Value × Funding Rate`
- **Position Notional Value** = Mark Price × Position Size (in contracts)
- **Funding rate composition**: `Funding Rate = Clamp(Premium Index + Interest Rate, -cap, +cap)`. The interest rate component is fixed at 0.01% per 8h (0.03%/day). The premium component reflects spot-perp price divergence
- **Rate caps**: Most symbols capped at ±0.75% per 8h, some have different caps (queryable via `GET /fapi/v1/fundingInfo`)
- **You must hold a position at the settlement timestamp** to pay/receive funding. Closing 1 second before = no funding

### The Arbitrage Strategy

**Cash-and-carry (funding rate harvest):**
1. When funding rate is positive (longs pay shorts):
   - **Buy spot** (go long the underlying asset)
   - **Short perpetual** (go short the perp contract)
   - Net directional exposure ≈ 0 (delta-neutral)
   - **Collect funding payment** every 8 hours as the short side
2. When funding rate flips negative:
   - Close both positions (or reverse if model predicts sustained negative funding)

**Why it works**: The spot position offsets the perp's price movement. You're not betting on direction — you're harvesting the carry premium that leveraged long speculators pay to maintain their positions.

**Risk factors**:
- **Basis risk**: Spot and perp prices don't move in perfect lockstep. Slippage on entry/exit of both legs
- **Liquidation risk**: The short perp position has a liquidation price. If price spikes sharply before funding settlement, you could get liquidated even though your net position is flat (spot gains aren't cross-margined with futures on Binance unless using Portfolio Margin)
- **Funding rate reversal**: Rate can flip between when you enter and when settlement occurs
- **Exchange risk**: Counterparty risk of holding assets on a centralized exchange
- **Execution risk**: Both legs must fill near-simultaneously. If spot fills but perp doesn't (or vice versa), you have unhedged exposure

### What Makes This ML-Suitable

Most retail traders doing funding arb just look at the **current** rate. They enter when it's high and exit when it drops. This is naive because:

1. **Funding rates are predictable** — they correlate with open interest, long/short ratio, spot-perp basis, recent liquidations, order book skew, and broader market sentiment
2. **The next rate matters more than the current rate** — you need the rate to still be positive at the next settlement (8h away) to profit
3. **Cross-asset opportunities** — the same underlying can have different funding rates on different exchanges (Binance vs Bybit vs Hyperliquid). Predicting which venue will have the highest rate enables venue selection
4. **Optimal entry/exit timing** — entering just before a high-funding settlement and exiting just after maximizes the ratio of funding collected to trading fees paid

An ML model that predicts the funding rate 8h ahead (direction + magnitude) turns this from a passive carry trade into an active alpha strategy.

---

## Platform: Binance USDT-Margined Futures

### Why Binance First

- **Largest liquidity**: ~$15B+ daily futures volume, tight spreads on majors
- **Full testnet**: `https://testnet.binancefuture.com` — same API, fake USDT, real-time prices. Proper paper trading sandbox
- **Official Python SDK**: `pip install binance-futures-connector` (v4.1.0). Classes: `UMFutures` (USDT-M), `CMFutures` (COIN-M)
- **EU-accessible**: Binance France holds EU license. Standard KYC with NIE. EUR deposits via bank transfer
- **Comprehensive WebSocket streams**: Real-time mark price + funding rate updates

### Fee Structure

```python
# Binance Futures fees (VIP 0 / regular user)
FEES = {
    "futures_maker": Decimal("0.0002"),    # 0.02% — limit orders that add liquidity
    "futures_taker": Decimal("0.0005"),    # 0.05% — market orders that remove liquidity
    "spot_maker":    Decimal("0.001"),     # 0.10% — spot limit orders
    "spot_taker":    Decimal("0.001"),     # 0.10% — spot market orders
}

# BNB discount: 10% off futures fees, 25% off spot fees (if paying fees in BNB)
# With BNB: futures_maker=0.018%, futures_taker=0.045%, spot=0.075%

# Funding rate: NOT a fee. Peer-to-peer transfer between longs and shorts.
# Typical range: -0.01% to +0.03% per 8h settlement
# Collected/paid on position notional value at mark price

# Total cost per round-trip (enter + exit both legs):
# Spot: 2 × taker fee = 2 × 0.10% = 0.20%
# Perp: 2 × taker fee = 2 × 0.05% = 0.10%
# Total trading cost per cycle ≈ 0.30% (0.225% with BNB)
# Break-even: need cumulative funding > 0.30% before closing
# At 0.01% per 8h, break-even ≈ 30 funding periods (10 days)
# At 0.03% per 8h, break-even ≈ 10 funding periods (3.3 days)
# Strategy: only enter when predicted avg funding > 0.02%/8h
```

### API Endpoints (USDT-Margined Futures — `/fapi/`)

**Market Data (no auth required):**
- `GET /fapi/v1/premiumIndex` — Mark price, index price, current funding rate, next funding time. Weight: 1 (single symbol), 10 (all symbols)
- `GET /fapi/v1/fundingRate` — Historical funding rates. Params: `symbol`, `startTime`, `endTime`, `limit` (max 1000). Weight: shared 500/5min/IP with fundingInfo
- `GET /fapi/v1/fundingInfo` — Funding rate caps/floors and interval hours per symbol
- `GET /fapi/v1/ticker/24hr` — 24h stats including volume, price change
- `GET /fapi/v1/openInterest` — Current open interest for a symbol
- `GET /fapi/v1/depth` — Order book. Limits: 5, 10, 20, 50, 100, 500, 1000
- `GET /fapi/v1/klines` — Candlestick data. Intervals: 1m to 1M

**Account & Trading (HMAC-SHA256 signed):**
- `POST /fapi/v1/order` — Place order. Required: symbol, side, type. Optional: quantity, price, timeInForce, positionSide, newClientOrderId
- `DELETE /fapi/v1/order` — Cancel order
- `GET /fapi/v2/account` — Account info including balances and positions
- `GET /fapi/v2/positionRisk` — Current positions with PnL, liquidation price, leverage
- `POST /fapi/v1/leverage` — Set leverage for a symbol
- `POST /fapi/v1/marginType` — Set ISOLATED or CROSSED margin

**WebSocket Streams:**
- `<symbol>@markPrice@1s` — Real-time mark price + funding rate (1s updates)
- `<symbol>@kline_<interval>` — Real-time candlesticks
- `<symbol>@depth@100ms` — Order book updates
- `<symbol>@aggTrade` — Aggregated trades
- `!markPrice@arr@1s` — All symbols mark price + funding rate (bulk)

**Testnet:**
- REST base URL: `https://testnet.binancefuture.com`
- WebSocket: `wss://fstream.binancefuture.com`
- Create testnet account at `https://testnet.binancefuture.com`
- Generate API key from testnet account settings
- Same endpoints, fake USDT balance, real-time price feeds
- SDK usage: `UMFutures(base_url="https://testnet.binancefuture.com")`

### Spot API (for the long leg)

The spot buy uses Binance Spot API (`/api/v3/`), separate from futures:
- `POST /api/v3/order` — Place spot order
- `GET /api/v3/account` — Spot balances
- Spot testnet: `https://testnet.binance.vision`
- SDK: `pip install binance-connector` → `from binance.spot import Spot`

**Important**: Spot and futures wallets are separate on Binance. You transfer USDT between them via internal transfer (free, instant). In testnet, both wallets have pre-loaded balances.

---

## Paper Trading Design

**Binance provides a real testnet** — this is a major advantage over skins and Betfair.

### Phase 1: Testnet Paper Trading
- Connect to testnet with testnet API keys
- Execute real orders against testnet order book
- Testnet has real-time price feeds mirroring production
- Testnet simulates funding settlements (verify this during implementation — if testnet doesn't settle funding, fall back to Phase 2)
- Track P&L against testnet balances

### Phase 2: Local Simulation (fallback if testnet funding doesn't settle)
- Pull real funding rates from production API (public, no auth)
- Simulate positions locally
- Calculate hypothetical funding payments using real rates
- Same pattern as Betfair/skins paper executor

### Validation Criteria Before Going Live
- 2+ weeks of paper trading data
- Funding rate predictor accuracy > 65% directional (does it predict positive/negative correctly?)
- Paper P&L positive after accounting for realistic trading fees
- No edge cases: position sizing, leverage management, simultaneous leg execution all tested

---

## File Structure

```
betfair-arb-engine/
├── CLAUDE.md                    # update: add funding module reference
├── PLAN-FUNDING.md              # this file
├── config.py                    # extend with FUNDING_ prefixed constants
│
├── data/                        # existing betfair data modules
├── core/                        # existing betfair core modules
├── execution/                   # existing betfair execution modules
├── strategy/                    # existing betfair strategy modules
├── monitoring/                  # SHARED — extend dashboard for funding
│
├── funding/                     # NEW — entire funding rate vertical
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── binance_futures_client.py  # Binance USDT-M Futures REST + WebSocket client
│   │   ├── binance_spot_client.py     # Binance Spot REST client (for long leg)
│   │   ├── funding_rate_fetcher.py    # Historical + real-time funding rate collection
│   │   ├── market_data_stream.py      # WebSocket manager: mark price, funding, order book
│   │   ├── open_interest_fetcher.py   # Open interest + long/short ratio data
│   │   └── price_cache.py            # Redis-backed, keyed by symbol + data_type
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── fee_calculator.py          # Trading fee + funding payment calculator [OPUS]
│   │   ├── opportunity_scanner.py     # Scan all perp symbols for high funding [OPUS]
│   │   ├── hedge_calculator.py        # Calculate spot/perp position sizes for delta-neutral
│   │   ├── risk_manager.py            # Exposure limits, leverage, liquidation distance [OPUS]
│   │   └── schemas.py                 # dataclasses: FundingOpportunity, HedgePosition, FundingSnapshot
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── executor.py                # Routes paper/live based on FUNDING_MODE env [OPUS]
│   │   ├── paper_executor.py          # Testnet execution or local simulation
│   │   ├── hedge_executor.py          # Simultaneous spot buy + perp short execution [OPUS]
│   │   ├── position_manager.py        # Track open hedges, funding collected, P&L per position
│   │   └── exit_manager.py            # Close both legs when model signals exit
│   │
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── symbol_selector.py         # Filter to high-volume perps worth monitoring
│   │   ├── entry_strategy.py          # When to open a hedge (ML-gated)
│   │   ├── exit_strategy.py           # When to close (funding flip, model signal, max hold time)
│   │   └── backtester.py              # Replay historical funding rates
│   │
│   ├── ml/                            # ML models — Phase 2+
│   │   ├── __init__.py
│   │   ├── funding_predictor.py       # Predict next funding rate (8h ahead)
│   │   ├── feature_engineer.py        # Build features from raw market data
│   │   └── venue_selector.py          # Phase 3: cross-exchange funding rate comparison
│   │
│   └── tests/
│       ├── unit/
│       │   ├── test_fee_calculator.py
│       │   ├── test_opportunity_scanner.py
│       │   ├── test_hedge_calculator.py
│       │   └── test_risk_manager.py
│       └── integration/
│           ├── test_binance_futures_client.py
│           ├── test_testnet_execution.py
│           └── test_paper_mode.py
```

---

## Config Constants (add to `config.py`)

```python
# === FUNDING MODULE ===
FUNDING_MODE = "paper"                             # "paper" or "live"
FUNDING_EXCHANGE = "binance"                       # binance (Phase 1), bybit/hyperliquid (Phase 3)
FUNDING_MIN_RATE_PER_8H = Decimal("0.0002")        # 0.02% min predicted funding rate to enter
FUNDING_MIN_ANNUALIZED_YIELD = Decimal("0.10")     # 10% min annualized yield to justify entry
FUNDING_MAX_POSITION_USD = Decimal("500.00")       # max notional per hedge position
FUNDING_MAX_TOTAL_EXPOSURE_USD = Decimal("2000.00")# max total across all open hedges
FUNDING_MAX_OPEN_HEDGES = 5                        # max simultaneous hedge positions
FUNDING_LEVERAGE = 2                               # low leverage — we're hedged, not speculating
FUNDING_MAX_LEVERAGE = 5                           # absolute cap, never exceed
FUNDING_MARGIN_TYPE = "ISOLATED"                   # ISOLATED preferred for risk containment
FUNDING_MIN_LIQUIDATION_DISTANCE = Decimal("0.30") # 30% — perp liquidation must be >30% away
FUNDING_MAX_HOLD_HOURS = 168                       # 7 days max hold before forced review
FUNDING_ENTRY_WINDOW_MINUTES = 30                  # enter within 30min before funding settlement
FUNDING_POLL_INTERVAL_SECONDS = 60                 # REST polling fallback interval
FUNDING_SYMBOLS_WATCHLIST_SIZE = 50                # top N perps by volume to monitor
FUNDING_MIN_24H_VOLUME_USD = Decimal("50000000")   # $50M min daily volume for a symbol

# Testnet config
BINANCE_FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"
BINANCE_SPOT_TESTNET_URL = "https://testnet.binance.vision"
BINANCE_FUTURES_WS_TESTNET = "wss://fstream.binancefuture.com"

# API keys (from .env)
BINANCE_FUTURES_API_KEY = os.getenv("BINANCE_FUTURES_API_KEY")
BINANCE_FUTURES_API_SECRET = os.getenv("BINANCE_FUTURES_API_SECRET")
BINANCE_SPOT_API_KEY = os.getenv("BINANCE_SPOT_API_KEY")
BINANCE_SPOT_API_SECRET = os.getenv("BINANCE_SPOT_API_SECRET")
# Note: testnet uses separate keys from production
BINANCE_FUTURES_TESTNET_API_KEY = os.getenv("BINANCE_FUTURES_TESTNET_API_KEY")
BINANCE_FUTURES_TESTNET_API_SECRET = os.getenv("BINANCE_FUTURES_TESTNET_API_SECRET")
BINANCE_SPOT_TESTNET_API_KEY = os.getenv("BINANCE_SPOT_TESTNET_API_KEY")
BINANCE_SPOT_TESTNET_API_SECRET = os.getenv("BINANCE_SPOT_TESTNET_API_SECRET")
```

---

## Development Phases

### Phase 0 — Accounts & Data Pipeline (Prompts 1–3)

**Prompt 1**: Setup & schemas
```
@PLAN-FUNDING.md

Create the funding/ directory structure as specified in the plan.
Add the FUNDING_ config constants to config.py.
Add the new env vars to .env.example.

Create funding/core/schemas.py with these dataclasses:
- FundingSnapshot(symbol, funding_rate, next_funding_time, mark_price, index_price, 
  open_interest, timestamp)
- FundingOpportunity(symbol, current_rate, predicted_rate, annualized_yield, 
  entry_price_spot, entry_price_perp, position_size, expected_funding_payment, timestamp)
- HedgePosition(id, symbol, side_spot, side_perp, entry_price_spot, entry_price_perp, 
  quantity_spot, quantity_perp, leverage, margin_type, entry_time, 
  funding_collected, trading_fees_paid, status, exit_time, exit_pnl)

Create funding/core/fee_calculator.py:
- Calculate trading fees for spot + perp round trip
- Calculate expected funding payment given rate and position size
- Calculate break-even: how many funding periods needed to cover entry/exit fees
- Calculate annualized yield from a given funding rate
- All math uses decimal.Decimal
- Follow same pattern as core/commission.py

Write unit tests for fee_calculator covering:
- Standard fee calculation (maker and taker)
- BNB discount scenarios
- Funding payment calculation (positive and negative rates)
- Break-even period calculation
- Annualized yield from 8h rate
```

**Prompt 2**: Binance Futures client
```
@PLAN-FUNDING.md

Implement funding/data/binance_futures_client.py:
- Wrapper around binance-futures-connector UMFutures class
- Constructor takes base_url param to switch between testnet and production
- Methods needed (all async-compatible via httpx or threading):
  - get_funding_rate_history(symbol, start_time, end_time, limit) → list[FundingSnapshot]
    Endpoint: GET /fapi/v1/fundingRate
  - get_premium_index(symbol=None) → FundingSnapshot or list[FundingSnapshot]
    Endpoint: GET /fapi/v1/premiumIndex
  - get_funding_info() → dict of symbol → {cap, floor, interval_hours}
    Endpoint: GET /fapi/v1/fundingInfo
  - get_open_interest(symbol) → Decimal
    Endpoint: GET /fapi/v1/openInterest
  - get_ticker_24h(symbol=None) → dict with volume, price change
    Endpoint: GET /fapi/v1/ticker/24hr
  - get_exchange_info() → list of tradeable perp symbols with filters
    Endpoint: GET /fapi/v1/exchangeInfo
  - get_klines(symbol, interval, limit) → list of OHLCV candles
    Endpoint: GET /fapi/v1/klines
  - place_order(symbol, side, type, quantity, **kwargs) → order response
    Endpoint: POST /fapi/v1/order
  - cancel_order(symbol, order_id) → cancellation response
    Endpoint: DELETE /fapi/v1/order
  - get_position_risk(symbol=None) → list of positions with PnL
    Endpoint: GET /fapi/v2/positionRisk
  - set_leverage(symbol, leverage) → confirmation
    Endpoint: POST /fapi/v1/leverage
  - set_margin_type(symbol, margin_type) → confirmation
    Endpoint: POST /fapi/v1/marginType
- Auth: HMAC-SHA256 signed requests (handled by SDK)
- Rate limiting: respect X-MBX-USED-WEIGHT headers, implement backoff
- All prices/rates converted to Decimal on output

Write integration test that connects to TESTNET and:
- Fetches premium index for BTCUSDT (read-only, safe)
- Fetches last 10 funding rate entries for BTCUSDT
- Verifies response parsing into FundingSnapshot
```

**Prompt 3**: Spot client & WebSocket streams
```
@PLAN-FUNDING.md

Implement funding/data/binance_spot_client.py:
- Wrapper around binance-connector Spot class
- Constructor takes base_url for testnet switching
- Methods:
  - get_price(symbol) → Decimal
  - place_order(symbol, side, type, quantity, **kwargs) → order response
  - get_account_balance(asset) → Decimal
  - transfer_to_futures(asset, amount) → confirmation
    Endpoint: POST /sapi/v1/futures/transfer (type=1 for spot→futures)
  - transfer_from_futures(asset, amount) → confirmation
    Endpoint: POST /sapi/v1/futures/transfer (type=2 for futures→spot)

Implement funding/data/market_data_stream.py:
- WebSocket manager using binance-futures-connector WebSocket client
- Subscribe to !markPrice@arr@1s (all symbols mark price + funding rate, 1s)
- On each message: update price_cache with latest mark_price, index_price, 
  funding_rate, next_funding_time for each symbol
- Handle reconnection, heartbeat (server pings every 3min, auto-pong by SDK)
- Emit events when funding rate crosses threshold (for scanner to react)
- Testnet WebSocket URL: wss://fstream.binancefuture.com

Implement funding/data/price_cache.py:
- Redis-backed, same pattern as data/price_cache.py
- Key format: funding:{symbol}:mark_price, funding:{symbol}:funding_rate, etc.
- TTL = 10 seconds (data is streaming, should be fresh)
- Method: get_all_funding_snapshots() → dict[symbol, FundingSnapshot]

Write integration test that:
- Connects to testnet WebSocket
- Receives and parses at least one markPrice update
- Verifies cache is populated
```

### Phase 1 — Scanner & Testnet Paper Trading (Prompts 4–6)

**Prompt 4**: Opportunity scanner
```
@PLAN-FUNDING.md

Implement funding/core/opportunity_scanner.py:
- Input: all FundingSnapshots from price_cache
- For each symbol in watchlist:
  - Get current funding rate from cache
  - Calculate annualized yield: rate_per_8h × 3 × 365
  - Check: rate > FUNDING_MIN_RATE_PER_8H
  - Check: annualized_yield > FUNDING_MIN_ANNUALIZED_YIELD
  - Check: 24h volume > FUNDING_MIN_24H_VOLUME_USD
  - Calculate position size respecting FUNDING_MAX_POSITION_USD
  - Calculate expected funding payment per settlement
  - Calculate break-even number of settlements
  - Emit FundingOpportunity if all checks pass
- Output: sorted list of FundingOpportunity by annualized_yield descending
- Mark as OPUS-reviewed: this is the core detection logic

Implement funding/core/hedge_calculator.py:
- Given a FundingOpportunity, calculate:
  - Exact spot quantity to buy (respecting lot size filters from exchangeInfo)
  - Exact perp quantity to short (matching spot quantity for delta-neutral)
  - Required USDT: spot_cost + perp_margin (at configured leverage) + buffer
  - Liquidation price of the short perp at configured leverage
  - Verify liquidation distance > FUNDING_MIN_LIQUIDATION_DISTANCE
- All calculations in Decimal

Write unit tests for scanner with realistic fixtures:
- A high-funding opportunity (should trigger)
- A low-funding opportunity below threshold (should NOT trigger)
- A symbol with insufficient volume (should NOT trigger)
```

**Prompt 5**: Paper executor (testnet)
```
@PLAN-FUNDING.md

Implement funding/execution/paper_executor.py:
- Connects to Binance TESTNET (both spot and futures)
- On opportunity:
  1. Set leverage for symbol on testnet
  2. Set margin type to ISOLATED on testnet
  3. Execute spot buy (market order on testnet spot)
  4. Execute perp short (market order on testnet futures)
  5. Log both fills with actual execution prices
  6. Create HedgePosition record in PostgreSQL
- On exit signal:
  1. Close perp short (market buy to close)
  2. Sell spot (market sell)
  3. Update HedgePosition with exit prices and realized P&L
- Track funding payments:
  - After each funding settlement, query testnet account and record funding income
  - If testnet doesn't process funding, simulate locally using production rates

Implement funding/execution/position_manager.py:
- Track all open HedgePositions
- Methods:
  - open_positions() → list of active hedges
  - get_position(symbol) → HedgePosition or None
  - total_exposure() → Decimal (sum of all position notional values)
  - record_funding(symbol, amount, timestamp) → update HedgePosition.funding_collected
  - close_position(symbol, exit_prices) → finalize P&L
- Persist to PostgreSQL, cache active positions in Redis

Implement funding/execution/executor.py:
- Routes to paper_executor or live executor based on FUNDING_MODE
- Same routing pattern as execution/executor.py
```

**Prompt 6**: Main loop & symbol selector
```
@PLAN-FUNDING.md

Implement funding/strategy/symbol_selector.py:
- Fetch exchange info for all USDT-M perpetual contracts
- Filter: contractType == "PERPETUAL" and status == "TRADING"
- Rank by 24h volume descending
- Take top FUNDING_SYMBOLS_WATCHLIST_SIZE symbols with volume > FUNDING_MIN_24H_VOLUME_USD
- Cache watchlist in Redis, refresh every 6 hours
- Expected result: ~50 symbols (BTC, ETH, SOL, XRP, DOGE, etc.)

Implement funding/strategy/entry_strategy.py:
- Called by main loop with list of FundingOpportunity from scanner
- Decision logic (Phase 1, simple rules — ML replaces this in Phase 2):
  - Check risk_manager: total exposure < max, open hedges < max
  - Check: time until next funding settlement < FUNDING_ENTRY_WINDOW_MINUTES
  - Check: funding rate has been consistently positive for last 3 settlements (fetch history)
  - If all checks pass → signal entry
- Output: list of (symbol, position_size) to execute

Implement funding/strategy/exit_strategy.py:
- Called periodically for each open HedgePosition
- Exit signals:
  - Current funding rate is negative (funding flip)
  - Position held > FUNDING_MAX_HOLD_HOURS
  - Liquidation distance < FUNDING_MIN_LIQUIDATION_DISTANCE (price moved against us)
  - ML model predicts negative funding for next settlement (Phase 2)
  - Manual override via config flag
- Output: list of symbols to close

Implement funding/main.py — the main event loop:
1. Initialize: load watchlist, connect WebSocket stream, warm price cache
2. Continuous loop:
   a. WebSocket keeps price_cache updated in real-time
   b. Every FUNDING_POLL_INTERVAL_SECONDS:
      - Run opportunity_scanner against cached data
      - Run entry_strategy against opportunities
      - For each entry signal: check risk_manager → if approved, execute via executor
      - Run exit_strategy against open positions
      - For each exit signal: execute close via executor
   c. On funding settlement times (00:00, 08:00, 16:00 UTC):
      - Record funding payments for all open positions
      - Log settlement stats to dashboard
3. Wire into existing monitoring/dashboard.py as a new tab/section

Implement funding/core/risk_manager.py [OPUS]:
- Check total exposure < FUNDING_MAX_TOTAL_EXPOSURE_USD
- Check open hedges < FUNDING_MAX_OPEN_HEDGES
- Check single position < FUNDING_MAX_POSITION_USD
- Check leverage ≤ FUNDING_MAX_LEVERAGE
- Check liquidation distance > FUNDING_MIN_LIQUIDATION_DISTANCE
- Check: not already in a hedge for this symbol
- Reject if any check fails, log reason
```

### Phase 2 — ML Models & Backtesting (Prompts 7–9)

**Prompt 7**: Historical data collection
```
@PLAN-FUNDING.md

Implement funding/data/funding_rate_fetcher.py:
- Bulk download historical funding rates from Binance production API
- For each symbol in watchlist:
  - GET /fapi/v1/fundingRate with pagination (limit=1000, cursor via startTime/endTime)
  - Store in PostgreSQL: table funding_rate_history(symbol, funding_rate, funding_time, mark_price)
  - Go back as far as API allows (typically 12+ months)
- Also collect:
  - 1h klines for spot-perp basis calculation
  - Open interest snapshots (GET /fapi/v1/openInterest, poll hourly and store)
  - 24h ticker stats (volume, price change)
- Run as one-off data collection script
- Estimate: 50 symbols × 3 settlements/day × 365 days = ~55,000 funding rate records
- Also download from Binance Data Collection bulk CSVs if available:
  https://data.binance.vision/ → futures/um/daily/ for historical klines and trades
```

**Prompt 8**: Funding rate predictor
```
@PLAN-FUNDING.md

Implement funding/ml/feature_engineer.py:
- Build feature vectors from raw data, aligned to 8h funding periods
- Features per symbol per period:
  - last_3_funding_rates (lag features)
  - funding_rate_8h_change, funding_rate_24h_change
  - spot_perp_basis (mark_price - index_price) / index_price
  - open_interest (absolute and % change over 8h, 24h)
  - volume_24h, volume_8h
  - long_short_ratio (if available from API, else derive from funding rate sign)
  - price_return_8h, price_return_24h, price_volatility_24h
  - order_book_imbalance (bid_volume - ask_volume) / total — snapshot at feature time
  - hour_of_day (0/8/16 UTC — categorical)
  - day_of_week (weekend effects)
  - btc_funding_rate (BTC as market-wide sentiment proxy)
- Output: DataFrame with features + target (next_funding_rate)
- Store engineered features in PostgreSQL for reproducibility

Implement funding/ml/funding_predictor.py:
- Problem: predict next_funding_rate for each symbol, 8h ahead
- Model: gradient boosted trees (LightGBM — fast on M2 Max, handles mixed feature types well)
- Two sub-models:
  1. Direction classifier: will funding be positive or negative? (binary classification)
  2. Magnitude regressor: how large will the rate be? (regression)
- Combined output: predicted_rate with confidence score
- Training:
  - Walk-forward validation: train on expanding window, test on next month
  - Train per-symbol for top 10 symbols, pooled model for the rest
  - Hyperparameter tuning via Optuna on validation set
- Evaluation metrics:
  - Direction accuracy (>65% is actionable)
  - MAE of rate prediction
  - Profit simulation: if we only entered when model predicted rate > threshold, what's the P&L?
- Integration into entry_strategy: replace simple "3 consecutive positive" rule with model prediction
- Model artifacts saved to disk, loaded at startup

Write unit tests:
- Feature engineering produces expected shape and no NaN for complete data
- Model can train on small fixture dataset and produce predictions
- Prediction output has expected schema (rate, confidence)
```

**Prompt 9**: Backtester
```
@PLAN-FUNDING.md

Implement funding/strategy/backtester.py:
- Replay historical funding rate data from funding_rate_history table
- Simulate the full strategy:
  1. At each funding period, run opportunity_scanner with historical data
  2. Apply entry_strategy (simple rules or ML-gated)
  3. Simulate hedge entry at historical spot/perp prices (use kline close as proxy)
  4. Track funding payments using actual historical rates
  5. Apply exit_strategy (funding flip, max hold, etc.)
  6. Simulate hedge exit at historical prices
  7. Calculate P&L: funding_collected - trading_fees - entry_exit_slippage
- Account for realistic trading fees (0.30% round-trip)
- Account for slippage estimate (configurable, default 0.05% per leg)
- Output:
  - Total funding collected, total fees paid, net P&L
  - Sharpe ratio equivalent (daily returns)
  - Win rate (% of hedges that were net profitable)
  - Average hold duration
  - Max drawdown
  - Distribution of per-trade P&L
- Compare: simple threshold strategy vs ML-gated strategy
- Generate report as markdown file

Write tests:
- Backtester produces consistent results on deterministic fixture data
- Fee calculation matches fee_calculator output
```

### Phase 3 — Live Execution & Multi-Venue (Prompts 10–12)

**Prompt 10**: Live hedge executor
```
@PLAN-FUNDING.md

Implement funding/execution/hedge_executor.py [OPUS]:
- Simultaneous execution of spot buy + perp short
- Critical: both legs must fill. If one fails, immediately unwind the other
- Execution flow:
  1. Pre-flight: verify balances (spot wallet for buy, futures wallet for margin)
  2. Set leverage and margin type for the symbol
  3. Place perp short order FIRST (market order)
  4. On perp fill: immediately place spot buy (market order)
  5. If spot fails: immediately close perp (market buy to unwind)
  6. Record both fill prices
  7. Calculate actual slippage vs expected prices
  8. Create HedgePosition with real execution data
- Safety checks:
  - Never exceed FUNDING_MAX_POSITION_USD
  - Never exceed FUNDING_MAX_TOTAL_EXPOSURE_USD
  - Verify liquidation price is safe distance away
  - Use newClientOrderId for idempotency
  - Timeout: if either leg doesn't fill within 5 seconds, cancel and abort
- Handle partial fills: if perp partially fills, match spot quantity to filled perp quantity
- OPUS territory: this handles real money

Implement funding/execution/exit_manager.py [OPUS]:
- Close both legs of a hedge
- Flow:
  1. Close perp short (market buy to close)
  2. On perp close: sell spot (market sell)
  3. Calculate realized P&L including all funding collected during hold
  4. Update HedgePosition status to CLOSED
- Same safety checks as entry
```

**Prompt 11**: Production wiring
```
@PLAN-FUNDING.md

Update funding/execution/executor.py to support live mode:
- When FUNDING_MODE == "live":
  - Use production API URLs
  - Use production API keys
  - Route to hedge_executor for entry/exit
  - All safety checks from risk_manager enforced
- Add kill switch: FUNDING_KILL_SWITCH env var, if true, close all positions and stop

Add alerting:
- Telegram/Discord webhook notifications for:
  - New hedge opened (symbol, size, expected yield)
  - Hedge closed (symbol, P&L, hold duration)
  - Funding payment received (symbol, amount)
  - Risk alert (liquidation distance shrinking, funding flip detected)
  - Error (execution failure, API error)
- Reuse monitoring/alerting.py pattern

Add to monitoring/dashboard.py:
- Funding module tab showing:
  - Open hedges with real-time P&L
  - Cumulative funding collected
  - Current funding rates for watchlist (heatmap)
  - ML model prediction vs actual accuracy tracker
```

**Prompt 12**: Multi-venue & optimization
```
@PLAN-FUNDING.md

Implement funding/ml/venue_selector.py:
- Phase 3 extension: compare funding rates across exchanges
- Add Bybit client (same structure as binance_futures_client.py):
  - Bybit also has testnet (testnet.bybit.com)
  - Bybit funding: also 8h settlements, similar API structure
  - Bybit fees: maker 0.02%, taker 0.055%
- For each symbol, compare funding rates between Binance and Bybit
- Strategy: go short on the exchange with the HIGHER funding rate
- This captures cross-venue funding divergence as additional alpha
- Future: add Hyperliquid (no testnet, add carefully)

Optimization tasks:
- Implement position rebalancing: if spot/perp drift apart, rebalance to maintain delta-neutral
- Implement funding rate prediction model retraining pipeline (weekly)
- Implement per-symbol fee optimization: use limit orders (maker fee) instead of market orders
  when time-to-settlement allows
- Track and log basis (spot-perp spread) as additional signal
```

---

## Environment Variables (add to .env.example)

```bash
# === FUNDING MODULE ===
FUNDING_MODE=paper                              # paper | live
FUNDING_KILL_SWITCH=false                       # true = close all positions and stop

# Binance Production (leave empty for paper trading)
BINANCE_FUTURES_API_KEY=
BINANCE_FUTURES_API_SECRET=
BINANCE_SPOT_API_KEY=
BINANCE_SPOT_API_SECRET=

# Binance Testnet
BINANCE_FUTURES_TESTNET_API_KEY=your_key_here
BINANCE_FUTURES_TESTNET_API_SECRET=your_key_here
BINANCE_SPOT_TESTNET_API_KEY=your_key_here
BINANCE_SPOT_TESTNET_API_SECRET=your_key_here
```

---

## Dependencies (add to requirements.txt)

```
# Funding module
binance-futures-connector>=4.1.0  # Official Binance Futures SDK (USDT-M + COIN-M)
binance-connector>=3.12.0         # Official Binance Spot SDK
lightgbm>=4.0.0                   # ML: funding rate prediction
optuna>=3.5.0                     # ML: hyperparameter tuning
scikit-learn>=1.4.0               # ML: preprocessing, evaluation
```

---

## Key Differences from Betfair and Skins Modules

| Aspect | Betfair | Skins | Funding |
|--------|---------|-------|---------|
| Execution speed | Milliseconds (streaming) | Minutes (REST polling) | Seconds (REST + WebSocket) |
| Settlement | Instant on market close | 7-day trade hold | 8-hour funding periods |
| Risk model | Risk-free if both legs fill | Price risk during hold | Near-zero if hedged, liquidation tail risk |
| Data freshness | Real-time WebSocket | Polled every 60s | Real-time WebSocket (1s mark price) |
| Fee structure | % of net winnings | % of sale price (both sides) | % of notional per trade + funding payments |
| Paper trading | Simulated locally | Simulated locally | **Real testnet** with sandbox API |
| ML relevance | Low (arb is mechanical) | High (price prediction for hold risk) | High (funding rate prediction is the strategy) |
| Capital required | €50+ | $100+ | $200–500 |
| Holding period | Minutes to hours | 7 days (trade hold) | Hours to days (funding cycles) |
| Revenue model | Spread capture | Spread - hold risk | Carry/yield harvesting |

---

## CLAUDE.md Updates

When starting this module, update CLAUDE.md to include:

```markdown
## Funding Module

- All funding code lives in `funding/` directory
- Follow same patterns as betfair modules (fee calc, paper executor, risk manager)
- OPUS-required files: `fee_calculator.py`, `opportunity_scanner.py`, `risk_manager.py`, 
  `executor.py`, `hedge_executor.py`, `exit_manager.py`
- All money math uses `decimal.Decimal` — no floats
- All prices stored as Decimal, funding rates as Decimal (8 decimal places)
- Two separate API clients: futures (UMFutures) and spot (Spot) — they use different base URLs and keys
- Testnet URLs: futures=https://testnet.binancefuture.com, spot=https://testnet.binance.vision
- Funding settlement at 00:00, 08:00, 16:00 UTC — timing-critical for entry/exit
- Both legs (spot + perp) must execute together — if one fails, unwind the other immediately
- See PLAN-FUNDING.md for full context and prompting guide
```

---

## Success Criteria

**Phase 0–1 (testnet paper trading):**
- [ ] Streaming real-time funding rates for ≥50 USDT-M perp symbols
- [ ] Opportunity scanner detecting ≥3 high-funding symbols per day
- [ ] Testnet execution of spot buy + perp short working correctly
- [ ] Funding payments being tracked (testnet or simulated)
- [ ] 2+ weeks of paper trading data before considering live

**Phase 2 (ML):**
- [ ] 12+ months of historical funding rates collected for top 50 symbols
- [ ] Funding rate direction predictor with accuracy > 65%
- [ ] ML-gated strategy outperforms simple threshold strategy in backtest
- [ ] Backtester showing positive net P&L after fees for ML strategy

**Phase 3 (live + multi-venue):**
- [ ] Live execution on Binance with $200 initial capital
- [ ] ≥80% of paper-trading predicted P&L realized in live
- [ ] No execution failures (both legs filling) in first 20 hedge cycles
- [ ] Bybit added as second venue for cross-exchange comparison

---

## Getting Started (First Steps)

1. **Create Binance testnet accounts:**
   - Go to `https://testnet.binancefuture.com` → Register/Login with GitHub
   - Generate API key from account settings → copy to .env
   - Go to `https://testnet.binance.vision` → same process for spot testnet
   - Testnet comes pre-loaded with fake USDT

2. **Verify API access:**
   ```python
   from binance.um_futures import UMFutures
   client = UMFutures(
       key="your_testnet_key",
       secret="your_testnet_secret",
       base_url="https://testnet.binancefuture.com"
   )
   print(client.mark_price("BTCUSDT"))  # Should return current mark price + funding rate
   ```

3. **Start with Prompt 1** from Phase 0
