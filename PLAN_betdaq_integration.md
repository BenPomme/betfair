# Betdaq Integration Plan — Betfair ↔ Betdaq Cross-Exchange Arbitrage

**Date:** 2026-02-28
**Status:** RESEARCH COMPLETE — Ready for human action + agent build

---

## Critical Finding: Smarkets Is Out

**Smarkets blocks Spain.** They explicitly ban residents of Spain, France, Italy, Portugal, Netherlands, Belgium, and many other EU countries. On top of that, they are **not accepting new API users** (indefinitely). Smarkets is eliminated.

**Betdaq works for Spain.** Betdaq explicitly accepts Spanish residents, is EU-compliant, and API access is available.

---

## What YOU (Ben) Need To Do First

These are human-only steps. No agent can do them. **Do these before we write any code.**

### Step 1: Create Betdaq Account
- Go to: `https://www.betdaq.com`
- Register with your Spanish ID/passport
- Complete KYC (passport + proof of address, same as Betfair)
- Make a first deposit (minimum ~€20 to activate the account)
- **Use promo code `0COMM100`** → 0% commission for first 100 days

### Step 2: Request API Access
- Contact Betdaq support: `https://betdaq.zendesk.com`
- Request API access credentials
- **Cost: £250 one-time fee** (non-refundable, required before credentials are issued)
- They'll ask for your account ID and verify your KYC
- Expected turnaround: 24–48 hours after payment

### Step 3: Receive API Credentials
- You'll get: username, password (for SOAP header authentication)
- The API is SOAP/XML (not REST) — different from Betfair's JSON
- Base URL: `https://api.betdaq.com/v2.0/`
- WSDL: `https://api.betdaq.com/v2.0/API.wsdl`
- Docs: `https://api.betdaq.com/v2.0/Docs/default.aspx`

### Step 4: Fund Both Accounts
- Keep at least €500 on Betfair and €500 on Betdaq for initial paper+live testing
- Cross-exchange arbs require capital on BOTH sides simultaneously

### Step 5: Tell Me Your Credentials Are Ready
- Once you have API access, add to your `.env`:
```bash
BD_USERNAME=your_betdaq_username
BD_PASSWORD=your_betdaq_password
BD_COMMISSION_RATE=0.00  # 0% for first 100 days with promo, then 0.05 for Spain
```

---

## What The Agents Will Build

### Architecture Overview

```
                    ┌──────────────┐
                    │  PriceCache  │  ← Exchange-agnostic in-memory store
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
    ┌─────────▼──┐  ┌──────▼─────┐  ┌──▼──────────┐
    │  Betfair   │  │  Betdaq    │  │  (Future     │
    │  Poller    │  │  Poller    │  │  exchanges)  │
    └─────────┬──┘  └──────┬─────┘  └─────────────┘
              │            │
              │     All produce PriceSnapshot
              │            │
              └────────────┼────────────┐
                           │            │
                    ┌──────▼───────┐    │
                    │   Scanner    │    │  ← UNCHANGED (exchange-agnostic)
                    │  (back-back  │    │
                    │   lay-lay)   │    │
                    └──────┬───────┘    │
                           │            │
              ┌────────────┤            │
              │            │            │
    ┌─────────▼──┐  ┌──────▼─────┐     │
    │  SAME-     │  │  CROSS-    │     │
    │  EXCHANGE  │  │  EXCHANGE  │ ◄───┘  NEW: Betfair back + Betdaq lay
    │  arbs      │  │  arbs      │        (or vice versa)
    └────────────┘  └──────┬─────┘
                           │
                    ┌──────▼───────┐
                    │  Executor    │  ← Routes to correct exchange
                    │  Router      │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
    ┌─────────▼──┐  ┌──────▼─────┐  ┌──▼──────────┐
    │  Betfair   │  │  Betdaq    │  │  Paper       │
    │  Executor  │  │  Executor  │  │  Executor    │
    └────────────┘  └────────────┘  └──────────────┘
```

### The Big Win: Cross-Exchange Arbitrage

This is the new capability that makes Betdaq worth adding:

**Same event, same market, different exchange:**
- Back "Real Madrid" on Betfair at 2.10
- Lay "Real Madrid" on Betdaq at 2.04
- If 2.10 > 2.04 + combined commission → guaranteed profit

**Commission advantage:**
- Betfair: 5% MBR (Spain)
- Betdaq: 0% for first 100 days, then 5% (Spain, non-UK residents)
- During the 100-day promo: your commission cost is halved on cross-exchange arbs
- Even after promo: price divergences between exchanges create arbs

**Why this works:**
- Betdaq has lower liquidity → prices lag Betfair by seconds
- Different user bases create different price discovery
- Commission differences create structural edges (especially during 0% promo)

---

## Agent Work Plan — Module Breakdown

### Module 1: Betdaq SOAP Client (`data/betdaq_client.py`)
**Tier:** 2 (Qwen 3.5 via Ollama, or Claude Sonnet)
**Why Tier 2:** SOAP/XML parsing is moderately complex, no financial math

**Implements:**
```python
class BetdaqClient:
    def __init__(self, username: str, password: str)
    def list_event_types(self) -> List[dict]
    def get_market_information(self, market_ids: List[str]) -> List[dict]
    def get_prices(self, market_ids: List[str]) -> List[dict]
    def place_order(self, market_id, selection_id, price, stake, side) -> str
    def cancel_orders(self, order_ids: List[str]) -> None
    def get_account_balance(self) -> Decimal
```

**Dependencies:** `zeep` (Python SOAP client) or raw `requests` + XML parsing
**Estimated effort:** 4–6 hours

---

### Module 2: Betdaq Price Converter + Poller (`data/betdaq_poller.py`)
**Tier:** 2

**Implements:**
```python
def betdaq_market_to_snapshot(market_data: dict) -> Optional[PriceSnapshot]
async def run_betdaq_price_poller(client, market_ids, price_cache, interval=2.0)
```

**Key:** Must convert Betdaq's odds format and selection IDs into `PriceSnapshot` / `SelectionPrice` types that the scanner already understands.

**Estimated effort:** 2–3 hours

---

### Module 3: Cross-Exchange Scanner (`core/cross_exchange_scanner.py`)
**Tier:** 3 (Claude Opus ONLY — financially critical)

**This is the new, high-value module.** It:
1. Takes a PriceSnapshot from Betfair AND a PriceSnapshot from Betdaq for the **same event**
2. Matches selections by name (fuzzy matching needed — "Real Madrid" vs "Real Madrid CF")
3. Checks: can we back on Exchange A and lay on Exchange B (or vice versa) profitably?
4. Accounts for DIFFERENT commission rates on each exchange
5. Calculates equal-profit stakes across two exchanges

**Implements:**
```python
def scan_cross_exchange(
    bf_snapshot: PriceSnapshot,      # Betfair prices
    bd_snapshot: PriceSnapshot,      # Betdaq prices
    bf_commission: Decimal,          # Betfair MBR
    bd_commission: Decimal,          # Betdaq commission rate
    bf_discount: Decimal,            # Betfair discount
    bd_discount: Decimal,            # Betdaq discount
    min_net_profit_eur: Decimal,
    min_liquidity_eur: Decimal,
    max_stake_eur: Decimal,
) -> Optional[Opportunity]:
    """
    Find best cross-exchange arb:
      - Back on Betfair, Lay on Betdaq
      - Back on Betdaq, Lay on Betfair
    Return best by net profit, or None.
    """
```

**Commission math is different here:** Each leg has its own exchange's commission. Must use Decimal throughout.

**Estimated effort:** 6–8 hours (Tier 3, includes thorough testing)

---

### Module 4: Event Matcher (`data/event_matcher.py`)
**Tier:** 2

**Problem:** Betfair's "Real Madrid v Barcelona" might be Betdaq's "Real Madrid CF vs FC Barcelona". Same event, different naming.

**Implements:**
```python
def match_events(
    bf_markets: List[dict],    # Betfair market catalogue
    bd_markets: List[dict],    # Betdaq market info
) -> List[Tuple[str, str]]:   # [(bf_market_id, bd_market_id), ...]
    """
    Match same events across exchanges.
    Uses: event name fuzzy matching, start time matching (±5 min),
    sport type matching, number of selections matching.
    """
```

**Estimated effort:** 3–4 hours

---

### Module 5: Cross-Exchange Executor (`execution/cross_executor.py`)
**Tier:** 3 (Claude Opus — handles real orders on TWO exchanges)

**The hardest part:** Placing orders on two exchanges atomically is impossible. Must handle:
- Leg 1 fills on Betfair, Leg 2 fails on Betdaq → cancel Leg 1
- Partial fills on either side
- Latency between exchanges
- Circuit breaker if cross-exchange execution fails 3x

**Implements:**
```python
class CrossExchangeExecutor:
    def __init__(self, bf_client, bd_client)

    def execute_cross_arb(self, opportunity: Opportunity) -> Optional[dict]:
        """
        1. Place back order on exchange A
        2. Place lay order on exchange B
        3. If leg 2 fails: cancel leg 1, log failure, alert
        4. If both fill: log success, update risk manager
        """
```

**Paper mode:** Simulates both legs using existing PaperExecutor pattern.

**Estimated effort:** 8–10 hours (Tier 3, critical path)

---

### Module 6: Config Extension + Main Loop (`config.py`, `main.py`)
**Tier:** 1 (Qwen 2.5 via Ollama, or Claude Haiku)

**Simple additions:**
```python
# config.py additions
BD_USERNAME, BD_PASSWORD
BD_COMMISSION_RATE: Decimal
CROSS_EXCHANGE_ENABLED: bool
CROSS_EXCHANGE_MIN_PROFIT_EUR: Decimal
```

**Main loop:** Run both pollers in parallel, add cross-exchange scan step.

**Estimated effort:** 1–2 hours

---

### Module 7: Tests
**Tier:** 1 (specs given) + Tier 3 (cross-exchange commission tests)

| Test file | Tier | What it tests |
|-----------|------|---------------|
| `test_betdaq_client.py` | 1 | SOAP request/response mocking |
| `test_betdaq_poller.py` | 1 | Price conversion to PriceSnapshot |
| `test_event_matcher.py` | 1 | Fuzzy name matching |
| `test_cross_exchange_scanner.py` | **3** | Commission math across 2 exchanges |
| `test_cross_executor.py` | **3** | Failure modes (leg 1 fills, leg 2 fails) |
| `test_paper_mode_cross.py` | 2 | End-to-end paper trading cross-exchange |

**Estimated effort:** 4–6 hours total

---

## Agent Dispatch Plan

### Phase 1 — Parallel (no dependencies)

| Agent | Tier | Module | Model |
|-------|------|--------|-------|
| Agent A | 2 | `data/betdaq_client.py` — SOAP client | `qwen3.5:27b` (Ollama) |
| Agent B | 1 | Config extension + `.env.example` update | `qwen2.5:32b` (Ollama) |
| Agent C | 1 | `test_betdaq_client.py` — mock tests | `qwen2.5:32b` (Ollama) |

### Phase 2 — After Phase 1 completes

| Agent | Tier | Module | Model |
|-------|------|--------|-------|
| Agent D | 2 | `data/betdaq_poller.py` + `data/event_matcher.py` | `qwen3.5:27b` (Ollama) |
| Agent E | 1 | `test_betdaq_poller.py` + `test_event_matcher.py` | `qwen2.5:32b` (Ollama) |

### Phase 3 — Tier 3 (sequential, Claude Opus only)

| Agent | Tier | Module | Model |
|-------|------|--------|-------|
| Agent F | **3** | `core/cross_exchange_scanner.py` | Claude Opus |
| Agent G | **3** | `execution/cross_executor.py` | Claude Opus |
| Agent H | **3** | `test_cross_exchange_scanner.py` + `test_cross_executor.py` | Claude Opus |

### Phase 4 — Integration

| Agent | Tier | Module | Model |
|-------|------|--------|-------|
| Agent I | 2 | Wire into `main.py`, integration test | `qwen3.5:27b` (Ollama) |
| Agent J | **3** | Review all code in `/core/` and `/execution/` | Claude Opus |

---

## Timeline

| Week | What happens |
|------|-------------|
| **Week 0 (now)** | Ben: Register Betdaq, pay £250 API fee, complete KYC |
| **Week 1** | Agents: Build Betdaq SOAP client, poller, event matcher (Phase 1+2) |
| **Week 2** | Agents: Build cross-exchange scanner + executor (Phase 3, Tier 3) |
| **Week 3** | Agents: Integration, testing, code review (Phase 4) |
| **Week 3–5** | Paper trading: Cross-exchange arbs logged, not executed |
| **Week 5+** | Analyze paper results → if profitable, go live with small stakes |

---

## Expected Returns (Honest Assessment)

### During 100-day 0% commission promo:
- Commission edge: 5% (Betfair) vs 0% (Betdaq) = massive advantage
- Cross-exchange price divergences: 0.5–2% additional
- **Combined edge: 2–5% per qualifying arb**
- Expected arbs per day: 2–5 (major football/racing)
- Stake per arb: €50–200
- **Expected daily profit: €2–20**
- **Expected monthly: €60–600**

### After promo ends (both 5% commission):
- Edge is purely from price divergences between exchanges
- Expected: 0.5–1.5% per arb
- **Expected monthly: €30–200**

### Key constraint: Betdaq liquidity
- Major football (EPL, La Liga, Champions League): adequate
- Major racing (UK/Ireland): adequate
- Everything else: thin, may not match
- **Best window: 2 hours before event start onwards**

---

## Betdaq API Quick Reference

| Operation | SOAP Method | Auth |
|-----------|-------------|------|
| List sports | `ListEventTypes` | Read-only (username) |
| Get markets | `GetMarketInformation` (max 50/call) | Read-only |
| Get prices | `GetMarketInformation` (includes runner prices) | Read-only |
| Place order | `PlaceOrdersNoReceipt` (max 50/call) | Secure (user+pass) |
| Cancel orders | `CancelAllOrders` | Secure |
| Account balance | `GetAccountBalances` | Secure |
| Order history | `GetOrderHistory` | Secure |

**Python SOAP library:** `zeep` (`pip install zeep`)
**Community wrapper:** `pip install betdaq` (github.com/rozzac90/betdaq)
**AAPI streaming:** Push-based, requires demonstrated sophistication + B2B account

---

## Dependencies To Install

```bash
pip install zeep           # SOAP client for Betdaq API
pip install betdaq         # Community Python wrapper (rozzac90)
pip install fuzzywuzzy     # Fuzzy string matching for event names
pip install python-Levenshtein  # Fast string matching backend
```

---

## Risk Checklist

- [ ] Betdaq account funded with ≥€500
- [ ] API credentials working (can call `ListEventTypes`)
- [ ] Paper trading cross-exchange for ≥2 weeks
- [ ] ≥20 simulated cross-exchange arbs logged
- [ ] Commission math verified manually on 5+ paper trades
- [ ] Event matcher accuracy verified (>95% correct matches)
- [ ] Cross-executor failure handling tested (leg 1 fills, leg 2 fails)
- [ ] Telegram alerts working for cross-exchange trades
- [ ] Daily loss limit covers both exchanges combined
- [ ] Only go live after all above checked
