# Betfair Exchange Arbitrage Engine — Project Brief

## Context

Platform: **Betfair Spain** (`betfair.es`)
Jurisdiction: Spain (regulated, EUR-denominated, no crypto required)
Strategy: Automated detection and execution of arbitrage opportunities within Betfair Exchange
Approach: Paper trading first → live with small capital → scale

---

## How Betfair Exchange Works

Betfair is not a bookmaker. It's a peer-to-peer exchange where users bet against each other:

- **Back bet**: You bet that something *will* happen (buying a position)
- **Lay bet**: You bet that something *won't* happen (selling, or taking the other side)
- Betfair matches backers with layers and charges commission only on net winnings

Prices are decimal odds (e.g. 2.50 means you win €1.50 profit on a €1 stake). Every selection has a best back price and a best lay price — the spread between them is where the exchange makes no money (only commission does).

---

## Betfair Commission Structure (Spain)

**Market Base Rate (MBR): 5%** for Spain
Commission is charged **only on net winnings per market** — losing bets pay nothing.

```
Commission = Net Winnings × MBR × (1 − Discount Rate)
```

**Discount Rate**: Starts at 0%. You earn Betfair Points (1 point per €0.10 commission paid, win or lose). Milestones:
- 0 points → 0% discount → pay full 5%
- 1,000 points → 2% discount → effective rate 4.9%
- 10,000 points → 15% discount → effective rate 4.25%
- 150,000 points → 60% discount → effective rate 2.0%

**Expert Fee** (replaced Premium Charge in Jan 2025): Only kicks in when your rolling 52-week gross profit exceeds £25,000. Not relevant at MVP. At scale:
- £25k–£100k profit band → 20% fee on top of standard commission
- £100k–£250k → 40%, £250k+ → 60%

**Key implication**: With 5% commission, you need the raw overround (sum of implied probabilities) to be low enough that even after paying commission on the winning leg, you still net a profit. This sets your minimum spread threshold.

---

## How Opportunities Are Identified Automatically

### Core Concept: Overround

For any market, the **overround** is:

```
Overround = Σ (1 / best_back_price[i])  for all selections i
```

In a fair market, overround = 1.00 (100%). In practice Betfair markets are very efficient, but they do drift:
- Heavily traded match: overround ~1.01–1.03 (1–3% over, no arb)
- Mid-table, lower-liquidity match: occasionally drops near or below 1.00
- Fast-moving in-play market: can spike up or down sharply for seconds

**Arb condition (raw)**: `overround < 1.00`
**Arb condition (net of 5% commission)**: much stricter — see calculation below

### Exact Commission-Adjusted Threshold

You back all outcomes and exactly one wins. Commission is paid on the profit from the winning leg. For a 2-outcome market:

```
Stake on A:  s_A = K × (1/p_A) / (1/p_A + 1/p_B)
Stake on B:  s_B = K × (1/p_B) / (1/p_A + 1/p_B)
Total:       K = s_A + s_B

Gross profit if A wins:  G_A = s_A × p_A − K
Net profit if A wins:    N_A = G_A × (1 − 0.05) = G_A × 0.95
```

For both outcomes to be profitable after commission:

```
min(N_A, N_B) > 0
```

In practice this requires the raw overround to be around **0.92–0.94** (6–8% below 1.00). Opportunities at that level are rare but real — your scanner runs 24/7 and catches them as they appear.

### The Scanner Loop

The streaming API pushes price changes in real time. On every update:

```
For each market in watchlist:
  1. Pull best_back_price for every selection from the stream
  2. Compute overround = Σ(1 / price[i])
  3. If overround < PRE_FILTER_THRESHOLD (e.g. 0.97):
     4. Compute optimal stakes (equal-profit method)
     5. Compute net profit after 5% commission on winning leg
     6. Check available_to_back volume at that price
     7. If net_profit > MIN_PROFIT_EUR and volume > MIN_LIQUIDITY_EUR:
        8. PAPER MODE → log opportunity, simulate fill, update virtual P&L
           LIVE MODE  → place limit orders, monitor fill status
```

### Types of Arb Opportunities on Betfair

**Type 1 — Back-Back (multi-outcome)**
Back all outcomes of a market (e.g. Home / Draw / Away in football). If the sum of implied probabilities < 1 net of commission, back all three. This is the cleanest and most common type. Same event settles all legs, zero resolution risk.

**Type 2 — Correlated constraint violation**
Two separate but logically linked markets must obey probability ordering. Examples:
- "Real Madrid to win LaLiga" must be ≥ "Real Madrid to win LaLiga and not concede" (a subset)
- "Player X to reach semi-final" must be ≥ "Player X to win the tournament"
- Mutually exclusive outcomes (only one team can win a match) must not sum above 100%

When these are violated, you back the underpriced side in one market and lay (or back the opposite) in the other. Same data resolves both.

**Type 3 — In-play dislocation**
During a fast-moving moment (goal scored, player injured, game changes momentum), prices spike before they fully reprice. A brief window exists where the overround is favourable. High reward, very hard to execute reliably — requires sub-second latency. Treat as Phase 4, not MVP.

### Best Markets to Watch (Spain-based, ranked by opportunity quality)

| Market | Why it works | Difficulty |
|---|---|---|
| LaLiga — mid/lower table Match Odds | Enough volume to fill, less sharp than top-6 matches | Low |
| Segunda División Match Odds | Less efficient, more price lags | Low-Medium |
| Tennis outright match winner | 2-outcome = simpler math, frequent dislocations | Low |
| Spanish Copa del Rey | Seasonal, mixed-quality teams | Medium |
| Tournament outrights (multi-runner) | More selections = more constraint violations | Medium |
| In-play football | Highest frequency, but execution is hardest | High |

**Start with:** pre-match Match Odds on LaLiga and Segunda División, 30–90 minutes before kick-off (peak liquidity, pre-in-play).

---

## Paper Trading Architecture

Paper mode must be **code-identical to live mode** except the final order placement is intercepted. This is critical — if paper and live modes diverge in logic, your paper results are meaningless.

```python
# Environment flag controls mode
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"

def execute_opportunity(opp):
    if PAPER_TRADING:
        paper_executor.log(opp)       # log + simulate fill
    else:
        live_executor.place(opp)      # real API call
```

**Paper executor behaviour:**
- Assumes fill at **best available price** (optimistic scenario)
- Also simulates fill at **1 Betfair tick worse** (realistic scenario — log both)
- Tracks virtual account balance, open bets, settled P&L
- Logs every simulated outcome when the market resolves (compare to what would have happened)

**Paper trading log schema:**
```json
{
  "ts": "2026-02-26T15:22:41Z",
  "mode": "paper",
  "market_id": "1.234567890",
  "event": "Getafe vs Villarreal — Match Odds",
  "market_start": "2026-02-26T16:00:00Z",
  "selections": [
    {"name": "Getafe",    "back_price": 3.60, "stake_eur": 27.78},
    {"name": "Draw",      "back_price": 3.50, "stake_eur": 28.57},
    {"name": "Villarreal","back_price": 2.30, "stake_eur": 43.48}
  ],
  "total_stake_eur": 99.83,
  "overround_raw": 0.9287,
  "gross_profit_eur": 0.17,
  "commission_eur": 0.009,
  "net_profit_eur": 0.161,
  "net_roi_pct": 0.16,
  "liquidity_a_eur": 3420.0,
  "liquidity_b_eur": 890.0,
  "liquidity_c_eur": 1240.0,
  "fill_simulated_optimistic": true,
  "fill_simulated_realistic_net": 0.08,
  "outcome_at_settlement": "Draw",
  "actual_pnl_if_live": 0.152
}
```

**Paper trading success criteria before going live:**
- Minimum 2 weeks of data
- At least 30 simulated trades logged
- Simulated net ROI > 0 after realistic slippage adjustment
- No systematic false positives (opportunities that never would have filled)
- Commission math validated against at least 5 manually checked calculations

---

## Commission Calculation Module (exact, Decimal-based)

```python
from decimal import Decimal, ROUND_HALF_UP

MBR = Decimal("0.05")

def effective_rate(mbr: Decimal, discount: Decimal) -> Decimal:
    return mbr * (1 - discount)

def commission(net_winnings: Decimal, mbr: Decimal, discount: Decimal) -> Decimal:
    return (net_winnings * effective_rate(mbr, discount)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

def evaluate_back_back_arb(prices: list[Decimal], total_stake: Decimal,
                             mbr: Decimal, discount: Decimal) -> dict | None:
    overround = sum(Decimal("1") / p for p in prices)
    stakes = [(Decimal("1") / p / overround) * total_stake for p in prices]
    profits = [(s * p - total_stake) for s, p in zip(stakes, prices)]
    net_profits = [g - commission(g, mbr, discount) for g in profits]

    if all(n > 0 for n in net_profits):
        return {
            "overround": overround,
            "stakes": stakes,
            "net_profits": net_profits,
            "min_net_profit": min(net_profits),
            "roi": min(net_profits) / total_stake,
        }
    return None  # not profitable after commission
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  BETFAIR ARBITRAGE ENGINE                     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Betfair Streaming API (WebSocket)             │   │
│  │  EX_ALL_OFFERS (best back/lay + depth)                │   │
│  │  EX_MARKET_DEF (runner names, status, start time)     │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │              Market Catalogue (REST poll)              │   │
│  │  Filter: football (ES), tennis                        │   │
│  │  Exclude: horse racing, in-play only, < €500 volume   │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │            Opportunity Scanner                        │   │
│  │  Compute overround → commission-adj threshold check   │   │
│  │  Compute stakes → check liquidity → rank by net ROI   │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │               Risk Manager                            │   │
│  │  Max stake per market, max open bets, daily loss cap  │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                         │                                    │
│          ┌──────────────┴─────────────┐                     │
│          ▼                            ▼                      │
│  ┌───────────────┐          ┌──────────────────┐            │
│  │ Paper Executor │          │  Live Executor    │            │
│  │ Log + simulate │          │  Place limit      │            │
│  │ Virtual P&L    │          │  orders via API   │            │
│  └───────┬────────┘          └────────┬─────────┘            │
│          └──────────┬─────────────────┘                     │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Position Monitor                            │   │
│  │  Track open bets, unmatched orders, market settlement │   │
│  │  Cancel stale arb legs (>5s unmatched)                │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │           Dashboard + Alerting                        │   │
│  │  Real-time P&L | Opportunity log | Telegram alerts    │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
betfair-arb-engine/
├── CLAUDE.md
├── brief.md
├── .env.example
├── config.py                # all thresholds and constants
├── requirements.txt
│
├── data/
│   ├── betfair_stream.py    # WebSocket stream client
│   ├── market_catalogue.py  # REST poll for event/runner metadata
│   └── price_cache.py       # Redis-backed latest price store
│
├── core/
│   ├── scanner.py           # overround computation + opportunity detection [OPUS]
│   ├── commission.py        # Decimal-based commission math [OPUS]
│   ├── stake_calculator.py  # equal-profit stake sizing [OPUS]
│   └── risk_manager.py      # exposure limits, daily cap [OPUS]
│
├── execution/
│   ├── executor.py          # routes to paper or live [OPUS]
│   ├── paper_executor.py    # virtual ledger + trade log
│   ├── live_executor.py     # real order placement [OPUS]
│   └── order_monitor.py     # track fills, cancel stale legs
│
├── strategy/
│   ├── market_selector.py   # build watchlist from catalogue
│   ├── liquidity_filter.py  # drop thin markets
│   └── backtester.py        # replay historical stream files
│
├── monitoring/
│   ├── dashboard.py         # FastAPI real-time UI
│   ├── alerting.py          # Telegram bot
│   └── logger.py            # structured JSON → PostgreSQL
│
└── tests/
    ├── unit/
    │   ├── test_commission.py
    │   ├── test_scanner.py
    │   └── test_stake_calculator.py
    └── integration/
        └── test_paper_mode.py
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.12 |
| Betfair API client | `betfairlightweight` (official) |
| High-level strategy runner (optional) | `flumine` |
| Async | `asyncio` |
| Price cache | Redis |
| Trade/P&L log | PostgreSQL |
| Dashboard | FastAPI + lightweight HTML |
| Alerts | Telegram Bot API |
| Arithmetic (money) | `decimal.Decimal` |
| Backtesting data | Betfair Historical Data (purchasable) |
| Infrastructure | VPS in Madrid or Frankfurt, ~€15/month |
| Local LLM (Tier 1 / lightweight agents) | Qwen via Ollama (e.g. `qwen25-32b`); see CLAUDE.md |

---

## Getting Started (account setup)

1. Register at **betfair.es** — Spanish exchange, regulated by DGOJ
2. Complete KYC (DNI or NIE + Spanish bank details)
3. Fund via bank transfer or Visa/Mastercard in EUR
4. Get API app key at `developer.betfair.com` (free for personal use)
5. Email `automation@betfair.com` to activate **live streaming** on your key
6. Set up SSL certificates OR use interactive login (simpler for dev)
7. Buy a small package of **Betfair Historical Data** for backtesting (Basic tier is fine)

**Capital needed:**
- Paper phase: €0
- Live MVP: €300–€500 (enough to cover realistic stake sizes of €50–€100 per arb)
- Infrastructure: ~€15/month VPS

---

## Development Phases

| Phase | Weeks | Goal |
|---|---|---|
| 0 — Setup | 1 | Account, API key, streaming connected, raw data logging |
| 1 — Scanner | 1–2 | Overround calculator + commission math + backtest on historical data |
| 2 — Paper trading | 3–4 | Full paper mode running live, 2+ weeks of logged simulated trades |
| 3 — Live (small) | 5–6 | Switch to live, €100 max daily exposure, validate fills vs paper |
| 4 — Optimise | 7–8 | Tune thresholds, add tennis markets, add correlated markets scanner |

---

## Key Risks

| Risk | Mitigation |
|---|---|
| Betfair flags/limits account for arbing | Use limit orders (maker), moderate volume, occasional non-arb bet |
| Opportunity disappears before fill | Use streaming not polling, cancel unmatched legs within 5s |
| Partial fill on one leg leaves unhedged exposure | Order monitor: cancel other legs immediately on partial fail |
| Commission miscalculation eats profit | All math via `Decimal`, exhaustively unit-tested before any live trading |
| Stale price creates false arb signal | Reject prices older than 2 seconds from stream timestamp |
| Expert Fee at high profitability | Track rolling 52-week profit; factor in when approaching £25k threshold |

---

## Important Note on Account Risk

Betfair has historically restricted accounts that exclusively arb. The Spanish exchange (`betfair.es`) operates under separate Spanish regulation (DGOJ) and has a different user base, which may mean less aggressive detection. However, best practices are:
- Always use **limit orders** (you're providing liquidity, not taking it — Betfair tolerates this)
- Never exclusively bet on arb opportunities — occasional normal bets help your profile
- Keep stakes proportional and consistent (don't suddenly bet €1,000 after months of €50 bets)
- Operate during normal hours when markets are liquid enough that your orders don't stand out
