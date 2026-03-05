# CLAUDE.md — Betfair Arbitrage Engine

## Project Context

This is a **production trading system** operating on real capital via the Betfair Exchange. Bugs in commission math, opportunity detection, or order execution translate directly into financial loss. Code correctness is non-negotiable.

The system runs in two modes controlled by `PAPER_TRADING` env var:
- `PAPER_TRADING=true` — log and simulate, no real orders placed
- `PAPER_TRADING=false` — live execution via Betfair API

**Never disable paper mode without at least 2 weeks of clean paper trading data.**

---

## Multi-Agent Rules (MANDATORY)

Subagents may be **any provider** (e.g. Claude, OpenAI, other). Assign by **capability tier**, not by vendor. Use the cheapest tier that can do the task; escalate only when financial correctness requires it.

### When to Spawn Agents

The primary agent must autonomously dispatch multiple subagents when tasks are parallelizable or require different capability tiers. Do not do all work sequentially in one context when parallel agents would be faster and cheaper.

**Trigger conditions for multi-agent dispatch:**
- Implementing 2+ independent modules with no shared state
- Writing unit tests while building the next feature
- Scaffolding boilerplate while implementing core logic
- Any task where two workstreams have zero dependencies on each other

**Example parallel dispatch:**
```
Task: "Build the streaming client and the commission module"

Agent A (Tier 1): betfair_stream.py — boilerplate WebSocket setup,
                  reconnect logic, message parsing skeleton

Agent B (Tier 3): commission.py — exact Decimal arithmetic,
                  all edge cases, full unit tests included

(After both complete)
Agent C (Tier 2): integrate both into scanner.py first pass
```

---

### Agent Tier Assignment — STRICT

Assign by **tier**. Tiers are provider-agnostic; map to your available models.

**Available local models (Ollama):**
- `qwen3.5:27b` — 17 GB, Qwen 3.5 series. Performance on par with Claude Sonnet / GPT-4o. **Use for Tier 2.**
- `qwen3.5:35b-a3b` — 23 GB, Qwen 3.5 MoE (3B active params). Similar capability tier. **Use for Tier 2.**
- `qwen2.5:32b` / `qwen25-32b` — 19 GB, Qwen 2.5 series. Good for straightforward coding. **Use for Tier 1.**

**Cost strategy:** Use local Qwen models for Tier 1 and Tier 2 to avoid cloud API costs. Reserve cloud models (Claude Opus) exclusively for Tier 3 — financially critical code where correctness is non-negotiable.

| Tier | Role | Use when | Model |
|------|------|----------|-------|
| **Tier 1** | Fast, low-cost | Boilerplate, scaffolding, tests from spec, non-critical UI | `qwen2.5:32b` (Ollama) or Claude Haiku |
| **Tier 2** | Balanced | Integration, streaming, backtester, paper executor, debugging | `qwen3.5:27b` or `qwen3.5:35b-a3b` (Ollama) |
| **Tier 3** | Highest capability | Financially critical: `/core/`, `/execution/`, reviews, architecture | Claude Opus only |

#### Tier 1 — Use for:
- File scaffolding, class stubs, `__init__.py`, `requirements.txt`
- Simple REST client wrappers (market catalogue fetcher, account balance)
- Config file generation (`.env.example`, `config.py` constants skeleton)
- Docstring generation, code formatting, type hint addition
- Telegram bot message formatting
- Writing tests **for already fully specified logic** (given exact expected outputs)
- Dashboard HTML/CSS, non-critical UI code
- Logger setup, Redis cache wrapper boilerplate

#### Tier 2 — Use for:
- WebSocket streaming client with reconnect and state management
- Market catalogue integration and watchlist builder
- Backtester against historical Betfair stream files
- Order monitor (poll unmatched orders, cancel stale legs)
- FastAPI dashboard endpoints
- Integration tests
- Debugging moderate complexity issues
- Market selector and liquidity filter logic
- `paper_executor.py` — virtual ledger and trade log

#### Tier 3 — Use ONLY for (financially critical):
- **`core/scanner.py`** — overround computation, opportunity detection, threshold logic
- **`core/commission.py`** — Decimal-based commission math; a 0.1% error destroys profitability
- **`core/stake_calculator.py`** — equal-profit stake sizing; wrong sizing = unhedged exposure
- **`core/risk_manager.py`** — exposure limits, daily loss cap, position tracking
- **`execution/executor.py`** — routing logic between paper and live; a bug here places real orders in paper mode or vice versa
- **`execution/live_executor.py`** — Betfair API order placement, retry logic, partial fill handling
- Any code that reads `PAPER_TRADING` env var and branches accordingly
- Architectural decisions with long-term consequences
- Reviewing any PR that touches `/core/` or `/execution/`
- Debugging any issue where real money has been affected

> **Decision rule**: Does a bug in this code cost money or place an unintended real order? Yes → Tier 3. No → Tier 2 or Tier 1.

---

### Agent Coordination Protocol

1. **Define contracts first**: Before dispatching parallel agents, specify exact function signatures, input types, and expected outputs. Agents working in parallel must have zero ambiguity about interfaces.

2. **No shared mutable state between parallel agents**: Each agent owns a specific file or module. Never have two agents edit the same file concurrently.

3. **Integration review after parallel completion**: After Tier 1/Tier 2 agents complete parallel work, run a Tier 2 integration pass to wire modules together.

4. **Tier 3 review before any `/core/` or `/execution/` code is used**: Even if a Tier 2 agent wrote a first draft of `scanner.py`, a Tier 3 agent must review and approve before it runs — even in paper mode.

5. **Paper mode first, always**: No code ever touches live execution paths until paper mode has been validated. The `PAPER_TRADING` flag must be checked in CI.

---

## Code Quality Rules

### Financial Arithmetic
- **Never use `float` for monetary values** — use `decimal.Decimal` everywhere money or odds math is involved
- Set `decimal.getcontext().prec = 10` at module level
- All stake amounts rounded to 2 decimal places using `ROUND_HALF_UP`
- Betfair prices have specific valid tick increments — validate before placing orders (e.g. 2.00–3.00 range: 0.02 increments; 3.00–4.00: 0.05 increments)

### Paper/Live Separation
- The `PAPER_TRADING` env var is the **single source of truth** — read once at startup, never override in code
- Paper executor and live executor must expose **identical interfaces** — the executor router calls them identically
- Unit tests must test both paper and live executor paths

### Error Handling
- Every live order placement wrapped in try/except with explicit unwind logic
- If leg 1 of an arb is filled but leg 2 fails → immediately cancel leg 1 if possible, log as failed trade, alert via Telegram
- Never silently swallow exceptions in scanner, executor, or order monitor
- Circuit breaker: after 3 consecutive execution failures, set a `trading_halted` flag, alert, and stop sending orders until manually cleared

### Async Patterns
- Use `asyncio` throughout the hot path (stream → scanner → executor)
- Use `asyncio.timeout()` on all Betfair API calls — never hang indefinitely
- WebSocket client must implement exponential backoff reconnection with jitter

### Configuration
- All secrets (API key, username, password) via environment variables only — never hardcoded
- All thresholds (`MIN_LIQUIDITY_EUR`, `MIN_NET_PROFIT_EUR`, `MAX_STAKE_EUR`, `DAILY_LOSS_LIMIT_EUR`, `MBR`, `DISCOUNT_RATE`) in `config.py` — never magic numbers in logic
- `config.py` reads from env vars with sensible defaults; all values documented inline

---

## Testing Rules

- Unit tests required for: `commission.py`, `scanner.py`, `stake_calculator.py`, `paper_executor.py`
- `test_commission.py` must include: zero discount, 60% discount, 3-outcome market, edge case where overround is exactly at threshold
- `test_scanner.py` must include: valid arb detected, valid arb missed (below min profit), false positive prevented (stale price)
- All tests pass before any code enters `/execution/` path
- Paper mode must be separately tested end-to-end in `tests/integration/test_paper_mode.py`

---

## Quick Agent Lookup

Assign by **tier**; use whatever agent provider you have for that tier (Claude, OpenAI, etc.).

| Task | Tier |
|---|---|
| Create `requirements.txt` | 1 |
| Scaffold new Python file with class stubs | 1 |
| `.env.example` with all config keys documented | 1 |
| REST client for market catalogue | 1 |
| Telegram alert message formatting | 1 |
| Write unit tests for `commission.py` (given spec) | 1 |
| FastAPI dashboard endpoints | 2 |
| WebSocket streaming client with reconnect | 2 |
| Backtester for historical stream files | 2 |
| `order_monitor.py` — poll and cancel stale orders | 2 |
| `market_selector.py` and `liquidity_filter.py` | 2 |
| `paper_executor.py` — virtual ledger | 2 |
| Integration tests | 2 |
| Debug streaming reconnect issue | 2 |
| `core/commission.py` | **3** |
| `core/scanner.py` | **3** |
| `core/stake_calculator.py` | **3** |
| `core/risk_manager.py` | **3** |
| `execution/executor.py` (paper/live router) | **3** |
| `execution/live_executor.py` | **3** |
| Review any code in `/core/` or `/execution/` | **3** |
| Decide overround threshold tuning strategy | **3** |
| Any bug where real money was affected | **3** |
| Architectural decision on concurrency model | **3** |

**Tier ↔ model mapping:**
- Tier 1 → `qwen2.5:32b` via Ollama (local, free). Fallback: Claude Haiku (cloud).
- Tier 2 → `qwen3.5:27b` or `qwen3.5:35b-a3b` via Ollama (local, free, Sonnet-equivalent). Fallback: Claude Sonnet (cloud).
- Tier 3 → Claude Opus only (cloud). No local substitute — financial correctness demands highest capability.

**Implementation note:** Claude Code subagents only support Claude models (haiku/sonnet/opus). For Tier 1/2, dispatch Qwen via `ollama run <model> "<prompt>"` in Bash when the task is self-contained (code generation, scaffolding). Use Claude Haiku/Sonnet subagents when the task requires file editing, tool use, or multi-step reasoning within the Claude Code environment.

---

## Environment Variables Reference

```bash
# Betfair credentials
BF_USERNAME=your_betfair_email
BF_PASSWORD=your_betfair_password
BF_APP_KEY=your_app_key
BF_CERTS_PATH=./certs             # optional, for non-interactive login

# Trading mode (ALWAYS start with true)
PAPER_TRADING=true

# Commission
MBR=0.05                          # Market Base Rate for Spain
DISCOUNT_RATE=0.00                # Update as you earn Betfair Points

# Thresholds
MIN_NET_PROFIT_EUR=0.10           # minimum net profit to act on
MIN_LIQUIDITY_EUR=50.00           # minimum available at best back price
MAX_STAKE_EUR=100.00              # max total stake per arb
DAILY_LOSS_LIMIT_EUR=50.00        # halt trading if virtual/real loss exceeds this

# Infrastructure
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/betfair_arb
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## Workflow — Before Any Live Trading

```
[ ] Paper mode validated for ≥ 2 weeks
[ ] ≥ 30 simulated trades logged
[ ] Commission math validated manually on 5+ trade logs
[ ] Simulated net ROI > 0 after realistic slippage
[ ] Unit tests for commission.py, scanner.py, stake_calculator.py all pass
[ ] Integration test for paper mode passes
[ ] Risk limits set conservatively (MAX_STAKE_EUR=50, DAILY_LOSS_LIMIT_EUR=25 for first week live)
[ ] Telegram alerts tested and working
[ ] Order monitor tested (cancel stale orders within 5s)
[ ] PAPER_TRADING=false only set after all above checked
```

---

## Funding Module

- All funding code lives in `funding/` directory
- Follow same patterns as betfair modules (fee calc, paper executor, risk manager)
- OPUS-required files: `fee_calculator.py`, `opportunity_scanner.py`, `risk_manager.py`, `executor.py`
- All money math uses `decimal.Decimal` — no floats
- All prices stored as Decimal, funding rates as Decimal (8 decimal places)
- Two separate API clients: futures (UMFutures) and spot (Spot) — they use different base URLs and keys
- Testnet URLs: futures=https://testnet.binancefuture.com, spot=https://testnet.binance.vision
- Funding settlement at 00:00, 08:00, 16:00 UTC — timing-critical for entry/exit
- Both legs (spot + perp) must execute together — if one fails, unwind the other immediately
- `FUNDING_MODE` env var controls paper/live (separate from Betfair `PAPER_TRADING`)
- See PLAN-FUNDING.md for full context and prompting guide

| Task | Tier |
|---|---|
| `funding/core/fee_calculator.py` | **3** |
| `funding/core/opportunity_scanner.py` | **3** |
| `funding/core/risk_manager.py` | **3** |
| `funding/execution/executor.py` | **3** |
| `funding/execution/hedge_executor.py` (Phase 3) | **3** |
| `funding/execution/exit_manager.py` (Phase 3) | **3** |
| `funding/data/*` | 2 |
| `funding/strategy/*` | 2 |
| Dashboard HTML/endpoints | 1 |


<claude-mem-context>
# Recent Activity

<!-- This section is auto-generated by claude-mem. Edit content outside the tags. -->

*No recent activity*
</claude-mem-context>