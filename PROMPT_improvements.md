# Betfair Arbitrage Engine — Improvement Prompt

**Purpose**: This document is a detailed engineering specification for Claude Code agents. Each section is a self-contained task with exact file paths, function signatures, tier assignment, and acceptance criteria. Implement them in the order listed — later tasks depend on earlier ones.

**Reference**: Follow all rules in `CLAUDE.md` (agent tiers, Decimal-only finance, paper/live separation). This prompt supplements but never overrides `CLAUDE.md`.

---

## Task 1 — Wire Prediction Engine into Arbitrage Scoring (HIGH PRIORITY)

**Tier**: 3 (Opus) — this changes the core execution decision path.

### Problem

The `MultiModelPredictionManager` in `strategy/prediction_engine.py` runs 3 parallel paper accounts (`implied_market`, `residual_logit`, `pure_logit`) but their output **never feeds back** into the arbitrage execution decision. The prediction engine and the scoring engine (`strategy/model_inference.py`) are completely disconnected. The prediction engine's superior probability estimates are wasted — they run in parallel logging data but don't influence whether an opportunity gets EXECUTE/DEFER/SKIP.

### What to Build

**File**: `strategy/model_inference.py`

Modify `score_opportunity()` to accept an optional `prediction_confidence: Optional[Dict[str, float]]` parameter. This dict contains the best-performing prediction model's output for the current market snapshot:

```python
def score_opportunity(
    opportunity: Opportunity,
    features: FeatureVector,
    prediction_confidence: Optional[Dict[str, float]] = None,
) -> ScoredOpportunity:
```

The `prediction_confidence` dict has this shape (provided by the caller in `main.py`):
```python
{
    "best_model_id": "residual_logit_1",
    "predicted_prob": 0.58,       # model's probability for the best selection
    "edge_vs_market": 0.04,       # predicted_prob - implied_prob
    "model_brier": 0.21,          # avg Brier score of that model (lower = better calibration)
    "settled_bets": 142,          # how many bets this model has settled (confidence in calibration)
}
```

**Scoring integration logic** (add inside `_heuristic_score` and `_linear_score`):

1. If `prediction_confidence` is None or `settled_bets < 30`, ignore it entirely (not enough data to trust).
2. If `model_brier > 0.28`, ignore it (poorly calibrated model).
3. Otherwise, blend the prediction signal into the scoring:
   - If `edge_vs_market > 0.05` (model sees strong positive edge on this selection), boost `edge_score` by `+0.10` (clamped to 1.0).
   - If `edge_vs_market < -0.03` (model sees the market is overpriced for this selection — i.e. arb might not hold), penalize `edge_score` by `-0.15` and set `fill_prob *= 0.80`.
   - Adjust `expected_net_profit_eur` accordingly after the fill_prob change.
4. Log the prediction influence in a new field on `ScoredOpportunity`. Add `prediction_influence: str` to the `ScoredOpportunity` dataclass in `core/types.py`:
   ```python
   prediction_influence: str = "none"  # "none", "boosted", "penalized", "ignored_insufficient_data"
   ```

**File**: `main.py`

In the scan loop (around line 165-176), after calling `prediction_manager.process_snapshot()`, extract the best-performing model's state and build the `prediction_confidence` dict. The "best performing" model is the one with the lowest `avg_brier` among models with `settled_bets >= 30`. Pass this to `score_opportunity()` on line 230.

```python
# After prediction_payload = prediction_manager.process_snapshot(...)
prediction_confidence = _extract_best_prediction(prediction_manager, market_id, snapshot)
# ...
scored = score_opportunity(opp, features, prediction_confidence=prediction_confidence)
```

Write the helper `_extract_best_prediction(prediction_manager, market_id, snapshot) -> Optional[Dict[str, float]]` in `main.py`. It should:
1. Get `prediction_manager.initial_state()` to find which model has the lowest `avg_brier` with `settled_bets >= 30`.
2. For that model, get its engine from `prediction_manager.engines[best_model_id]`.
3. Use the engine's `_predict_prob()` with the best back price from the snapshot to get `predicted_prob`.
4. Compute `edge_vs_market = predicted_prob - (1.0 / best_back_odds)`.
5. Return the dict, or None if no model qualifies.

### Acceptance Criteria

- [ ] `ScoredOpportunity` in `core/types.py` has `prediction_influence` field with default `"none"`.
- [ ] `score_opportunity()` signature accepts `prediction_confidence` kwarg.
- [ ] When prediction data is available and model is well-calibrated (Brier < 0.28, settled >= 30), it modifies edge_score and fill_prob.
- [ ] When prediction data is missing or model is uncalibrated, scoring behaves exactly as before (backward compatible).
- [ ] `main.py` passes prediction data to `score_opportunity` for single-market scans.
- [ ] `build_scan_record` in `candidate_logger.py` logs the `prediction_influence` field.
- [ ] All existing unit tests in `tests/unit/` still pass.
- [ ] New unit test in `tests/unit/test_model_inference.py` covers: prediction boost, prediction penalty, prediction ignored (insufficient data), prediction None.

---

## Task 2 — Train ML Scoring Model from Candidate Logger Data (HIGH PRIORITY)

**Tier**: 3 (Opus) — this builds the trained model that replaces the heuristic.

### Problem

The candidate logger (`data/candidate_logger.py`) writes JSONL files to `data/candidates/` with every scan outcome (executed, deferred, skipped, no arb). This is a supervised learning goldmine: each record has features + outcome. But there's no training pipeline — `model_inference.py` either uses a hand-tuned heuristic or loads a linear model from a JSON file that nobody creates.

### What to Build

**New file**: `strategy/train_scoring_model.py`

A standalone CLI script that:

1. Reads all `data/candidates/*.jsonl` files.
2. Filters to records where `executed == true` AND the trade has a known outcome. For now, `executed == true` is the positive signal (the heuristic thought it was good enough). Records with `reason == "no_arb_after_filters"` or `reason == "stale_or_missing_snapshot"` are excluded.
3. Feature columns (all must be present in the JSONL; skip records missing any):
   - `net_roi_pct` (from opportunity)
   - `net_profit_eur` (from opportunity)
   - `overround_back` (from snapshot)
   - `overround_lay` (from snapshot)
   - `selection_count` (from snapshot)
   - `edge_score` (from scored — the heuristic's own score, for bootstrapping)
   - `fill_prob` (from scored)
4. Label: `y = 1` if `executed == true`, `y = 0` otherwise. (Initially this is a "should we have executed?" classifier trained on the heuristic's own decisions. Once we add post-trade CLV data in Task 6, this becomes a real profit predictor.)
5. Train a logistic regression using **only standard library + numpy** (no sklearn dependency). If numpy is not available, use pure Python SGD (like `predictive_model.py` already does).
6. Output: Save model weights as JSON to `data/models/scoring_linear_v2.json` in the format that `_load_linear_model()` in `model_inference.py` already expects:
   ```json
   {
     "bias": 0.12,
     "roi": 2.3,
     "profit": 0.8,
     "depth": 0.0001,
     "spread": -3.1,
     "volatility": -1.5
   }
   ```
   Note: map feature names to the stable keys used in `_linear_score()`: `net_roi_pct` → `"roi"`, `net_profit_eur` → `"profit"`, `overround_back` as proxy for depth → `"depth"`, spread_mean → `"spread"`, short_volatility → `"volatility"`.
7. Print walk-forward cross-validation metrics: accuracy, Brier score, and a confusion matrix.
8. Add CLI args: `--input-dir data/candidates --output data/models/scoring_linear_v2.json --min-samples 100`

**File**: `config.py`

Add:
```python
ML_LINEAR_MODEL_PATH: str = os.getenv("ML_LINEAR_MODEL_PATH", "")
```
(Already exists via `os.getenv` in model_inference.py line 32, but make it explicit in config.py for consistency.)

### Acceptance Criteria

- [ ] `python -m strategy.train_scoring_model --input-dir data/candidates --output data/models/scoring_linear_v2.json` works end-to-end.
- [ ] Output JSON can be loaded by `_load_linear_model()` in `model_inference.py` without changes.
- [ ] Walk-forward validation prints accuracy and Brier score.
- [ ] Script handles edge cases: no JSONL files, fewer than `--min-samples` records, missing fields in some records.
- [ ] No sklearn or torch dependency. Pure Python or numpy only.

---

## Task 3 — Implement Order Monitor for Live Trading (CRITICAL for live)

**Tier**: 3 (Opus) — wrong order handling = unhedged exposure = direct money loss.

### Problem

`execution/order_monitor.py` is a skeleton. The `_poll_loop()` method is a placeholder with no Betfair API calls. When going live, unmatched order legs sit indefinitely, creating unhedged exposure. If leg 1 of an arb fills but leg 2 doesn't, you're holding a naked position.

### What to Build

**File**: `execution/order_monitor.py`

Replace the placeholder `_poll_loop()` with a real implementation:

1. Every 1 second, call `client.betting.list_current_orders()` from `betfairlightweight` with `orderBy='BY_PLACE_TIME'` and `dateRange` covering the last hour.
2. For each order in `self._order_ids`:
   - If status is `EXECUTABLE` (unmatched) and `placed_at_ts` is older than `STALE_ORDER_SECONDS` (default 5s):
     - Call `client.betting.cancel_orders(market_id=order.market_id, instructions=[CancelInstruction(bet_id=order.bet_id)])`.
     - Log cancellation via `logger.warning()`.
     - Call `alert_stale_order_cancelled(order)` (new function in `monitoring/alerting.py`).
   - If status is `EXECUTION_COMPLETE` (fully matched): remove from tracking.
   - If status is `EXECUTABLE` but partially matched: log the partial fill amount but don't cancel yet (wait for STALE_ORDER_SECONDS from original placement).
3. Remove cancelled and completed orders from `self._order_ids`.

**Constructor change**: Accept a `client` parameter (the betfairlightweight APIClient instance):
```python
def __init__(self, client=None) -> None:
    self._client = client
    # ... rest stays the same
```

**File**: `monitoring/alerting.py`

Add:
```python
def alert_stale_order_cancelled(order_info: dict) -> None:
    send_telegram(
        f"[Arb] Stale order cancelled: market={order_info.get('market_id')} "
        f"bet_id={order_info.get('bet_id')} age={order_info.get('age_seconds')}s"
    )

def alert_partial_fill(order_info: dict) -> None:
    send_telegram(
        f"[Arb] Partial fill detected: market={order_info.get('market_id')} "
        f"matched={order_info.get('size_matched')} / {order_info.get('size_total')}"
    )
```

**File**: `main.py`

In `main()` (around line 365), instantiate the OrderMonitor with the client:
```python
order_monitor = OrderMonitor(client=client)
```
In `_main()`, call `await order_monitor.start()` before the main loop, and `await order_monitor.stop()` in the finally block.

### Acceptance Criteria

- [ ] In paper mode, `_poll_loop` is still a no-op (no API calls).
- [ ] In live mode, stale unmatched orders (>5s) are cancelled via Betfair API.
- [ ] Telegram alert fires on every cancellation.
- [ ] Partial fills are logged but not cancelled prematurely.
- [ ] `register_orders()` accepts order dicts with `{"bet_id": str, "market_id": str}`.
- [ ] Integration test: mock the betfairlightweight client, register an order, advance time by 6s, assert cancel was called.
- [ ] All error paths wrapped in try/except — a failed cancel must not crash the monitor loop.

---

## Task 4 — Expand Telegram Alerting (MEDIUM PRIORITY)

**Tier**: 1 (Haiku or qwen2.5:32b) — straightforward, no financial logic.

### Problem

`monitoring/alerting.py` only has 3 alert types: execution failure, circuit breaker, daily loss cap. For live trading and ongoing monitoring, we need richer alerts.

### What to Build

**File**: `monitoring/alerting.py`

Add these functions:

```python
def alert_trade_executed(opp: dict, result: dict, scored: dict) -> None:
    """Fire on every successful trade (paper or live)."""
    send_telegram(
        f"[Trade] {opp.get('arb_type')} on {opp.get('event_name')}\n"
        f"Net profit: {opp.get('net_profit_eur')} | ROI: {opp.get('net_roi_pct')}%\n"
        f"Edge score: {scored.get('edge_score')} | Fill prob: {scored.get('fill_prob')}\n"
        f"Prediction: {scored.get('prediction_influence', 'none')}"
    )

def alert_daily_summary(paper_executor, prediction_manager=None) -> None:
    """Send once daily at midnight UTC. Summarize P&L, trades, model performance."""
    balance = paper_executor.balance
    trade_count = paper_executor.trade_count  # you may need to add this attr
    msg = f"[Daily] Balance: {balance} | Trades today: {trade_count}"
    if prediction_manager:
        for model_id, engine in prediction_manager.engines.items():
            state = engine.get_state()
            msg += (
                f"\n  {model_id}: ROI={state['roi_pct']}% "
                f"Brier={state['avg_brier']} "
                f"Bets={state['settled_bets']}"
            )
    send_telegram(msg)

def alert_model_degradation(model_id: str, metric: str, value: float, threshold: float) -> None:
    """Fire when a prediction model's performance drops below threshold."""
    send_telegram(
        f"[Model] {model_id} degradation: {metric}={value:.4f} (threshold={threshold:.4f})"
    )

def alert_prediction_bet(payload: dict) -> None:
    """Fire when prediction engine opens or settles a paper bet."""
    for event in payload.get("events", []):
        kind = event.get("kind", "")
        if kind == "prediction_open":
            send_telegram(
                f"[Pred] {event['model_id']} bet on {event['selection']} @ {event['odds']} "
                f"edge={event['edge']} stake={event['stake_eur']}"
            )
        elif kind == "prediction_settle":
            emoji = "W" if event.get("won") else ("V" if event.get("void") else "L")
            send_telegram(
                f"[Pred] {event['model_id']} {emoji} {event['selection']} "
                f"PnL={event['pnl_eur']} Balance={event['balance_eur']}"
            )
```

**File**: `main.py`

Wire the new alerts into the callbacks:
- `on_trade` callback → call `alert_trade_executed()`
- `on_prediction` callback → call `alert_prediction_bet()`

**New**: Add a daily summary scheduler. In `_main()`, create an asyncio task that fires `alert_daily_summary()` every 24 hours (or at midnight UTC):
```python
async def _daily_summary_task():
    while _running:
        now = datetime.now(timezone.utc)
        # Sleep until next midnight UTC
        next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        await asyncio.sleep((next_midnight - now).total_seconds())
        alert_daily_summary(paper_executor, prediction_manager)
```

### Acceptance Criteria

- [ ] All new alert functions added to `monitoring/alerting.py`.
- [ ] `alert_trade_executed` called from `on_trade` callback in `main.py`.
- [ ] `alert_prediction_bet` called from `on_prediction` callback in `main.py`.
- [ ] Daily summary task runs in the async loop.
- [ ] All alerts are no-ops if Telegram tokens are not configured (existing `send_telegram` handles this).
- [ ] No changes to financial logic.

---

## Task 5 — Add Closing Line Value (CLV) Tracking (HIGH PRIORITY)

**Tier**: 2 (Sonnet or qwen3.5:27b) — involves data pipeline, not financial math.

### Problem

The single most important metric for evaluating a sports bettor is **Closing Line Value (CLV)**: did you get odds better than the final market price before the event started? Currently we don't track this. We know `entry_odds` (from the prediction engine's `_PendingBet`) and `entry_odds` equivalent for arb legs (from `Opportunity.selections`), but we never record the closing odds.

### What to Build

**New file**: `data/clv_tracker.py`

```python
class CLVTracker:
    """
    Track closing line value for prediction bets and arbitrage trades.

    When a market transitions to CLOSED or in-play, record the last
    pre-close prices as "closing odds" for each selection. Then compute
    CLV = (closing_implied_prob - entry_implied_prob) for each bet.
    Positive CLV = you got a better price than the market settled at.
    """

    def __init__(self, log_dir: str = "data/clv"):
        ...

    def record_entry(self, bet_id: str, market_id: str, selection_id: str,
                     entry_odds: float, entry_timestamp: str) -> None:
        """Called when a bet is placed (prediction or arb)."""
        ...

    def record_closing_prices(self, market_id: str, snapshot: PriceSnapshot) -> None:
        """Called on each snapshot. Stores the latest prices as potential closing prices.
        When market_status transitions to CLOSED or in_play transitions from False to True,
        freeze the previous snapshot's prices as the closing line."""
        ...

    def compute_clv(self, bet_id: str) -> Optional[float]:
        """Returns CLV for a settled bet. Positive = beat the closing line."""
        ...

    def get_summary(self) -> dict:
        """Return aggregate CLV stats: avg CLV, % of bets with positive CLV, etc."""
        ...
```

**Integration points**:

1. In `prediction_engine.py` `_open_bet()`: after creating `_PendingBet`, call `clv_tracker.record_entry()`.
2. In `prediction_engine.py` `process_snapshot()`: call `clv_tracker.record_closing_prices()` on every snapshot.
3. In `prediction_engine.py` `_settle_pending()`: after settling, call `clv_tracker.compute_clv()` and include the CLV value in the settle event dict.
4. In `main.py`: instantiate `CLVTracker` and pass to prediction engines.

**File**: `strategy/prediction_engine.py`

Add `clv_tracker: Optional[CLVTracker] = None` parameter to `OnlinePredictionEngine.__init__()` and `MultiModelPredictionManager.__init__()`.

### Acceptance Criteria

- [ ] CLV is computed for every settled prediction bet.
- [ ] CLV data is logged in the settle events (`prediction_settle` events get a `"clv"` field).
- [ ] `get_summary()` returns `{"avg_clv": float, "positive_clv_pct": float, "total_tracked": int}`.
- [ ] CLV log persisted to `data/clv/*.jsonl` partitioned by day.
- [ ] Works in paper mode with no external dependencies.
- [ ] Unit test: create mock entries, mock closing prices, verify CLV computation.

---

## Task 6 — Cross-Market Expansion (MEDIUM PRIORITY)

**Tier**: 2 (Sonnet or qwen3.5:27b) for the scanner; Tier 3 review before merge.

### Problem

`core/cross_market_scanner.py` only handles `MATCH_ODDS` vs `DRAW_NO_BET`. Football markets offer several other cross-market arb types that the engine should detect.

### What to Build

**File**: `data/event_grouper.py`

Extend `get_cross_market_pairs()` to also return pairs for:
- `MATCH_ODDS` + `OVER_UNDER_25` (Over/Under 2.5 goals)
- `MATCH_ODDS` + `BOTH_TEAMS_TO_SCORE`
- `CORRECT_SCORE` vs `MATCH_ODDS` (correct score prices often misprice the underlying match odds)

The function signature stays the same but returns a richer structure:
```python
def get_cross_market_pairs(market_ids, meta) -> List[Tuple[str, str, str]]:
    """Returns (market_id_a, market_id_b, pair_type) tuples."""
```

Where `pair_type` is one of: `"mo_dnb"`, `"mo_ou25"`, `"mo_btts"`, `"cs_mo"`.

**File**: `core/cross_market_scanner.py`

Add new scanner functions:
```python
def scan_cross_market_ou25(mo_snap, ou25_snap, ...) -> Optional[Opportunity]:
    """MATCH_ODDS vs OVER_UNDER_25 cross-market arb."""
    ...

def scan_cross_market_btts(mo_snap, btts_snap, ...) -> Optional[Opportunity]:
    """MATCH_ODDS vs BOTH_TEAMS_TO_SCORE cross-market arb."""
    ...
```

The mathematical relationship for MO + OU25:
- If the sum of backing all MO outcomes and backing Under 2.5 costs less than backing Over 2.5 and all MO outcomes at their lay prices, there's an arbitrage.
- This requires careful enumeration of score-implied probabilities, which is complex. Start with the simpler approach: check if implied probability sums across combined markets < 1.0 after commission.

**File**: `main.py`

In the cross-market scan section (line 259-315), dispatch to the correct scanner function based on `pair_type`.

**File**: `data/market_catalogue.py`

In `discover_markets()`, ensure the API query includes `OVER_UNDER_25` and `BOTH_TEAMS_TO_SCORE` market types. Check if the current `marketFilter` in the `listMarketCatalogue` call already covers these or needs to be expanded.

### Acceptance Criteria

- [ ] `get_cross_market_pairs()` returns MO+OU25 and MO+BTTS pairs when both markets exist for the same event.
- [ ] Scanner functions for OU25 and BTTS compute overround correctly and detect arbs.
- [ ] Commission is applied correctly (same as existing `scan_cross_market` for DNB).
- [ ] `main.py` dispatches to correct scanner per pair type.
- [ ] Arb type in `Opportunity` uses descriptive strings: `"cross_mo_ou25"`, `"cross_mo_btts"`.
- [ ] Unit tests for each new scanner function with mock price data.
- [ ] **Tier 3 review required** before merging — cross-market commission math must be verified.

---

## Task 7 — Harden Learning Architect (LOW PRIORITY)

**Tier**: 2 (Sonnet or qwen3.5:27b)

### Problem

The learning architect (`strategy/learning_architect.py`) has two modes: rules-based and LLM (Ollama). The LLM path is risky: if Ollama returns garbage JSON, the bounded-step logic catches it, but the LLM could propose subtly harmful changes that pass validation (e.g., setting `min_edge` to the minimum allowed, which opens the floodgates to bad bets).

### What to Build

**File**: `strategy/learning_architect.py`

1. **Make rules-based primary, LLM secondary**: Flip the priority. Run rules first. Only consult LLM if rules produce no proposals AND `ARCHITECT_LLM_ENABLED` is true. The LLM is a tiebreaker, not the primary decision-maker.

2. **Add safety constraints to LLM proposals**:
   - Never allow `min_edge` to go below `0.02` regardless of `ARCHITECT_MIN_EDGE` config (hard floor).
   - Never allow `stake_fraction` to increase by more than 50% of its current value in a single step.
   - If a model has `roi_pct < -5%`, force `stake_fraction` to the minimum, regardless of what the LLM says.

3. **Add model degradation detection**: In `evaluate_and_apply()`, after computing states, check each model for:
   - Brier score > 0.30 with > 50 settled bets → fire `alert_model_degradation()` from `monitoring/alerting.py`.
   - ROI < -10% with > 50 settled bets → fire degradation alert.
   - Win rate < 35% with > 50 settled bets → fire degradation alert.

4. **Cooldown**: After applying proposals, don't run again for `2 * interval_seconds` (double the normal interval) to let changes take effect before re-evaluating.

### Acceptance Criteria

- [ ] Rules-based runs first; LLM only consulted when rules produce empty proposals.
- [ ] Hard floor on `min_edge` of 0.02 enforced even if config says lower.
- [ ] Stake increase capped at 50% of current per step.
- [ ] Models with ROI < -5% forced to minimum stake.
- [ ] Degradation alerts fire via Telegram.
- [ ] Cooldown doubles after applying changes.
- [ ] All changes logged to `data/architect/decisions.jsonl`.
- [ ] Unit test: verify hard floor, verify forced minimum, verify cooldown.

---

## Task 8 — Add New Features to Feature Vector (MEDIUM PRIORITY)

**Tier**: 2 (Sonnet or qwen3.5:27b) — feature engineering, no financial math.

### Problem

`strategy/features.py` extracts 7 microstructure features. Several valuable signals are missing.

### What to Build

**File**: `strategy/features.py`

Add to `build_market_microstructure()`:

1. **`weighted_spread`**: Volume-weighted average spread (weight each selection's spread by its back liquidity as proportion of total).
   ```python
   weighted_spread = sum(spread_i * back_liq_i / total_back) for each selection
   ```

2. **`lay_back_ratio`**: `total_lay / total_back` — indicates market maker activity. High ratio = lots of lay liquidity = professional market.

3. **`top_of_book_concentration`**: What fraction of total liquidity sits at the best price? High concentration = thin book, susceptible to slippage.
   ```python
   top_concentration = max(sel.available_to_back for sel in selections) / total_back
   ```

4. **`selection_count`**: Number of runners. 2-way markets behave differently from 15-horse races.

**File**: `core/types.py`

Add the new fields to `MarketMicrostructure`:
```python
@dataclass(frozen=True)
class MarketMicrostructure:
    # ... existing fields ...
    weighted_spread: Decimal = Decimal("0")
    lay_back_ratio: Decimal = Decimal("0")
    top_of_book_concentration: Decimal = Decimal("0")
    selection_count: int = 0
```

Use default values so existing code that creates `MarketMicrostructure` without these fields still works.

**File**: `strategy/prediction_engine.py`

Add the new features to `FEATURE_NAMES` list (line 19-27):
```python
FEATURE_NAMES = [
    "spread_mean",
    "imbalance",
    "depth_total_eur",
    "price_velocity",
    "short_volatility",
    "time_to_start_sec",
    "in_play",
    "weighted_spread",
    "lay_back_ratio",
    "top_of_book_concentration",
    "selection_count",
]
```

And update `_features_from_snapshot()` to extract them.

**IMPORTANT**: Existing model weights files (`data/prediction/models/*.json`) will have weights only for the old 7 features. The logistic models in `predictive_model.py` use `features.get(name, 0.0)` which returns 0.0 for missing features. So new features will start with zero weight and learn organically — no migration needed. But verify this is the case before shipping.

### Acceptance Criteria

- [ ] 4 new features added to `MarketMicrostructure` with defaults.
- [ ] `build_market_microstructure()` computes all 4 new features.
- [ ] `FEATURE_NAMES` in prediction_engine.py includes all 11 features.
- [ ] Existing model files load without errors (new features default to weight 0.0).
- [ ] Unit test: build microstructure with mock snapshot, verify new features are computed correctly.
- [ ] `FeatureVector` and downstream consumers unaffected (microstructure is nested, not flat).

---

## Task 9 — Fill Probability Model (MEDIUM-HIGH PRIORITY)

**Tier**: 3 (Opus) — directly affects execution decision.

### Problem

`fill_prob` in `model_inference.py` is computed heuristically: `0.50 + depth_boost - spread_penalty - volatility_penalty/2`. This has no empirical basis. With candidate logger data accumulating, we can build an actual fill probability estimator.

### What to Build

**New file**: `strategy/fill_model.py`

A lightweight logistic model that predicts P(trade fills at expected price) given:
- `spread_mean`
- `depth_total_eur`
- `short_volatility`
- `time_to_start_sec`
- `in_play` (binary)
- `total_stake_eur` (larger stakes are harder to fill)

Training data comes from candidate logger:
- Records where `executed == true` → label 1 (we assume paper executor filled it; in live mode we'll have actual fill data).
- Records where `reason == "risk_blocked"` or `reason == "fill_prob_below_min"` → label 0 (conditions were unfavorable).

This is approximate until we have real live fill data, but it's better than the current heuristic.

**File**: `strategy/model_inference.py`

Add a `_fill_model_score()` that:
1. Tries to load a fill model from `data/models/fill_model_v1.json`.
2. If available, uses it instead of the heuristic fill_prob computation.
3. Falls back to heuristic if model not found.

### Acceptance Criteria

- [ ] `strategy/fill_model.py` trains from candidate logger data.
- [ ] CLI: `python -m strategy.fill_model --input-dir data/candidates --output data/models/fill_model_v1.json`
- [ ] `model_inference.py` loads and uses fill model when available.
- [ ] Heuristic fallback when model file doesn't exist.
- [ ] Unit test with mock data.

---

## Task 10 — Paper Trading Data Collection Automation (LOW PRIORITY)

**Tier**: 1 (Haiku or qwen2.5:32b) — scripting, no financial logic.

### Problem

The system needs to run paper trading for 2+ weeks continuously to collect enough data for Tasks 2 and 9. Currently you have to manually start `main.py` and keep it running. Need a simple systemd service or screen/tmux script.

### What to Build

**New file**: `scripts/run_paper.sh`

```bash
#!/usr/bin/env bash
# Run paper trading in a tmux session with auto-restart on crash.
# Usage: ./scripts/run_paper.sh
```

The script should:
1. Source `.env` file.
2. Verify `PAPER_TRADING=true`.
3. Start `python main.py` in a tmux session named `arb-paper`.
4. Auto-restart on crash with 30s delay.
5. Redirect stdout/stderr to `logs/paper_$(date +%Y%m%d).log`.

**New file**: `scripts/retrain_models.sh`

```bash
#!/usr/bin/env bash
# Retrain scoring and fill models from accumulated candidate data.
# Run weekly after 2+ weeks of paper trading data collection.
```

1. Run `python -m strategy.train_scoring_model --input-dir data/candidates --output data/models/scoring_linear_v2.json`.
2. Run `python -m strategy.fill_model --input-dir data/candidates --output data/models/fill_model_v1.json`.
3. Print summary metrics.
4. Optionally send Telegram alert with metrics.

### Acceptance Criteria

- [ ] `scripts/run_paper.sh` starts paper trading in tmux with auto-restart.
- [ ] `scripts/retrain_models.sh` retrains both models and prints metrics.
- [ ] Both scripts are executable (`chmod +x`).
- [ ] `run_paper.sh` refuses to start if `PAPER_TRADING != true`.

---

## Implementation Order

Execute tasks in this order (dependencies flow downward):

```
Phase 1 (can be parallel):
  Task 8 (new features) — Tier 2
  Task 4 (Telegram alerts) — Tier 1
  Task 10 (paper trading scripts) — Tier 1

Phase 2 (depends on Phase 1):
  Task 1 (wire prediction → scoring) — Tier 3
  Task 5 (CLV tracking) — Tier 2

Phase 3 (depends on Phase 2 + accumulated data):
  Task 2 (train ML scoring model) — Tier 3
  Task 9 (fill probability model) — Tier 3

Phase 4 (depends on Phase 1):
  Task 3 (order monitor) — Tier 3
  Task 6 (cross-market expansion) — Tier 2 + Tier 3 review
  Task 7 (harden learning architect) — Tier 2
```

**Critical path**: Tasks 8 → 1 → 2 → 9 is the ML improvement pipeline.
**Live trading critical path**: Task 3 must be complete before `PAPER_TRADING=false`.

---

## Config Additions Summary

Add these to `config.py` and `.env.example`:

```python
# --- CLV tracking ---
CLV_LOG_DIR: str = os.getenv("CLV_LOG_DIR", "data/clv")
CLV_ENABLED: bool = os.getenv("CLV_ENABLED", "true").lower() == "true"

# --- Fill model ---
FILL_MODEL_PATH: str = os.getenv("FILL_MODEL_PATH", "")

# --- Scoring model (explicit) ---
ML_LINEAR_MODEL_PATH: str = os.getenv("ML_LINEAR_MODEL_PATH", "")
```
