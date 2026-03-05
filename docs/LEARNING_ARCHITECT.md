# Learning Architect

The Learning Architect is a safe meta-controller that tunes prediction-model
runtime parameters at fixed intervals.

## Purpose

- Propose bounded parameter updates from deterministic rules.
- Keep strict hard limits so no uncontrolled behavior reaches execution.
- Log every decision for auditability.

## What it can change

Per prediction model account:

- `stake_fraction`
- `min_edge`

Both are bounded by config and max step size per cycle.

## Decision flow

1. Collect latest model states from all prediction accounts.
2. Generate proposals using deterministic rule policy.
3. Apply bounded deltas only.
4. Emit architect event to dashboard.
5. Append decision log to `data/architect/decisions.jsonl`.

## Safety limits

- Minimum settled bets before tuning (`ARCHITECT_MIN_SETTLED_BETS`).
- Maximum parameter change per cycle:
  - `ARCHITECT_MAX_STAKE_STEP`
  - `ARCHITECT_MAX_EDGE_STEP`
- Absolute bounds:
  - `ARCHITECT_MIN_STAKE_FRACTION` to `ARCHITECT_MAX_STAKE_FRACTION`
  - `ARCHITECT_MIN_EDGE` to `ARCHITECT_MAX_EDGE`

## Config

```bash
ARCHITECT_ENABLED=true
ARCHITECT_INTERVAL_SECONDS=900
ARCHITECT_MIN_SETTLED_BETS=30
ARCHITECT_LOG_DIR=data/architect
```

## Dashboard

The dashboard shows:
- last architect run time
- mode (`rules`)
- whether changes were applied
- proposals table by model
