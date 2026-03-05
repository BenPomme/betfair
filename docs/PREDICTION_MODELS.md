# Prediction Models (Parallel Paper Accounts)

The dashboard runs multiple prediction models in parallel.  
Each model has its own fake account with the same starting bankroll (**€100,000** by default), so performance is directly comparable.

## Active model families

1. `implied_market`
- Approach: uses market implied probability only (`1 / odds`).
- Learning: none.
- Role: control baseline.

2. `residual_logit`
- Approach: logistic model on `logit(base_prob)` plus learned residual from microstructure/context features.
- Learning: online SGD update after each short-horizon settlement.
- Role: learns when market prices are temporarily mis-calibrated.

3. `pure_logit`
- Approach: logistic model only on microstructure/context features.
- Learning: online SGD update after each settlement.
- Role: tests whether structure alone can beat market-implied baseline.

## Shared features

- `spread_mean`
- `imbalance`
- `depth_total_eur`
- `price_velocity`
- `short_volatility`
- `time_to_start_sec`
- `in_play`

## Settlement and learning signal

Online labels are now **outcome-based**:
- model positions settle only when market status is `CLOSED`
- selection label is taken from runner status (`WINNER` / `LOSER`)
- voided runners refund stake and are tracked as void outcomes

This aligns paper P&L and model learning with real-money deployment behavior.

## Dashboard comparison metrics

- Balance: current paper account balance.
- P&L: cumulative paper profit/loss in EUR.
- ROI %: P&L relative to initial bankroll.
- Bets: number of opened positions.
- Win Rate: wins / settled bets.
- Brier: probability calibration error (lower is better).
- Updates: count of online model parameter updates.
- Resets: account reset count after reaching 0.

## Configuration

Use `.env`:

```bash
PREDICTION_ENABLED=true
PREDICTION_MODEL_KINDS=implied_market,residual_logit,pure_logit
PREDICTION_INITIAL_BALANCE_EUR=100000.00
PREDICTION_STAKE_FRACTION=0.05
PREDICTION_MIN_EDGE=0.03
PREDICTION_MODEL_DIR=data/prediction/models
```

## Notes

- All models run on the same incoming snapshots.
- They are intentionally isolated by bankroll and model artifact path.
- This lets you compare approaches without strategy interference.
