# Operator Notes

## Enable Polymarket Quantum-Fold

`polymarket_quantum_fold` is disabled by default.

To turn it on:

1. Set `POLYMARKET_QF_ENABLED=true` in `.env`.
2. Optionally tune:
   `POLYMARKET_QF_INITIAL_BALANCE_USD`
   `POLYMARKET_QF_SPORTS_FILTER`
   `POLYMARKET_QF_MAX_OPEN_POSITIONS`
   `POLYMARKET_QF_MAX_NOTIONAL_PER_TRADE_USD`
   `POLYMARKET_QF_LABEL_HORIZONS_SECONDS`
   `POLYMARKET_QF_MIN_EDGE_AFTER_COSTS`
3. Start it from the command center, or run:
   `python3 scripts/run_portfolio.py --portfolio polymarket_quantum_fold`
4. Monitor it in the dashboard under the `Polymarket` group.

Default operating mode is sports-only, paper-only, and public-data-only. It does not place authenticated Polymarket orders.
