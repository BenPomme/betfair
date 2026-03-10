# Binance Futures Auth Blocker

Last verified: March 10, 2026

## Current Status

- `spot_prod`: authenticated OK
- `spot_testnet`: authenticated OK
- `futures_prod`: failing with Binance error `-2015`
- `futures_testnet`: failing with Binance error `-2015`

The factory can continue in research mode without futures trading auth, but authenticated Binance futures runtime validation is still blocked until the key/IP/permission setup is corrected on Binance.

## Verification Command

```bash
python3 scripts/check_binance_auth.py
```

This command is read-only and prints redacted status only. It does not print API keys or secrets.

## Likely Causes

- Futures permission is not enabled on the Binance API key.
- The trusted IP list on Binance does not include the machine currently running the repo.
- A futures-specific key/secret pair was copied incorrectly or saved in the wrong env variables.
- Testnet futures permissions were not enabled separately from production spot/testnet spot.

## Required Follow-Up

1. Re-check the Binance API key settings for the futures key.
2. Confirm the machine IP is listed in Binance trusted IPs if IP restriction is enabled.
3. Re-run `python3 scripts/check_binance_auth.py`.
4. Do not attempt live futures execution until both `futures_prod` and `futures_testnet` authenticate successfully.

## Relevant Env Vars

```env
BINANCE_FUTURES_API_KEY=
BINANCE_FUTURES_API_SECRET=
BINANCE_FUTURES_TESTNET_API_KEY=
BINANCE_FUTURES_TESTNET_API_SECRET=
```
