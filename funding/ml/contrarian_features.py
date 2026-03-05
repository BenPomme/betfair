"""
Feature engineering for the contrarian (directional) trading model.

Reuses loaders from funding.ml.feature_engineer and adds new data sources
(long/short ratio, fear & greed) to build a feature matrix targeting
forward price direction rather than the next funding rate.

Each row corresponds to one 8h funding period.
Targets:
  - price_return_24h_target: forward 24h return (3 × 8h periods)
  - price_return_72h_target: forward 72h return (9 × 8h periods)
  - direction_24h: binary (1 if forward 24h return > 0, else 0)
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from funding.ml.feature_engineer import (
    load_funding_rates,
    load_klines,
    load_open_interest,
    _aggregate_klines_to_8h,
    _aggregate_oi_to_8h,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/funding_history")

# Thresholds for extremity flags
EXTREME_POSITIVE_THRESHOLD = 0.0005
EXTREME_NEGATIVE_THRESHOLD = -0.0005
LS_EXTREME_LONG = 2.0
LS_EXTREME_SHORT = 0.5
FEAR_GREED_EXTREME_FEAR = 25
FEAR_GREED_EXTREME_GREED = 75


# ---------------------------------------------------------------------------
# New loaders
# ---------------------------------------------------------------------------

def load_long_short_ratio(symbol: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load long/short ratio CSV for a symbol.

    Columns: timestamp, long_short_ratio, long_account, short_account,
             top_long_short_ratio, top_long_account, top_short_account
    """
    path = (data_dir or DATA_DIR) / "long_short_ratio" / f"{symbol}.csv"
    if not path.exists():
        logger.warning("Long/short ratio file not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["long_short_ratio", "long_account", "short_account",
                "top_long_short_ratio", "top_long_account", "top_short_account"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_fear_greed(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load fear & greed index CSV.

    Columns: timestamp, value, value_classification
    """
    path = (data_dir or DATA_DIR) / "fear_greed" / "index.csv"
    if not path.exists():
        logger.warning("Fear & greed file not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["value"] = df["value"].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Feature group helpers
# ---------------------------------------------------------------------------

def _funding_extremity_features(rates_df: pd.DataFrame) -> pd.DataFrame:
    """Compute funding extremity features from the base rates DataFrame.

    Operates on a DataFrame already indexed by funding_time_dt with a
    'funding_rate' column.

    Returns a DataFrame of new feature columns (same index).
    """
    r = rates_df["funding_rate"]

    feats = pd.DataFrame(index=rates_df.index)

    rolling_30 = r.rolling(30)
    feats["funding_zscore_30"] = (
        (r - rolling_30.mean()) / rolling_30.std().replace(0, np.nan)
    )
    feats["funding_percentile_30"] = r.rolling(30).rank(pct=True)

    # Consecutive extreme positive periods
    is_extreme_pos = r > EXTREME_POSITIVE_THRESHOLD
    feats["consecutive_extreme_positive"] = (
        is_extreme_pos
        .groupby((~is_extreme_pos).cumsum())
        .cumcount()
    )

    # Consecutive extreme negative periods
    is_extreme_neg = r < EXTREME_NEGATIVE_THRESHOLD
    feats["consecutive_extreme_negative"] = (
        is_extreme_neg
        .groupby((~is_extreme_neg).cumsum())
        .cumcount()
    )

    feats["funding_rate_abs"] = r.abs()

    feats["rate_lag_1"] = r.shift(1)
    feats["rate_lag_2"] = r.shift(2)
    feats["rate_lag_3"] = r.shift(3)

    return feats


def _long_short_features(ls_df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute long/short ratio features aligned to funding_time_dt index.

    Uses merge_asof to align on nearest timestamp.
    """
    feats = pd.DataFrame(index=index)

    if ls_df.empty:
        for col in ["ls_ratio", "ls_ratio_momentum_3", "ls_ratio_zscore_10",
                    "ls_extreme_long", "ls_extreme_short"]:
            feats[col] = np.nan
        return feats

    # Build left frame from index for merge_asof
    left = pd.DataFrame({"funding_time_dt": index}).sort_values("funding_time_dt")
    right = ls_df[["timestamp_dt", "long_short_ratio"]].sort_values("timestamp_dt")

    merged = pd.merge_asof(
        left,
        right,
        left_on="funding_time_dt",
        right_on="timestamp_dt",
        direction="nearest",
    ).set_index("funding_time_dt")

    feats["ls_ratio"] = merged["long_short_ratio"]

    feats["ls_ratio_momentum_3"] = feats["ls_ratio"].diff(3)

    rolling_10 = feats["ls_ratio"].rolling(10)
    feats["ls_ratio_zscore_10"] = (
        (feats["ls_ratio"] - rolling_10.mean()) / rolling_10.std().replace(0, np.nan)
    )

    feats["ls_extreme_long"] = (feats["ls_ratio"] > LS_EXTREME_LONG).astype(int)
    feats["ls_extreme_short"] = (feats["ls_ratio"] < LS_EXTREME_SHORT).astype(int)

    return feats


def _taker_imbalance_features(klines_df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute taker buy/sell imbalance features aligned to funding_time_dt index."""
    feats = pd.DataFrame(index=index)

    if klines_df.empty:
        feats["taker_buy_ratio"] = np.nan
        feats["taker_imbalance_3"] = np.nan
        return feats

    kline_8h = _aggregate_klines_to_8h(klines_df)
    if kline_8h.empty:
        feats["taker_buy_ratio"] = np.nan
        feats["taker_imbalance_3"] = np.nan
        return feats

    # taker_buy_ratio already computed in _aggregate_klines_to_8h
    tbr = kline_8h["taker_buy_ratio"].reindex(index).ffill()

    feats["taker_buy_ratio"] = tbr
    feats["taker_imbalance_3"] = (tbr - 0.5).rolling(3).mean()

    return feats


def _fear_greed_features(fg_df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute fear & greed features aligned to funding_time_dt index."""
    feats = pd.DataFrame(index=index)

    if fg_df.empty:
        for col in ["fear_greed_value", "fear_greed_change_24h",
                    "fear_greed_extreme_fear", "fear_greed_extreme_greed"]:
            feats[col] = np.nan
        return feats

    left = pd.DataFrame({"funding_time_dt": index}).sort_values("funding_time_dt")
    right = fg_df[["timestamp_dt", "value"]].sort_values("timestamp_dt")

    merged = pd.merge_asof(
        left,
        right,
        left_on="funding_time_dt",
        right_on="timestamp_dt",
        direction="nearest",
    ).set_index("funding_time_dt")

    fgv = merged["value"]

    feats["fear_greed_value"] = fgv
    # 24h = 3 × 8h periods
    feats["fear_greed_change_24h"] = fgv.diff(3)
    feats["fear_greed_extreme_fear"] = (fgv < FEAR_GREED_EXTREME_FEAR).astype(int)
    feats["fear_greed_extreme_greed"] = (fgv > FEAR_GREED_EXTREME_GREED).astype(int)

    return feats


def _funding_dispersion_features(
    symbol: str,
    btc_rates: Optional[pd.DataFrame],
    index: pd.DatetimeIndex,
    data_dir: Optional[Path],
) -> pd.DataFrame:
    """Compute cross-asset funding dispersion.

    If btc_rates is a multi-symbol DataFrame (with 'symbol' column),
    compute std of funding rates across symbols per period.
    Otherwise fall back to a single-symbol approach using top-10 symbols
    loaded from disk.
    """
    feats = pd.DataFrame(index=index)

    if btc_rates is not None and not btc_rates.empty:
        # Multi-symbol path: btc_rates contains a 'symbol' column
        if "symbol" in btc_rates.columns:
            try:
                pivoted = btc_rates.pivot_table(
                    index="funding_time_dt",
                    columns="symbol",
                    values="funding_rate",
                    aggfunc="last",
                )
                dispersion = pivoted.std(axis=1).reindex(index).ffill()
                feats["funding_dispersion"] = dispersion
                return feats
            except Exception as exc:
                logger.warning("Could not compute multi-symbol dispersion: %s", exc)

        # Single additional asset (e.g. BTC) — dispersion not meaningful
        feats["funding_dispersion"] = np.nan
        return feats

    # No btc_rates supplied — attempt to load top-10 symbols from disk
    top10 = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    ]
    frames = []
    for sym in top10:
        if sym == symbol:
            continue
        df_sym = load_funding_rates(sym, data_dir)
        if df_sym.empty:
            continue
        df_sym = df_sym.set_index("funding_time_dt")
        df_sym = df_sym[~df_sym.index.duplicated(keep="last")]
        frames.append(df_sym["funding_rate"].rename(sym))

    if not frames:
        feats["funding_dispersion"] = np.nan
        return feats

    multi = pd.concat(frames, axis=1)
    dispersion = multi.std(axis=1).reindex(index).ffill()
    feats["funding_dispersion"] = dispersion
    return feats


def _price_features(rates_df: pd.DataFrame, klines_df: pd.DataFrame) -> pd.DataFrame:
    """Compute price return and volatility features.

    Uses mark_price from rates_df as the primary price series; augments
    with kline close prices when available.
    """
    feats = pd.DataFrame(index=rates_df.index)

    # Use mark_price from funding rates (already 8h-aligned)
    price = rates_df["mark_price"]

    feats["price_return_8h"] = price.pct_change(1)
    feats["price_return_24h"] = price.pct_change(3)   # 3 × 8h
    feats["price_return_72h"] = price.pct_change(9)   # 9 × 8h
    feats["price_volatility_24h"] = price.pct_change().rolling(3).std()

    return feats


def _oi_features(oi_df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute open interest change features aligned to index."""
    feats = pd.DataFrame(index=index)

    if oi_df.empty:
        feats["oi_change_8h"] = np.nan
        feats["oi_change_24h"] = np.nan
        return feats

    oi_8h = _aggregate_oi_to_8h(oi_df)
    if oi_8h.empty:
        feats["oi_change_8h"] = np.nan
        feats["oi_change_24h"] = np.nan
        return feats

    feats["oi_change_8h"] = oi_8h["oi_change_8h"].reindex(index).ffill()
    feats["oi_change_24h"] = oi_8h["oi_change_24h"].reindex(index).ffill()

    return feats


def _time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute time-based features from the index."""
    feats = pd.DataFrame(index=index)
    feats["hour_of_day"] = index.hour
    feats["day_of_week"] = index.dayofweek
    feats["is_weekend"] = (index.dayofweek >= 5).astype(int)
    return feats


def _targets(rates_df: pd.DataFrame) -> pd.DataFrame:
    """Compute forward-looking price direction targets.

    NOTE: These use shift(-N) so the last N rows will be NaN.
    Drop NaN targets before training.
    """
    feats = pd.DataFrame(index=rates_df.index)

    price = rates_df["mark_price"]

    # Forward 24h (3 × 8h periods): price at t+3 relative to price at t
    forward_24h = price.shift(-3)
    feats["price_return_24h_target"] = (forward_24h - price) / price

    # Forward 72h (9 × 8h periods)
    forward_72h = price.shift(-9)
    feats["price_return_72h_target"] = (forward_72h - price) / price

    feats["direction_24h"] = (feats["price_return_24h_target"] > 0).astype(int)
    # Mark rows where the target is NaN (end of series) with NaN on direction too
    feats.loc[feats["price_return_24h_target"].isna(), "direction_24h"] = np.nan

    return feats


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_contrarian_features(
    symbol: str,
    btc_rates: Optional[pd.DataFrame] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Build contrarian feature matrix for a single symbol.

    Each row corresponds to one 8h funding period indexed by funding_time_dt.
    Targets predict forward price direction, not the next funding rate.

    Args:
        symbol: Trading pair (e.g. "ETHUSDT").
        btc_rates: Optional pre-loaded funding rates for cross-asset dispersion.
                   May be a single-symbol DataFrame or a multi-symbol DataFrame
                   with a 'symbol' column.
        data_dir: Override data directory (default: data/funding_history).

    Returns:
        DataFrame with feature columns + target columns, indexed by funding_time_dt.
        Returns empty DataFrame when insufficient data is available.
    """
    rates_df = load_funding_rates(symbol, data_dir)
    klines_df = load_klines(symbol, data_dir)
    oi_df = load_open_interest(symbol, data_dir)
    ls_df = load_long_short_ratio(symbol, data_dir)
    fg_df = load_fear_greed(data_dir)

    if rates_df.empty or len(rates_df) < 30:
        logger.warning(
            "Insufficient funding rate data for %s (%d rows); need ≥ 30",
            symbol, len(rates_df),
        )
        return pd.DataFrame()

    # Base index: funding_time_dt, deduplicated, sorted
    base = rates_df[["funding_time_dt", "funding_rate", "mark_price"]].copy()
    base = base.set_index("funding_time_dt")
    base = base[~base.index.duplicated(keep="last")].sort_index()

    index = base.index

    # --- Feature groups ---
    group_funding = _funding_extremity_features(base)
    group_ls = _long_short_features(ls_df, index)
    group_taker = _taker_imbalance_features(klines_df, index)
    group_fg = _fear_greed_features(fg_df, index)
    group_dispersion = _funding_dispersion_features(symbol, btc_rates, index, data_dir)
    group_price = _price_features(base, klines_df)
    group_oi = _oi_features(oi_df, index)
    group_time = _time_features(index)
    group_targets = _targets(base)

    # --- Assemble ---
    df = pd.concat(
        [
            base[["funding_rate", "mark_price"]],
            group_funding,
            group_ls,
            group_taker,
            group_fg,
            group_dispersion,
            group_price,
            group_oi,
            group_time,
            group_targets,
        ],
        axis=1,
    )

    # Drop rows that have NaN targets (end of series, no future prices available)
    df = df.dropna(subset=["price_return_24h_target", "direction_24h"])

    df["symbol"] = symbol

    logger.info("%s: built %d contrarian feature rows", symbol, len(df))
    return df


# ---------------------------------------------------------------------------
# Multi-symbol builder
# ---------------------------------------------------------------------------

def build_contrarian_features_all(
    symbols: List[str],
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Build contrarian feature matrix for multiple symbols, pooled.

    Loads BTC funding rates once and passes as btc_rates to each symbol
    for cross-asset dispersion. Also constructs a multi-symbol rates
    DataFrame for dispersion across the full symbol list.

    Args:
        symbols: List of trading pairs to process.
        data_dir: Override data directory.

    Returns:
        Combined DataFrame for all symbols, sorted by index (funding_time_dt).
    """
    # Build multi-symbol funding rate frame for dispersion
    all_rates_frames = []
    for sym in symbols:
        r = load_funding_rates(sym, data_dir)
        if not r.empty:
            r = r[["funding_time_dt", "funding_rate"]].copy()
            r["symbol"] = sym
            all_rates_frames.append(r)

    if all_rates_frames:
        multi_rates = pd.concat(all_rates_frames, ignore_index=True)
    else:
        multi_rates = None

    all_dfs = []
    for sym in symbols:
        df = build_contrarian_features(sym, btc_rates=multi_rates, data_dir=data_dir)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, axis=0).sort_index()
    logger.info(
        "Built pooled contrarian features: %d rows, %d columns, %d symbols",
        len(combined), len(combined.columns), len(all_dfs),
    )
    return combined


# ---------------------------------------------------------------------------
# Column selector
# ---------------------------------------------------------------------------

def get_contrarian_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of feature column names, excluding targets and metadata.

    Args:
        df: Output of build_contrarian_features or build_contrarian_features_all.

    Returns:
        List of column names suitable for use as model inputs (X).
    """
    exclude = {
        "symbol",
        "funding_rate",
        "mark_price",
        "price_return_24h_target",
        "price_return_72h_target",
        "direction_24h",
    }
    numeric_dtypes = (np.float64, np.int64, np.float32, np.int32, float, int)
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype.type in {
            np.float64, np.int64, np.float32, np.int32,
        }
    ]
