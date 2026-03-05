"""
Feature engineering for funding rate prediction.
Builds feature vectors aligned to 8h funding periods from raw CSV data.

Features per symbol per period:
  - Lag features: last 3 funding rates
  - Rate momentum: 8h change, 24h change
  - Spot-perp basis
  - Open interest (absolute + changes)
  - Volume metrics
  - Price returns and volatility
  - Time features (hour, day of week)
  - BTC funding rate as market sentiment proxy
"""
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/funding_history")


def load_funding_rates(symbol: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load funding rates CSV into DataFrame."""
    path = (data_dir or DATA_DIR) / "funding_rates" / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["funding_time_dt"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True)
    df["funding_rate"] = df["funding_rate"].astype(float)
    df["mark_price"] = df["mark_price"].astype(float)
    df = df.sort_values("funding_time").reset_index(drop=True)
    return df


def load_klines(symbol: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load klines CSV into DataFrame."""
    path = (data_dir or DATA_DIR) / "klines" / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["open_time_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                 "taker_buy_volume", "taker_buy_quote_volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def load_open_interest(symbol: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load open interest CSV into DataFrame."""
    path = (data_dir or DATA_DIR) / "open_interest" / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["sum_open_interest"] = df["sum_open_interest"].astype(float)
    df["sum_open_interest_value"] = df["sum_open_interest_value"].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _aggregate_klines_to_8h(klines_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1h klines to 8h periods aligned with funding settlements."""
    if klines_df.empty:
        return pd.DataFrame()

    df = klines_df.copy()
    df = df.set_index("open_time_dt")

    # Resample to 8h starting at 00:00 UTC (funding settlement aligned)
    agg = df.resample("8h", offset="0h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "trades": "sum",
        "taker_buy_volume": "sum",
        "taker_buy_quote_volume": "sum",
    }).dropna()

    # Derived features
    agg["return_8h"] = agg["close"].pct_change()
    agg["volatility_8h"] = (agg["high"] - agg["low"]) / agg["close"]
    agg["taker_buy_ratio"] = agg["taker_buy_volume"] / agg["volume"].replace(0, np.nan)

    return agg


def _aggregate_oi_to_8h(oi_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate open interest to 8h periods."""
    if oi_df.empty:
        return pd.DataFrame()

    df = oi_df.copy()
    df = df.set_index("timestamp_dt")

    agg = df.resample("8h", offset="0h").agg({
        "sum_open_interest": "last",
        "sum_open_interest_value": "last",
    }).dropna()

    agg["oi_change_8h"] = agg["sum_open_interest"].pct_change()
    agg["oi_change_24h"] = agg["sum_open_interest"].pct_change(3)  # 3 × 8h = 24h

    return agg


def build_features_for_symbol(
    symbol: str,
    btc_rates: Optional[pd.DataFrame] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Build feature matrix for a single symbol.

    Each row corresponds to one 8h funding period.
    Target: next_funding_rate (what we're predicting).

    Args:
        symbol: Trading pair (e.g. "ETHUSDT").
        btc_rates: Pre-loaded BTC funding rates for cross-asset feature.
        data_dir: Override data directory.

    Returns:
        DataFrame with features + target, indexed by funding time.
    """
    rates_df = load_funding_rates(symbol, data_dir)
    klines_df = load_klines(symbol, data_dir)
    oi_df = load_open_interest(symbol, data_dir)

    if rates_df.empty or len(rates_df) < 10:
        logger.warning("Insufficient funding rate data for %s (%d rows)", symbol, len(rates_df))
        return pd.DataFrame()

    # Start with funding rates as base
    df = rates_df[["funding_time_dt", "funding_rate", "mark_price"]].copy()
    df = df.set_index("funding_time_dt")
    df = df[~df.index.duplicated(keep="last")]

    # === Lag features ===
    df["rate_lag_1"] = df["funding_rate"].shift(1)
    df["rate_lag_2"] = df["funding_rate"].shift(2)
    df["rate_lag_3"] = df["funding_rate"].shift(3)

    # Rate momentum
    df["rate_change_8h"] = df["funding_rate"] - df["rate_lag_1"]
    df["rate_change_24h"] = df["funding_rate"] - df["funding_rate"].shift(3)

    # Rate statistics
    df["rate_mean_3"] = df["funding_rate"].rolling(3).mean()
    df["rate_std_3"] = df["funding_rate"].rolling(3).std()
    df["rate_mean_9"] = df["funding_rate"].rolling(9).mean()  # 3-day average
    df["rate_positive_streak"] = (
        df["funding_rate"]
        .gt(0)
        .groupby((~df["funding_rate"].gt(0)).cumsum())
        .cumcount()
    )

    # === Price features ===
    df["price_return_8h"] = df["mark_price"].pct_change()
    df["price_return_24h"] = df["mark_price"].pct_change(3)
    df["price_volatility_24h"] = df["mark_price"].pct_change().rolling(3).std()

    # === Kline features ===
    if not klines_df.empty:
        kline_8h = _aggregate_klines_to_8h(klines_df)
        if not kline_8h.empty:
            # Align klines to funding times (nearest 8h period)
            kline_features = kline_8h[["volume", "quote_volume", "return_8h",
                                       "volatility_8h", "taker_buy_ratio"]].copy()
            kline_features.columns = ["kline_volume", "kline_quote_volume",
                                      "kline_return_8h", "kline_volatility_8h",
                                      "kline_taker_buy_ratio"]
            df = df.join(kline_features, how="left")
            # Forward-fill kline features (kline timestamps may not align exactly)
            for col in kline_features.columns:
                if col in df.columns:
                    df[col] = df[col].ffill()

            # Volume momentum
            if "kline_quote_volume" in df.columns:
                df["volume_change_8h"] = df["kline_quote_volume"].pct_change(fill_method=None)
                df["volume_mean_3"] = df["kline_quote_volume"].rolling(3).mean()

    # === Open interest features ===
    if not oi_df.empty:
        oi_8h = _aggregate_oi_to_8h(oi_df)
        if not oi_8h.empty:
            oi_features = oi_8h[["sum_open_interest_value", "oi_change_8h",
                                  "oi_change_24h"]].copy()
            oi_features.columns = ["open_interest_value", "oi_change_8h", "oi_change_24h"]
            df = df.join(oi_features, how="left")
            for col in oi_features.columns:
                if col in df.columns:
                    df[col] = df[col].ffill()

    # === Time features ===
    df["hour_of_day"] = df.index.hour  # 0, 8, or 16
    df["day_of_week"] = df.index.dayofweek  # 0=Mon, 6=Sun
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # === BTC funding rate (market sentiment proxy) ===
    if btc_rates is not None and not btc_rates.empty and symbol != "BTCUSDT":
        btc_feature = btc_rates[["funding_time_dt", "funding_rate"]].copy()
        btc_feature = btc_feature.set_index("funding_time_dt")
        btc_feature = btc_feature[~btc_feature.index.duplicated(keep="last")]
        btc_feature.columns = ["btc_funding_rate"]
        df = df.join(btc_feature, how="left")

        # BTC rate momentum
        if "btc_funding_rate" in df.columns:
            df["btc_rate_change"] = df["btc_funding_rate"].diff()
            df["btc_rate_mean_3"] = df["btc_funding_rate"].rolling(3).mean()

    # === Rate relative to BTC (spread) ===
    if "btc_funding_rate" in df.columns:
        df["rate_vs_btc"] = df["funding_rate"] - df["btc_funding_rate"]

    # === Rate regime features ===
    df["rate_zscore_9"] = (
        (df["funding_rate"] - df["funding_rate"].rolling(9).mean())
        / df["funding_rate"].rolling(9).std().replace(0, np.nan)
    )
    df["rate_percentile_30"] = df["funding_rate"].rolling(30).rank(pct=True)

    # === Target: next funding rate ===
    df["next_funding_rate"] = df["funding_rate"].shift(-1)
    df["next_rate_positive"] = (df["next_funding_rate"] > 0).astype(int)

    # Drop rows with NaN in key features (from lag/shift)
    df = df.dropna(subset=["rate_lag_3", "next_funding_rate"])

    # Add symbol column
    df["symbol"] = symbol

    logger.info("%s: built %d feature rows", symbol, len(df))
    return df


def build_features_all_symbols(
    symbols: List[str],
    data_dir: Optional[Path] = None,
    pool: bool = True,
) -> pd.DataFrame:
    """Build feature matrix for multiple symbols.

    Args:
        symbols: List of symbols to process.
        data_dir: Override data directory.
        pool: If True, combine all symbols into one DataFrame.

    Returns:
        Combined DataFrame with features + target for all symbols.
    """
    # Load BTC rates first (used as cross-asset feature)
    btc_rates = load_funding_rates("BTCUSDT", data_dir)

    all_dfs = []
    for symbol in symbols:
        df = build_features_for_symbol(symbol, btc_rates=btc_rates, data_dir=data_dir)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    if pool:
        combined = pd.concat(all_dfs, axis=0).sort_index()
        logger.info(
            "Built pooled features: %d rows, %d columns, %d symbols",
            len(combined), len(combined.columns), len(all_dfs),
        )
        return combined
    else:
        return all_dfs


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature column names (excluding target and metadata)."""
    exclude = {
        "symbol", "funding_rate", "mark_price",
        "next_funding_rate", "next_rate_positive",
    }
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]
