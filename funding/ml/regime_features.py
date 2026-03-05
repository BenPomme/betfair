"""
Feature engineering for the volatility regime classification model.

Predicts 4 regimes derived from realized volatility quantiles:
  0 — low     (bottom 25%)
  1 — medium  (25–50%)
  2 — high    (50–75%)
  3 — crisis  (top 25%)

Feature groups:
  - Multi-timescale realized volatility (8h / 24h / 72h) from 1h klines
  - Volatility-of-volatility
  - Funding rate cross-symbol dispersion
  - Open interest rate of change
  - Volume z-score and surge flag
  - Cross-asset (BTC/ETH) return correlation
  - Liquidation notional volume from JSONL files
  - Bid-ask depth imbalance from gzipped JSONL snapshots
  - Taker buy ratio

All features are indexed at 8h granularity aligned to Binance funding
settlement times (00:00, 08:00, 16:00 UTC).

Usage:
    df = build_regime_features(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    df["regime_label"] = assign_regime_labels(df)
    feature_cols = get_regime_feature_columns(df)
"""
import gzip
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from funding.ml.feature_engineer import load_funding_rates, load_klines, load_open_interest

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = Path("data/funding_history")

# ------------------------------------------------------------------
# Optional import: contrarian features (may not exist yet)
# ------------------------------------------------------------------
try:
    from funding.ml.contrarian_features import load_long_short_ratio, load_fear_greed  # noqa: F401
    _CONTRARIAN_AVAILABLE = True
except ImportError:
    _CONTRARIAN_AVAILABLE = False
    logger.debug("contrarian_features not available; long/short and fear/greed skipped")


# ==================================================================
# Internal helpers — kline-based volatility
# ==================================================================

def _compute_realized_vol_from_klines(
    klines_df: pd.DataFrame,
    period_label: str = "8h",
) -> pd.DataFrame:
    """
    Compute multi-timescale realized volatility from 1h klines.

    Returns a DataFrame indexed at 8h periods with columns:
        realized_vol_8h, realized_vol_24h, realized_vol_72h

    Each realized vol is the standard deviation of 1h log-returns over
    the trailing N hours, resampled to the period boundary.

    Args:
        klines_df:    Raw 1h klines DataFrame (from load_klines()).
        period_label: Unused label for logging.

    Returns:
        DataFrame with realized vol columns, indexed at 8h UTC boundaries.
    """
    if klines_df.empty:
        return pd.DataFrame()

    df = klines_df.copy()
    df = df.set_index("open_time_dt").sort_index()

    # 1h log-returns from close prices (more stationary than pct_change for vol)
    close = df["close"].copy()
    log_ret = np.log(close / close.shift(1))

    # Rolling stds at 1h frequency
    vol_8h_1h = log_ret.rolling(8).std()    # 8 × 1h = 8h
    vol_24h_1h = log_ret.rolling(24).std()  # 24 × 1h = 24h
    vol_72h_1h = log_ret.rolling(72).std()  # 72 × 1h = 72h

    vol_df = pd.DataFrame({
        "realized_vol_8h":  vol_8h_1h,
        "realized_vol_24h": vol_24h_1h,
        "realized_vol_72h": vol_72h_1h,
    })

    # Resample to 8h: take the last observation in each 8h window (end-of-period)
    vol_8h = vol_df.resample("8h", offset="0h").last().dropna(how="all")
    return vol_8h


def _compute_taker_buy_ratio_8h(klines_df: pd.DataFrame) -> pd.Series:
    """
    Compute taker buy ratio resampled to 8h periods.

    taker_buy_ratio_8h = sum(taker_buy_volume) / sum(volume) over the 8h window.

    Returns:
        pd.Series named "taker_buy_ratio_8h", indexed at 8h UTC boundaries.
    """
    if klines_df.empty:
        return pd.Series(dtype=float, name="taker_buy_ratio_8h")

    df = klines_df.copy()
    df = df.set_index("open_time_dt").sort_index()

    tbv = df["taker_buy_volume"].resample("8h", offset="0h").sum()
    vol = df["volume"].resample("8h", offset="0h").sum()
    ratio = (tbv / vol.replace(0, np.nan)).rename("taker_buy_ratio_8h")
    return ratio.dropna()


def _compute_volume_features_8h(klines_df: pd.DataFrame, window: int = 9) -> pd.DataFrame:
    """
    Compute volume z-score and surge flag at 8h granularity.

    Columns returned:
        volume_zscore_9:  z-score of quote_volume over *window* 8h periods
        volume_surge:     int flag — 1 if quote_volume > 2 × rolling mean

    Args:
        klines_df: Raw 1h klines DataFrame.
        window:    Rolling window size in 8h periods (default 9 ≈ 3 days).

    Returns:
        DataFrame with volume feature columns, indexed at 8h UTC boundaries.
    """
    if klines_df.empty:
        return pd.DataFrame()

    df = klines_df.copy()
    df = df.set_index("open_time_dt").sort_index()

    qvol_8h = df["quote_volume"].resample("8h", offset="0h").sum().dropna()

    roll_mean = qvol_8h.rolling(window).mean()
    roll_std = qvol_8h.rolling(window).std()

    zscore = ((qvol_8h - roll_mean) / roll_std.replace(0, np.nan)).rename("volume_zscore_9")
    surge = (qvol_8h > 2 * roll_mean).astype(int).rename("volume_surge")

    return pd.concat([zscore, surge], axis=1)


# ==================================================================
# Internal helpers — open interest
# ==================================================================

def _compute_oi_changes_8h(oi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute OI rate-of-change at 8h and 24h granularity.

    Columns:
        oi_change_rate_8h:  pct_change(1)  of OI over one 8h period
        oi_change_rate_24h: pct_change(3)  of OI over three 8h periods

    Returns:
        DataFrame, indexed at 8h UTC boundaries.
    """
    if oi_df.empty:
        return pd.DataFrame()

    df = oi_df.copy()
    df = df.set_index("timestamp_dt").sort_index()

    oi_8h = df["sum_open_interest"].resample("8h", offset="0h").last().dropna()

    result = pd.DataFrame(index=oi_8h.index)
    result["oi_change_rate_8h"] = oi_8h.pct_change(1)
    result["oi_change_rate_24h"] = oi_8h.pct_change(3)
    return result


# ==================================================================
# Internal helpers — liquidation JSONL
# ==================================================================

def _load_liquidation_volume_8h(
    symbol: str,
    data_dir: Path,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Load total liquidation notional for *symbol* from daily JSONL files and
    aggregate into 8h buckets aligned to *index*.

    Files are expected at:
        <data_dir>/liquidations/YYYY-MM-DD.jsonl

    Each line is a dict with fields:
        symbol, filled_qty, avg_price, timestamp (ISO 8601 UTC)

    Notional = filled_qty * avg_price.

    Args:
        symbol:   Uppercase symbol string (e.g. "BTCUSDT").
        data_dir: Base data directory.
        index:    DatetimeIndex of 8h period starts to align output to.

    Returns:
        pd.Series named "liquidation_volume_8h" aligned to *index*.
        Fills missing periods with 0.
    """
    liq_dir = data_dir / "liquidations"
    if not liq_dir.exists():
        logger.debug("No liquidation directory at %s; filling with 0", liq_dir)
        return pd.Series(0.0, index=index, name="liquidation_volume_8h")

    records = []
    for jsonl_path in sorted(liq_dir.glob("*.jsonl")):
        try:
            with open(jsonl_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("symbol", "").upper() != symbol.upper():
                        continue
                    records.append({
                        "timestamp": event.get("timestamp", ""),
                        "notional": float(event.get("filled_qty", 0)) * float(event.get("avg_price", 0)),
                    })
        except OSError as exc:
            logger.warning("Could not read %s: %s", jsonl_path, exc)

    if not records:
        logger.debug("No liquidation records found for %s; filling with 0", symbol)
        return pd.Series(0.0, index=index, name="liquidation_volume_8h")

    liq_df = pd.DataFrame(records)
    liq_df["ts"] = pd.to_datetime(liq_df["timestamp"], utc=True, errors="coerce")
    liq_df = liq_df.dropna(subset=["ts"])
    liq_df = liq_df.set_index("ts").sort_index()

    # Resample to 8h — sum notional within each window
    liq_8h = liq_df["notional"].resample("8h", offset="0h").sum()

    # Reindex to match main index, fill missing periods with 0
    liq_aligned = liq_8h.reindex(index, fill_value=0.0)
    liq_aligned.name = "liquidation_volume_8h"
    return liq_aligned


# ==================================================================
# Internal helpers — depth imbalance
# ==================================================================

def _load_depth_imbalance_8h(
    symbol: str,
    data_dir: Path,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Load bid-ask imbalance from gzipped JSONL snapshots and average over
    each 8h period.

    Files are at:
        <data_dir>/depth/<SYMBOL>/YYYY-MM-DD.jsonl.gz

    Each line is a dict with "timestamp" (ISO 8601 UTC) and "bid_ask_imbalance" (float).

    Args:
        symbol:   Uppercase symbol string.
        data_dir: Base data directory.
        index:    DatetimeIndex of 8h period starts to align output to.

    Returns:
        pd.Series named "avg_bid_ask_imbalance" aligned to *index*.
        Fills missing periods with NaN (filled forward later by caller).
    """
    depth_dir = data_dir / "depth" / symbol.upper()
    if not depth_dir.exists():
        logger.debug("No depth directory for %s; skipping depth imbalance", symbol)
        return pd.Series(np.nan, index=index, name="avg_bid_ask_imbalance")

    records = []
    for gz_path in sorted(depth_dir.glob("*.jsonl.gz")):
        try:
            with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        snap = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    records.append({
                        "timestamp": snap.get("timestamp", ""),
                        "imbalance": float(snap.get("bid_ask_imbalance", np.nan)),
                    })
        except OSError as exc:
            logger.warning("Could not read %s: %s", gz_path, exc)

    if not records:
        logger.debug("No depth snapshots found for %s; skipping depth imbalance", symbol)
        return pd.Series(np.nan, index=index, name="avg_bid_ask_imbalance")

    depth_df = pd.DataFrame(records)
    depth_df["ts"] = pd.to_datetime(depth_df["timestamp"], utc=True, errors="coerce")
    depth_df = depth_df.dropna(subset=["ts"])
    depth_df = depth_df.set_index("ts").sort_index()

    # Average imbalance within each 8h period
    avg_imbalance = depth_df["imbalance"].resample("8h", offset="0h").mean()
    aligned = avg_imbalance.reindex(index)
    aligned.name = "avg_bid_ask_imbalance"
    return aligned


# ==================================================================
# Internal helpers — cross-asset correlation
# ==================================================================

def _compute_btc_eth_return_corr(
    btc_klines: pd.DataFrame,
    eth_klines: pd.DataFrame,
    window: int = 9,
) -> pd.Series:
    """
    Compute rolling correlation between BTC and ETH 8h returns.

    Args:
        btc_klines: Raw 1h klines DataFrame for BTCUSDT.
        eth_klines: Raw 1h klines DataFrame for ETHUSDT.
        window:     Rolling window in 8h periods (default 9 ≈ 3 days).

    Returns:
        pd.Series named "btc_eth_return_corr_9" at 8h frequency.
    """
    if btc_klines.empty or eth_klines.empty:
        return pd.Series(dtype=float, name="btc_eth_return_corr_9")

    def _8h_returns(klines_df: pd.DataFrame) -> pd.Series:
        df = klines_df.copy().set_index("open_time_dt").sort_index()
        close_8h = df["close"].resample("8h", offset="0h").last().dropna()
        return close_8h.pct_change()

    btc_ret = _8h_returns(btc_klines)
    eth_ret = _8h_returns(eth_klines)

    # Align on common index
    combined = pd.DataFrame({"btc": btc_ret, "eth": eth_ret}).dropna()

    corr = combined["btc"].rolling(window).corr(combined["eth"]).rename("btc_eth_return_corr_9")
    return corr


# ==================================================================
# Internal helpers — funding rate dispersion
# ==================================================================

def _compute_funding_dispersion(
    all_rates: dict,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Compute cross-symbol funding rate dispersion at each 8h period.

    Dispersion = standard deviation of funding rates across all symbols
    at the same timestamp.

    Args:
        all_rates: Dict of symbol -> pd.DataFrame from load_funding_rates().
        index:     DatetimeIndex of 8h period starts to align output to.

    Returns:
        pd.Series named "funding_dispersion" aligned to *index*.
    """
    rate_series = {}
    for sym, df in all_rates.items():
        if df.empty:
            continue
        s = df.set_index("funding_time_dt")["funding_rate"]
        s = s[~s.index.duplicated(keep="last")]
        rate_series[sym] = s

    if not rate_series:
        return pd.Series(np.nan, index=index, name="funding_dispersion")

    aligned = pd.DataFrame(rate_series).reindex(index)
    dispersion = aligned.std(axis=1).rename("funding_dispersion")
    return dispersion


# ==================================================================
# Public API
# ==================================================================

def build_regime_features(
    symbols: List[str],
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build a feature matrix for the volatility regime classification model.

    The first symbol in *symbols* is treated as the **primary** asset.
    All remaining symbols contribute to cross-asset features (dispersion,
    BTC/ETH correlation).

    Each row corresponds to one 8h funding period.  The index is a
    UTC-aware DatetimeIndex.

    Args:
        symbols:  Non-empty list of symbols (e.g. ["BTCUSDT", "ETHUSDT"]).
                  At least one symbol is required; BTC and ETH should be
                  included for the cross-asset correlation feature.
        data_dir: Override for the base data directory.
                  Defaults to data/funding_history/.

    Returns:
        DataFrame with all regime features.  Rows with insufficient
        history (i.e., leading NaNs from rolling windows) are **not**
        dropped here — callers should decide their own dropna strategy.

    Raises:
        ValueError: If *symbols* is empty.
    """
    if not symbols:
        raise ValueError("build_regime_features() requires at least one symbol")

    base_dir = data_dir or _DEFAULT_DATA_DIR
    primary = symbols[0].upper()
    all_syms = [s.upper() for s in symbols]

    logger.info("Building regime features: primary=%s, all=%s", primary, all_syms)

    # ------------------------------------------------------------------
    # 1. Load raw data for all symbols
    # ------------------------------------------------------------------
    all_klines: dict = {}
    all_oi: dict = {}
    all_rates: dict = {}

    for sym in all_syms:
        all_klines[sym] = load_klines(sym, base_dir)
        all_oi[sym] = load_open_interest(sym, base_dir)
        all_rates[sym] = load_funding_rates(sym, base_dir)

    primary_klines = all_klines[primary]
    primary_oi = all_oi[primary]

    # ------------------------------------------------------------------
    # 2. Establish the common 8h index from primary klines
    # ------------------------------------------------------------------
    if primary_klines.empty:
        logger.warning("No klines data for primary symbol %s; returning empty DataFrame", primary)
        return pd.DataFrame()

    tmp_close = (
        primary_klines.set_index("open_time_dt")["close"]
        .resample("8h", offset="0h").last()
        .dropna()
    )
    common_index: pd.DatetimeIndex = tmp_close.index

    # ------------------------------------------------------------------
    # 3. Multi-timescale realized volatility
    # ------------------------------------------------------------------
    vol_df = _compute_realized_vol_from_klines(primary_klines)
    # Align to common index
    vol_df = vol_df.reindex(common_index)

    # ------------------------------------------------------------------
    # 4. Vol-of-vol: rolling std of realized_vol_24h over 9 periods
    # ------------------------------------------------------------------
    if "realized_vol_24h" in vol_df.columns:
        vol_df["vol_of_vol_9"] = vol_df["realized_vol_24h"].rolling(9).std()

    # ------------------------------------------------------------------
    # 5. Start assembling the master DataFrame
    # ------------------------------------------------------------------
    df = vol_df.copy()

    # ------------------------------------------------------------------
    # 6. Funding rate dispersion across all symbols
    # ------------------------------------------------------------------
    df["funding_dispersion"] = _compute_funding_dispersion(all_rates, common_index)

    # ------------------------------------------------------------------
    # 7. Open interest rate of change
    # ------------------------------------------------------------------
    oi_changes = _compute_oi_changes_8h(primary_oi)
    for col in ["oi_change_rate_8h", "oi_change_rate_24h"]:
        if col in oi_changes.columns:
            df[col] = oi_changes[col].reindex(common_index)
        else:
            df[col] = np.nan

    # ------------------------------------------------------------------
    # 8. Volume features
    # ------------------------------------------------------------------
    vol_feat = _compute_volume_features_8h(primary_klines, window=9)
    for col in ["volume_zscore_9", "volume_surge"]:
        if col in vol_feat.columns:
            df[col] = vol_feat[col].reindex(common_index)
        else:
            df[col] = np.nan

    # ------------------------------------------------------------------
    # 9. Cross-asset BTC/ETH return correlation
    # ------------------------------------------------------------------
    btc_klines = all_klines.get("BTCUSDT", pd.DataFrame())
    eth_klines = all_klines.get("ETHUSDT", pd.DataFrame())

    if not btc_klines.empty and not eth_klines.empty:
        corr_series = _compute_btc_eth_return_corr(btc_klines, eth_klines, window=9)
        df["btc_eth_return_corr_9"] = corr_series.reindex(common_index)
    else:
        logger.debug(
            "BTC or ETH klines not available; btc_eth_return_corr_9 will be NaN "
            "(pass both BTCUSDT and ETHUSDT in symbols for this feature)"
        )
        df["btc_eth_return_corr_9"] = np.nan

    # ------------------------------------------------------------------
    # 10. Liquidation volume
    # ------------------------------------------------------------------
    df["liquidation_volume_8h"] = _load_liquidation_volume_8h(
        primary, base_dir, common_index
    )

    # ------------------------------------------------------------------
    # 11. Depth imbalance
    # ------------------------------------------------------------------
    df["avg_bid_ask_imbalance"] = _load_depth_imbalance_8h(
        primary, base_dir, common_index
    )
    # Forward-fill sparse depth snapshots (may only exist during live collection)
    df["avg_bid_ask_imbalance"] = df["avg_bid_ask_imbalance"].ffill()

    # ------------------------------------------------------------------
    # 12. Taker buy ratio
    # ------------------------------------------------------------------
    taker = _compute_taker_buy_ratio_8h(primary_klines)
    df["taker_buy_ratio_8h"] = taker.reindex(common_index)

    # ------------------------------------------------------------------
    # 13. Metadata
    # ------------------------------------------------------------------
    df["symbol"] = primary

    logger.info(
        "Regime features built: symbol=%s, rows=%d, columns=%d",
        primary,
        len(df),
        len(df.columns),
    )
    return df


# ==================================================================
# Target labeling
# ==================================================================

def assign_regime_labels(
    df: pd.DataFrame,
    vol_col: str = "realized_vol_24h",
) -> pd.Series:
    """
    Assign discrete volatility regime labels based on quantiles of *vol_col*.

    Labels:
        0 — low     (vol <= 25th percentile)
        1 — medium  (25th < vol <= 50th percentile)
        2 — high    (50th < vol <= 75th percentile)
        3 — crisis  (vol > 75th percentile)

    NaN rows in *vol_col* are assigned NaN labels.

    Args:
        df:      DataFrame containing *vol_col*.
        vol_col: Column name to compute quantiles from.
                 Default: "realized_vol_24h".

    Returns:
        pd.Series named "regime_label" with dtype float64 (NaN-safe).

    Raises:
        KeyError: If *vol_col* is not found in *df*.
    """
    if vol_col not in df.columns:
        raise KeyError(
            f"Column '{vol_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    vol = df[vol_col]
    q25 = vol.quantile(0.25)
    q50 = vol.quantile(0.50)
    q75 = vol.quantile(0.75)

    logger.debug(
        "Regime quantiles (%s): q25=%.6f, q50=%.6f, q75=%.6f",
        vol_col, q25, q50, q75,
    )

    def _label(v: float) -> float:
        if pd.isna(v):
            return np.nan
        if v <= q25:
            return 0.0
        if v <= q50:
            return 1.0
        if v <= q75:
            return 2.0
        return 3.0

    labels = vol.map(_label)
    labels.name = "regime_label"
    return labels


# ==================================================================
# Feature column selector
# ==================================================================

def get_regime_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of feature column names suitable for model training.

    Excludes metadata and label columns:
        symbol, regime_label

    Includes only numeric columns (float or int dtypes).

    Args:
        df: Feature DataFrame produced by build_regime_features().

    Returns:
        List of column name strings.
    """
    exclude = {"symbol", "regime_label"}
    numeric_dtypes = {np.float64, np.int64, np.float32, np.int32, float, int}
    return [
        col for col in df.columns
        if col not in exclude and df[col].dtype.type in numeric_dtypes
    ]
