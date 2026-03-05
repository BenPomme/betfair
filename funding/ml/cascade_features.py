"""
Feature engineering for liquidation cascade prediction at 1h resolution.

A "cascade" is a self-reinforcing chain of liquidations that drives sharp
price dislocations (typically >5% in 4h). The model predicts whether one
will occur in the next 4 hours.

Features (indexed at 1h resolution):
  1.  OI concentration      — Herfindahl-Hirschman index across symbols
  2.  Liquidation count      — count of liquidation events in sliding 1h window
  3.  Liquidation volume     — USD volume of liquidations in sliding 1h window
  4.  Depth depletion        — change in bid/ask depth USD over time
  5.  Funding extremity      — max |funding rate| across symbols
  6.  Cross-asset PC1        — first principal component of returns across symbols
  7.  Volume surge z-score   — current volume vs rolling mean (z-score)
  8.  Price acceleration     — second derivative of price (d²price/dt²)
  9.  Leverage proxy         — OI value / volume ratio

Data sources:
  - data/funding_history/liquidations/YYYY-MM-DD.jsonl   (LiquidationStream)
  - data/funding_history/depth/{symbol}/YYYY-MM-DD.jsonl.gz  (DepthCollector)
  - data/funding_history/klines/{symbol}.csv             (feature_engineer loaders)
  - data/funding_history/open_interest/{symbol}.csv      (feature_engineer loaders)
  - data/funding_history/funding_rates/{symbol}.csv      (feature_engineer loaders)

Known historical cascade dates (UTC):
  "2020-03-12" — Black Thursday COVID crash
  "2021-05-19" — China mining ban sell-off
  "2022-06-13" — LUNA / 3AC contagion
  "2022-11-08" — FTX collapse
  "2025-10-15" — user-provided

Usage:
    from funding.ml.cascade_features import build_cascade_features, label_cascade_events

    df   = build_cascade_features(["BTCUSDT", "ETHUSDT"])
    labels = label_cascade_events(df)
    feat_cols = get_cascade_feature_columns(df)
"""
import gzip
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR = Path("data/funding_history")
_LIQ_DIR  = _DATA_DIR / "liquidations"
_DEPTH_DIR = _DATA_DIR / "depth"

# ---------------------------------------------------------------------------
# Known cascade event dates (UTC midnight)
# ---------------------------------------------------------------------------

_KNOWN_CASCADE_DATES: List[str] = [
    "2020-03-12",
    "2021-05-19",
    "2022-06-13",
    "2022-11-08",
    "2025-10-15",
]

# Cascade detection parameters
_CASCADE_HORIZON_HOURS = 4          # Label window: 4h before a cascade
_AUTO_DETECT_DRAWDOWN_THRESHOLD = -0.05   # 5% drawdown within 4h → cascade


# ===========================================================================
# Low-level loaders
# ===========================================================================

def _load_klines_1h(symbol: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load 1h kline CSV for *symbol*.

    Reuses the same file layout as feature_engineer.load_klines but returns
    raw 1h bars (no aggregation).

    Returns:
        DataFrame indexed by open_time_dt (UTC), sorted ascending.
        Empty DataFrame if no file found.
    """
    path = (data_dir or _DATA_DIR) / "klines" / f"{symbol}.csv"
    if not path.exists():
        logger.debug("No klines file for %s at %s", symbol, path)
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["open_time_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_volume", "taker_buy_quote_volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    df = df.sort_values("open_time_dt").set_index("open_time_dt")
    df = df[~df.index.duplicated(keep="last")]
    return df


def _load_oi_1h(symbol: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load open interest CSV for *symbol* and resample to 1h.

    Returns:
        DataFrame indexed by hourly UTC timestamps with columns:
            sum_open_interest_value (USD notional of OI)
        Empty DataFrame if no file found.
    """
    path = (data_dir or _DATA_DIR) / "open_interest" / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["sum_open_interest_value"] = df["sum_open_interest_value"].astype(float)
    df = df.sort_values("timestamp_dt").set_index("timestamp_dt")
    df = df[~df.index.duplicated(keep="last")]

    # Resample to 1h (forward-fill within hour)
    oi_1h = df["sum_open_interest_value"].resample("1h").last().ffill()
    return oi_1h.to_frame(name="sum_open_interest_value")


def _load_funding_rates_1h(symbol: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load funding rates CSV and forward-fill to 1h resolution.

    Funding rates settle at 00:00, 08:00, 16:00 UTC.  Between settlements
    the rate is constant — we forward-fill it to each 1h bar.

    Returns:
        Series indexed at hourly UTC timestamps with column 'funding_rate'.
        Empty DataFrame if no file found.
    """
    path = (data_dir or _DATA_DIR) / "funding_rates" / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["funding_time_dt"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True)
    df["funding_rate"] = df["funding_rate"].astype(float)
    df = df.sort_values("funding_time_dt").set_index("funding_time_dt")
    df = df[~df.index.duplicated(keep="last")]

    # Upsample to 1h, forward-fill
    rate_1h = df["funding_rate"].resample("1h").last().ffill()
    return rate_1h.to_frame(name="funding_rate")


def _load_liquidations(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load all historical liquidation events from JSONL files.

    Reads every ``YYYY-MM-DD.jsonl`` file in the liquidations directory
    (written by LiquidationStream).

    Returns:
        DataFrame with columns:
            timestamp_dt (UTC), symbol, side, filled_qty, avg_price, notional_usd
        Sorted ascending by timestamp_dt.
        Empty DataFrame if no files found.
    """
    liq_dir = (data_dir or _DATA_DIR) / "liquidations"
    if not liq_dir.exists():
        logger.debug("Liquidation directory not found: %s", liq_dir)
        return pd.DataFrame()

    records = []
    for jsonl_path in sorted(liq_dir.glob("*.jsonl")):
        try:
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    records.append({
                        "timestamp_dt": ev.get("timestamp", ""),
                        "symbol":       ev.get("symbol", ""),
                        "side":         ev.get("side", ""),
                        "filled_qty":   float(ev.get("filled_qty", 0)),
                        "avg_price":    float(ev.get("avg_price", 0)),
                    })
        except OSError as exc:
            logger.warning("Could not read liquidation file %s: %s", jsonl_path, exc)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp_dt"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_dt"])
    df["notional_usd"] = df["filled_qty"] * df["avg_price"]
    df = df.sort_values("timestamp_dt").reset_index(drop=True)
    logger.info("Loaded %d liquidation events from %s", len(df), liq_dir)
    return df


def _load_depth_1h(symbol: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load DepthCollector gzipped JSONL snapshots for *symbol*, resampled to 1h.

    Reads all ``YYYY-MM-DD.jsonl.gz`` files from
    ``data/funding_history/depth/{symbol}/``.

    Returns:
        DataFrame indexed at hourly UTC timestamps with columns:
            bid_depth_usd, ask_depth_usd, bid_ask_imbalance
        Empty DataFrame if no files found.
    """
    depth_dir = (data_dir or _DATA_DIR) / "depth" / symbol
    if not depth_dir.exists():
        return pd.DataFrame()

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
                        "timestamp_dt":     snap.get("timestamp", ""),
                        "bid_depth_usd":    float(snap.get("bid_depth_usd", 0)),
                        "ask_depth_usd":    float(snap.get("ask_depth_usd", 0)),
                        "bid_ask_imbalance": float(snap.get("bid_ask_imbalance", 0)),
                    })
        except (OSError, EOFError) as exc:
            logger.warning("Could not read depth file %s: %s", gz_path, exc)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp_dt"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_dt"])
    df = df.sort_values("timestamp_dt").set_index("timestamp_dt")
    df = df[~df.index.duplicated(keep="last")]

    # Resample to 1h: use last snapshot within each hour
    depth_1h = df[["bid_depth_usd", "ask_depth_usd", "bid_ask_imbalance"]].resample("1h").last().ffill()
    return depth_1h


# ===========================================================================
# Individual feature constructors
# ===========================================================================

def _oi_concentration_feature(
    symbols: List[str],
    hourly_index: pd.DatetimeIndex,
    data_dir: Optional[Path],
) -> pd.Series:
    """Herfindahl-Hirschman index of OI concentration across *symbols*.

    HHI = sum(share_i^2) where share_i = OI_i / sum(OI).
    Range: [1/N, 1.0]. High value → OI concentrated in few symbols → systemic risk.

    Returns:
        Series indexed at *hourly_index* named "oi_concentration_hhi".
    """
    frames: List[pd.Series] = []
    for sym in symbols:
        oi = _load_oi_1h(sym, data_dir)
        if oi.empty:
            continue
        s = oi["sum_open_interest_value"].reindex(hourly_index).ffill()
        s.name = sym
        frames.append(s)

    if not frames:
        return pd.Series(np.nan, index=hourly_index, name="oi_concentration_hhi")

    oi_df = pd.concat(frames, axis=1).clip(lower=0)
    total = oi_df.sum(axis=1).replace(0, np.nan)
    shares = oi_df.div(total, axis=0)
    hhi = (shares ** 2).sum(axis=1)
    hhi.name = "oi_concentration_hhi"
    return hhi


def _liquidation_features(
    liq_df: pd.DataFrame,
    hourly_index: pd.DatetimeIndex,
    symbols: List[str],
) -> pd.DataFrame:
    """Sliding 1h liquidation count and volume for specified symbols.

    Each 1h bar aggregates all liquidation events that fall within that hour.
    The "sliding" aspect comes from the hourly resampling.

    Returns:
        DataFrame indexed at *hourly_index* with columns:
            liq_count_1h, liq_volume_usd_1h
    """
    feats = pd.DataFrame(
        {"liq_count_1h": 0.0, "liq_volume_usd_1h": 0.0},
        index=hourly_index,
    )

    if liq_df.empty:
        return feats

    # Filter to requested symbols if non-trivial
    symbol_set = {s.upper() for s in symbols}
    filtered = liq_df[liq_df["symbol"].isin(symbol_set)] if symbol_set else liq_df

    if filtered.empty:
        return feats

    ts = filtered["timestamp_dt"]
    notional = filtered["notional_usd"]

    # Resample to 1h
    count_1h = ts.groupby(ts.dt.floor("1h")).count().rename("liq_count_1h")
    volume_1h = notional.groupby(ts.dt.floor("1h")).sum().rename("liq_volume_usd_1h")

    # Reindex to our standard index
    feats["liq_count_1h"] = count_1h.reindex(hourly_index).fillna(0.0)
    feats["liq_volume_usd_1h"] = volume_1h.reindex(hourly_index).fillna(0.0)

    return feats


def _depth_depletion_feature(
    symbols: List[str],
    hourly_index: pd.DatetimeIndex,
    data_dir: Optional[Path],
) -> pd.DataFrame:
    """Change in total bid/ask depth USD across symbols over the last 1h.

    A sudden drop in depth → market becoming thin → cascades become easier.

    Returns:
        DataFrame with columns:
            depth_bid_change_1h, depth_ask_change_1h, depth_imbalance_mean
    """
    bid_frames, ask_frames, imb_frames = [], [], []

    for sym in symbols:
        depth = _load_depth_1h(sym, data_dir)
        if depth.empty:
            continue
        bid_frames.append(depth["bid_depth_usd"].reindex(hourly_index).ffill().rename(sym))
        ask_frames.append(depth["ask_depth_usd"].reindex(hourly_index).ffill().rename(sym))
        imb_frames.append(depth["bid_ask_imbalance"].reindex(hourly_index).ffill().rename(sym))

    feats = pd.DataFrame(index=hourly_index)

    if bid_frames:
        total_bid = pd.concat(bid_frames, axis=1).sum(axis=1)
        total_ask = pd.concat(ask_frames, axis=1).sum(axis=1)
        feats["depth_bid_change_1h"] = total_bid.diff(1)
        feats["depth_ask_change_1h"] = total_ask.diff(1)
    else:
        feats["depth_bid_change_1h"] = np.nan
        feats["depth_ask_change_1h"] = np.nan

    if imb_frames:
        feats["depth_imbalance_mean"] = pd.concat(imb_frames, axis=1).mean(axis=1)
    else:
        feats["depth_imbalance_mean"] = np.nan

    return feats


def _funding_extremity_feature(
    symbols: List[str],
    hourly_index: pd.DatetimeIndex,
    data_dir: Optional[Path],
) -> pd.Series:
    """Max |funding rate| across symbols at each hour.

    Extreme funding rates → highly leveraged market → cascade risk.

    Returns:
        Series named "funding_extremity_max_abs".
    """
    frames: List[pd.Series] = []
    for sym in symbols:
        rates = _load_funding_rates_1h(sym, data_dir)
        if rates.empty:
            continue
        s = rates["funding_rate"].abs().reindex(hourly_index).ffill().rename(sym)
        frames.append(s)

    if not frames:
        return pd.Series(np.nan, index=hourly_index, name="funding_extremity_max_abs")

    combined = pd.concat(frames, axis=1)
    result = combined.max(axis=1)
    result.name = "funding_extremity_max_abs"
    return result


def _cross_asset_pc1_feature(
    symbols: List[str],
    hourly_index: pd.DatetimeIndex,
    data_dir: Optional[Path],
    window: int = 24,
) -> pd.Series:
    """Rolling first principal component of hourly returns across symbols.

    Measures how much of cross-asset variance is explained by a single
    systemic factor.  High PC1 loading → correlation spike → cascade risk.

    Uses a rolling *window*-hour PCA, iterating row-by-row for robustness.

    Returns:
        Series named "cross_asset_pc1_loading".
    """
    ret_frames: List[pd.Series] = []
    for sym in symbols:
        klines = _load_klines_1h(sym, data_dir)
        if klines.empty or "close" not in klines.columns:
            continue
        close = klines["close"].reindex(hourly_index).ffill()
        ret = close.pct_change()
        ret.name = sym
        ret_frames.append(ret)

    if len(ret_frames) < 2:
        return pd.Series(np.nan, index=hourly_index, name="cross_asset_pc1_loading")

    ret_df = pd.concat(ret_frames, axis=1).fillna(0.0)

    # Rolling PCA: compute explained variance ratio of first PC over rolling window
    pc1_values = np.full(len(hourly_index), np.nan)

    for i in range(window, len(ret_df) + 1):
        window_data = ret_df.iloc[i - window: i].values
        # Remove constant columns to avoid singular matrix
        std = window_data.std(axis=0)
        valid = std > 1e-10
        if valid.sum() < 2:
            pc1_values[i - 1] = np.nan
            continue
        W = window_data[:, valid]
        try:
            cov = np.cov(W.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            total_var = eigenvalues.sum()
            if total_var > 0:
                pc1_values[i - 1] = eigenvalues[-1] / total_var
        except np.linalg.LinAlgError:
            pc1_values[i - 1] = np.nan

    result = pd.Series(pc1_values, index=hourly_index, name="cross_asset_pc1_loading")
    return result


def _volume_surge_zscore_feature(
    symbols: List[str],
    hourly_index: pd.DatetimeIndex,
    data_dir: Optional[Path],
    window: int = 168,  # 7 days
) -> pd.Series:
    """Volume surge z-score: (volume - rolling_mean) / rolling_std.

    Aggregates quote volume across symbols.  A high z-score indicates
    abnormal trading activity, a common cascade precursor.

    Returns:
        Series named "volume_surge_zscore".
    """
    vol_frames: List[pd.Series] = []
    for sym in symbols:
        klines = _load_klines_1h(sym, data_dir)
        if klines.empty or "quote_volume" not in klines.columns:
            continue
        vol = klines["quote_volume"].reindex(hourly_index).ffill().fillna(0.0)
        vol.name = sym
        vol_frames.append(vol)

    if not vol_frames:
        return pd.Series(np.nan, index=hourly_index, name="volume_surge_zscore")

    total_vol = pd.concat(vol_frames, axis=1).sum(axis=1)
    rolling_mean = total_vol.rolling(window, min_periods=window // 2).mean()
    rolling_std  = total_vol.rolling(window, min_periods=window // 2).std()

    zscore = (total_vol - rolling_mean) / rolling_std.replace(0, np.nan)
    zscore.name = "volume_surge_zscore"
    return zscore


def _price_acceleration_feature(
    symbols: List[str],
    hourly_index: pd.DatetimeIndex,
    data_dir: Optional[Path],
) -> pd.Series:
    """Second derivative of BTC price (rate of change of 1h return).

    Acceleration (d²price/dt²) spikes before and during cascades.
    Uses BTC as the primary market-wide price signal.

    Returns:
        Series named "price_acceleration".
    """
    # Prefer BTC; fall back to first available symbol
    ordered = ["BTCUSDT"] + [s for s in symbols if s != "BTCUSDT"]

    for sym in ordered:
        klines = _load_klines_1h(sym, data_dir)
        if klines.empty or "close" not in klines.columns:
            continue
        close = klines["close"].reindex(hourly_index).ffill()
        ret = close.pct_change()           # first derivative
        accel = ret.diff()                 # second derivative
        accel.name = "price_acceleration"
        return accel

    return pd.Series(np.nan, index=hourly_index, name="price_acceleration")


def _leverage_proxy_feature(
    symbols: List[str],
    hourly_index: pd.DatetimeIndex,
    data_dir: Optional[Path],
) -> pd.Series:
    """Leverage proxy: total OI value / total quote volume.

    A high ratio means the market is highly leveraged relative to turnover —
    a necessary (but not sufficient) condition for cascades.

    Returns:
        Series named "leverage_proxy_oi_vol".
    """
    oi_total  = pd.Series(0.0, index=hourly_index)
    vol_total = pd.Series(0.0, index=hourly_index)
    n_oi = 0
    n_vol = 0

    for sym in symbols:
        oi = _load_oi_1h(sym, data_dir)
        if not oi.empty:
            oi_total += oi["sum_open_interest_value"].reindex(hourly_index).ffill().fillna(0.0)
            n_oi += 1

        klines = _load_klines_1h(sym, data_dir)
        if not klines.empty and "quote_volume" in klines.columns:
            vol = klines["quote_volume"].reindex(hourly_index).ffill().fillna(0.0)
            vol_total += vol
            n_vol += 1

    if n_oi == 0 or n_vol == 0:
        return pd.Series(np.nan, index=hourly_index, name="leverage_proxy_oi_vol")

    proxy = oi_total / vol_total.replace(0, np.nan)
    proxy.name = "leverage_proxy_oi_vol"
    return proxy


# ===========================================================================
# Main builder
# ===========================================================================

def build_cascade_features(
    symbols: List[str],
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Build cascade prediction feature matrix at 1h resolution.

    Loads all available raw data for the given *symbols*, computes eight
    feature groups, and returns a DataFrame aligned to an hourly DatetimeIndex
    spanning the full historical data range.

    Args:
        symbols:  List of Binance USDT-M futures symbols (e.g. ["BTCUSDT", "ETHUSDT"]).
        data_dir: Optional override for the base data directory
                  (default: data/funding_history).

    Returns:
        pd.DataFrame indexed at hourly UTC timestamps with columns:

            oi_concentration_hhi     — Herfindahl index of OI across symbols
            liq_count_1h             — number of liquidations in the 1h window
            liq_volume_usd_1h        — USD volume of liquidations
            depth_bid_change_1h      — 1h change in aggregate bid depth (USD)
            depth_ask_change_1h      — 1h change in aggregate ask depth (USD)
            depth_imbalance_mean     — mean bid/ask imbalance across symbols
            funding_extremity_max_abs— max |funding rate| across symbols
            cross_asset_pc1_loading  — rolling PCA first PC explained variance ratio
            volume_surge_zscore      — volume z-score vs rolling 168h baseline
            price_acceleration       — second derivative of BTC hourly return
            leverage_proxy_oi_vol    — OI / volume ratio

        Returns an empty DataFrame if no data is found for any symbol.
    """
    if not symbols:
        logger.warning("build_cascade_features: no symbols provided.")
        return pd.DataFrame()

    # Build the hourly index from the union of all kline data
    min_ts, max_ts = None, None
    for sym in symbols:
        klines = _load_klines_1h(sym, data_dir)
        if klines.empty:
            continue
        t_min = klines.index.min()
        t_max = klines.index.max()
        if min_ts is None or t_min < min_ts:
            min_ts = t_min
        if max_ts is None or t_max > max_ts:
            max_ts = t_max

    if min_ts is None:
        logger.warning("build_cascade_features: no kline data found for any symbol.")
        return pd.DataFrame()

    hourly_index = pd.date_range(start=min_ts, end=max_ts, freq="1h", tz="UTC")
    logger.info(
        "build_cascade_features: hourly_index from %s to %s (%d hours, %d symbols)",
        min_ts.isoformat(), max_ts.isoformat(), len(hourly_index), len(symbols),
    )

    # Load liquidation data once
    liq_df = _load_liquidations(data_dir)

    # --- Feature 1: OI concentration ---
    logger.debug("Computing OI concentration (HHI)...")
    f_oi_hhi = _oi_concentration_feature(symbols, hourly_index, data_dir)

    # --- Features 2 & 3: Liquidation clustering ---
    logger.debug("Computing liquidation clustering features...")
    f_liq = _liquidation_features(liq_df, hourly_index, symbols)

    # --- Features 4, 5, 6: Depth depletion ---
    logger.debug("Computing depth depletion features...")
    f_depth = _depth_depletion_feature(symbols, hourly_index, data_dir)

    # --- Feature 7: Funding extremity ---
    logger.debug("Computing funding extremity...")
    f_funding_ext = _funding_extremity_feature(symbols, hourly_index, data_dir)

    # --- Feature 8: Cross-asset PC1 ---
    logger.debug("Computing cross-asset PCA first component...")
    f_pc1 = _cross_asset_pc1_feature(symbols, hourly_index, data_dir)

    # --- Feature 9: Volume surge z-score ---
    logger.debug("Computing volume surge z-score...")
    f_vol_zscore = _volume_surge_zscore_feature(symbols, hourly_index, data_dir)

    # --- Feature 10: Price acceleration ---
    logger.debug("Computing price acceleration...")
    f_price_accel = _price_acceleration_feature(symbols, hourly_index, data_dir)

    # --- Feature 11: Leverage proxy ---
    logger.debug("Computing leverage proxy (OI / volume)...")
    f_leverage = _leverage_proxy_feature(symbols, hourly_index, data_dir)

    # --- Assemble ---
    df = pd.concat(
        [
            f_oi_hhi,
            f_liq,
            f_depth,
            f_funding_ext,
            f_pc1,
            f_vol_zscore,
            f_price_accel,
            f_leverage,
        ],
        axis=1,
    )

    df.index.name = "timestamp"

    logger.info(
        "build_cascade_features: built %d rows x %d columns.",
        len(df), len(df.columns),
    )
    return df


# ===========================================================================
# Target labeling
# ===========================================================================

def label_cascade_events(
    df: pd.DataFrame,
    known_events: Optional[List[str]] = None,
) -> pd.Series:
    """Create binary cascade labels for each 1h row in *df*.

    A label of 1 means: a cascade occurred within the next 4 hours.

    Two labeling passes are combined (logical OR):

    1. **Known events**: any hour that falls within *CASCADE_HORIZON_HOURS*
       before a known cascade date gets label=1.

    2. **Auto-detection**: load BTC 1h klines (or the first available symbol);
       any 4h rolling window with a cumulative return < -5% labels the *start*
       of that window as 1.

    Args:
        df:           Feature DataFrame returned by build_cascade_features().
                      Must have a UTC DatetimeIndex at 1h resolution.
        known_events: List of "YYYY-MM-DD" strings for known cascade dates.
                      Defaults to the module-level _KNOWN_CASCADE_DATES list.

    Returns:
        pd.Series of dtype int (0/1) with the same index as *df*, named "cascade_label".
    """
    labels = pd.Series(0, index=df.index, name="cascade_label", dtype=int)

    # --- Pass 1: Known events ---
    events = known_events if known_events is not None else _KNOWN_CASCADE_DATES
    for date_str in events:
        try:
            event_ts = pd.Timestamp(date_str, tz="UTC")
        except Exception:
            logger.warning("Invalid cascade event date: %s", date_str)
            continue
        # Mark all hours in [event_ts - 4h, event_ts)
        window_start = event_ts - pd.Timedelta(hours=_CASCADE_HORIZON_HOURS)
        mask = (df.index >= window_start) & (df.index < event_ts)
        labels[mask] = 1
        logger.debug(
            "Known cascade %s: labelled %d hours", date_str, int(mask.sum())
        )

    # --- Pass 2: Auto-detect from BTC price (>5% drawdown in 4h) ---
    # Try to load BTC klines from the data dir (no explicit data_dir here,
    # use the default path which matches what build_cascade_features uses).
    _btc_klines_path = _DATA_DIR / "klines" / "BTCUSDT.csv"
    close: Optional[pd.Series] = None

    if _btc_klines_path.exists():
        try:
            raw = pd.read_csv(_btc_klines_path)
            raw["open_time_dt"] = pd.to_datetime(raw["open_time"], unit="ms", utc=True)
            raw["close"] = raw["close"].astype(float)
            raw = raw.set_index("open_time_dt").sort_index()
            close = raw["close"].reindex(df.index).ffill()
        except Exception as exc:
            logger.warning("Could not load BTC klines for auto-detection: %s", exc)

    if close is not None and not close.isna().all():
        # Rolling 4h forward return: ret(t) = (close[t+4] - close[t]) / close[t]
        horizon = _CASCADE_HORIZON_HOURS
        forward_close = close.shift(-horizon)
        forward_ret = (forward_close - close) / close

        auto_mask = forward_ret < _AUTO_DETECT_DRAWDOWN_THRESHOLD
        auto_count = int(auto_mask.sum())
        logger.debug(
            "Auto-detected %d cascade hours (>%.0f%% drawdown in %dh)",
            auto_count, abs(_AUTO_DETECT_DRAWDOWN_THRESHOLD) * 100, horizon,
        )

        # Combine: any row that is either known or auto-detected
        labels = (labels | auto_mask.reindex(df.index).fillna(False).astype(int))
    else:
        logger.warning(
            "label_cascade_events: BTC close prices not available for auto-detection. "
            "Only known event labels will be used."
        )

    labels = labels.astype(int)
    total_positive = int(labels.sum())
    logger.info(
        "label_cascade_events: %d cascade hours out of %d total (%.2f%%)",
        total_positive, len(labels),
        100.0 * total_positive / max(len(labels), 1),
    )
    return labels


# ===========================================================================
# Column selector
# ===========================================================================

def get_cascade_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return feature column names from a cascade feature DataFrame.

    Excludes any label columns ("cascade_label") and non-numeric columns.

    Args:
        df: Output of build_cascade_features() (with optional label column).

    Returns:
        List of column names suitable for model input (X).
    """
    exclude = {"cascade_label", "symbol"}
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
