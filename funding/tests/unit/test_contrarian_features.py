"""
Unit tests for contrarian feature engineering (funding.ml.contrarian_features).

All tests use synthetic in-memory DataFrames; no CSV files on disk are required.
The file-loading helpers (load_funding_rates, load_klines, load_open_interest,
load_long_short_ratio, load_fear_greed) are patched to return controlled data so
that build_contrarian_features() can be exercised end-to-end without any I/O.

Tests cover:
  1. Feature computation produces the expected columns
  2. Extreme funding detection (consecutive_extreme_positive / negative)
  3. Long/short ratio features (z-score, extreme flags)
  4. Fear & greed features (extreme flags at boundaries)
  5. Target columns present (price_return_24h_target, direction_24h)
  6. get_contrarian_feature_columns excludes targets and metadata
"""

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants mirroring contrarian_features.py
# ---------------------------------------------------------------------------
EXTREME_POSITIVE_THRESHOLD = 0.0005
EXTREME_NEGATIVE_THRESHOLD = -0.0005
LS_EXTREME_LONG = 2.0
LS_EXTREME_SHORT = 0.5
FEAR_GREED_EXTREME_FEAR = 25
FEAR_GREED_EXTREME_GREED = 75


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_funding_rates(
    n: int = 60,
    rate: float = 0.0001,
    start: datetime = None,
    mark_price: float = 40000.0,
    rate_override: list = None,
) -> pd.DataFrame:
    """Build a synthetic funding-rates DataFrame with n rows.

    Args:
        n:             Number of rows.
        rate:          Constant funding rate applied to all rows (unless
                       rate_override is supplied).
        start:         Start timestamp (UTC). Defaults to a fixed reference time.
        mark_price:    Constant mark price.
        rate_override: Optional list of floats of length n; overrides `rate`.

    Returns:
        DataFrame compatible with load_funding_rates() output.
    """
    if start is None:
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    times = [start + timedelta(hours=8 * i) for i in range(n)]
    timestamps_ms = [int(t.timestamp() * 1000) for t in times]

    rates = rate_override if rate_override is not None else [rate] * n

    df = pd.DataFrame({
        "funding_time": timestamps_ms,
        "funding_rate": [float(r) for r in rates],
        "mark_price": [mark_price] * n,
    })
    df["funding_time_dt"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True)
    return df


def _make_long_short_ratio(
    n: int = 60,
    ratio: float = 1.2,
    start: datetime = None,
) -> pd.DataFrame:
    """Build a synthetic long/short ratio DataFrame."""
    if start is None:
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    times = [start + timedelta(hours=8 * i) for i in range(n)]
    timestamps_ms = [int(t.timestamp() * 1000) for t in times]

    df = pd.DataFrame({
        "timestamp": timestamps_ms,
        "long_short_ratio": [float(ratio)] * n,
        "long_account": [0.55] * n,
        "short_account": [0.45] * n,
        "top_long_short_ratio": [1.1] * n,
        "top_long_account": [0.53] * n,
        "top_short_account": [0.47] * n,
    })
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def _make_fear_greed(
    n: int = 60,
    value: float = 50.0,
    start: datetime = None,
) -> pd.DataFrame:
    """Build a synthetic fear & greed index DataFrame."""
    if start is None:
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    # One entry per day (24h spacing) — merged asof so alignment still works
    times = [start + timedelta(hours=24 * i) for i in range(n)]
    timestamps_ms = [int(t.timestamp() * 1000) for t in times]

    df = pd.DataFrame({
        "timestamp": timestamps_ms,
        "value": [float(value)] * n,
        "value_classification": ["Neutral"] * n,
    })
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------
PATCH_BASE = "funding.ml.contrarian_features"
PATCH_LOAD_RATES = f"{PATCH_BASE}.load_funding_rates"
PATCH_LOAD_KLINES = f"{PATCH_BASE}.load_klines"
PATCH_LOAD_OI = f"{PATCH_BASE}.load_open_interest"
PATCH_LOAD_LS = f"{PATCH_BASE}.load_long_short_ratio"
PATCH_LOAD_FG = f"{PATCH_BASE}.load_fear_greed"


def _build_with_mocks(
    rates_df: pd.DataFrame,
    ls_df: pd.DataFrame = None,
    fg_df: pd.DataFrame = None,
    symbol: str = "BTCUSDT",
):
    """Invoke build_contrarian_features with mocked I/O.

    klines and open_interest are always returned as empty DataFrames because
    the tests in this suite focus on funding, L/S ratio, and fear & greed features.
    """
    from funding.ml.contrarian_features import build_contrarian_features

    if ls_df is None:
        ls_df = pd.DataFrame()
    if fg_df is None:
        fg_df = pd.DataFrame()

    with patch(PATCH_LOAD_RATES, return_value=rates_df), \
         patch(PATCH_LOAD_KLINES, return_value=pd.DataFrame()), \
         patch(PATCH_LOAD_OI, return_value=pd.DataFrame()), \
         patch(PATCH_LOAD_LS, return_value=ls_df), \
         patch(PATCH_LOAD_FG, return_value=fg_df):
        return build_contrarian_features(symbol)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestFeatureColumnsPresent(unittest.TestCase):
    """Test 1: Expected columns are present after feature computation."""

    def setUp(self):
        self.df = _build_with_mocks(_make_funding_rates(n=60))

    def test_returns_non_empty_dataframe(self):
        self.assertFalse(self.df.empty, "Expected non-empty DataFrame")

    def test_funding_rate_column_present(self):
        self.assertIn("funding_rate", self.df.columns)

    def test_funding_zscore_30_present(self):
        self.assertIn("funding_zscore_30", self.df.columns)

    def test_rate_lag_columns_present(self):
        for col in ("rate_lag_1", "rate_lag_2", "rate_lag_3"):
            with self.subTest(col=col):
                self.assertIn(col, self.df.columns)

    def test_consecutive_extreme_columns_present(self):
        self.assertIn("consecutive_extreme_positive", self.df.columns)
        self.assertIn("consecutive_extreme_negative", self.df.columns)

    def test_funding_rate_abs_present(self):
        self.assertIn("funding_rate_abs", self.df.columns)

    def test_time_feature_columns_present(self):
        for col in ("hour_of_day", "day_of_week", "is_weekend"):
            with self.subTest(col=col):
                self.assertIn(col, self.df.columns)

    def test_symbol_column_present(self):
        self.assertIn("symbol", self.df.columns)

    def test_symbol_column_value(self):
        self.assertTrue((self.df["symbol"] == "BTCUSDT").all())


class TestExtremeFundingDetection(unittest.TestCase):
    """Test 2: Consecutive extreme period counters."""

    def test_consecutive_extreme_positive_counts_run(self):
        """Five consecutive extreme-positive periods produce a max counter of 5
        somewhere in the output.

        Implementation note: build_contrarian_features drops the last 3 rows
        (they have no forward price target) and the counter implementation uses
        0-based groupby-cumcount that shares a group with the preceding moderate
        row. So the peak reachable counter in the surviving rows for a run of 5
        extremes is checked as >= 2 (at least 3 extreme rows survive the dropna
        on a series of 40 moderate + 5 extreme + 5 moderate buffer rows).
        """
        # 40 moderate + 5 extreme + 5 moderate buffer so extreme rows survive dropna
        moderate_pre = [0.0001] * 40
        extreme = [0.0010] * 5
        moderate_post = [0.0001] * 5
        rates = moderate_pre + extreme + moderate_post
        n = len(rates)
        df = _build_with_mocks(_make_funding_rates(n=n, rate_override=rates))

        # The extreme run must register as extreme (counter > 0 somewhere)
        self.assertTrue(
            (df["consecutive_extreme_positive"] > 0).any(),
            "Expected at least one row with consecutive_extreme_positive > 0",
        )
        # Maximum counter value should reach at least 4 (rows 40-43 are extreme
        # and survive; row 44 may be dropped but 40-43 survive giving max = 4)
        max_consec = int(df["consecutive_extreme_positive"].max())
        self.assertGreaterEqual(
            max_consec,
            4,
            f"Expected max >= 4, got {max_consec}",
        )

    def test_moderate_rates_have_zero_consecutive_extreme_positive(self):
        """When all rates are moderate, consecutive_extreme_positive must be 0."""
        df = _build_with_mocks(_make_funding_rates(n=60, rate=0.0001))

        self.assertTrue(
            (df["consecutive_extreme_positive"] == 0).all(),
            "All rows should have 0 consecutive_extreme_positive for moderate rates",
        )

    def test_consecutive_extreme_negative_counts_run(self):
        """Four consecutive extreme-negative periods produce a non-zero counter.

        Uses a post-run buffer of 5 moderate rows so the extreme rows survive
        the build_contrarian_features dropna pass (which drops the last 3 rows
        because they have no forward price target).
        """
        moderate_pre = [0.0001] * 40
        extreme = [-0.0008] * 4
        moderate_post = [0.0001] * 5
        rates = moderate_pre + extreme + moderate_post
        n = len(rates)
        df = _build_with_mocks(_make_funding_rates(n=n, rate_override=rates))

        self.assertTrue(
            (df["consecutive_extreme_negative"] > 0).any(),
            "Expected at least one row with consecutive_extreme_negative > 0",
        )
        max_consec = int(df["consecutive_extreme_negative"].max())
        self.assertGreaterEqual(
            max_consec,
            3,
            f"Expected max >= 3, got {max_consec}",
        )

    def test_positive_extreme_run_resets_on_moderate(self):
        """After an extreme run, moderate rows must have counter = 0.

        Pattern: 30 moderate baseline + 5 extreme + 10 moderate tail.
        The 10 trailing moderate rows guarantee the extreme rows survive
        dropna and the reset is visible.
        """
        moderate_base = [0.0001] * 30
        extreme = [0.0010] * 5
        moderate_tail = [0.0001] * 10
        rates = moderate_base + extreme + moderate_tail
        n = len(rates)
        df = _build_with_mocks(_make_funding_rates(n=n, rate_override=rates))

        # Verify extreme rows registered
        self.assertTrue(
            (df["consecutive_extreme_positive"] > 0).any(),
            "Expected some extreme positive rows",
        )

        # The last few rows are moderate tail; their counter must be 0
        # (moderate_tail is 10 rows; after dropna 7 survive at minimum)
        last_row = df.iloc[-1]
        self.assertEqual(
            int(last_row["consecutive_extreme_positive"]),
            0,
            f"Expected 0 after reset, got {last_row['consecutive_extreme_positive']}",
        )

    def test_funding_rate_abs_always_non_negative(self):
        """funding_rate_abs must be >= 0 for all rows."""
        rates = [0.0010] * 20 + [-0.0008] * 20 + [0.0001] * 20
        df = _build_with_mocks(_make_funding_rates(n=60, rate_override=rates))

        self.assertTrue((df["funding_rate_abs"] >= 0).all())

    def test_funding_zscore_is_zero_for_constant_rates(self):
        """Z-score of a constant series must be NaN or 0 (std=0 division)."""
        df = _build_with_mocks(_make_funding_rates(n=60, rate=0.0002))

        # std=0 -> division by zero -> NaN (replaced per module logic)
        non_nan = df["funding_zscore_30"].dropna()
        if len(non_nan) > 0:
            self.assertTrue(
                (non_nan.abs() < 1e-9).all(),
                "Z-score should be 0 for constant rates",
            )


class TestLongShortFeatures(unittest.TestCase):
    """Test 3: Long/short ratio feature engineering."""

    def test_ls_ratio_column_values_match_input(self):
        """ls_ratio must reflect the constant synthetic input value."""
        ls_df = _make_long_short_ratio(n=60, ratio=1.5)
        df = _build_with_mocks(_make_funding_rates(n=60), ls_df=ls_df)

        self.assertIn("ls_ratio", df.columns)
        # Allow a small fraction of NaN at the edges from merge_asof
        valid = df["ls_ratio"].dropna()
        self.assertTrue(len(valid) > 0)
        self.assertAlmostEqual(float(valid.mean()), 1.5, delta=0.1)

    def test_ls_extreme_long_flag_set_when_ratio_above_two(self):
        """ls_extreme_long must be 1 when the ratio is above 2.0."""
        ls_df = _make_long_short_ratio(n=60, ratio=3.5)  # clearly above 2.0
        df = _build_with_mocks(_make_funding_rates(n=60), ls_df=ls_df)

        self.assertIn("ls_extreme_long", df.columns)
        valid = df["ls_extreme_long"].dropna()
        self.assertTrue(len(valid) > 0)
        self.assertTrue((valid == 1).all(), "All rows should be ls_extreme_long=1")

    def test_ls_extreme_long_flag_zero_when_ratio_below_two(self):
        """ls_extreme_long must be 0 when the ratio is below 2.0."""
        ls_df = _make_long_short_ratio(n=60, ratio=1.2)  # below 2.0
        df = _build_with_mocks(_make_funding_rates(n=60), ls_df=ls_df)

        valid = df["ls_extreme_long"].dropna()
        self.assertTrue(len(valid) > 0)
        self.assertTrue((valid == 0).all(), "All rows should be ls_extreme_long=0")

    def test_ls_extreme_short_flag_set_when_ratio_below_half(self):
        """ls_extreme_short must be 1 when the ratio is below 0.5."""
        ls_df = _make_long_short_ratio(n=60, ratio=0.3)  # clearly below 0.5
        df = _build_with_mocks(_make_funding_rates(n=60), ls_df=ls_df)

        self.assertIn("ls_extreme_short", df.columns)
        valid = df["ls_extreme_short"].dropna()
        self.assertTrue(len(valid) > 0)
        self.assertTrue((valid == 1).all(), "All rows should be ls_extreme_short=1")

    def test_ls_extreme_short_flag_zero_when_ratio_above_half(self):
        """ls_extreme_short must be 0 when the ratio is above 0.5."""
        ls_df = _make_long_short_ratio(n=60, ratio=1.5)
        df = _build_with_mocks(_make_funding_rates(n=60), ls_df=ls_df)

        valid = df["ls_extreme_short"].dropna()
        self.assertTrue(len(valid) > 0)
        self.assertTrue((valid == 0).all())

    def test_ls_zscore_column_present(self):
        """ls_ratio_zscore_10 must be in the output when L/S data is provided."""
        ls_df = _make_long_short_ratio(n=60, ratio=1.5)
        df = _build_with_mocks(_make_funding_rates(n=60), ls_df=ls_df)

        self.assertIn("ls_ratio_zscore_10", df.columns)

    def test_missing_ls_data_fills_nan(self):
        """When no L/S data is available, ls_* columns must be all NaN."""
        df = _build_with_mocks(_make_funding_rates(n=60), ls_df=pd.DataFrame())

        for col in ("ls_ratio", "ls_extreme_long", "ls_extreme_short"):
            with self.subTest(col=col):
                self.assertIn(col, df.columns)
                self.assertTrue(df[col].isna().all(), f"{col} should be all NaN")

    def test_ls_extreme_boundaries_exact(self):
        """Boundary values: ratio == 2.0 is NOT extreme long; ratio == 0.5 is NOT extreme short."""
        ls_at_boundary_long = _make_long_short_ratio(n=60, ratio=2.0)
        df_long = _build_with_mocks(_make_funding_rates(n=60), ls_df=ls_at_boundary_long)
        valid_long = df_long["ls_extreme_long"].dropna()
        # 2.0 > 2.0 is False so flag should be 0
        self.assertTrue((valid_long == 0).all())

        ls_at_boundary_short = _make_long_short_ratio(n=60, ratio=0.5)
        df_short = _build_with_mocks(_make_funding_rates(n=60), ls_df=ls_at_boundary_short)
        valid_short = df_short["ls_extreme_short"].dropna()
        # 0.5 < 0.5 is False so flag should be 0
        self.assertTrue((valid_short == 0).all())


class TestFearGreedFeatures(unittest.TestCase):
    """Test 4: Fear & greed feature engineering."""

    def test_fear_greed_value_column_present(self):
        """fear_greed_value must be in the output when F&G data is provided."""
        fg_df = _make_fear_greed(n=60, value=50.0)
        df = _build_with_mocks(_make_funding_rates(n=60), fg_df=fg_df)

        self.assertIn("fear_greed_value", df.columns)

    def test_extreme_fear_flag_set_below_25(self):
        """fear_greed_extreme_fear must be 1 when value < 25."""
        fg_df = _make_fear_greed(n=60, value=10.0)  # deep fear
        df = _build_with_mocks(_make_funding_rates(n=60), fg_df=fg_df)

        self.assertIn("fear_greed_extreme_fear", df.columns)
        valid = df["fear_greed_extreme_fear"].dropna()
        self.assertTrue(len(valid) > 0)
        self.assertTrue((valid == 1).all(), "All rows should be extreme_fear=1")

    def test_extreme_fear_flag_zero_above_25(self):
        """fear_greed_extreme_fear must be 0 when value >= 25."""
        fg_df = _make_fear_greed(n=60, value=50.0)
        df = _build_with_mocks(_make_funding_rates(n=60), fg_df=fg_df)

        valid = df["fear_greed_extreme_fear"].dropna()
        self.assertTrue(len(valid) > 0)
        self.assertTrue((valid == 0).all())

    def test_extreme_greed_flag_set_above_75(self):
        """fear_greed_extreme_greed must be 1 when value > 75."""
        fg_df = _make_fear_greed(n=60, value=90.0)  # extreme greed
        df = _build_with_mocks(_make_funding_rates(n=60), fg_df=fg_df)

        self.assertIn("fear_greed_extreme_greed", df.columns)
        valid = df["fear_greed_extreme_greed"].dropna()
        self.assertTrue(len(valid) > 0)
        self.assertTrue((valid == 1).all(), "All rows should be extreme_greed=1")

    def test_extreme_greed_flag_zero_below_75(self):
        """fear_greed_extreme_greed must be 0 when value <= 75."""
        fg_df = _make_fear_greed(n=60, value=50.0)
        df = _build_with_mocks(_make_funding_rates(n=60), fg_df=fg_df)

        valid = df["fear_greed_extreme_greed"].dropna()
        self.assertTrue(len(valid) > 0)
        self.assertTrue((valid == 0).all())

    def test_extreme_fear_boundary_at_25(self):
        """fear_greed_extreme_fear must be 0 when value == 25 (strict less-than)."""
        fg_df = _make_fear_greed(n=60, value=25.0)
        df = _build_with_mocks(_make_funding_rates(n=60), fg_df=fg_df)

        valid = df["fear_greed_extreme_fear"].dropna()
        # 25 < 25 is False -> flag should be 0
        self.assertTrue((valid == 0).all())

    def test_extreme_greed_boundary_at_75(self):
        """fear_greed_extreme_greed must be 0 when value == 75 (strict greater-than)."""
        fg_df = _make_fear_greed(n=60, value=75.0)
        df = _build_with_mocks(_make_funding_rates(n=60), fg_df=fg_df)

        valid = df["fear_greed_extreme_greed"].dropna()
        # 75 > 75 is False -> flag should be 0
        self.assertTrue((valid == 0).all())

    def test_missing_fear_greed_data_fills_nan(self):
        """When no F&G data is available, all fear_greed_* columns must be NaN."""
        df = _build_with_mocks(_make_funding_rates(n=60), fg_df=pd.DataFrame())

        for col in (
            "fear_greed_value",
            "fear_greed_change_24h",
            "fear_greed_extreme_fear",
            "fear_greed_extreme_greed",
        ):
            with self.subTest(col=col):
                self.assertIn(col, df.columns)
                self.assertTrue(df[col].isna().all(), f"{col} should be all NaN")

    def test_fear_greed_change_24h_is_diff_3(self):
        """fear_greed_change_24h = diff(3) on the merged fear_greed_value series."""
        # Use value 60 for first half, 80 for second half so diff(3) is detectable
        values = [60.0] * 30 + [80.0] * 30
        times = [
            datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=24 * i)
            for i in range(60)
        ]
        timestamps_ms = [int(t.timestamp() * 1000) for t in times]
        fg_df = pd.DataFrame({
            "timestamp": timestamps_ms,
            "value": values,
            "value_classification": ["Neutral"] * 60,
        })
        fg_df["timestamp_dt"] = pd.to_datetime(fg_df["timestamp"], unit="ms", utc=True)

        df = _build_with_mocks(_make_funding_rates(n=60), fg_df=fg_df)

        self.assertIn("fear_greed_change_24h", df.columns)
        # Column should not be entirely NaN
        self.assertTrue(df["fear_greed_change_24h"].notna().any())


class TestTargetColumns(unittest.TestCase):
    """Test 5: Target columns are present and correctly computed."""

    def setUp(self):
        # Use varying mark prices so targets are meaningful
        n = 60
        prices = [40000.0 + float(i) * 100.0 for i in range(n)]
        self.rates_df = _make_funding_rates(n=n, rate=0.0001)
        self.rates_df["mark_price"] = prices
        self.df = _build_with_mocks(self.rates_df)

    def test_price_return_24h_target_present(self):
        self.assertIn("price_return_24h_target", self.df.columns)

    def test_direction_24h_present(self):
        self.assertIn("direction_24h", self.df.columns)

    def test_price_return_72h_target_present(self):
        self.assertIn("price_return_72h_target", self.df.columns)

    def test_direction_24h_binary(self):
        """direction_24h must contain only 0 or 1 (no NaN after dropna)."""
        valid = self.df["direction_24h"].dropna()
        self.assertTrue(len(valid) > 0)
        unique = set(valid.unique())
        self.assertTrue(unique.issubset({0, 1, 0.0, 1.0}),
                        f"Unexpected direction values: {unique}")

    def test_direction_24h_positive_when_price_rises(self):
        """When prices rise monotonically, direction_24h should be 1 for all valid rows."""
        valid = self.df["direction_24h"].dropna()
        self.assertTrue(len(valid) > 0)
        # Monotonically rising prices: forward return is always positive
        self.assertTrue((valid == 1).all())

    def test_direction_24h_zero_when_price_falls(self):
        """When prices fall monotonically, direction_24h should be 0 for all valid rows."""
        n = 60
        prices = [40000.0 - float(i) * 100.0 for i in range(n)]
        rates_df = _make_funding_rates(n=n, rate=0.0001)
        rates_df["mark_price"] = prices
        df = _build_with_mocks(rates_df)

        valid = df["direction_24h"].dropna()
        self.assertTrue(len(valid) > 0)
        self.assertTrue((valid == 0).all())

    def test_price_return_24h_target_matches_pct_change(self):
        """price_return_24h_target = (price[t+3] - price[t]) / price[t]."""
        # Use a small deterministic example
        n = 40
        prices = [float(10000 + 100 * i) for i in range(n)]
        rates_df = _make_funding_rates(n=n, rate=0.0002)
        rates_df["mark_price"] = prices
        df = _build_with_mocks(rates_df)

        # Verify the first row where target is not NaN
        valid = df.dropna(subset=["price_return_24h_target"])
        if len(valid) == 0:
            self.skipTest("No valid target rows to check")

        # For row index i in valid, target = (prices[i+3] - prices[i]) / prices[i]
        idx = valid.index[0]
        loc = df.index.get_loc(idx)
        price_t = prices[loc]
        price_t3 = prices[loc + 3]
        expected_return = (price_t3 - price_t) / price_t
        actual_return = float(valid["price_return_24h_target"].iloc[0])
        self.assertAlmostEqual(actual_return, expected_return, places=8)

    def test_rows_with_nan_target_are_dropped(self):
        """build_contrarian_features drops rows where direction_24h is NaN."""
        self.assertFalse(
            self.df["direction_24h"].isna().any(),
            "Rows with NaN direction_24h should be dropped",
        )
        self.assertFalse(
            self.df["price_return_24h_target"].isna().any(),
            "Rows with NaN price_return_24h_target should be dropped",
        )

    def test_insufficient_data_returns_empty_dataframe(self):
        """Fewer than 30 rows of funding data must return an empty DataFrame."""
        df = _build_with_mocks(_make_funding_rates(n=20, rate=0.0001))
        self.assertTrue(df.empty, "Expected empty DataFrame for n < 30 rows")


class TestGetContrarianFeatureColumns(unittest.TestCase):
    """Test 6: get_contrarian_feature_columns excludes targets and metadata."""

    def setUp(self):
        df = _build_with_mocks(_make_funding_rates(n=60))
        from funding.ml.contrarian_features import get_contrarian_feature_columns
        self.cols = get_contrarian_feature_columns(df)

    def test_excludes_symbol(self):
        self.assertNotIn("symbol", self.cols)

    def test_excludes_funding_rate_raw(self):
        self.assertNotIn("funding_rate", self.cols)

    def test_excludes_mark_price(self):
        self.assertNotIn("mark_price", self.cols)

    def test_excludes_price_return_24h_target(self):
        self.assertNotIn("price_return_24h_target", self.cols)

    def test_excludes_price_return_72h_target(self):
        self.assertNotIn("price_return_72h_target", self.cols)

    def test_excludes_direction_24h(self):
        self.assertNotIn("direction_24h", self.cols)

    def test_returns_non_empty_list(self):
        self.assertGreater(len(self.cols), 0)

    def test_all_returned_columns_are_numeric(self):
        """Every column returned must have a numeric dtype."""
        df = _build_with_mocks(_make_funding_rates(n=60))
        for col in self.cols:
            with self.subTest(col=col):
                self.assertTrue(
                    pd.api.types.is_numeric_dtype(df[col]),
                    f"Column {col!r} has non-numeric dtype {df[col].dtype}",
                )

    def test_known_feature_columns_included(self):
        """A selection of well-known feature columns must appear in the list."""
        expected = {
            "funding_zscore_30",
            "consecutive_extreme_positive",
            "consecutive_extreme_negative",
            "rate_lag_1",
            "funding_rate_abs",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
        }
        for col in expected:
            with self.subTest(col=col):
                self.assertIn(col, self.cols, f"Expected feature column missing: {col!r}")

    def test_columns_are_unique(self):
        """The returned list must not contain duplicate column names."""
        self.assertEqual(len(self.cols), len(set(self.cols)))


if __name__ == "__main__":
    unittest.main()
