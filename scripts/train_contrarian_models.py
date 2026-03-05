#!/usr/bin/env python3
"""
Train contrarian and regime ML models.

Usage:
    python scripts/train_contrarian_models.py --all              # Train everything
    python scripts/train_contrarian_models.py --contrarian       # Just contrarian (XGB + TFT)
    python scripts/train_contrarian_models.py --regime           # Just regime (HMM + Transformer)
    python scripts/train_contrarian_models.py --contrarian-xgb   # Just XGBoost
    python scripts/train_contrarian_models.py --tft              # Just TFT
    python scripts/train_contrarian_models.py --hmm              # Just HMM
    python scripts/train_contrarian_models.py --symbols ETHUSDT,BTCUSDT  # Specify symbols
    python scripts/train_contrarian_models.py --n-trials 50      # More Optuna trials
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is importable when script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

# Load .env from project root before importing config or any module that reads env vars.
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass  # dotenv optional; env vars may already be set in shell

# ---------------------------------------------------------------------------
# Logging setup (must happen before module-level loggers are created)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_contrarian_models")

# ---------------------------------------------------------------------------
# Fallback default symbols (used when historical data volume detection fails)
# ---------------------------------------------------------------------------

_FALLBACK_SYMBOLS: List[str] = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
]

_FUNDING_HISTORY_DIR = _PROJECT_ROOT / "data" / "funding_history"
_MODEL_DIR = _PROJECT_ROOT / "data" / "funding_models"
_MIN_ROWS = 500


# ---------------------------------------------------------------------------
# Symbol discovery
# ---------------------------------------------------------------------------

def _discover_symbols_by_volume(n: int = 10) -> List[str]:
    """Return the top *n* symbols ranked by number of funding-rate rows on disk.

    Falls back to _FALLBACK_SYMBOLS if the funding_history directory is absent
    or no CSV files are found.

    Args:
        n: Maximum number of symbols to return.

    Returns:
        List of symbol strings (upper-case), sorted by row count descending.
    """
    rates_dir = _FUNDING_HISTORY_DIR / "funding_rates"
    if not rates_dir.exists():
        logger.warning(
            "Funding rates directory not found at %s; using fallback symbols.", rates_dir
        )
        return list(_FALLBACK_SYMBOLS)

    candidates: List[tuple[int, str]] = []
    for csv_path in sorted(rates_dir.glob("*.csv")):
        symbol = csv_path.stem.upper()
        try:
            # Count lines minus header rather than loading the full frame.
            with open(csv_path, "r", encoding="utf-8") as fh:
                row_count = sum(1 for _ in fh) - 1  # subtract header
            if row_count > 0:
                candidates.append((row_count, symbol))
        except OSError as exc:
            logger.debug("Could not read %s: %s", csv_path, exc)

    if not candidates:
        logger.warning("No funding rate CSVs found; using fallback symbols.")
        return list(_FALLBACK_SYMBOLS)

    candidates.sort(key=lambda x: x[0], reverse=True)
    symbols = [sym for _, sym in candidates[:n]]
    logger.info(
        "Discovered top-%d symbols by row count: %s",
        n,
        ", ".join(f"{sym}({cnt})" for cnt, sym in candidates[:n]),
    )
    return symbols


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def _build_contrarian_features(symbols: List[str]):
    """Import and call build_contrarian_features_all.

    Deferred import keeps the CLI help fast even when optional deps are absent.
    """
    from funding.ml.contrarian_features import build_contrarian_features_all  # noqa: PLC0415

    logger.info("Building contrarian features for symbols: %s", symbols)
    df = build_contrarian_features_all(symbols)
    return df


def _build_regime_features(symbols: List[str]):
    """Import and call build_regime_features + assign_regime_labels.

    Deferred import keeps the CLI help fast even when optional deps are absent.
    """
    from funding.ml.regime_features import (  # noqa: PLC0415
        assign_regime_labels,
        build_regime_features,
    )

    logger.info("Building regime features for symbols: %s", symbols)
    df = build_regime_features(symbols)
    if not df.empty and "regime_label" not in df.columns:
        df["regime_label"] = assign_regime_labels(df)
    return df


# ---------------------------------------------------------------------------
# Individual model trainers
# ---------------------------------------------------------------------------

def train_contrarian_xgb(df, n_trials: int, model_dir: Path) -> dict:
    """Train only the ContrarianXGBoost model.

    Args:
        df:        Feature DataFrame from build_contrarian_features_all.
        n_trials:  Optuna trials for hyperparameter search.
        model_dir: Root model output directory.

    Returns:
        Metrics dict returned by ContrarianXGBoost.train().
    """
    from funding.ml.contrarian_xgb import ContrarianXGBoost  # noqa: PLC0415

    output_dir = model_dir / "contrarian_xgb"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Training ContrarianXGBoost — rows: %d, n_trials: %d, output: %s",
        len(df), n_trials, output_dir,
    )
    model = ContrarianXGBoost(model_dir=str(output_dir))
    metrics = model.train(df, tune=True, n_trials=n_trials, test_months=2)
    logger.info("ContrarianXGBoost training complete — metrics: %s", metrics)
    return metrics


def train_tft(df, tft_epochs: int, model_dir: Path) -> dict:
    """Train only the TFTPredictor model.

    Args:
        df:         Feature DataFrame from build_contrarian_features_all.
        tft_epochs: Maximum training epochs.
        model_dir:  Root model output directory.

    Returns:
        Metrics dict returned by TFTPredictor.train(), or empty dict on failure.
    """
    from funding.ml.tft_predictor import TFTPredictor, _DEEP_LEARNING_AVAILABLE  # noqa: PLC0415

    if not _DEEP_LEARNING_AVAILABLE:
        logger.warning(
            "TFT dependencies (torch / pytorch_lightning / pytorch_forecasting) "
            "are not installed. Skipping TFT training.\n"
            "Install with: pip install torch pytorch-lightning pytorch-forecasting"
        )
        return {}

    output_dir = model_dir / "tft"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Training TFTPredictor — rows: %d, max_epochs: %d, output: %s",
        len(df), tft_epochs, output_dir,
    )
    model = TFTPredictor(model_dir=str(output_dir))
    metrics = model.train(df, max_epochs=tft_epochs)
    logger.info("TFTPredictor training complete — metrics: %s", metrics)
    return metrics


def train_hmm(df, model_dir: Path) -> dict:
    """Train only the RegimeHMM model.

    Args:
        df:        Feature DataFrame from build_regime_features (with regime_label).
        model_dir: Root model output directory.

    Returns:
        Metrics dict returned by RegimeHMM.fit().
    """
    import config  # noqa: PLC0415
    from funding.ml.regime_hmm import RegimeHMM  # noqa: PLC0415
    from funding.ml.regime_features import get_regime_feature_columns  # noqa: PLC0415

    output_dir = model_dir / "regime"
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = get_regime_feature_columns(df)
    clean = df[cols + ["regime_label"]].dropna()

    logger.info(
        "Training RegimeHMM — rows: %d (clean), features: %d, n_states: %d, output: %s",
        len(clean), len(cols), config.REGIME_N_STATES, output_dir,
    )
    model = RegimeHMM(n_states=config.REGIME_N_STATES, model_dir=str(output_dir))
    metrics = model.fit(clean, feature_columns=cols)
    logger.info("RegimeHMM training complete — metrics: %s", metrics)
    return metrics


def train_regime_transformer(df, model_dir: Path) -> dict:
    """Train only the RegimeTransformer model.

    Args:
        df:        Feature DataFrame from build_regime_features (with regime_label).
        model_dir: Root model output directory.

    Returns:
        Metrics dict returned by RegimeTransformer.fit().
    """
    import config  # noqa: PLC0415
    from funding.ml.regime_transformer import RegimeTransformer  # noqa: PLC0415
    from funding.ml.regime_features import get_regime_feature_columns  # noqa: PLC0415

    output_dir = model_dir / "regime"
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = get_regime_feature_columns(df)
    clean = df[cols + ["regime_label"]].dropna()

    logger.info(
        "Training RegimeTransformer — rows: %d (clean), features: %d, "
        "n_states: %d, output: %s",
        len(clean), len(cols), config.REGIME_N_STATES, output_dir,
    )
    model = RegimeTransformer(n_states=config.REGIME_N_STATES, model_dir=str(output_dir))
    metrics = model.fit(clean, feature_columns=cols)
    logger.info("RegimeTransformer training complete — metrics: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Suite runners (ModelSelector / RegimeSelector)
# ---------------------------------------------------------------------------

def run_contrarian_suite(symbols: List[str], n_trials: int, tft_epochs: int) -> dict:
    """Run full ModelSelector comparison (XGB + TFT) and return the comparison dict.

    Args:
        symbols:    Symbols to build features for.
        n_trials:   Optuna trials passed to ContrarianXGBoost.
        tft_epochs: Max epochs passed to TFTPredictor.

    Returns:
        ModelSelector comparison dict (see ModelSelector.compare() docstring).

    Raises:
        SystemExit: If the feature DataFrame has fewer than _MIN_ROWS rows.
    """
    from funding.ml.model_selector import ModelSelector  # noqa: PLC0415

    df = _build_contrarian_features(symbols)
    _assert_sufficient_data(df, "contrarian", symbols)

    logger.info(
        "ModelSelector.compare() — %d rows, n_trials=%d, tft_epochs=%d",
        len(df), n_trials, tft_epochs,
    )
    selector = ModelSelector(model_dir=str(_MODEL_DIR))
    comparison = selector.compare(df, n_trials=n_trials, tft_epochs=tft_epochs)
    return comparison


def run_regime_suite(symbols: List[str]) -> dict:
    """Run full RegimeSelector comparison (HMM + Transformer) and return the comparison dict.

    Args:
        symbols: Symbols to build features for.

    Returns:
        RegimeSelector comparison dict (see RegimeSelector.compare() docstring).

    Raises:
        SystemExit: If the feature DataFrame has fewer than _MIN_ROWS rows.
    """
    from funding.ml.regime_selector import RegimeSelector  # noqa: PLC0415
    from funding.ml.regime_features import assign_regime_labels  # noqa: PLC0415

    df = _build_regime_features(symbols)
    _assert_sufficient_data(df, "regime", symbols)

    if "regime_label" not in df.columns:
        df["regime_label"] = assign_regime_labels(df)

    logger.info("RegimeSelector.compare() — %d rows", len(df))
    selector = RegimeSelector(model_dir=str(_MODEL_DIR / "regime"))
    comparison = selector.compare(df)
    best = selector.select_best()
    selector.save_comparison()
    logger.info("RegimeSelector selected: %s", best)
    return {"comparison": comparison, "selected": best}


# ---------------------------------------------------------------------------
# Data validation helper
# ---------------------------------------------------------------------------

def _assert_sufficient_data(df, task: str, symbols: List[str]) -> None:
    """Exit with a clear message when *df* is too small to train on.

    Args:
        df:      Feature DataFrame to check.
        task:    Human-readable task name (e.g. "contrarian", "regime").
        symbols: Symbol list used to produce *df*.

    Raises:
        SystemExit(1): When len(df) < _MIN_ROWS or df is empty.
    """
    if df is None or df.empty:
        logger.error(
            "No data returned for %s features (symbols: %s). "
            "Ensure that data/funding_history/ contains funding_rates and klines "
            "CSV files for the requested symbols.",
            task,
            ", ".join(symbols),
        )
        sys.exit(1)

    if len(df) < _MIN_ROWS:
        logger.error(
            "Insufficient data for %s training: got %d rows, need >= %d. "
            "Symbols: %s. "
            "Fetch more historical data before training.",
            task,
            len(df),
            _MIN_ROWS,
            ", ".join(symbols),
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------

def _print_contrarian_summary(comparison: dict) -> None:
    """Print a human-readable contrarian model comparison summary."""
    print()
    print("=" * 60)
    print("  CONTRARIAN MODEL COMPARISON RESULTS")
    print("=" * 60)

    for model_name in ("xgboost", "tft"):
        entry = comparison.get(model_name, {})
        trained = entry.get("trained", False)
        if not trained:
            err = entry.get("error", "unknown error")
            print(f"  {model_name.upper():10s}  SKIPPED  ({err})")
            continue
        sharpe = entry.get("sharpe_simulated", float("nan"))
        auc = entry.get("auc", float("nan"))
        acc = entry.get("direction_accuracy", float("nan"))
        pf = entry.get("profit_factor", float("nan"))
        print(
            f"  {model_name.upper():10s}  "
            f"Sharpe={sharpe:+.3f}  AUC={auc:.3f}  "
            f"Acc={acc:.3f}  ProfitFactor={pf:.3f}"
        )

    selected = comparison.get("selected", "N/A")
    print()
    print(f"  --> Best model selected: {selected.upper()}")
    print("=" * 60)
    print(f"  Artifacts saved to: {_MODEL_DIR}")
    print("=" * 60)
    print()


def _print_regime_summary(result: dict) -> None:
    """Print a human-readable regime model comparison summary."""
    comparison = result.get("comparison", {})
    selected = result.get("selected", "N/A")

    print()
    print("=" * 60)
    print("  REGIME MODEL COMPARISON RESULTS")
    print("=" * 60)

    for model_name in ("hmm", "transformer"):
        entry = comparison.get(model_name, {})
        if "error" in entry and not entry.get("accuracy", 0):
            err = entry.get("error", "unknown error")
            print(f"  {model_name.upper():12s}  FAILED  ({err})")
            continue
        acc = entry.get("accuracy", float("nan"))
        stab = entry.get("stability", float("nan"))
        crisis = entry.get("crisis_corr", float("nan"))
        print(
            f"  {model_name.upper():12s}  "
            f"Accuracy={acc:.4f}  Stability={stab:.2f}  "
            f"CrisisCorr={crisis:.3f}"
        )

    print()
    print(f"  --> Best regime model selected: {selected.upper()}")
    print("=" * 60)
    print(f"  Artifacts saved to: {_MODEL_DIR / 'regime'}")
    print("=" * 60)
    print()


def _print_single_model_summary(model_name: str, metrics: dict) -> None:
    """Print a summary for a single-model training run."""
    if not metrics:
        print(f"\n  {model_name}: no metrics returned (model may have been skipped).\n")
        return

    print()
    print("=" * 60)
    print(f"  {model_name.upper()} TRAINING RESULTS")
    print("=" * 60)
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key:35s} {value:.6g}")
        else:
            print(f"  {key:35s} {value}")
    print("=" * 60)
    print(f"  Artifacts saved to: {_MODEL_DIR}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train contrarian (XGBoost + TFT) and regime (HMM + Transformer) ML models "
            "for the Binance funding-rate arbitrage strategy."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # What to train
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Train all models: contrarian (XGB + TFT) and regime (HMM + Transformer).",
    )
    group.add_argument(
        "--contrarian",
        action="store_true",
        help="Train the contrarian suite (XGBoost + TFT) via ModelSelector.",
    )
    group.add_argument(
        "--regime",
        action="store_true",
        help="Train the regime suite (HMM + Transformer) via RegimeSelector.",
    )
    group.add_argument(
        "--contrarian-xgb",
        action="store_true",
        dest="contrarian_xgb",
        help="Train only ContrarianXGBoost (no ModelSelector comparison).",
    )
    group.add_argument(
        "--tft",
        action="store_true",
        help="Train only TFTPredictor (no ModelSelector comparison).",
    )
    group.add_argument(
        "--hmm",
        action="store_true",
        help="Train only RegimeHMM (no RegimeSelector comparison).",
    )
    group.add_argument(
        "--regime-transformer",
        action="store_true",
        dest="regime_transformer",
        help="Train only RegimeTransformer (no RegimeSelector comparison).",
    )

    # Symbols
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        metavar="SYM1,SYM2,...",
        help=(
            "Comma-separated list of symbols to use (e.g. ETHUSDT,BTCUSDT). "
            "Defaults to the top 10 symbols by historical data volume, "
            "falling back to BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT."
        ),
    )

    # Hyperparameters
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        dest="n_trials",
        help="Number of Optuna trials for ContrarianXGBoost hyperparameter search (default: 30).",
    )
    parser.add_argument(
        "--tft-epochs",
        type=int,
        default=50,
        dest="tft_epochs",
        help="Maximum training epochs for TFTPredictor (default: 50).",
    )

    # Verbosity
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse arguments, build features, train models, print summary.

    Returns:
        0 on success, 1 on error (data insufficient, import failure, etc.).
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("DEBUG logging enabled.")

    # ------------------------------------------------------------------ #
    # 1. Resolve symbol list                                               #
    # ------------------------------------------------------------------ #
    if args.symbols:
        symbols: List[str] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if not symbols:
            logger.error("--symbols produced an empty list. Check the format: SYM1,SYM2,...")
            return 1
        logger.info("Using user-supplied symbols: %s", symbols)
    else:
        symbols = _discover_symbols_by_volume(n=10)
        logger.info("Auto-discovered symbols: %s", symbols)

    # ------------------------------------------------------------------ #
    # 2. Ensure model output directory exists                             #
    # ------------------------------------------------------------------ #
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 3. Dispatch training based on CLI flags                             #
    # ------------------------------------------------------------------ #
    exit_code = 0

    try:
        if args.all:
            # --- Contrarian suite ---
            logger.info("--- Contrarian suite (XGB + TFT) ---")
            contrarian_comparison = run_contrarian_suite(
                symbols=symbols,
                n_trials=args.n_trials,
                tft_epochs=args.tft_epochs,
            )
            _print_contrarian_summary(contrarian_comparison)

            # --- Regime suite ---
            logger.info("--- Regime suite (HMM + Transformer) ---")
            # Regime features require at least BTC+ETH for cross-asset features.
            regime_symbols = _ensure_btc_eth(symbols)
            regime_result = run_regime_suite(symbols=regime_symbols)
            _print_regime_summary(regime_result)

        elif args.contrarian:
            contrarian_comparison = run_contrarian_suite(
                symbols=symbols,
                n_trials=args.n_trials,
                tft_epochs=args.tft_epochs,
            )
            _print_contrarian_summary(contrarian_comparison)

        elif args.regime:
            regime_symbols = _ensure_btc_eth(symbols)
            regime_result = run_regime_suite(symbols=regime_symbols)
            _print_regime_summary(regime_result)

        elif args.contrarian_xgb:
            df = _build_contrarian_features(symbols)
            _assert_sufficient_data(df, "contrarian-xgb", symbols)
            metrics = train_contrarian_xgb(df, n_trials=args.n_trials, model_dir=_MODEL_DIR)
            _print_single_model_summary("ContrarianXGBoost", metrics)

        elif args.tft:
            df = _build_contrarian_features(symbols)
            _assert_sufficient_data(df, "tft", symbols)
            metrics = train_tft(df, tft_epochs=args.tft_epochs, model_dir=_MODEL_DIR)
            _print_single_model_summary("TFTPredictor", metrics)

        elif args.hmm:
            regime_symbols = _ensure_btc_eth(symbols)
            df = _build_regime_features(regime_symbols)
            _assert_sufficient_data(df, "hmm", regime_symbols)
            metrics = train_hmm(df, model_dir=_MODEL_DIR)
            _print_single_model_summary("RegimeHMM", metrics)

        elif args.regime_transformer:
            regime_symbols = _ensure_btc_eth(symbols)
            df = _build_regime_features(regime_symbols)
            _assert_sufficient_data(df, "regime-transformer", regime_symbols)
            metrics = train_regime_transformer(df, model_dir=_MODEL_DIR)
            _print_single_model_summary("RegimeTransformer", metrics)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        return 1
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during training: %s", exc)
        return 1

    logger.info("All requested models trained successfully.")
    return exit_code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_btc_eth(symbols: List[str]) -> List[str]:
    """Return *symbols* with BTCUSDT and ETHUSDT prepended if not present.

    The regime feature builder uses BTC+ETH for cross-asset correlation.
    If they are missing from the user-supplied list, insert them silently.

    Args:
        symbols: Input symbol list.

    Returns:
        Augmented symbol list with BTCUSDT at position 0 and ETHUSDT at 1
        (existing occurrences are de-duplicated and moved to the front).
    """
    augmented = list(symbols)
    for anchor in reversed(["ETHUSDT", "BTCUSDT"]):
        if anchor not in augmented:
            logger.debug(
                "%s not in symbol list; prepending for cross-asset regime features.", anchor
            )
            augmented.insert(0, anchor)
    return augmented


if __name__ == "__main__":
    raise SystemExit(main())
