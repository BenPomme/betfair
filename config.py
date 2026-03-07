"""
Configuration loaded from environment. All thresholds and secrets come from env;
no magic numbers in the rest of the codebase.
"""
import os
from decimal import Decimal
from pathlib import Path

# Load .env if present (e.g. for local dev)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Trading mode (single source of truth) ---
PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "true").lower() == "true"

# --- Commission (Spain) ---
MBR: Decimal = Decimal(os.getenv("MBR", "0.05"))
DISCOUNT_RATE: Decimal = Decimal(os.getenv("DISCOUNT_RATE", "0.00"))

# --- Thresholds ---
MIN_NET_PROFIT_EUR: Decimal = Decimal(os.getenv("MIN_NET_PROFIT_EUR", "0.10"))
MIN_LIQUIDITY_EUR: Decimal = Decimal(os.getenv("MIN_LIQUIDITY_EUR", "50.00"))
MAX_STAKE_EUR: Decimal = Decimal(os.getenv("MAX_STAKE_EUR", "100.00"))
DAILY_LOSS_LIMIT_EUR: Decimal = Decimal(os.getenv("DAILY_LOSS_LIMIT_EUR", "50.00"))
PRE_FILTER_THRESHOLD: Decimal = Decimal(os.getenv("PRE_FILTER_THRESHOLD", "0.97"))
STALE_PRICE_SECONDS: int = int(os.getenv("STALE_PRICE_SECONDS", "2"))
STALE_PRICE_SECONDS_PREMATCH: int = int(os.getenv("STALE_PRICE_SECONDS_PREMATCH", "4"))
STALE_PRICE_SECONDS_INPLAY: int = int(os.getenv("STALE_PRICE_SECONDS_INPLAY", "2"))

# --- Infrastructure ---
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/betfair_arb")
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
DISCORD_ENABLED: bool = os.getenv("DISCORD_ENABLED", "false").lower() == "true"
DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
DISCORD_WEBHOOK_USERNAME: str = os.getenv("DISCORD_WEBHOOK_USERNAME", "Strategy Command Center")
DISCORD_WEBHOOK_AVATAR_URL: str = os.getenv("DISCORD_WEBHOOK_AVATAR_URL", "")
DISCORD_DIGEST_ENABLED: bool = os.getenv("DISCORD_DIGEST_ENABLED", "true").lower() == "true"
DISCORD_DIGEST_INTERVAL_MINUTES: int = int(os.getenv("DISCORD_DIGEST_INTERVAL_MINUTES", "30"))
DISCORD_DAILY_DIGEST_ENABLED: bool = os.getenv("DISCORD_DAILY_DIGEST_ENABLED", "true").lower() == "true"
DISCORD_DAILY_DIGEST_UTC_HOUR: int = int(os.getenv("DISCORD_DAILY_DIGEST_UTC_HOUR", "18"))
DISCORD_DIGEST_TIMEZONE: str = os.getenv("DISCORD_DIGEST_TIMEZONE", "Europe/Madrid")
DISCORD_DAILY_DIGEST_LOCAL_HOURS: str = os.getenv("DISCORD_DAILY_DIGEST_LOCAL_HOURS", "9,21")
DISCORD_NOTIFY_CRITICAL_ONLY: bool = os.getenv("DISCORD_NOTIFY_CRITICAL_ONLY", "false").lower() == "true"
DISCORD_NOTIFY_PORTFOLIOS: str = os.getenv(
    "DISCORD_NOTIFY_PORTFOLIOS", "betfair_core,hedge_validation,cascade_alpha,mev_scout_sol,contrarian_legacy,command_center"
)
DISCORD_NOTIFY_EVENT_TYPES: str = os.getenv("DISCORD_NOTIFY_EVENT_TYPES", "trade_closed")
DISCORD_MIN_TRADE_ALERT_PNL_USD: Decimal = Decimal(
    os.getenv("DISCORD_MIN_TRADE_ALERT_PNL_USD", "5")
)
DISCORD_MIN_TRADE_ALERT_PNL_EUR: Decimal = Decimal(
    os.getenv("DISCORD_MIN_TRADE_ALERT_PNL_EUR", "2")
)
DISCORD_MODEL_ALERT_MIN_AUC_DELTA: Decimal = Decimal(
    os.getenv("DISCORD_MODEL_ALERT_MIN_AUC_DELTA", "0.02")
)
DISCORD_MODEL_ALERT_MIN_BRIER_LIFT_DELTA: Decimal = Decimal(
    os.getenv("DISCORD_MODEL_ALERT_MIN_BRIER_LIFT_DELTA", "0.01")
)
DISCORD_BOT_ENABLED: bool = os.getenv("DISCORD_BOT_ENABLED", "false").lower() == "true"
DISCORD_BOT_TOKEN: str = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_BOT_PREFIX: str = os.getenv("DISCORD_BOT_PREFIX", "!")
DISCORD_BOT_ALLOWED_USER_IDS: str = os.getenv("DISCORD_BOT_ALLOWED_USER_IDS", "")
DISCORD_BOT_ALLOWED_GUILD_IDS: str = os.getenv("DISCORD_BOT_ALLOWED_GUILD_IDS", "")
DISCORD_BOT_ALLOWED_CHANNEL_IDS: str = os.getenv("DISCORD_BOT_ALLOWED_CHANNEL_IDS", "")
DISCORD_BOT_STATUS_PORTFOLIOS: str = os.getenv(
    "DISCORD_BOT_STATUS_PORTFOLIOS",
    "betfair_core,hedge_validation,hedge_research,cascade_alpha,contrarian_legacy,mev_scout_sol",
)
NOTIFICATIONS_ENABLED: bool = os.getenv("NOTIFICATIONS_ENABLED", "true").lower() == "true"
NOTIFY_DEDUPE_WINDOW_SECONDS: int = int(os.getenv("NOTIFY_DEDUPE_WINDOW_SECONDS", "600"))

# --- Betfair ---
BF_USERNAME: str = os.getenv("BF_USERNAME", "")
BF_PASSWORD: str = os.getenv("BF_PASSWORD", "")
BF_APP_KEY: str = os.getenv("BF_APP_KEY", "")
BF_CERTS_PATH: str = os.getenv("BF_CERTS_PATH", str(Path.cwd() / "certs"))
# Locale for API identity/exchange: "spain" for betfair.es, None/default for .com
BF_LOCALE: str = os.getenv("BF_LOCALE", "spain")

# --- Paper trading ---
INITIAL_BALANCE_EUR: Decimal = Decimal(os.getenv("INITIAL_BALANCE_EUR", "1000.00"))
PAPER_STATE_PATH: str = os.getenv("PAPER_STATE_PATH", "data/state/paper_executor_state.json")
PAPER_TRADES_LOG_PATH: str = os.getenv("PAPER_TRADES_LOG_PATH", "data/state/paper_trades.jsonl")

# --- Stake sizing ---
STAKE_FRACTION: Decimal = Decimal(os.getenv("STAKE_FRACTION", "0.10"))  # 10% of balance per arb

# --- Market scanning ---
SCAN_SPORTS: str = os.getenv("SCAN_SPORTS", "all")  # "all" = auto-discover, or comma-separated IDs
SCAN_COUNTRIES: str = os.getenv("SCAN_COUNTRIES", "")  # empty = no country filter (global)
SCAN_MAX_MARKETS: int = int(os.getenv("SCAN_MAX_MARKETS", "500"))
SCAN_MARKET_TYPES: str = os.getenv(
    "SCAN_MARKET_TYPES",
    "MATCH_ODDS,DRAW_NO_BET,OVER_UNDER_25,BOTH_TEAMS_TO_SCORE",
)
SCAN_MAX_HOURS_AHEAD: int = int(os.getenv("SCAN_MAX_HOURS_AHEAD", "12"))
SCAN_INCLUDE_IN_PLAY: bool = os.getenv("SCAN_INCLUDE_IN_PLAY", "true").lower() == "true"
SCAN_MIN_INPLAY_MARKETS: int = int(os.getenv("SCAN_MIN_INPLAY_MARKETS", "20"))
SCAN_INTERVAL_SECONDS: float = float(os.getenv("SCAN_INTERVAL_SECONDS", "2.0"))
MARKET_REFRESH_INTERVAL_SECONDS: int = int(os.getenv("MARKET_REFRESH_INTERVAL_SECONDS", "180"))
MARKET_NO_MOVEMENT_SECONDS: int = int(os.getenv("MARKET_NO_MOVEMENT_SECONDS", "900"))
MARKET_MISSING_RETIRE_CYCLES: int = int(os.getenv("MARKET_MISSING_RETIRE_CYCLES", "120"))
PRICE_POLL_INTERVAL_SECONDS: float = float(os.getenv("PRICE_POLL_INTERVAL_SECONDS", "1.0"))
POLLER_BATCH_SIZE: int = int(os.getenv("POLLER_BATCH_SIZE", "20"))
POLLER_CONCURRENT_BATCHES: int = int(os.getenv("POLLER_CONCURRENT_BATCHES", "8"))
POLLER_REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("POLLER_REQUEST_TIMEOUT_SECONDS", "12.0"))
BETFAIR_FEED_STALE_SECONDS: int = int(os.getenv("BETFAIR_FEED_STALE_SECONDS", "12"))
BETFAIR_ZERO_SNAPSHOT_CYCLE_LIMIT: int = int(os.getenv("BETFAIR_ZERO_SNAPSHOT_CYCLE_LIMIT", "5"))
BETFAIR_ZERO_BOOK_CYCLE_LIMIT: int = int(os.getenv("BETFAIR_ZERO_BOOK_CYCLE_LIMIT", "5"))
BETFAIR_TIMEOUT_CYCLE_LIMIT: int = int(os.getenv("BETFAIR_TIMEOUT_CYCLE_LIMIT", "3"))
BETFAIR_RETIRE_HEAVY_THRESHOLD: float = float(os.getenv("BETFAIR_RETIRE_HEAVY_THRESHOLD", "0.30"))
BETFAIR_STALE_HEAVY_THRESHOLD: float = float(os.getenv("BETFAIR_STALE_HEAVY_THRESHOLD", "0.40"))
BETFAIR_FORCE_REFRESH_ON_DEGRADE: bool = os.getenv("BETFAIR_FORCE_REFRESH_ON_DEGRADE", "true").lower() == "true"
BETFAIR_RELOGIN_ON_DEGRADE: bool = os.getenv("BETFAIR_RELOGIN_ON_DEGRADE", "true").lower() == "true"
BETFAIR_MAX_RECOVERY_ATTEMPTS: int = int(os.getenv("BETFAIR_MAX_RECOVERY_ATTEMPTS", "2"))

# --- Cross-market arbitrage ---
CROSS_MARKET_ENABLED: bool = os.getenv("CROSS_MARKET_ENABLED", "true").lower() == "true"
CROSS_MARKET_MO_DNB_ENABLED: bool = os.getenv("CROSS_MARKET_MO_DNB_ENABLED", "true").lower() == "true"
CROSS_MARKET_MO_OU25_ENABLED: bool = os.getenv("CROSS_MARKET_MO_OU25_ENABLED", "false").lower() == "true"
CROSS_MARKET_MO_BTTS_ENABLED: bool = os.getenv("CROSS_MARKET_MO_BTTS_ENABLED", "false").lower() == "true"
CROSS_MARKET_CS_MO_ENABLED: bool = os.getenv("CROSS_MARKET_CS_MO_ENABLED", "false").lower() == "true"

# --- ML scoring / candidate logging ---
ML_SCORING_ENABLED: bool = os.getenv("ML_SCORING_ENABLED", "true").lower() == "true"
ML_LINEAR_MODEL_PATH: str = os.getenv("ML_LINEAR_MODEL_PATH", "")
ML_BASE_DECISION_THRESHOLD_EUR: Decimal = Decimal(os.getenv("ML_BASE_DECISION_THRESHOLD_EUR", "0.08"))
ML_MIN_FILL_PROB: Decimal = Decimal(os.getenv("ML_MIN_FILL_PROB", "0.45"))
ML_STAKE_SIZING_ENABLED: bool = os.getenv("ML_STAKE_SIZING_ENABLED", "true").lower() == "true"
ML_STAKE_MIN_MULTIPLIER: Decimal = Decimal(os.getenv("ML_STAKE_MIN_MULTIPLIER", "0.35"))
ML_STAKE_MAX_MULTIPLIER: Decimal = Decimal(os.getenv("ML_STAKE_MAX_MULTIPLIER", "1.25"))
ML_STAKE_MIN_EUR: Decimal = Decimal(os.getenv("ML_STAKE_MIN_EUR", "2.00"))
EXECUTION_TTL_SECONDS: int = int(os.getenv("EXECUTION_TTL_SECONDS", "4"))
CANDIDATE_LOG_ENABLED: bool = os.getenv("CANDIDATE_LOG_ENABLED", "true").lower() == "true"
CANDIDATE_LOG_DIR: str = os.getenv("CANDIDATE_LOG_DIR", "data/candidates")
FILL_MODEL_PATH: str = os.getenv("FILL_MODEL_PATH", "")
CLV_ENABLED: bool = os.getenv("CLV_ENABLED", "true").lower() == "true"
CLV_LOG_DIR: str = os.getenv("CLV_LOG_DIR", "data/clv")

# --- Prediction paper account (separate from arbitrage) ---
PREDICTION_ENABLED: bool = os.getenv("PREDICTION_ENABLED", "true").lower() == "true"
PREDICTION_MODEL_KINDS: str = os.getenv("PREDICTION_MODEL_KINDS", "implied_market,residual_logit,pure_logit")
PREDICTION_INITIAL_BALANCE_EUR: Decimal = Decimal(os.getenv("PREDICTION_INITIAL_BALANCE_EUR", "100000.00"))
PREDICTION_STAKE_FRACTION: Decimal = Decimal(os.getenv("PREDICTION_STAKE_FRACTION", "0.05"))
PREDICTION_MIN_STAKE_EUR: Decimal = Decimal(os.getenv("PREDICTION_MIN_STAKE_EUR", "2.00"))
PREDICTION_MAX_STAKE_EUR: Decimal = Decimal(os.getenv("PREDICTION_MAX_STAKE_EUR", "15.00"))
PREDICTION_MIN_EDGE: Decimal = Decimal(os.getenv("PREDICTION_MIN_EDGE", "0.03"))
PREDICTION_MIN_LIQUIDITY_EUR: Decimal = Decimal(os.getenv("PREDICTION_MIN_LIQUIDITY_EUR", "50.00"))
PREDICTION_MODEL_DIR: str = os.getenv("PREDICTION_MODEL_DIR", "data/prediction/models")
PREDICTION_STATE_DIR: str = os.getenv("PREDICTION_STATE_DIR", "data/prediction/state")
PREDICTION_SAVE_EVERY: int = int(os.getenv("PREDICTION_SAVE_EVERY", "25"))
PREDICTION_STRICT_GATE_MIN_SETTLED: int = int(os.getenv("PREDICTION_STRICT_GATE_MIN_SETTLED", "100"))
PREDICTION_GATE_PASS_STREAK_REQUIRED: int = int(os.getenv("PREDICTION_GATE_PASS_STREAK_REQUIRED", "2"))
# observe | soft | strict
PREDICTION_GATE_ENFORCEMENT_MODE: str = os.getenv("PREDICTION_GATE_ENFORCEMENT_MODE", "observe").strip().lower()
PREDICTION_GATE_FAIL_STAKE_MULTIPLIER: Decimal = Decimal(
    os.getenv("PREDICTION_GATE_FAIL_STAKE_MULTIPLIER", "0.25")
)
PREDICTION_GATE_FAIL_MIN_EDGE_BUMP: Decimal = Decimal(
    os.getenv("PREDICTION_GATE_FAIL_MIN_EDGE_BUMP", "0.01")
)
PREDICTION_FEATURE_ABS_MAX: Decimal = Decimal(os.getenv("PREDICTION_FEATURE_ABS_MAX", "1000000"))
PREDICTION_DRIFT_Z_THRESHOLD: Decimal = Decimal(os.getenv("PREDICTION_DRIFT_Z_THRESHOLD", "4.0"))
PREDICTION_DRIFT_MIN_COUNT: int = int(os.getenv("PREDICTION_DRIFT_MIN_COUNT", "50"))
PREDICTION_DRIFT_SUSTAIN_COUNT: int = int(os.getenv("PREDICTION_DRIFT_SUSTAIN_COUNT", "5"))
PREDICTION_FROZEN_WINDOW: int = int(os.getenv("PREDICTION_FROZEN_WINDOW", "50"))
PREDICTION_FROZEN_STD_THRESHOLD: Decimal = Decimal(os.getenv("PREDICTION_FROZEN_STD_THRESHOLD", "0.002"))
PREDICTION_SATURATION_WINDOW: int = int(os.getenv("PREDICTION_SATURATION_WINDOW", "100"))
PREDICTION_SATURATION_LOW: Decimal = Decimal(os.getenv("PREDICTION_SATURATION_LOW", "0.01"))
PREDICTION_SATURATION_HIGH: Decimal = Decimal(os.getenv("PREDICTION_SATURATION_HIGH", "0.99"))
PREDICTION_SATURATION_RATE_THRESHOLD: Decimal = Decimal(
    os.getenv("PREDICTION_SATURATION_RATE_THRESHOLD", "0.7")
)
PREDICTION_EXPERIMENT_LOG_PATH: str = os.getenv(
    "PREDICTION_EXPERIMENT_LOG_PATH", "data/prediction/experiments.jsonl"
)
PREDICTION_EXPERIMENT_LOG_EVERY_SETTLED: int = int(
    os.getenv("PREDICTION_EXPERIMENT_LOG_EVERY_SETTLED", "20")
)

# --- Learning architect (meta-controller) ---
ARCHITECT_ENABLED: bool = os.getenv("ARCHITECT_ENABLED", "true").lower() == "true"
ARCHITECT_INTERVAL_SECONDS: int = int(os.getenv("ARCHITECT_INTERVAL_SECONDS", "900"))
ARCHITECT_MIN_SETTLED_BETS: int = int(os.getenv("ARCHITECT_MIN_SETTLED_BETS", "30"))
ARCHITECT_MAX_STAKE_STEP: Decimal = Decimal(os.getenv("ARCHITECT_MAX_STAKE_STEP", "0.01"))
ARCHITECT_MAX_EDGE_STEP: Decimal = Decimal(os.getenv("ARCHITECT_MAX_EDGE_STEP", "0.01"))
ARCHITECT_MIN_STAKE_FRACTION: Decimal = Decimal(os.getenv("ARCHITECT_MIN_STAKE_FRACTION", "0.01"))
ARCHITECT_MAX_STAKE_FRACTION: Decimal = Decimal(os.getenv("ARCHITECT_MAX_STAKE_FRACTION", "0.10"))
ARCHITECT_MIN_EDGE: Decimal = Decimal(os.getenv("ARCHITECT_MIN_EDGE", "0.01"))
ARCHITECT_MAX_EDGE: Decimal = Decimal(os.getenv("ARCHITECT_MAX_EDGE", "0.10"))
# --- Value betting (unhedged single-leg bets from prediction models) ---
VALUE_BET_ENABLED: bool = os.getenv("VALUE_BET_ENABLED", "true").lower() == "true"
VALUE_BET_MIN_ENSEMBLE_SIZE: int = int(os.getenv("VALUE_BET_MIN_ENSEMBLE_SIZE", "2"))
VALUE_BET_MIN_EDGE: Decimal = Decimal(os.getenv("VALUE_BET_MIN_EDGE", "0.05"))
VALUE_BET_MAX_BRIER: Decimal = Decimal(os.getenv("VALUE_BET_MAX_BRIER", "0.24"))
VALUE_BET_KELLY_FRACTION: Decimal = Decimal(os.getenv("VALUE_BET_KELLY_FRACTION", "0.25"))
VALUE_BET_MAX_STAKE_EUR: Decimal = Decimal(os.getenv("VALUE_BET_MAX_STAKE_EUR", "50.00"))
VALUE_BET_MIN_STAKE_EUR: Decimal = Decimal(os.getenv("VALUE_BET_MIN_STAKE_EUR", "2.00"))

ARCHITECT_LOG_DIR: str = os.getenv("ARCHITECT_LOG_DIR", "data/architect")

# --- Live QA agent (rules-only) ---
QA_AGENT_ENABLED: bool = os.getenv("QA_AGENT_ENABLED", "true").lower() == "true"
QA_AGENT_INTERVAL_SECONDS: int = int(os.getenv("QA_AGENT_INTERVAL_SECONDS", "300"))
QA_AGENT_AUTO_APPLY: bool = os.getenv("QA_AGENT_AUTO_APPLY", "true").lower() == "true"
QA_RESTART_ON_DEGRADED_ENABLED: bool = os.getenv("QA_RESTART_ON_DEGRADED_ENABLED", "true").lower() == "true"
QA_DEGRADED_MIN_AGE_SECONDS: int = int(os.getenv("QA_DEGRADED_MIN_AGE_SECONDS", "60"))
QA_RESTART_COOLDOWN_SECONDS: int = int(os.getenv("QA_RESTART_COOLDOWN_SECONDS", "900"))
QA_LOG_DIR: str = os.getenv("QA_LOG_DIR", "data/qa")

# === FUNDING MODULE ===
FUNDING_MODE: str = os.getenv("FUNDING_MODE", "paper")  # "paper" or "live"
FUNDING_EXCHANGE: str = os.getenv("FUNDING_EXCHANGE", "binance")
FUNDING_MIN_RATE_PER_8H: Decimal = Decimal(os.getenv("FUNDING_MIN_RATE_PER_8H", "0.0002"))
FUNDING_MIN_ANNUALIZED_YIELD: Decimal = Decimal(os.getenv("FUNDING_MIN_ANNUALIZED_YIELD", "0.10"))
FUNDING_MAX_POSITION_USD: Decimal = Decimal(os.getenv("FUNDING_MAX_POSITION_USD", "5000.00"))
FUNDING_MAX_TOTAL_EXPOSURE_USD: Decimal = Decimal(os.getenv("FUNDING_MAX_TOTAL_EXPOSURE_USD", "50000.00"))
FUNDING_MAX_OPEN_HEDGES: int = int(os.getenv("FUNDING_MAX_OPEN_HEDGES", "10"))
FUNDING_LEVERAGE: int = int(os.getenv("FUNDING_LEVERAGE", "2"))
FUNDING_MAX_LEVERAGE: int = int(os.getenv("FUNDING_MAX_LEVERAGE", "5"))
FUNDING_MARGIN_TYPE: str = os.getenv("FUNDING_MARGIN_TYPE", "ISOLATED")
FUNDING_MIN_LIQUIDATION_DISTANCE: Decimal = Decimal(os.getenv("FUNDING_MIN_LIQUIDATION_DISTANCE", "0.30"))
FUNDING_MAX_HOLD_HOURS: int = int(os.getenv("FUNDING_MAX_HOLD_HOURS", "168"))
FUNDING_ENTRY_WINDOW_MINUTES: int = int(os.getenv("FUNDING_ENTRY_WINDOW_MINUTES", "30"))
FUNDING_POLL_INTERVAL_SECONDS: int = int(os.getenv("FUNDING_POLL_INTERVAL_SECONDS", "60"))
FUNDING_SYMBOLS_WATCHLIST_SIZE: int = int(os.getenv("FUNDING_SYMBOLS_WATCHLIST_SIZE", "50"))
FUNDING_MIN_24H_VOLUME_USD: Decimal = Decimal(os.getenv("FUNDING_MIN_24H_VOLUME_USD", "50000000"))
FUNDING_KILL_SWITCH: bool = os.getenv("FUNDING_KILL_SWITCH", "false").lower() == "true"
FUNDING_STATE_PATH: str = os.getenv("FUNDING_STATE_PATH", "data/state/funding_positions.json")
FUNDING_VALIDATION_MODE: bool = os.getenv("FUNDING_VALIDATION_MODE", "true").lower() == "true"
FUNDING_VALIDATION_SCOPE: str = os.getenv("FUNDING_VALIDATION_SCOPE", "hedge_only").strip().lower()
FUNDING_PAPER_REQUIRE_TESTNET_FILLS: bool = (
    os.getenv("FUNDING_PAPER_REQUIRE_TESTNET_FILLS", "true").lower() == "true"
)
FUNDING_PAPER_ALLOW_SIM_FALLBACK: bool = (
    os.getenv("FUNDING_PAPER_ALLOW_SIM_FALLBACK", "false").lower() == "true"
)
FUNDING_PAPER_SPOT_SLIPPAGE_BPS: Decimal = Decimal(
    os.getenv("FUNDING_PAPER_SPOT_SLIPPAGE_BPS", "6")
)
FUNDING_PAPER_PERP_SLIPPAGE_BPS: Decimal = Decimal(
    os.getenv("FUNDING_PAPER_PERP_SLIPPAGE_BPS", "4")
)
FUNDING_PAPER_MIN_SPREAD_BPS_BUFFER: Decimal = Decimal(
    os.getenv("FUNDING_PAPER_MIN_SPREAD_BPS_BUFFER", "2")
)
FUNDING_MAX_BASIS_BPS: Decimal = Decimal(os.getenv("FUNDING_MAX_BASIS_BPS", "35"))
FUNDING_MAX_ESTIMATED_ROUND_TRIP_COST_BPS: Decimal = Decimal(
    os.getenv("FUNDING_MAX_ESTIMATED_ROUND_TRIP_COST_BPS", "20")
)
FUNDING_MIN_TOP_BOOK_NOTIONAL_MULTIPLE: Decimal = Decimal(
    os.getenv("FUNDING_MIN_TOP_BOOK_NOTIONAL_MULTIPLE", "3.0")
)
FUNDING_MIN_DEPTH_USD: Decimal = Decimal(os.getenv("FUNDING_MIN_DEPTH_USD", "150000"))
FUNDING_MAX_SPREAD_BPS: Decimal = Decimal(os.getenv("FUNDING_MAX_SPREAD_BPS", "12"))
FUNDING_VALIDATION_MIN_SETTLEMENT_EVENTS: int = int(
    os.getenv("FUNDING_VALIDATION_MIN_SETTLEMENT_EVENTS", "12")
)
FUNDING_VALIDATION_MIN_CLOSED_HEDGES: int = int(
    os.getenv("FUNDING_VALIDATION_MIN_CLOSED_HEDGES", "8")
)
FUNDING_VALIDATION_MIN_NET_PNL_USD: Decimal = Decimal(
    os.getenv("FUNDING_VALIDATION_MIN_NET_PNL_USD", "0")
)
FUNDING_VALIDATION_MAX_REJECT_RATE: float = float(
    os.getenv("FUNDING_VALIDATION_MAX_REJECT_RATE", "0.35")
)
FUNDING_VALIDATION_MAX_SLIPPAGE_BPS: Decimal = Decimal(
    os.getenv("FUNDING_VALIDATION_MAX_SLIPPAGE_BPS", "12")
)
FUNDING_VALIDATION_ARCHIVE_DIR: str = os.getenv(
    "FUNDING_VALIDATION_ARCHIVE_DIR", "data/state/archive/funding_validation_runs"
)
FUNDING_VALIDATION_ENTRY_MAX_MINUTES_BEFORE_SETTLEMENT: int = int(
    os.getenv("FUNDING_VALIDATION_ENTRY_MAX_MINUTES_BEFORE_SETTLEMENT", "12")
)
FUNDING_VALIDATION_ENTRY_MIN_MINUTES_BEFORE_SETTLEMENT: int = int(
    os.getenv("FUNDING_VALIDATION_ENTRY_MIN_MINUTES_BEFORE_SETTLEMENT", "3")
)
FUNDING_VALIDATION_MAX_SNAPSHOT_AGE_SECONDS: int = int(
    os.getenv("FUNDING_VALIDATION_MAX_SNAPSHOT_AGE_SECONDS", "5")
)

# Funding ML / online learning
FUNDING_MAKER_ORDERS: bool = os.getenv("FUNDING_MAKER_ORDERS", "true").lower() == "true"
FUNDING_BNB_DISCOUNT: bool = os.getenv("FUNDING_BNB_DISCOUNT", "true").lower() == "true"
FUNDING_ML_MIN_CONFIDENCE: float = float(os.getenv("FUNDING_ML_MIN_CONFIDENCE", "0.70"))
FUNDING_ML_MIN_PREDICTED_RATE: float = float(os.getenv("FUNDING_ML_MIN_PREDICTED_RATE", "0.0001"))
FUNDING_ML_DYNAMIC_SIZING: bool = os.getenv("FUNDING_ML_DYNAMIC_SIZING", "true").lower() == "true"
FUNDING_RETRAIN_INTERVAL_HOURS: int = int(os.getenv("FUNDING_RETRAIN_INTERVAL_HOURS", "24"))
FUNDING_RETRAIN_MIN_ROWS: int = int(os.getenv("FUNDING_RETRAIN_MIN_ROWS", "500"))
FUNDING_RETRAIN_MIN_AUC: float = float(os.getenv("FUNDING_RETRAIN_MIN_AUC", "0.75"))
FUNDING_CONTRARIAN_RETRAIN_MAX_SYMBOLS: int = int(
    os.getenv("FUNDING_CONTRARIAN_RETRAIN_MAX_SYMBOLS", "20")
)
FUNDING_STRICT_MIN_SETTLED: int = int(os.getenv("FUNDING_STRICT_MIN_SETTLED", "100"))
# observe | soft | full
FUNDING_GATE_MODE: str = os.getenv("FUNDING_GATE_MODE", "observe").strip().lower()
FUNDING_GATE_FAIL_SOFT_MULTIPLIER: float = float(os.getenv("FUNDING_GATE_FAIL_SOFT_MULTIPLIER", "0.5"))
FUNDING_GATE_FAIL_FULL_MULTIPLIER: float = float(os.getenv("FUNDING_GATE_FAIL_FULL_MULTIPLIER", "0.25"))
FUNDING_GATE_FAIL_EDGE_BUMP: float = float(os.getenv("FUNDING_GATE_FAIL_EDGE_BUMP", "0.05"))
FUNDING_FEATURE_ABS_MAX: float = float(os.getenv("FUNDING_FEATURE_ABS_MAX", "1000000000"))
FUNDING_DRIFT_Z_THRESHOLD: float = float(os.getenv("FUNDING_DRIFT_Z_THRESHOLD", "4.0"))
FUNDING_DRIFT_MIN_COUNT: int = int(os.getenv("FUNDING_DRIFT_MIN_COUNT", "50"))
FUNDING_DRIFT_SUSTAIN_COUNT: int = int(os.getenv("FUNDING_DRIFT_SUSTAIN_COUNT", "5"))
FUNDING_FROZEN_WINDOW: int = int(os.getenv("FUNDING_FROZEN_WINDOW", "50"))
FUNDING_FROZEN_STD_THRESHOLD: float = float(os.getenv("FUNDING_FROZEN_STD_THRESHOLD", "0.002"))
FUNDING_SATURATION_WINDOW: int = int(os.getenv("FUNDING_SATURATION_WINDOW", "100"))
FUNDING_SATURATION_LOW: float = float(os.getenv("FUNDING_SATURATION_LOW", "0.01"))
FUNDING_SATURATION_HIGH: float = float(os.getenv("FUNDING_SATURATION_HIGH", "0.99"))
FUNDING_SATURATION_RATE_THRESHOLD: float = float(os.getenv("FUNDING_SATURATION_RATE_THRESHOLD", "0.7"))
FUNDING_EXPERIMENT_LOG_PATH: str = os.getenv(
    "FUNDING_EXPERIMENT_LOG_PATH", "data/funding/experiments.jsonl"
)
FUNDING_EXPERIMENT_LOG_EVERY_SETTLED: int = int(
    os.getenv("FUNDING_EXPERIMENT_LOG_EVERY_SETTLED", "20")
)

# === LIVE ACTIVATION READINESS ===
# Betfair readiness thresholds
LIVE_READY_BETFAIR_MIN_MODEL_POOL: int = int(os.getenv("LIVE_READY_BETFAIR_MIN_MODEL_POOL", "2"))
LIVE_READY_BETFAIR_MIN_PASSING_MODELS: int = int(os.getenv("LIVE_READY_BETFAIR_MIN_PASSING_MODELS", "2"))
LIVE_READY_BETFAIR_MIN_PASS_RATE: float = float(os.getenv("LIVE_READY_BETFAIR_MIN_PASS_RATE", "1.0"))
LIVE_READY_BETFAIR_MIN_AVG_ROI_200_PCT: float = float(
    os.getenv("LIVE_READY_BETFAIR_MIN_AVG_ROI_200_PCT", "0.0")
)
LIVE_READY_BETFAIR_MIN_AVG_BRIER_LIFT_200: float = float(
    os.getenv("LIVE_READY_BETFAIR_MIN_AVG_BRIER_LIFT_200", "0.0")
)
LIVE_READY_BETFAIR_MIN_MODEL_WINDOW_SETTLED: int = int(
    os.getenv("LIVE_READY_BETFAIR_MIN_MODEL_WINDOW_SETTLED", "200")
)

# Binance/funding readiness thresholds
LIVE_READY_BINANCE_MIN_MODEL_POOL: int = int(os.getenv("LIVE_READY_BINANCE_MIN_MODEL_POOL", "2"))
LIVE_READY_BINANCE_MIN_PASSING_MODELS: int = int(os.getenv("LIVE_READY_BINANCE_MIN_PASSING_MODELS", "2"))
LIVE_READY_BINANCE_MIN_PASS_RATE: float = float(os.getenv("LIVE_READY_BINANCE_MIN_PASS_RATE", "1.0"))
LIVE_READY_BINANCE_MIN_AVG_ROI_200_PCT: float = float(
    os.getenv("LIVE_READY_BINANCE_MIN_AVG_ROI_200_PCT", "0.0")
)
LIVE_READY_BINANCE_MIN_AVG_BRIER_LIFT_200: float = float(
    os.getenv("LIVE_READY_BINANCE_MIN_AVG_BRIER_LIFT_200", "0.0")
)
LIVE_READY_BINANCE_MIN_MODEL_WINDOW_SETTLED: int = int(
    os.getenv("LIVE_READY_BINANCE_MIN_MODEL_WINDOW_SETTLED", "200")
)
LIVE_READY_BINANCE_MIN_REALIZED_ROI_PCT: float = float(
    os.getenv("LIVE_READY_BINANCE_MIN_REALIZED_ROI_PCT", "0.0")
)

# Operational policy for switching to live
LIVE_READY_REQUIRE_STRICT_ENFORCEMENT: bool = (
    os.getenv("LIVE_READY_REQUIRE_STRICT_ENFORCEMENT", "true").lower() == "true"
)
LIVE_READY_REQUIRE_FULL_FUNDING_GATE: bool = (
    os.getenv("LIVE_READY_REQUIRE_FULL_FUNDING_GATE", "true").lower() == "true"
)

# Binance testnet URLs
BINANCE_FUTURES_TESTNET_URL: str = os.getenv(
    "BINANCE_FUTURES_TESTNET_URL", "https://demo-fapi.binance.com"
)
BINANCE_SPOT_TESTNET_URL: str = os.getenv(
    "BINANCE_SPOT_TESTNET_URL", "https://testnet.binance.vision"
)
BINANCE_FUTURES_WS_TESTNET: str = os.getenv(
    "BINANCE_FUTURES_WS_TESTNET", "wss://fstream.binancefuture.com"
)

# Binance production URLs
BINANCE_FUTURES_PROD_URL: str = "https://fapi.binance.com"
BINANCE_SPOT_PROD_URL: str = "https://api.binance.com"
BINANCE_FUTURES_WS_PROD: str = "wss://fstream.binance.com"

# Binance API keys (from .env)
BINANCE_FUTURES_API_KEY: str = os.getenv("BINANCE_FUTURES_API_KEY", "")
BINANCE_FUTURES_API_SECRET: str = os.getenv("BINANCE_FUTURES_API_SECRET", "")
BINANCE_SPOT_API_KEY: str = os.getenv("BINANCE_SPOT_API_KEY", "")
BINANCE_SPOT_API_SECRET: str = os.getenv("BINANCE_SPOT_API_SECRET", "")
BINANCE_FUTURES_TESTNET_API_KEY: str = os.getenv("BINANCE_FUTURES_TESTNET_API_KEY", "")
BINANCE_FUTURES_TESTNET_API_SECRET: str = os.getenv("BINANCE_FUTURES_TESTNET_API_SECRET", "")
BINANCE_SPOT_TESTNET_API_KEY: str = os.getenv("BINANCE_SPOT_TESTNET_API_KEY", "")
BINANCE_SPOT_TESTNET_API_SECRET: str = os.getenv("BINANCE_SPOT_TESTNET_API_SECRET", "")

# === DATA COLLECTION ===
COLLECT_LIQUIDATIONS: bool = os.getenv("COLLECT_LIQUIDATIONS", "true").lower() == "true"
COLLECT_LONG_SHORT: bool = os.getenv("COLLECT_LONG_SHORT", "true").lower() == "true"
COLLECT_DEPTH: bool = os.getenv("COLLECT_DEPTH", "true").lower() == "true"
LONG_SHORT_POLL_SECONDS: int = int(os.getenv("LONG_SHORT_POLL_SECONDS", "300"))
DEPTH_POLL_SECONDS: int = int(os.getenv("DEPTH_POLL_SECONDS", "60"))
DEPTH_TOP_N_SYMBOLS: int = int(os.getenv("DEPTH_TOP_N_SYMBOLS", "10"))
FEAR_GREED_POLL_SECONDS: int = int(os.getenv("FEAR_GREED_POLL_SECONDS", "900"))

# === CONTRARIAN STRATEGY ===
CONTRARIAN_ENABLED: bool = os.getenv("CONTRARIAN_ENABLED", "true").lower() == "true"
CONTRARIAN_MIN_FUNDING_RATE: Decimal = Decimal(os.getenv("CONTRARIAN_MIN_FUNDING_RATE", "0.0005"))
CONTRARIAN_MIN_CONFIDENCE: float = float(os.getenv("CONTRARIAN_MIN_CONFIDENCE", "0.65"))
CONTRARIAN_CAPITAL_PER_TRADE_PCT: Decimal = Decimal(os.getenv("CONTRARIAN_CAPITAL_PER_TRADE_PCT", "0.025"))
CONTRARIAN_MAX_POSITIONS: int = int(os.getenv("CONTRARIAN_MAX_POSITIONS", "3"))
CONTRARIAN_STOP_LOSS_PCT: Decimal = Decimal(os.getenv("CONTRARIAN_STOP_LOSS_PCT", "0.025"))
CONTRARIAN_TAKE_PROFIT_RATIO: Decimal = Decimal(os.getenv("CONTRARIAN_TAKE_PROFIT_RATIO", "2.5"))
CONTRARIAN_MAX_HOLD_HOURS: int = int(os.getenv("CONTRARIAN_MAX_HOLD_HOURS", "72"))
CONTRARIAN_LEVERAGE: int = int(os.getenv("CONTRARIAN_LEVERAGE", "2"))
CONTRARIAN_MODEL: str = os.getenv("CONTRARIAN_MODEL", "auto")  # "auto" | "xgboost" | "tft"
CONTRARIAN_STATE_PATH: str = os.getenv("CONTRARIAN_STATE_PATH", "data/state/directional_positions.json")
CONTRARIAN_SCAN_INTERVAL_SECONDS: int = int(os.getenv("CONTRARIAN_SCAN_INTERVAL_SECONDS", "30"))
CONTRARIAN_HISTORY_SYMBOL_LIMIT: int = int(os.getenv("CONTRARIAN_HISTORY_SYMBOL_LIMIT", "25"))
CONTRARIAN_HISTORY_FETCH_CONCURRENCY: int = int(os.getenv("CONTRARIAN_HISTORY_FETCH_CONCURRENCY", "8"))
CONTRARIAN_PORTFOLIO_INITIAL_BALANCE_USD: Decimal = Decimal(
    os.getenv("CONTRARIAN_PORTFOLIO_INITIAL_BALANCE_USD", "5000")
)
CONTRARIAN_DAILY_LOSS_LIMIT_PCT: Decimal = Decimal(
    os.getenv("CONTRARIAN_DAILY_LOSS_LIMIT_PCT", "0.03")
)
CONTRARIAN_REQUIRE_MIN_UNIQUE_SYMBOLS: int = int(
    os.getenv("CONTRARIAN_REQUIRE_MIN_UNIQUE_SYMBOLS", "8")
)
CONTRARIAN_MAX_SYMBOL_CONCENTRATION: float = float(
    os.getenv("CONTRARIAN_MAX_SYMBOL_CONCENTRATION", "0.35")
)
CONTRARIAN_MAX_DUPLICATE_SIGNATURE_RATE: float = float(
    os.getenv("CONTRARIAN_MAX_DUPLICATE_SIGNATURE_RATE", "0.10")
)
CONTRARIAN_RETRAIN_XGB_TRIALS: int = int(os.getenv("CONTRARIAN_RETRAIN_XGB_TRIALS", "3"))
CONTRARIAN_RETRAIN_TFT_EPOCHS: int = int(os.getenv("CONTRARIAN_RETRAIN_TFT_EPOCHS", "5"))

# === REGIME MODEL ===
REGIME_ENABLED: bool = os.getenv("REGIME_ENABLED", "false").lower() == "true"
REGIME_MODEL_TYPE: str = os.getenv("REGIME_MODEL_TYPE", "auto")  # "auto" | "hmm" | "transformer"
REGIME_UPDATE_INTERVAL_HOURS: int = int(os.getenv("REGIME_UPDATE_INTERVAL_HOURS", "8"))
REGIME_N_STATES: int = int(os.getenv("REGIME_N_STATES", "4"))
REGIME_CRISIS_HALT_TRADING: bool = os.getenv("REGIME_CRISIS_HALT_TRADING", "true").lower() == "true"
REGIME_MODEL_PATH: str = os.getenv("REGIME_MODEL_PATH", "data/funding_models/regime")

# === STRATEGY ORCHESTRATOR (autonomous ML lifecycle) ===
ORCHESTRATOR_ENABLED: bool = os.getenv("ORCHESTRATOR_ENABLED", "true").lower() == "true"
ORCHESTRATOR_INTERVAL_SECONDS: int = int(os.getenv("ORCHESTRATOR_INTERVAL_SECONDS", "300"))
ORCHESTRATOR_MIN_DATA_ROWS: int = int(os.getenv("ORCHESTRATOR_MIN_DATA_ROWS", "500"))
ORCHESTRATOR_MAX_MODEL_AGE_HOURS: int = int(os.getenv("ORCHESTRATOR_MAX_MODEL_AGE_HOURS", "168"))
ORCHESTRATOR_MIN_WIN_RATE: float = float(os.getenv("ORCHESTRATOR_MIN_WIN_RATE", "0.40"))
ORCHESTRATOR_MIN_TRADES_FOR_EVAL: int = int(os.getenv("ORCHESTRATOR_MIN_TRADES_FOR_EVAL", "20"))
ORCHESTRATOR_LOG_DIR: str = os.getenv("ORCHESTRATOR_LOG_DIR", "data/funding_agents")

# === CASCADE PREDICTION ===
CASCADE_ENABLED: bool = os.getenv("CASCADE_ENABLED", "true").lower() == "true"
CASCADE_ALERT_THRESHOLD: float = float(os.getenv("CASCADE_ALERT_THRESHOLD", "0.6"))
CASCADE_ACTION_THRESHOLD: float = float(os.getenv("CASCADE_ACTION_THRESHOLD", "0.8"))
CASCADE_MAX_DEFENSIVE_POSITION_PCT: Decimal = Decimal(os.getenv("CASCADE_MAX_DEFENSIVE_POSITION_PCT", "0.05"))
CASCADE_MONITOR_INTERVAL_SECONDS: int = int(os.getenv("CASCADE_MONITOR_INTERVAL_SECONDS", "300"))
CASCADE_MODEL_PATH: str = os.getenv("CASCADE_MODEL_PATH", "data/funding_models/cascade")

# === PORTFOLIO COMMAND CENTER ===
PORTFOLIO_STATE_ROOT: str = os.getenv("PORTFOLIO_STATE_ROOT", "data/portfolios")
PORTFOLIO_RUNNER_HEARTBEAT_SECONDS: int = int(
    os.getenv("PORTFOLIO_RUNNER_HEARTBEAT_SECONDS", "5")
)
COMMAND_CENTER_HISTORY_INTERVAL_SECONDS: int = int(
    os.getenv("COMMAND_CENTER_HISTORY_INTERVAL_SECONDS", "300")
)
COMMAND_CENTER_ENABLED: bool = os.getenv("COMMAND_CENTER_ENABLED", "true").lower() == "true"
COMMAND_CENTER_HOST: str = os.getenv("COMMAND_CENTER_HOST", "0.0.0.0")
COMMAND_CENTER_PORT: int = int(os.getenv("COMMAND_CENTER_PORT", "8000"))
DEPLOY_WATCHER_ENABLED: bool = os.getenv("DEPLOY_WATCHER_ENABLED", "false").lower() == "true"
DEPLOY_WATCHER_INTERVAL_SECONDS: int = int(os.getenv("DEPLOY_WATCHER_INTERVAL_SECONDS", "120"))
DEPLOY_WATCHER_BRANCH: str = os.getenv("DEPLOY_WATCHER_BRANCH", "main")
DEPLOY_WATCHER_REMOTE: str = os.getenv("DEPLOY_WATCHER_REMOTE", "origin")
DEPLOY_WATCHER_AUTO_RESTART_DASHBOARD: bool = (
    os.getenv("DEPLOY_WATCHER_AUTO_RESTART_DASHBOARD", "false").lower() == "true"
)
DEPLOY_WATCHER_AUTOSTART: bool = os.getenv("DEPLOY_WATCHER_AUTOSTART", "true").lower() == "true"
DEPLOY_WATCHER_AUTO_RESTART_PORTFOLIOS: str = os.getenv(
    "DEPLOY_WATCHER_AUTO_RESTART_PORTFOLIOS",
    "betfair_core,hedge_validation,hedge_research,cascade_alpha,contrarian_legacy",
)
BETFAIR_PORTFOLIO_ID: str = os.getenv("BETFAIR_PORTFOLIO_ID", "betfair_core")
HEDGE_PORTFOLIO_ID: str = os.getenv("HEDGE_PORTFOLIO_ID", "hedge_validation")
HEDGE_PORTFOLIO_INITIAL_BALANCE_USD: Decimal = Decimal(
    os.getenv("HEDGE_PORTFOLIO_INITIAL_BALANCE_USD", "50000")
)
HEDGE_RESEARCH_PORTFOLIO_ID: str = os.getenv("HEDGE_RESEARCH_PORTFOLIO_ID", "hedge_research")
HEDGE_RESEARCH_ENABLED: bool = os.getenv("HEDGE_RESEARCH_ENABLED", "true").lower() == "true"
HEDGE_RESEARCH_INITIAL_BALANCE_USD: Decimal = Decimal(
    os.getenv("HEDGE_RESEARCH_INITIAL_BALANCE_USD", "25000")
)
HEDGE_RESEARCH_STATE_PATH: str = os.getenv(
    "HEDGE_RESEARCH_STATE_PATH", "data/state/funding_positions_research.json"
)
HEDGE_RESEARCH_ENTRY_WINDOW_MINUTES: int = int(
    os.getenv("HEDGE_RESEARCH_ENTRY_WINDOW_MINUTES", "120")
)
FUNDING_SHARED_LEARNER_READ_ONLY: bool = (
    os.getenv("FUNDING_SHARED_LEARNER_READ_ONLY", "false").lower() == "true"
)

# === CASCADE ALPHA PORTFOLIO ===
CASCADE_ALPHA_ENABLED: bool = os.getenv("CASCADE_ALPHA_ENABLED", "false").lower() == "true"
CASCADE_ALPHA_INITIAL_BALANCE_USD: Decimal = Decimal(
    os.getenv("CASCADE_ALPHA_INITIAL_BALANCE_USD", "15000")
)
CASCADE_ALPHA_MAX_OPEN_POSITIONS: int = int(
    os.getenv("CASCADE_ALPHA_MAX_OPEN_POSITIONS", "3")
)
CASCADE_ALPHA_MAX_NOTIONAL_PER_TRADE_USD: Decimal = Decimal(
    os.getenv("CASCADE_ALPHA_MAX_NOTIONAL_PER_TRADE_USD", "5000")
)
CASCADE_ALPHA_MAX_GROSS_EXPOSURE_USD: Decimal = Decimal(
    os.getenv("CASCADE_ALPHA_MAX_GROSS_EXPOSURE_USD", "30000")
)
CASCADE_ALPHA_DAILY_LOSS_LIMIT_PCT: Decimal = Decimal(
    os.getenv("CASCADE_ALPHA_DAILY_LOSS_LIMIT_PCT", "0.03")
)
CASCADE_ALPHA_MAX_HOLD_SECONDS: int = int(
    os.getenv("CASCADE_ALPHA_MAX_HOLD_SECONDS", "900")
)
CASCADE_ALPHA_EVENT_COOLDOWN_SECONDS: int = int(
    os.getenv("CASCADE_ALPHA_EVENT_COOLDOWN_SECONDS", "300")
)
CASCADE_ALPHA_REQUIRE_BETA_HEDGE: bool = (
    os.getenv("CASCADE_ALPHA_REQUIRE_BETA_HEDGE", "true").lower() == "true"
)
CASCADE_ALPHA_RULE_ONLY: bool = os.getenv("CASCADE_ALPHA_RULE_ONLY", "true").lower() == "true"
CASCADE_ALPHA_MAX_SPREAD_BPS: Decimal = Decimal(
    os.getenv("CASCADE_ALPHA_MAX_SPREAD_BPS", "15")
)
CASCADE_ALPHA_MAX_SLIPPAGE_BPS: Decimal = Decimal(
    os.getenv("CASCADE_ALPHA_MAX_SLIPPAGE_BPS", "20")
)
CASCADE_ALPHA_MIN_LIQUIDATION_Z: Decimal = Decimal(
    os.getenv("CASCADE_ALPHA_MIN_LIQUIDATION_Z", "2.5")
)
CASCADE_ALPHA_MIN_DEPTH_COLLAPSE_Z: Decimal = Decimal(
    os.getenv("CASCADE_ALPHA_MIN_DEPTH_COLLAPSE_Z", "2.0")
)
CASCADE_ALPHA_LEARNER_ENABLED: bool = (
    os.getenv("CASCADE_ALPHA_LEARNER_ENABLED", "true").lower() == "true"
)
CASCADE_ALPHA_MIN_SETTLED_FOR_CANDIDATE: int = int(
    os.getenv("CASCADE_ALPHA_MIN_SETTLED_FOR_CANDIDATE", "20")
)
CASCADE_ALPHA_MIN_WIN_RATE_PCT: Decimal = Decimal(
    os.getenv("CASCADE_ALPHA_MIN_WIN_RATE_PCT", "52")
)
CASCADE_ALPHA_MIN_AVG_NET_PNL_USD: Decimal = Decimal(
    os.getenv("CASCADE_ALPHA_MIN_AVG_NET_PNL_USD", "0")
)
CASCADE_ALPHA_POLICY_ACTIVATION_SETTLED: int = int(
    os.getenv("CASCADE_ALPHA_POLICY_ACTIVATION_SETTLED", "5")
)
CASCADE_ALPHA_POLICY_REDUCED_MAX_OPEN_POSITIONS: int = int(
    os.getenv("CASCADE_ALPHA_POLICY_REDUCED_MAX_OPEN_POSITIONS", "2")
)
CASCADE_ALPHA_POLICY_REDUCED_NOTIONAL_MULTIPLIER: Decimal = Decimal(
    os.getenv("CASCADE_ALPHA_POLICY_REDUCED_NOTIONAL_MULTIPLIER", "0.5")
)
CASCADE_ALPHA_POLICY_STRONG_MIN_SIGNAL_SCORE: Decimal = Decimal(
    os.getenv("CASCADE_ALPHA_POLICY_STRONG_MIN_SIGNAL_SCORE", "8.0")
)

# === SOLANA MEV SCOUT ===
MEV_SCOUT_SOL_ENABLED: bool = os.getenv("MEV_SCOUT_SOL_ENABLED", "false").lower() == "true"
MEV_SCOUT_SOL_SHADOW_BALANCE_USD: Decimal = Decimal(
    os.getenv("MEV_SCOUT_SOL_SHADOW_BALANCE_USD", "5000")
)
MEV_SCOUT_SOL_RPC_URL: str = os.getenv("MEV_SCOUT_SOL_RPC_URL", "")
MEV_SCOUT_SOL_WS_URL: str = os.getenv("MEV_SCOUT_SOL_WS_URL", "")
MEV_SCOUT_SOL_YELLOWSTONE_URL: str = os.getenv("MEV_SCOUT_SOL_YELLOWSTONE_URL", "")
MEV_SCOUT_SOL_JITO_REGION: str = os.getenv("MEV_SCOUT_SOL_JITO_REGION", "auto")
MEV_SCOUT_SOL_MIN_WHALE_USD: Decimal = Decimal(
    os.getenv("MEV_SCOUT_SOL_MIN_WHALE_USD", "250000")
)
MEV_SCOUT_SOL_REPLAY_PATH: str = os.getenv("MEV_SCOUT_SOL_REPLAY_PATH", "")
MEV_SCOUT_SOL_MAX_EVENTS_PER_POLL: int = int(os.getenv("MEV_SCOUT_SOL_MAX_EVENTS_PER_POLL", "25"))
MEV_SCOUT_SOL_LABEL_DELAY_SECONDS: int = int(os.getenv("MEV_SCOUT_SOL_LABEL_DELAY_SECONDS", "120"))
MEV_SCOUT_SOL_MIN_EXPECTED_EDGE_USD: Decimal = Decimal(
    os.getenv("MEV_SCOUT_SOL_MIN_EXPECTED_EDGE_USD", "5")
)

# === LEGACY PORTFOLIOS ===
CONTRARIAN_LEGACY_ENABLED: bool = (
    os.getenv("CONTRARIAN_LEGACY_ENABLED", "false").lower() == "true"
)
