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
FUNDING_MAX_POSITION_USD: Decimal = Decimal(os.getenv("FUNDING_MAX_POSITION_USD", "2000.00"))
FUNDING_MAX_TOTAL_EXPOSURE_USD: Decimal = Decimal(os.getenv("FUNDING_MAX_TOTAL_EXPOSURE_USD", "10000.00"))
FUNDING_MAX_OPEN_HEDGES: int = int(os.getenv("FUNDING_MAX_OPEN_HEDGES", "5"))
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

# Funding ML / online learning
FUNDING_MAKER_ORDERS: bool = os.getenv("FUNDING_MAKER_ORDERS", "true").lower() == "true"
FUNDING_BNB_DISCOUNT: bool = os.getenv("FUNDING_BNB_DISCOUNT", "true").lower() == "true"
FUNDING_ML_MIN_CONFIDENCE: float = float(os.getenv("FUNDING_ML_MIN_CONFIDENCE", "0.70"))
FUNDING_ML_MIN_PREDICTED_RATE: float = float(os.getenv("FUNDING_ML_MIN_PREDICTED_RATE", "0.0001"))
FUNDING_ML_DYNAMIC_SIZING: bool = os.getenv("FUNDING_ML_DYNAMIC_SIZING", "true").lower() == "true"
FUNDING_RETRAIN_INTERVAL_HOURS: int = int(os.getenv("FUNDING_RETRAIN_INTERVAL_HOURS", "24"))
FUNDING_RETRAIN_MIN_ROWS: int = int(os.getenv("FUNDING_RETRAIN_MIN_ROWS", "500"))
FUNDING_RETRAIN_MIN_AUC: float = float(os.getenv("FUNDING_RETRAIN_MIN_AUC", "0.75"))

# Binance testnet URLs
BINANCE_FUTURES_TESTNET_URL: str = "https://testnet.binancefuture.com"
BINANCE_SPOT_TESTNET_URL: str = "https://testnet.binance.vision"
BINANCE_FUTURES_WS_TESTNET: str = "wss://fstream.binancefuture.com"

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
