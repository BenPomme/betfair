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

# --- Stake sizing ---
STAKE_FRACTION: Decimal = Decimal(os.getenv("STAKE_FRACTION", "0.10"))  # 10% of balance per arb

# --- Market scanning ---
SCAN_SPORTS: str = os.getenv("SCAN_SPORTS", "all")  # "all" = auto-discover, or comma-separated IDs
SCAN_COUNTRIES: str = os.getenv("SCAN_COUNTRIES", "")  # empty = no country filter (global)
SCAN_MAX_MARKETS: int = int(os.getenv("SCAN_MAX_MARKETS", "500"))
SCAN_INCLUDE_IN_PLAY: bool = os.getenv("SCAN_INCLUDE_IN_PLAY", "true").lower() == "true"

# --- Cross-market arbitrage ---
CROSS_MARKET_ENABLED: bool = os.getenv("CROSS_MARKET_ENABLED", "true").lower() == "true"
