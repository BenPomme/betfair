"""
Historical data collector for funding rate arbitrage ML training.
Downloads from Binance PRODUCTION API (public endpoints, no auth required):
  - Funding rate history
  - 1h klines (for spot-perp basis, volatility, volume features)
  - Open interest snapshots

Stores data as CSV files in data/funding_history/ for offline ML training.
"""
import asyncio
import csv
import logging
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from binance.um_futures import UMFutures

logger = logging.getLogger(__name__)

# Production API (public, no auth)
PROD_BASE_URL = "https://fapi.binance.com"
DATA_DIR = Path("data/funding_history")


def _to_str(val: Any) -> str:
    return str(val)


class FundingRateFetcher:
    """Bulk download historical data from Binance production API."""

    def __init__(self, data_dir: Optional[Path] = None):
        self._client = UMFutures(base_url=PROD_BASE_URL)
        self._data_dir = data_dir or DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)
        (self._data_dir / "funding_rates").mkdir(exist_ok=True)
        (self._data_dir / "klines").mkdir(exist_ok=True)
        (self._data_dir / "open_interest").mkdir(exist_ok=True)

    def _rate_limit_pause(self, weight: int = 1) -> None:
        """Pause to respect API rate limits (2400 weight/min)."""
        time.sleep(weight * 0.05)  # Conservative: ~20 req/sec

    def fetch_funding_rates(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[dict]:
        """Fetch all historical funding rates for a symbol.

        Paginates using startTime/endTime to get full history.
        """
        if start_date is None:
            start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        all_rates = []
        current_start = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        while current_start < end_ms:
            try:
                result = self._client.funding_rate(
                    symbol=symbol,
                    startTime=current_start,
                    endTime=end_ms,
                    limit=limit,
                )
            except Exception as e:
                logger.warning("Error fetching rates for %s at %s: %s", symbol, current_start, e)
                self._rate_limit_pause(5)
                continue

            if not result:
                break

            for item in result:
                all_rates.append({
                    "symbol": item["symbol"],
                    "funding_rate": item["fundingRate"],
                    "funding_time": int(item["fundingTime"]),
                    "mark_price": item.get("markPrice", "0"),
                })

            # Move start past last result
            last_time = int(result[-1]["fundingTime"])
            if last_time <= current_start:
                break
            current_start = last_time + 1

            self._rate_limit_pause(1)

            if len(result) < limit:
                break

        return all_rates

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1500,
    ) -> List[dict]:
        """Fetch historical klines (candlesticks) for a symbol."""
        if start_date is None:
            start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        all_klines = []
        current_start = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        while current_start < end_ms:
            try:
                result = self._client.klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=current_start,
                    endTime=end_ms,
                    limit=limit,
                )
            except Exception as e:
                logger.warning("Error fetching klines for %s: %s", symbol, e)
                self._rate_limit_pause(5)
                continue

            if not result:
                break

            for k in result:
                all_klines.append({
                    "open_time": int(k[0]),
                    "open": k[1],
                    "high": k[2],
                    "low": k[3],
                    "close": k[4],
                    "volume": k[5],
                    "close_time": int(k[6]),
                    "quote_volume": k[7],
                    "trades": int(k[8]),
                    "taker_buy_volume": k[9],
                    "taker_buy_quote_volume": k[10],
                })

            last_time = int(result[-1][6])
            if last_time <= current_start:
                break
            current_start = last_time + 1

            self._rate_limit_pause(2)

            if len(result) < limit:
                break

        return all_klines

    def fetch_open_interest_history(
        self,
        symbol: str,
        period: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[dict]:
        """Fetch historical open interest for a symbol."""
        if start_date is None:
            start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        all_oi = []
        current_start = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        max_retries = 3
        while current_start < end_ms:
            try:
                result = self._client.open_interest_hist(
                    symbol=symbol,
                    period=period,
                    startTime=current_start,
                    endTime=end_ms,
                    limit=limit,
                )
            except Exception as e:
                max_retries -= 1
                if max_retries <= 0:
                    logger.warning("Skipping OI for %s after retries: %s", symbol, e)
                    break
                logger.debug("Error fetching OI for %s: %s", symbol, e)
                self._rate_limit_pause(5)
                continue

            if not result:
                break

            for item in result:
                all_oi.append({
                    "symbol": item["symbol"],
                    "sum_open_interest": item["sumOpenInterest"],
                    "sum_open_interest_value": item["sumOpenInterestValue"],
                    "timestamp": int(item["timestamp"]),
                })

            last_time = int(result[-1]["timestamp"])
            if last_time <= current_start:
                break
            current_start = last_time + 1

            self._rate_limit_pause(2)

            if len(result) < limit:
                break

        return all_oi

    def save_funding_rates(self, symbol: str, data: List[dict]) -> Path:
        """Save funding rates to CSV."""
        path = self._data_dir / "funding_rates" / f"{symbol}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["symbol", "funding_rate", "funding_time", "mark_price"])
            writer.writeheader()
            writer.writerows(data)
        return path

    def save_klines(self, symbol: str, data: List[dict]) -> Path:
        """Save klines to CSV."""
        path = self._data_dir / "klines" / f"{symbol}.csv"
        fieldnames = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_volume", "taker_buy_quote_volume",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return path

    def save_open_interest(self, symbol: str, data: List[dict]) -> Path:
        """Save open interest to CSV."""
        path = self._data_dir / "open_interest" / f"{symbol}.csv"
        fieldnames = ["symbol", "sum_open_interest", "sum_open_interest_value", "timestamp"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return path

    def collect_all(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        kline_start: Optional[datetime] = None,
    ) -> Dict[str, dict]:
        """Collect all historical data for a list of symbols.

        Args:
            symbols: List of symbol strings (e.g. ["BTCUSDT", "ETHUSDT"]).
            start_date: Start date for funding rates (default: 2020-01-01).
            kline_start: Start date for klines/OI (default: 12 months ago).

        Returns:
            Dict of symbol → {funding_count, kline_count, oi_count}.
        """
        if kline_start is None:
            kline_start = datetime.now(timezone.utc) - timedelta(days=365)

        stats = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            logger.info("[%d/%d] Collecting %s...", i + 1, total, symbol)

            # Funding rates (go back as far as possible)
            rates = self.fetch_funding_rates(symbol, start_date=start_date)
            if rates:
                self.save_funding_rates(symbol, rates)
            logger.info("  %s: %d funding rates", symbol, len(rates))

            # 1h klines (last 12 months)
            klines = self.fetch_klines(symbol, interval="1h", start_date=kline_start)
            if klines:
                self.save_klines(symbol, klines)
            logger.info("  %s: %d klines", symbol, len(klines))

            # Open interest (last 12 months)
            oi = self.fetch_open_interest_history(symbol, start_date=kline_start)
            if oi:
                self.save_open_interest(symbol, oi)
            logger.info("  %s: %d OI records", symbol, len(oi))

            stats[symbol] = {
                "funding_count": len(rates),
                "kline_count": len(klines),
                "oi_count": len(oi),
            }

        return stats


    def append_new_rates(self, symbol: str) -> int:
        """Fetch only new funding rates since last stored timestamp and append to CSV.

        Returns count of new rows added.
        """
        path = self._data_dir / "funding_rates" / f"{symbol}.csv"

        # Read last timestamp from existing CSV
        last_ts_ms = 0
        if path.exists():
            try:
                with open(path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ts = int(row["funding_time"])
                        if ts > last_ts_ms:
                            last_ts_ms = ts
            except Exception as e:
                logger.warning("Error reading last timestamp for %s: %s", symbol, e)

        # Fetch only new rows (start_date = last_ts + 1ms)
        if last_ts_ms > 0:
            start_date = datetime.fromtimestamp((last_ts_ms + 1) / 1000, tz=timezone.utc)
        else:
            start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)

        new_rates = self.fetch_funding_rates(symbol, start_date=start_date)
        if not new_rates:
            return 0

        # Filter out any duplicates (belt-and-suspenders)
        new_rates = [r for r in new_rates if int(r["funding_time"]) > last_ts_ms]
        if not new_rates:
            return 0

        # Append to CSV (create with header if file doesn't exist)
        file_exists = path.exists() and path.stat().st_size > 0
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["symbol", "funding_rate", "funding_time", "mark_price"]
            )
            if not file_exists:
                writer.writeheader()
            writer.writerows(new_rates)

        logger.info("%s: appended %d new funding rates", symbol, len(new_rates))
        return len(new_rates)

    def update_all(self, symbols: List[str]) -> Dict[str, int]:
        """Incrementally update funding rates for all symbols.

        Returns dict of symbol → new rows added.
        """
        results = {}
        for symbol in symbols:
            try:
                count = self.append_new_rates(symbol)
                results[symbol] = count
            except Exception as e:
                logger.warning("Failed to update %s: %s", symbol, e)
                results[symbol] = 0
        total = sum(results.values())
        logger.info("Updated %d symbols, %d total new rows", len(symbols), total)
        return results


def get_top_symbols(n: int = 50) -> List[str]:
    """Fetch top N perpetual symbols by 24h volume from production API."""
    client = UMFutures(base_url=PROD_BASE_URL)

    # Get exchange info for perpetual contracts
    info = client.exchange_info()
    perpetuals = {
        s["symbol"] for s in info.get("symbols", [])
        if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING"
    }

    # Get 24h tickers and sort by volume
    tickers = client.ticker_24hr_price_change()
    ranked = []
    for t in tickers:
        sym = t["symbol"]
        if sym in perpetuals:
            ranked.append((sym, float(t.get("quoteVolume", 0))))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in ranked[:n]]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    print("Fetching top 50 symbols by volume...")
    symbols = get_top_symbols(50)
    print(f"Top symbols: {symbols[:10]}...")

    fetcher = FundingRateFetcher()
    stats = fetcher.collect_all(symbols)

    print("\n=== Collection Summary ===")
    total_rates = sum(s["funding_count"] for s in stats.values())
    total_klines = sum(s["kline_count"] for s in stats.values())
    total_oi = sum(s["oi_count"] for s in stats.values())
    print(f"Symbols: {len(stats)}")
    print(f"Total funding rates: {total_rates:,}")
    print(f"Total klines: {total_klines:,}")
    print(f"Total OI records: {total_oi:,}")
