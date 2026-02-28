"""
Autonomous trade monitor: polls the dashboard API, sends trade data to local
Qwen 3.5 (Ollama) for analysis, logs findings. Runs continuously at configurable
intervals. Tier 2 task — uses free local model.

Usage: python scripts/trade_monitor.py [--interval 60] [--dashboard-url http://127.0.0.1:8000]
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MONITOR] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:32b"  # Fastest local model for monitoring
REPORT_DIR = Path(__file__).parent.parent / "monitoring" / "reports"


def fetch_state(dashboard_url: str) -> Optional[dict]:
    try:
        r = requests.get(f"{dashboard_url}/api/state", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Failed to fetch dashboard state: %s", e)
        return None


def verify_trade_math(trade: dict) -> List[str]:
    """Verify trade arithmetic locally (no LLM needed). Returns list of issues."""
    issues = []
    sels = trade.get("selections", [])
    if not sels:
        issues.append("No selections in trade")
        return issues

    sel = sels[0]
    arb_type = trade.get("arb_type", "")

    if arb_type == "cross_market":
        back_price = sel.get("back_price")
        lay_price = sel.get("lay_price")
        back_stake = sel.get("stake_eur")
        lay_stake = sel.get("lay_stake_eur")
        net_profit = trade.get("net_profit_eur")
        roi = trade.get("net_roi_pct")

        if back_price is None or lay_price is None:
            issues.append(f"Missing prices: back={back_price} lay={lay_price}")
            return issues

        if back_price <= lay_price:
            issues.append(f"BUG: back_price ({back_price}) <= lay_price ({lay_price}) — no arb exists")

        if back_stake and lay_stake and back_price and lay_price:
            expected_lay = round(back_stake * back_price / lay_price, 2)
            if abs(expected_lay - lay_stake) > 0.05:
                issues.append(f"Lay stake mismatch: expected {expected_lay}, got {lay_stake}")

        if back_stake and lay_stake:
            # Check scenario: selection wins
            back_win = back_stake * (back_price - 1)
            lay_loss = lay_stake * (lay_price - 1)
            gross_win = back_win - lay_loss
            comm_win = round(back_win * 0.05, 2)
            net_win = gross_win - comm_win

            # Check scenario: selection loses
            gross_lose = lay_stake - back_stake
            comm_lose = round(lay_stake * 0.05, 2)
            net_lose = gross_lose - comm_lose

            expected_net = min(net_win, net_lose)
            if net_profit is not None and abs(expected_net - net_profit) > 0.10:
                issues.append(
                    f"Net profit mismatch: expected ~{expected_net:.2f}, got {net_profit:.4f} "
                    f"(win scenario: {net_win:.2f}, lose scenario: {net_lose:.2f})"
                )

            if net_win < 0 or net_lose < 0:
                issues.append(f"BUG: Negative scenario — win: {net_win:.2f}, lose: {net_lose:.2f}")

        if roi is not None and roi > 0.25:
            issues.append(f"Suspicious ROI: {roi*100:.1f}% — likely stale/thin liquidity")

        # Check realistic net sanity
        realistic = trade.get("fill_simulated_realistic_net")
        if realistic is not None and net_profit is not None:
            if realistic > net_profit * 1.5:
                issues.append(
                    f"Realistic net ({realistic:.2f}) > optimistic ({net_profit:.4f}) — "
                    f"realistic calc may be buggy"
                )

        # Selection name check
        name = sel.get("name", "")
        if name.isdigit():
            issues.append(f"Selection name is numeric ID: '{name}' — runner name mapping failed")

    elif arb_type in ("back_back", "lay_lay"):
        # Basic checks for single-market arbs
        if net_profit is not None and net_profit < 0:
            issues.append(f"BUG: Negative net profit on {arb_type}: {net_profit}")

    return issues


def build_analysis_prompt(state: dict, trade_issues: Dict[int, List[str]], prev_summary: str) -> str:
    """Build a SHORT prompt for Qwen to analyze. Brevity = speed on local models."""
    trades = state.get("trades", [])
    ss = state.get("scan_stats", {})

    # Compact trade summary
    trade_lines = []
    for t in trades[-5:]:
        sel = t.get("selections", [{}])[0]
        trade_lines.append(
            f"{t.get('event','?')}: {sel.get('name','?')} B@{sel.get('back_price','?')} "
            f"L@{sel.get('lay_price','?')} stake={sel.get('stake_eur','?')} "
            f"net={t.get('net_profit_eur','?')} roi={t.get('net_roi_pct',0)*100:.1f}%"
        )

    issues_text = ""
    if trade_issues:
        for idx, issues in trade_issues.items():
            issues_text += f"Trade {idx+1}: {'; '.join(issues)}\n"

    prompt = f"""Betfair arb bot monitor. Balance: {state.get('balance_eur')} EUR, P&L: {state.get('daily_pnl_eur')} EUR, {len(trades)} trades.

Trades:
{chr(10).join(trade_lines)}

Math issues found:
{issues_text or 'None'}

Check: 1) lay_stake = back_stake*back_price/lay_price? 2) ROI>15% suspicious? 3) names not numeric IDs? 4) any bugs?
Reply: ISSUES, WARNINGS, STATUS. Max 150 words."""

    return prompt


def query_qwen(prompt: str) -> Optional[str]:
    """Query local Qwen 3.5 via Ollama API."""
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 500},
            },
            timeout=300,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.ConnectionError:
        logger.error("Ollama not running at %s — install/start Ollama first", OLLAMA_URL)
        return None
    except Exception as e:
        logger.error("Qwen query failed: %s", e)
        return None


def save_report(report: str, trade_issues: Dict[int, List[str]], state: dict) -> Path:
    """Save report to file."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"monitor_{ts}.json"

    data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "trade_count": len(state.get("trades", [])),
        "balance_eur": state.get("balance_eur"),
        "daily_pnl_eur": state.get("daily_pnl_eur"),
        "math_issues": {str(k): v for k, v in trade_issues.items()},
        "qwen_analysis": report,
        "scan_stats": state.get("scan_stats", {}),
    }

    path.write_text(json.dumps(data, indent=2))
    return path


def run_monitor(dashboard_url: str, interval: int) -> None:
    logger.info("Trade monitor started — polling every %ds", interval)
    logger.info("Dashboard: %s | Model: %s", dashboard_url, MODEL)

    prev_trade_count = 0
    prev_summary = ""
    check_count = 0

    while True:
        check_count += 1
        logger.info("--- Check #%d ---", check_count)

        state = fetch_state(dashboard_url)
        if state is None:
            logger.warning("Dashboard unreachable, retrying in %ds", interval)
            time.sleep(interval)
            continue

        trades = state.get("trades", [])
        if not state.get("running", False):
            logger.info("Engine not running. Waiting...")
            time.sleep(interval)
            continue

        # Math verification (always, no LLM cost)
        trade_issues: Dict[int, List[str]] = {}
        for i, t in enumerate(trades):
            issues = verify_trade_math(t)
            if issues:
                trade_issues[i] = issues

        if trade_issues:
            logger.warning("Math issues found in %d trades:", len(trade_issues))
            for idx, issues in trade_issues.items():
                event = trades[idx].get("event", "?")
                for issue in issues:
                    logger.warning("  Trade %d (%s): %s", idx + 1, event, issue)

        # Only query Qwen when there are new trades or issues
        new_trades = len(trades) > prev_trade_count
        has_issues = bool(trade_issues)

        if new_trades or has_issues or check_count % 5 == 0:
            logger.info("Querying Qwen 3.5 for analysis...")
            prompt = build_analysis_prompt(state, trade_issues, prev_summary)
            report = query_qwen(prompt)

            if report:
                logger.info("Qwen analysis:\n%s", report)
                path = save_report(report, trade_issues, state)
                logger.info("Report saved: %s", path)

                # Extract summary for next iteration context
                prev_summary = report[:200] if len(report) > 200 else report
            else:
                logger.warning("No response from Qwen")
        else:
            ss = state.get("scan_stats", {})
            logger.info(
                "No new trades. Balance: %.2f EUR | Scans: %d | Cross-market: %d",
                state.get("balance_eur", 0),
                ss.get("total_scans", 0),
                ss.get("cross_market_found", 0),
            )

        prev_trade_count = len(trades)
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Trade monitor with Qwen 3.5 analysis")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:8000", help="Dashboard URL")
    args = parser.parse_args()

    run_monitor(args.dashboard_url, args.interval)


if __name__ == "__main__":
    main()
