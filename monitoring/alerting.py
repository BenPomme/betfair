"""
Telegram Bot API alerts: execution failure, circuit breaker, daily loss cap.
"""
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)

_PLACEHOLDER_MARKERS = ("your_token", "your_chat_id", "changeme", "replace_me", "example")


def _is_placeholder(value: str) -> bool:
    v = (value or "").strip().lower()
    if not v:
        return True
    return any(m in v for m in _PLACEHOLDER_MARKERS)


def send_telegram(message: str) -> bool:
    """Send message to TELEGRAM_CHAT_ID via TELEGRAM_BOT_TOKEN. Returns True if sent."""
    if (
        not config.TELEGRAM_BOT_TOKEN
        or not config.TELEGRAM_CHAT_ID
        or _is_placeholder(config.TELEGRAM_BOT_TOKEN)
        or _is_placeholder(config.TELEGRAM_CHAT_ID)
    ):
        logger.debug("Telegram not configured; skipping alert")
        return False
    try:
        import httpx
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = httpx.post(url, json={"chat_id": config.TELEGRAM_CHAT_ID, "text": message}, timeout=10.0)
        return resp.status_code == 200
    except Exception as e:
        logger.exception("Telegram send failed: %s", e)
        return False


def alert_execution_failure(market_id: str, error: str) -> None:
    send_telegram(f"[Arb] Execution failed market_id={market_id}: {error}")


def alert_circuit_breaker() -> None:
    send_telegram("[Arb] Circuit breaker: trading halted after 3 consecutive failures.")


def alert_daily_loss_cap() -> None:
    send_telegram("[Arb] Daily loss limit reached. Trading halted.")


def alert_stale_order_cancelled(order_info: dict) -> None:
    send_telegram(
        f"[Arb] Stale order cancelled: market={order_info.get('market_id')} "
        f"bet_id={order_info.get('bet_id')} age={order_info.get('age_seconds')}s"
    )


def alert_partial_fill(order_info: dict) -> None:
    send_telegram(
        f"[Arb] Partial fill detected: market={order_info.get('market_id')} "
        f"matched={order_info.get('size_matched')} / {order_info.get('size_total')}"
    )


def alert_trade_executed(opp: dict, result: dict, scored: dict) -> None:
    send_telegram(
        f"[Trade] {opp.get('arb_type')} on {opp.get('event_name')}\n"
        f"Net profit: {opp.get('net_profit_eur')} | ROI: {opp.get('net_roi_pct')}%\n"
        f"Edge score: {scored.get('edge_score')} | Fill prob: {scored.get('fill_prob')}\n"
        f"Prediction: {scored.get('prediction_influence', 'none')}"
    )


def alert_daily_summary(paper_executor, prediction_manager=None) -> None:
    trade_count = len(getattr(paper_executor, "log_entries", []))
    msg = f"[Daily] Balance: {paper_executor.balance} | Trades today: {trade_count}"
    if prediction_manager:
        for model_id, engine in prediction_manager.engines.items():
            state = engine.get_state()
            msg += (
                f"\n  {model_id}: ROI={state['roi_pct']}% "
                f"Brier={state['avg_brier']} "
                f"Bets={state['settled_bets']}"
            )
    send_telegram(msg)


def alert_model_degradation(model_id: str, metric: str, value: float, threshold: float) -> None:
    send_telegram(
        f"[Model] {model_id} degradation: {metric}={value:.4f} (threshold={threshold:.4f})"
    )


def alert_prediction_bet(payload: dict) -> None:
    for event in payload.get("events", []):
        kind = event.get("kind", "")
        if kind == "prediction_open":
            send_telegram(
                f"[Pred] {event['model_id']} bet on {event['selection']} @ {event['odds']} "
                f"edge={event['edge']} stake={event['stake_eur']}"
            )
        elif kind == "prediction_settle":
            emoji = "W" if event.get("won") else ("V" if event.get("void") else "L")
            send_telegram(
                f"[Pred] {event['model_id']} {emoji} {event['selection']} "
                f"PnL={event['pnl_eur']} Balance={event['balance_eur']}"
            )
