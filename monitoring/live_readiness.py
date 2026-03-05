"""
Live-trading readiness evaluator for Betfair + Binance funding.

This module turns existing strict-gate and health metrics into a clear
"can switch to live now" decision with explicit blockers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import config


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _check(name: str, ok: bool, actual: Any, required: Any, reason: str) -> Dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "actual": actual,
        "required": required,
        "reason": reason,
    }


def _score(checks: List[Dict[str, Any]]) -> float:
    if not checks:
        return 0.0
    passed = sum(1 for c in checks if bool(c.get("ok")))
    return round((passed / len(checks)) * 100.0, 2)


def _label_from_score(score_pct: float) -> str:
    if score_pct >= 95.0:
        return "high"
    if score_pct >= 75.0:
        return "medium"
    return "low"


def _avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def evaluate_betfair_live_readiness(state: Dict[str, Any]) -> Dict[str, Any]:
    health = state.get("health") or {}
    cfg = state.get("config") or {}
    models = state.get("prediction_models") or {}

    strict_min_settled = _as_int(getattr(config, "PREDICTION_STRICT_GATE_MIN_SETTLED", 100), 100)
    min_pool = _as_int(getattr(config, "LIVE_READY_BETFAIR_MIN_MODEL_POOL", 2), 2)
    min_passing = _as_int(getattr(config, "LIVE_READY_BETFAIR_MIN_PASSING_MODELS", 2), 2)
    min_pass_rate = _as_float(getattr(config, "LIVE_READY_BETFAIR_MIN_PASS_RATE", 1.0), 1.0)
    min_roi_200 = _as_float(getattr(config, "LIVE_READY_BETFAIR_MIN_AVG_ROI_200_PCT", 0.0), 0.0)
    min_lift_200 = _as_float(getattr(config, "LIVE_READY_BETFAIR_MIN_AVG_BRIER_LIFT_200", 0.0), 0.0)
    min_model_window_settled = _as_int(getattr(config, "LIVE_READY_BETFAIR_MIN_MODEL_WINDOW_SETTLED", 200), 200)
    require_strict_mode = _as_bool(getattr(config, "LIVE_READY_REQUIRE_STRICT_ENFORCEMENT", True), True)

    prediction_mode = str(getattr(config, "PREDICTION_GATE_ENFORCEMENT_MODE", "observe")).lower()
    running = _as_bool(state.get("running"), False)
    feed_ok = _as_bool(health.get("feed_ok"), False)
    prediction_ok = _as_bool(health.get("prediction_ok"), False)
    risk_ok = _as_bool(health.get("risk_ok"), False)
    paper_mode = _as_bool(cfg.get("paper_trading"), _as_bool(getattr(config, "PAPER_TRADING", True), True))

    candidate_models: List[Dict[str, Any]] = []
    passing_models: List[Dict[str, Any]] = []
    for model in models.values():
        if str(model.get("model_kind", "")) == "implied_market":
            continue
        if _as_int(model.get("settled_bets"), 0) < strict_min_settled:
            continue
        candidate_models.append(model)
        if _as_bool(model.get("strict_gate_pass"), False):
            passing_models.append(model)

    pass_rate = (len(passing_models) / len(candidate_models)) if candidate_models else 0.0
    avg_roi_200 = _avg([_as_float((m.get("rolling_200") or {}).get("roi_pct"), 0.0) for m in passing_models])
    avg_lift_200 = _avg([_as_float((m.get("rolling_200") or {}).get("brier_lift_abs"), 0.0) for m in passing_models])
    min_window_settled = min(
        [_as_int((m.get("rolling_200") or {}).get("settled"), 0) for m in passing_models],
        default=0,
    )

    checks = [
        _check("engine_running", running, running, True, "Betfair runtime must be active"),
        _check(
            "health_feed_prediction_risk",
            (feed_ok and prediction_ok and risk_ok),
            {"feed_ok": feed_ok, "prediction_ok": prediction_ok, "risk_ok": risk_ok},
            True,
            "Feed, prediction loop, and risk controls must all be healthy",
        ),
        _check(
            "model_pool_depth",
            len(candidate_models) >= min_pool,
            len(candidate_models),
            f">={min_pool}",
            "Need enough mature non-control models",
        ),
        _check(
            "strict_gate_passing_models",
            len(passing_models) >= min_passing,
            len(passing_models),
            f">={min_passing}",
            "Need enough strict-pass models before live activation",
        ),
        _check(
            "strict_gate_pass_rate",
            pass_rate >= min_pass_rate,
            round(pass_rate, 4),
            f">={min_pass_rate}",
            "Strict-pass rate across mature models is too low",
        ),
        _check(
            "rolling_200_roi",
            avg_roi_200 >= min_roi_200,
            round(avg_roi_200, 4),
            f">={min_roi_200}",
            "Average ROI over rolling-200 settled bets must be non-negative",
        ),
        _check(
            "rolling_200_brier_lift",
            avg_lift_200 > min_lift_200,
            round(avg_lift_200, 6),
            f">{min_lift_200}",
            "Average rolling-200 Brier lift must beat baseline",
        ),
        _check(
            "window_coverage_per_model",
            min_window_settled >= min_model_window_settled,
            min_window_settled,
            f">={min_model_window_settled}",
            "Each passing model needs deep enough rolling-200 coverage",
        ),
        _check(
            "enforcement_mode",
            (prediction_mode == "strict") if require_strict_mode else (prediction_mode in {"soft", "strict"}),
            prediction_mode,
            "strict" if require_strict_mode else "soft|strict",
            "Gate enforcement should be active before switching to live",
        ),
    ]

    validation_ready = all(c["ok"] for c in checks)
    can_switch_to_live = bool(validation_ready and paper_mode)
    blockers = [c["name"] for c in checks if not c["ok"]]
    if validation_ready and not paper_mode:
        blockers.append("already_live_mode")

    return {
        "system": "betfair",
        "mode": "paper" if paper_mode else "live",
        "validation_ready": validation_ready,
        "can_switch_to_live_now": can_switch_to_live,
        "score_pct": _score(checks),
        "confidence": _label_from_score(_score(checks)),
        "candidate_models": len(candidate_models),
        "passing_models": len(passing_models),
        "strict_pass_rate": round(pass_rate, 4),
        "avg_rolling_200_roi_pct": round(avg_roi_200, 4),
        "avg_rolling_200_brier_lift": round(avg_lift_200, 6),
        "checks": checks,
        "blockers": blockers,
    }


def evaluate_binance_live_readiness(state: Dict[str, Any]) -> Dict[str, Any]:
    running = _as_bool(state.get("running"), False)
    ws_connected = _as_bool(state.get("ws_connected"), False)
    trading_halted = _as_bool(state.get("trading_halted"), False)
    mode = str(state.get("mode", getattr(config, "FUNDING_MODE", "paper"))).lower()
    paper_mode = mode == "paper"
    gate_mode = str(getattr(config, "FUNDING_GATE_MODE", "observe")).lower()

    min_pool = _as_int(getattr(config, "LIVE_READY_BINANCE_MIN_MODEL_POOL", 2), 2)
    min_passing = _as_int(getattr(config, "LIVE_READY_BINANCE_MIN_PASSING_MODELS", 2), 2)
    min_pass_rate = _as_float(getattr(config, "LIVE_READY_BINANCE_MIN_PASS_RATE", 1.0), 1.0)
    min_roi_200 = _as_float(getattr(config, "LIVE_READY_BINANCE_MIN_AVG_ROI_200_PCT", 0.0), 0.0)
    min_lift_200 = _as_float(getattr(config, "LIVE_READY_BINANCE_MIN_AVG_BRIER_LIFT_200", 0.0), 0.0)
    min_model_window_settled = _as_int(getattr(config, "LIVE_READY_BINANCE_MIN_MODEL_WINDOW_SETTLED", 200), 200)
    min_realized_roi = _as_float(getattr(config, "LIVE_READY_BINANCE_MIN_REALIZED_ROI_PCT", 0.0), 0.0)
    require_full_gate_mode = _as_bool(getattr(config, "LIVE_READY_REQUIRE_FULL_FUNDING_GATE", True), True)

    learners = []
    ol = state.get("online_learner")
    cl = state.get("contrarian_learner")
    if isinstance(ol, dict):
        learners.append(("funding_online_learner", ol))
    if isinstance(cl, dict):
        learners.append(("contrarian_online_learner", cl))

    candidate_models: List[Dict[str, Any]] = []
    passing_models: List[Dict[str, Any]] = []
    for model_id, learner in learners:
        r200 = learner.get("rolling_200") or {}
        settled = _as_int(r200.get("settled"), 0)
        candidate = {
            "model_id": model_id,
            "strict_gate_pass": _as_bool(learner.get("strict_gate_pass"), False),
            "rolling_200_settled": settled,
            "rolling_200_roi": _as_float(r200.get("roi_pct"), 0.0),
            "rolling_200_brier_lift": _as_float(r200.get("brier_lift_abs"), 0.0),
        }
        candidate_models.append(candidate)
        if candidate["strict_gate_pass"]:
            passing_models.append(candidate)

    pass_rate = (len(passing_models) / len(candidate_models)) if candidate_models else 0.0
    avg_roi_200 = _avg([m["rolling_200_roi"] for m in passing_models])
    avg_lift_200 = _avg([m["rolling_200_brier_lift"] for m in passing_models])
    min_window_settled = min([m["rolling_200_settled"] for m in passing_models], default=0)
    realized_roi_pct = _as_float(state.get("realized_roi_pct"), 0.0)

    checks = [
        _check("engine_running", running, running, True, "Funding runtime must be active"),
        _check("ws_connected", ws_connected, ws_connected, True, "Binance websocket should be connected"),
        _check("risk_not_halted", not trading_halted, trading_halted, False, "Risk circuit breaker must be clear"),
        _check(
            "model_pool_depth",
            len(candidate_models) >= min_pool,
            len(candidate_models),
            f">={min_pool}",
            "Need enough funding learners online",
        ),
        _check(
            "strict_gate_passing_models",
            len(passing_models) >= min_passing,
            len(passing_models),
            f">={min_passing}",
            "Need enough strict-pass funding learners",
        ),
        _check(
            "strict_gate_pass_rate",
            pass_rate >= min_pass_rate,
            round(pass_rate, 4),
            f">={min_pass_rate}",
            "Strict-pass rate across funding learners is too low",
        ),
        _check(
            "rolling_200_roi",
            avg_roi_200 >= min_roi_200,
            round(avg_roi_200, 4),
            f">={min_roi_200}",
            "Average rolling-200 ROI must be non-negative",
        ),
        _check(
            "rolling_200_brier_lift",
            avg_lift_200 > min_lift_200,
            round(avg_lift_200, 6),
            f">{min_lift_200}",
            "Average rolling-200 Brier lift must beat baseline",
        ),
        _check(
            "window_coverage_per_model",
            min_window_settled >= min_model_window_settled,
            min_window_settled,
            f">={min_model_window_settled}",
            "Each passing learner needs rolling-200 depth",
        ),
        _check(
            "realized_roi_floor",
            realized_roi_pct >= min_realized_roi,
            round(realized_roi_pct, 4),
            f">={min_realized_roi}",
            "Realized hedge ROI should not be negative",
        ),
        _check(
            "enforcement_mode",
            (gate_mode == "full") if require_full_gate_mode else (gate_mode in {"soft", "full"}),
            gate_mode,
            "full" if require_full_gate_mode else "soft|full",
            "Funding gate mode should be fully enforced before going live",
        ),
    ]

    validation_ready = all(c["ok"] for c in checks)
    can_switch_to_live = bool(validation_ready and paper_mode)
    blockers = [c["name"] for c in checks if not c["ok"]]
    if validation_ready and not paper_mode:
        blockers.append("already_live_mode")

    return {
        "system": "binance",
        "mode": mode,
        "validation_ready": validation_ready,
        "can_switch_to_live_now": can_switch_to_live,
        "score_pct": _score(checks),
        "confidence": _label_from_score(_score(checks)),
        "candidate_models": len(candidate_models),
        "passing_models": len(passing_models),
        "strict_pass_rate": round(pass_rate, 4),
        "avg_rolling_200_roi_pct": round(avg_roi_200, 4),
        "avg_rolling_200_brier_lift": round(avg_lift_200, 6),
        "realized_roi_pct": round(realized_roi_pct, 4),
        "checks": checks,
        "blockers": blockers,
    }


def evaluate_live_trading_readiness(
    betfair_state: Dict[str, Any],
    funding_state: Dict[str, Any],
) -> Dict[str, Any]:
    betfair = evaluate_betfair_live_readiness(betfair_state or {})
    binance = evaluate_binance_live_readiness(funding_state or {})

    overall_score = round((float(betfair["score_pct"]) + float(binance["score_pct"])) / 2.0, 2)
    validation_ready = bool(betfair["validation_ready"] and binance["validation_ready"])
    can_activate = bool(betfair["can_switch_to_live_now"] and binance["can_switch_to_live_now"])

    blockers: List[str] = []
    if not betfair["can_switch_to_live_now"]:
        blockers.extend([f"betfair:{b}" for b in betfair.get("blockers", [])])
    if not binance["can_switch_to_live_now"]:
        blockers.extend([f"binance:{b}" for b in binance.get("blockers", [])])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_ready": validation_ready,
        "can_activate_live_now": can_activate,
        "score_pct": overall_score,
        "confidence": _label_from_score(overall_score),
        "blockers": blockers,
        "betfair": betfair,
        "binance": binance,
        "activation_message": (
            "Both systems are validated and still in paper mode. You can switch to live."
            if can_activate
            else "Live activation is blocked until all validation checks pass for both systems."
        ),
    }

