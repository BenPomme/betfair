"""
Opportunity scoring API with model fallback.
"""
from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import Any, Dict, Optional

import config
from core.types import Opportunity, ScoredOpportunity
from strategy.features import FeatureVector


def _clip01(value: Decimal) -> Decimal:
    if value < Decimal("0"):
        return Decimal("0")
    if value > Decimal("1"):
        return Decimal("1")
    return value


def _as_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return default


def _load_linear_model() -> Dict[str, Decimal]:
    path = os.getenv("ML_LINEAR_MODEL_PATH", "").strip()
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: _as_decimal(v) for k, v in raw.items()}
    except Exception:
        return {}


_LINEAR_MODEL = _load_linear_model()


def _load_fill_model() -> Dict[str, Decimal]:
    path = config.FILL_MODEL_PATH.strip() if hasattr(config, "FILL_MODEL_PATH") else ""
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: _as_decimal(v) for k, v in raw.items()}
    except Exception:
        return {}


_FILL_MODEL = _load_fill_model()


def _dynamic_threshold(features: FeatureVector) -> Decimal:
    base = _as_decimal(os.getenv("ML_BASE_DECISION_THRESHOLD_EUR", "0.08"), Decimal("0.08"))
    if features.microstructure.in_play:
        base *= _as_decimal(os.getenv("ML_INPLAY_THRESHOLD_MULTIPLIER", "1.4"), Decimal("1.4"))
    volatility = features.microstructure.short_volatility
    if volatility > Decimal("0.03"):
        base *= _as_decimal(os.getenv("ML_HIGH_VOL_THRESHOLD_MULTIPLIER", "1.2"), Decimal("1.2"))
    return base


def _fill_model_score(features: FeatureVector, opportunity: Opportunity) -> Optional[Decimal]:
    """Return fill probability from trained fill model if configured."""
    if not _FILL_MODEL:
        return None
    z = _FILL_MODEL.get("bias", Decimal("0"))
    z += _FILL_MODEL.get("spread_mean", Decimal("0")) * features.microstructure.spread_mean
    z += _FILL_MODEL.get("depth_total_eur", Decimal("0")) * features.microstructure.depth_total_eur
    z += _FILL_MODEL.get("short_volatility", Decimal("0")) * features.microstructure.short_volatility
    z += _FILL_MODEL.get("time_to_start_sec", Decimal("0")) * Decimal(features.microstructure.time_to_start_sec)
    z += _FILL_MODEL.get("in_play", Decimal("0")) * (Decimal("1") if features.microstructure.in_play else Decimal("0"))
    z += _FILL_MODEL.get("total_stake_eur", Decimal("0")) * opportunity.total_stake_eur
    # Approx logistic squash around 0 using linear clipping.
    return _clip01(Decimal("0.5") + (z / Decimal("8")))


def _stake_multiplier(
    edge_score: Decimal,
    fill_prob: Decimal,
    expected_net: Decimal,
    threshold: Decimal,
    prediction_influence: str,
) -> Decimal:
    """
    Convert model confidence + EV into a per-bet stake multiplier.
    Keeps final multiplier in configured bounds.
    """
    if not bool(getattr(config, "ML_STAKE_SIZING_ENABLED", True)):
        return Decimal("1")

    min_mult = max(
        Decimal("0.05"),
        _as_decimal(getattr(config, "ML_STAKE_MIN_MULTIPLIER", Decimal("0.35")), Decimal("0.35")),
    )
    max_mult = max(
        min_mult,
        _as_decimal(getattr(config, "ML_STAKE_MAX_MULTIPLIER", Decimal("1.25")), Decimal("1.25")),
    )

    safe_threshold = max(Decimal("0.01"), threshold)
    # Saturate at 2x threshold so outsized edges do not explode stake sizing.
    ev_norm = _clip01(expected_net / (safe_threshold * Decimal("2")))
    quality = _clip01((edge_score * Decimal("0.60")) + (fill_prob * Decimal("0.40")))
    base = _clip01((quality * Decimal("0.60")) + (ev_norm * Decimal("0.40")))
    mult = min_mult + ((max_mult - min_mult) * base)

    influence = str(prediction_influence or "none")
    if influence == "boosted":
        mult *= Decimal("1.10")
    elif influence == "penalized":
        mult *= Decimal("0.80")
    elif influence == "ignored_insufficient_data":
        mult *= Decimal("0.90")

    if mult < min_mult:
        return min_mult
    if mult > max_mult:
        return max_mult
    return mult


def _apply_prediction_influence(
    edge_score: Decimal,
    fill_prob: Decimal,
    expected_net: Decimal,
    prediction_confidence: Optional[Dict[str, float]],
) -> tuple[Decimal, Decimal, Decimal, str]:
    if prediction_confidence is None:
        return edge_score, fill_prob, expected_net, "none"
    settled = int(prediction_confidence.get("settled_bets", 0))
    brier = _as_decimal(prediction_confidence.get("model_brier", 1.0), Decimal("1.0"))
    if settled < 30:
        return edge_score, fill_prob, expected_net, "ignored_insufficient_data"
    if brier > Decimal("0.28"):
        return edge_score, fill_prob, expected_net, "ignored_insufficient_data"
    edge_vs_market = _as_decimal(prediction_confidence.get("edge_vs_market", 0), Decimal("0"))
    if edge_vs_market > Decimal("0.05"):
        new_edge = _clip01(edge_score + Decimal("0.10"))
        return new_edge, fill_prob, expected_net, "boosted"
    if edge_vs_market < Decimal("-0.03"):
        new_edge = _clip01(edge_score - Decimal("0.15"))
        new_fill = _clip01(fill_prob * Decimal("0.80"))
        new_expected = expected_net * (new_fill / fill_prob) if fill_prob > Decimal("0") else expected_net
        return new_edge, new_fill, new_expected, "penalized"
    return edge_score, fill_prob, expected_net, "none"


def _heuristic_score(
    opportunity: Opportunity,
    features: FeatureVector,
    prediction_confidence: Optional[Dict[str, float]] = None,
) -> ScoredOpportunity:
    spread_penalty = features.microstructure.spread_mean * Decimal("5")
    volatility_penalty = features.microstructure.short_volatility * Decimal("3")
    depth_boost = min(features.microstructure.depth_total_eur / Decimal("2000"), Decimal("0.25"))
    roi_boost = min(opportunity.net_roi_pct * Decimal("8"), Decimal("0.45"))

    edge_score = _clip01(Decimal("0.35") + roi_boost + depth_boost - spread_penalty - volatility_penalty)
    fill_prob = _clip01(Decimal("0.50") + depth_boost - spread_penalty - (volatility_penalty / Decimal("2")))
    fill_model_prob = _fill_model_score(features, opportunity)
    if fill_model_prob is not None:
        fill_prob = fill_model_prob
    slippage_factor = _clip01(Decimal("0.95") - spread_penalty - (volatility_penalty / Decimal("2")))
    expected_net = opportunity.net_profit_eur * fill_prob * slippage_factor
    edge_score, fill_prob, expected_net, prediction_influence = _apply_prediction_influence(
        edge_score=edge_score,
        fill_prob=fill_prob,
        expected_net=expected_net,
        prediction_confidence=prediction_confidence,
    )

    threshold = _dynamic_threshold(features)
    min_fill_prob = _as_decimal(os.getenv("ML_MIN_FILL_PROB", "0.45"), Decimal("0.45"))
    stake_multiplier = _stake_multiplier(
        edge_score=edge_score,
        fill_prob=fill_prob,
        expected_net=expected_net,
        threshold=threshold,
        prediction_influence=prediction_influence,
    )
    adjusted_expected_net = expected_net * stake_multiplier

    if fill_prob < min_fill_prob:
        decision = "SKIP"
        policy = "defer"
        reason = "fill_prob_below_min"
    elif adjusted_expected_net >= threshold:
        decision = "EXECUTE"
        policy = "best" if edge_score >= Decimal("0.65") else "improve"
        reason = "expected_value_pass"
    else:
        decision = "DEFER"
        policy = "defer"
        reason = "expected_value_below_threshold"

    ttl_seconds = int(_as_decimal(os.getenv("EXECUTION_TTL_SECONDS", "4"), Decimal("4")))
    return ScoredOpportunity(
        edge_score=edge_score,
        fill_prob=fill_prob,
        expected_net_profit_eur=adjusted_expected_net,
        decision=decision,
        dynamic_threshold_eur=threshold,
        model_version="heuristic_v1",
        confidence=edge_score,
        order_policy=policy,
        ttl_seconds=ttl_seconds,
        reason=reason,
        prediction_influence=prediction_influence,
        stake_multiplier=stake_multiplier,
    )


def _linear_score(
    opportunity: Opportunity,
    features: FeatureVector,
    prediction_confidence: Optional[Dict[str, float]] = None,
) -> ScoredOpportunity:
    # Optional linear model loaded from JSON (weights + bias).
    if not _LINEAR_MODEL:
        return _heuristic_score(opportunity, features, prediction_confidence=prediction_confidence)

    # These names are intentionally stable to allow model artifact upgrades.
    score_raw = _LINEAR_MODEL.get("bias", Decimal("0"))
    score_raw += _LINEAR_MODEL.get("roi", Decimal("0")) * opportunity.net_roi_pct
    score_raw += _LINEAR_MODEL.get("profit", Decimal("0")) * opportunity.net_profit_eur
    score_raw += _LINEAR_MODEL.get("depth", Decimal("0")) * features.microstructure.depth_total_eur
    score_raw += _LINEAR_MODEL.get("spread", Decimal("0")) * features.microstructure.spread_mean
    score_raw += _LINEAR_MODEL.get("volatility", Decimal("0")) * features.microstructure.short_volatility

    # Logistic squashing without float dependencies.
    fill_prob = _clip01(Decimal("0.5") + (score_raw / Decimal("8")))
    fill_model_prob = _fill_model_score(features, opportunity)
    if fill_model_prob is not None:
        fill_prob = fill_model_prob
    expected_net = opportunity.net_profit_eur * fill_prob
    edge_score = _clip01(score_raw)
    edge_score, fill_prob, expected_net, prediction_influence = _apply_prediction_influence(
        edge_score=edge_score,
        fill_prob=fill_prob,
        expected_net=expected_net,
        prediction_confidence=prediction_confidence,
    )
    threshold = _dynamic_threshold(features)

    stake_multiplier = _stake_multiplier(
        edge_score=edge_score,
        fill_prob=fill_prob,
        expected_net=expected_net,
        threshold=threshold,
        prediction_influence=prediction_influence,
    )
    adjusted_expected_net = expected_net * stake_multiplier

    if fill_prob < _as_decimal(os.getenv("ML_MIN_FILL_PROB", "0.45"), Decimal("0.45")):
        decision = "SKIP"
        policy = "defer"
        reason = "model_low_fill_prob"
    elif adjusted_expected_net >= threshold:
        decision = "EXECUTE"
        policy = "best"
        reason = "model_expected_value_pass"
    else:
        decision = "DEFER"
        policy = "defer"
        reason = "model_expected_value_below_threshold"

    return ScoredOpportunity(
        edge_score=edge_score,
        fill_prob=fill_prob,
        expected_net_profit_eur=adjusted_expected_net,
        decision=decision,
        dynamic_threshold_eur=threshold,
        model_version="linear_v1",
        confidence=_clip01(score_raw),
        order_policy=policy,
        ttl_seconds=int(_as_decimal(os.getenv("EXECUTION_TTL_SECONDS", "4"), Decimal("4"))),
        reason=reason,
        prediction_influence=prediction_influence,
        stake_multiplier=stake_multiplier,
    )


def score_opportunity(
    opportunity: Opportunity,
    features: FeatureVector,
    prediction_confidence: Optional[Dict[str, float]] = None,
) -> ScoredOpportunity:
    """
    Main inference API.
    Falls back to deterministic heuristic when no model artifact is configured.
    """
    if os.getenv("ML_SCORING_ENABLED", "true").lower() != "true":
        return ScoredOpportunity(
            edge_score=Decimal("1"),
            fill_prob=Decimal("1"),
            expected_net_profit_eur=opportunity.net_profit_eur,
            decision="EXECUTE",
            dynamic_threshold_eur=Decimal("0"),
            model_version="disabled",
            confidence=Decimal("1"),
            order_policy="best",
            ttl_seconds=0,
            reason="scoring_disabled",
            prediction_influence="none",
            stake_multiplier=Decimal("1"),
        )
    return _linear_score(opportunity, features, prediction_confidence=prediction_confidence)
