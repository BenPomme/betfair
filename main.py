"""
Main entry: login -> get markets -> poll prices -> scan -> execute (paper or live).
Runs a single async loop: price poller task + scan/execute loop. No stream required.
"""
import asyncio
import logging
import os
import signal
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Callable, Dict, List, Optional

import config
from data.betfair_client import create_and_login
from data.candidate_logger import CandidateLogger, build_scan_record
from data.clv_tracker import CLVTracker
from data.price_cache import PriceCache
from data.price_poller import run_price_poller
from data.market_catalogue import discover_markets
from core.scanner import scan_snapshot, scan_snapshot_lay
from core.types import Opportunity, ScoredOpportunity
from core.cross_market_scanner import (
    scan_cross_market,
    scan_cross_market_btts,
    scan_cross_market_cs_mo,
    scan_cross_market_ou25,
)
from core.risk_manager import RiskManager
from data.event_grouper import group_by_event, get_cross_market_pairs
from execution.executor import execute_opportunity, init_live_executor
from execution.order_monitor import OrderMonitor
from execution.paper_executor import PaperExecutor
from monitoring.alerting import alert_daily_summary, alert_prediction_bet, alert_trade_executed
from strategy.features import build_feature_vector
from strategy.model_inference import score_opportunity
from strategy.learning_architect import LearningArchitect
from strategy.prediction_engine import MultiModelPredictionManager
from qa.live_qa_agent import LiveQAAgent
from strategy.audit_agent import AuditAgent
from betfair.strategies.information_arb_manager import BetfairInformationArbManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set to False on shutdown so poller and loop exit
_running = True


def _shutdown() -> None:
    global _running
    _running = False


def _parse_market_start(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _extract_best_prediction(
    prediction_manager: Optional[MultiModelPredictionManager],
    snapshot,
    market_start: Optional[datetime] = None,
) -> Optional[Dict[str, float]]:
    """
    Build prediction confidence via inverse-Brier-weighted ensemble.
    Uses two trust tiers:
    - trusted: strict-pass mature models
    - exploratory: early models with positive learning evidence
    """
    if prediction_manager is None or snapshot is None or not snapshot.selections:
        return None
    states = prediction_manager.initial_state()
    trusted_candidates = [
        (model_id, st)
        for model_id, st in states.items()
        if int(st.get("settled_bets", 0)) >= 30
        and float(st.get("avg_brier", 1.0)) < 0.28
        and str(st.get("model_kind", "")) != "implied_market"
        and bool(st.get("strict_gate_pass", False))
    ]
    exploratory_candidates = [
        (model_id, st)
        for model_id, st in states.items()
        if int(st.get("learning_settled", 0) or st.get("settled_bets", 0) or 0) >= 10
        and float(st.get("avg_brier", 1.0)) < 0.40
        and str(st.get("model_kind", "")) != "implied_market"
        and (
            float(st.get("recent_learning_brier_lift", 0.0) or 0.0) > 0.0
            or float(st.get("recent_brier_lift", 0.0) or 0.0) > 0.0
        )
    ]
    tier = "trusted"
    candidates = trusted_candidates
    if not candidates:
        tier = "exploratory"
        candidates = exploratory_candidates
    if not candidates:
        return None
    backable = [s for s in snapshot.selections if s.best_back_price > Decimal("1.01")]
    if not backable:
        return None
    best_sel = max(backable, key=lambda s: s.available_to_back)
    best_back_odds = float(best_sel.best_back_price)
    implied_prob = 1.0 / best_back_odds

    # Collect predictions from all calibrated models with inverse-Brier weights
    weighted_sum = 0.0
    weight_total = 0.0
    model_predictions = []
    for model_id, st in candidates:
        engine = prediction_manager.engines.get(model_id)
        if engine is None:
            continue
        brier = float(st.get("avg_brier", 1.0))
        inv_brier = 1.0 / max(brier, 0.001)  # Inverse Brier as weight
        predicted_prob = float(
            engine._predict_prob(
                implied_prob,
                engine._features_from_snapshot(snapshot, market_start),
            )
        )  # noqa: SLF001
        weighted_sum += predicted_prob * inv_brier
        weight_total += inv_brier
        model_predictions.append((model_id, predicted_prob, brier))

    if weight_total <= 0 or not model_predictions:
        return None

    ensemble_prob = weighted_sum / weight_total
    # Best individual model for reporting
    best_model_id, _, best_brier = min(model_predictions, key=lambda x: x[2])
    avg_ensemble_brier = sum(b for _, _, b in model_predictions) / len(model_predictions)

    return {
        "best_model_id": best_model_id,
        "predicted_prob": ensemble_prob,
        "edge_vs_market": ensemble_prob - implied_prob,
        "model_brier": avg_ensemble_brier,
        "settled_bets": sum(int(st.get("settled_bets", 0)) for _, st in candidates),
        "ensemble_size": len(model_predictions),
        "tier": tier,
        "strict_models": len(trusted_candidates),
        "exploratory_models": len(exploratory_candidates),
    }


def _round_eur(amount: Decimal) -> Decimal:
    return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _round_pnl(amount: Decimal) -> Decimal:
    return amount.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def _effective_base_stake(balance: Decimal) -> Decimal:
    stake = balance * config.STAKE_FRACTION
    if stake > config.MAX_STAKE_EUR:
        stake = config.MAX_STAKE_EUR
    return _round_eur(stake)


def _scale_opportunity_to_total_stake(
    opportunity: Opportunity,
    target_total_stake_eur: Decimal,
) -> Opportunity:
    """
    Scale all stake-bearing fields so execution can use dynamic per-bet sizing.
    Profits are scaled proportionally to keep ROI stable.
    """
    current_total = Decimal(str(opportunity.total_stake_eur))
    if current_total <= Decimal("0"):
        return opportunity
    if target_total_stake_eur <= Decimal("0"):
        return opportunity

    raw_scale = target_total_stake_eur / current_total
    if raw_scale <= Decimal("0"):
        return opportunity

    # Keep scaled stakes within top-of-book liquidity if provided.
    max_scale = raw_scale
    for sel in opportunity.selections:
        if "stake_eur" not in sel or "liquidity_eur" not in sel:
            continue
        try:
            stake = Decimal(str(sel["stake_eur"]))
            liquidity = Decimal(str(sel["liquidity_eur"]))
        except Exception:
            continue
        if stake > Decimal("0") and liquidity > Decimal("0"):
            liq_scale = liquidity / stake
            if liq_scale < max_scale:
                max_scale = liq_scale

    if max_scale <= Decimal("0"):
        return opportunity

    scale = min(raw_scale, max_scale)
    if abs(scale - Decimal("1")) < Decimal("0.001"):
        return opportunity

    new_total = _round_eur(current_total * scale)
    if new_total <= Decimal("0"):
        return opportunity
    scale = new_total / current_total

    scaled_selections = []
    for sel in opportunity.selections:
        out = dict(sel)
        for key in ("stake_eur", "lay_stake_eur", "liability_eur"):
            if key not in out or out[key] is None:
                continue
            try:
                out[key] = float(_round_eur(Decimal(str(out[key])) * scale))
            except Exception:
                continue
        scaled_selections.append(out)

    return Opportunity(
        market_id=opportunity.market_id,
        event_name=opportunity.event_name,
        market_start=opportunity.market_start,
        arb_type=opportunity.arb_type,
        selections=tuple(scaled_selections),
        total_stake_eur=new_total,
        overround_raw=opportunity.overround_raw,
        gross_profit_eur=_round_pnl(opportunity.gross_profit_eur * scale),
        commission_eur=_round_pnl(opportunity.commission_eur * scale),
        net_profit_eur=_round_pnl(opportunity.net_profit_eur * scale),
        net_roi_pct=opportunity.net_roi_pct,
        liquidity_by_selection=opportunity.liquidity_by_selection,
    )


async def run_loop(
    market_ids: List[str],
    price_cache: PriceCache,
    risk_manager: RiskManager,
    paper_executor: PaperExecutor,
    scan_interval_seconds: float = 2.0,
    on_scan: Optional[Callable] = None,
    on_opportunity: Optional[Callable[..., None]] = None,
    on_trade: Optional[Callable[..., None]] = None,
    on_prediction: Optional[Callable[..., None]] = None,
    on_architect: Optional[Callable[..., None]] = None,
    on_strategy_state: Optional[Callable[..., None]] = None,
    market_metadata: Optional[dict] = None,
    candidate_logger: Optional[CandidateLogger] = None,
    prediction_manager: Optional[MultiModelPredictionManager] = None,
    info_arb_manager: Optional[BetfairInformationArbManager] = None,
    learning_architect: Optional[LearningArchitect] = None,
    poller_metrics_provider: Optional[Callable[[], dict]] = None,
) -> None:
    """Scan/execute loop: every scan_interval_seconds check each market for arbs and execute if allowed."""
    meta = market_metadata or {}

    # Track executed cross-market pairs and value bets to avoid repeated trades
    _executed_cross_pairs: set = set()
    _executed_value_bets: set = set()
    _prev_snapshots: Dict[str, object] = {}
    _market_last_fingerprint: Dict[str, tuple] = {}
    _market_last_motion_ts: Dict[str, float] = {}
    _market_missing_cycles: Dict[str, int] = {}
    market_no_movement_seconds = max(0, int(getattr(config, "MARKET_NO_MOVEMENT_SECONDS", 900)))
    market_missing_retire_cycles = max(1, int(getattr(config, "MARKET_MISSING_RETIRE_CYCLES", 120)))

    def _execute_opp(opp, scored, on_trade_cb, features=None):
        """Common path: check risk, execute, log."""
        if not risk_manager.can_execute(opp):
            if candidate_logger:
                candidate_logger.log(
                    build_scan_record(
                        market_id=opp.market_id,
                        event_name=opp.event_name,
                        has_snapshot=True,
                        reason="risk_blocked",
                        opportunity=opp,
                        scored=scored,
                        features=features,
                        executed=False,
                    )
                )
            return
        result = execute_opportunity(opp, paper_executor=paper_executor, scored=scored)
        if result:
            try:
                alert_trade_executed(
                    {
                        "arb_type": opp.arb_type,
                        "event_name": opp.event_name,
                        "net_profit_eur": float(opp.net_profit_eur),
                        "net_roi_pct": float(opp.net_roi_pct),
                    },
                    result,
                    {
                        "edge_score": float(getattr(scored, "edge_score", 0)),
                        "fill_prob": float(getattr(scored, "fill_prob", 0)),
                        "prediction_influence": getattr(scored, "prediction_influence", "none"),
                    },
                )
            except Exception:
                pass
            if on_trade_cb:
                try:
                    on_trade_cb(opp, result, scored)
                except Exception:
                    pass
            logger.info(
                "Opportunity executed: market_id=%s net_profit_eur=%s arb_type=%s decision=%s",
                opp.market_id,
                result.get("net_profit_eur"),
                opp.arb_type,
                getattr(scored, "decision", "EXECUTE"),
            )
            risk_manager.register_execution(opp, opp.net_profit_eur)
            if candidate_logger:
                candidate_logger.log(
                    build_scan_record(
                        market_id=opp.market_id,
                        event_name=opp.event_name,
                        has_snapshot=True,
                        reason="executed",
                        opportunity=opp,
                        scored=scored,
                        features=features,
                        executed=True,
                    )
                )

    while _running:
        if not market_ids:
            await asyncio.sleep(max(1.0, scan_interval_seconds))
            continue
        if info_arb_manager is not None:
            try:
                strategy_state = await info_arb_manager.evaluate_cycle(
                    market_ids=market_ids,
                    market_metadata=meta,
                    price_cache=price_cache,
                    candidate_logger=candidate_logger,
                )
                if on_strategy_state:
                    on_strategy_state(strategy_state)
            except Exception as e:
                logger.exception("Information-arbitrage cycle error: %s", e)
        poller_metrics = poller_metrics_provider() if poller_metrics_provider else {}
        cycle_sec = float((poller_metrics or {}).get("cycle_duration_sec", 0.0) or 0.0)
        dynamic_prematch_stale = max(
            int(config.STALE_PRICE_SECONDS_PREMATCH),
            int(cycle_sec * 3.0 + scan_interval_seconds * 2.0),
        )
        dynamic_inplay_stale = max(
            int(config.STALE_PRICE_SECONDS_INPLAY),
            int(cycle_sec * 2.0 + scan_interval_seconds * 1.5),
        )
        # Dynamic base stake: fraction of current balance, bounded by max stake.
        effective_stake = _effective_base_stake(paper_executor.balance)
        ml_stake_min_eur = Decimal(str(getattr(config, "ML_STAKE_MIN_EUR", Decimal("2.00"))))
        if effective_stake < ml_stake_min_eur:
            await asyncio.sleep(scan_interval_seconds)
            continue

        def _sized_for_execution(
            opp: Opportunity,
            scored: Optional[ScoredOpportunity],
        ) -> Opportunity:
            if scored is None or not bool(getattr(config, "ML_STAKE_SIZING_ENABLED", True)):
                return opp
            try:
                multiplier = Decimal(str(scored.stake_multiplier))
            except Exception:
                multiplier = Decimal("1")
            if multiplier <= Decimal("0"):
                return opp
            target = _round_eur(effective_stake * multiplier)
            if target > config.MAX_STAKE_EUR:
                target = config.MAX_STAKE_EUR
            balance_cap = _round_eur(paper_executor.balance)
            if target > balance_cap:
                target = balance_cap
            if target < ml_stake_min_eur:
                return opp
            return _scale_opportunity_to_total_stake(opp, target)

        # --- Single-market scans (back-back, lay-lay) ---
        markets_to_retire: set = set()
        for market_id in list(market_ids):
            if not _running:
                break
            try:
                m_meta = meta.get(market_id, {})
                event_name = m_meta.get("event_name", "")
                market_start = _parse_market_start(m_meta.get("market_start"))

                snapshot = price_cache.get_prices_by_regime(
                    market_id=market_id,
                    market_start=market_start,
                    stale_prematch_seconds=dynamic_prematch_stale,
                    stale_inplay_seconds=dynamic_inplay_stale,
                )
                if on_scan:
                    try:
                        on_scan(market_id, snapshot is not None)
                    except Exception:
                        pass

                if snapshot is None:
                    missing = _market_missing_cycles.get(market_id, 0) + 1
                    _market_missing_cycles[market_id] = missing
                    if candidate_logger:
                        candidate_logger.log(
                            build_scan_record(
                                market_id=market_id,
                                event_name=event_name,
                                has_snapshot=False,
                                reason="stale_or_missing_snapshot",
                            )
                        )
                    if missing >= market_missing_retire_cycles:
                        markets_to_retire.add(market_id)
                        if candidate_logger:
                            candidate_logger.log(
                                build_scan_record(
                                    market_id=market_id,
                                    event_name=event_name,
                                    has_snapshot=False,
                                    reason="retired_missing_snapshot",
                                )
                            )
                    continue
                _market_missing_cycles[market_id] = 0

                if prediction_manager is not None:
                    try:
                        prediction_payload = prediction_manager.process_snapshot(
                            market_id=market_id,
                            snapshot=snapshot,
                            event_name=event_name,
                            market_start=market_start,
                        )
                        if on_prediction:
                            on_prediction(prediction_payload)
                        try:
                            alert_prediction_bet(prediction_payload)
                        except Exception:
                            pass
                    except Exception as e:
                        logger.exception("Prediction engine error for %s: %s", market_id, e)
                prediction_confidence = _extract_best_prediction(prediction_manager, snapshot, market_start=market_start)

                # --- Value betting from prediction ensemble ---
                if (
                    config.VALUE_BET_ENABLED
                    and prediction_confidence is not None
                    and int(prediction_confidence.get("ensemble_size", 0)) >= config.VALUE_BET_MIN_ENSEMBLE_SIZE
                    and float(prediction_confidence.get("model_brier", 1.0)) < float(config.VALUE_BET_MAX_BRIER)
                    and float(prediction_confidence.get("edge_vs_market", 0)) >= float(config.VALUE_BET_MIN_EDGE)
                    and market_id not in _executed_value_bets
                ):
                    try:
                        edge = float(prediction_confidence["edge_vs_market"])
                        pred_prob = float(prediction_confidence["predicted_prob"])
                        tier = str(prediction_confidence.get("tier", "trusted"))
                        exploratory = tier != "trusted"
                        # Find best backable selection (most liquid)
                        vb_backable = [s for s in snapshot.selections if s.best_back_price > Decimal("1.01")]
                        if vb_backable:
                            vb_sel = max(vb_backable, key=lambda s: s.available_to_back)
                            vb_odds = float(vb_sel.best_back_price)
                            # Fractional Kelly: f* = (bp - q) / b where b=odds-1, p=pred_prob, q=1-p
                            b = vb_odds - 1.0
                            kelly_f = (b * pred_prob - (1.0 - pred_prob)) / b if b > 0 else 0
                            kelly_f = max(0.0, kelly_f) * float(config.VALUE_BET_KELLY_FRACTION)
                            if exploratory:
                                if edge < max(float(config.VALUE_BET_MIN_EDGE) + 0.02, 0.07):
                                    raise ValueError("exploratory_edge_too_small")
                                kelly_f *= 0.35
                            vb_stake = Decimal(str(round(float(paper_executor.balance) * kelly_f, 2)))
                            vb_stake = min(vb_stake, config.VALUE_BET_MAX_STAKE_EUR)
                            if vb_stake >= config.VALUE_BET_MIN_STAKE_EUR:
                                vb_opp = Opportunity(
                                    market_id=market_id,
                                    event_name=event_name,
                                    market_start=market_start,
                                    arb_type="value_bet",
                                    selections=(
                                        {
                                            "name": vb_sel.name,
                                            "back_price": float(vb_sel.best_back_price),
                                            "stake_eur": float(vb_stake),
                                            "selection_id": vb_sel.selection_id,
                                            "predicted_prob": pred_prob,
                                            "edge": edge,
                                        },
                                    ),
                                    total_stake_eur=vb_stake,
                                    overround_raw=Decimal("0"),
                                    gross_profit_eur=vb_stake * (Decimal(str(vb_odds)) - Decimal("1")),
                                    commission_eur=Decimal("0"),
                                    net_profit_eur=vb_stake * (Decimal(str(vb_odds)) - Decimal("1")),
                                    net_roi_pct=Decimal(str(round(edge * 100, 2))),
                                    liquidity_by_selection=(vb_sel.available_to_back,),
                                )
                                if risk_manager.can_execute(vb_opp):
                                    vb_result = execute_opportunity(vb_opp, paper_executor=paper_executor, scored=None)
                                    if vb_result:
                                        _executed_value_bets.add(market_id)
                                        logger.info(
                                            "Value bet executed: market_id=%s event=%s odds=%.2f edge=%.4f stake=%.2f",
                                            market_id, event_name, vb_odds, edge, float(vb_stake),
                                        )
                                        try:
                                            alert_trade_executed(
                                                {"arb_type": "value_bet", "event_name": event_name,
                                                 "net_profit_eur": float(vb_opp.net_profit_eur),
                                                 "net_roi_pct": float(vb_opp.net_roi_pct)},
                                                vb_result,
                                                {"edge_score": edge, "fill_prob": 1.0,
                                                 "prediction_influence": "ensemble_value_bet"},
                                            )
                                        except Exception:
                                            pass
                                        if on_trade:
                                            try:
                                                on_trade(vb_opp, vb_result, None)
                                            except Exception:
                                                pass
                                        risk_manager.register_execution(vb_opp, vb_opp.net_profit_eur)
                    except Exception as e:
                        logger.exception("Value bet error for %s: %s", market_id, e)

                market_status = str(getattr(snapshot, "market_status", "OPEN") or "OPEN").upper()
                if market_status == "CLOSED":
                    markets_to_retire.add(market_id)
                    if candidate_logger:
                        candidate_logger.log(
                            build_scan_record(
                                market_id=market_id,
                                event_name=event_name,
                                has_snapshot=True,
                                reason="retired_market_closed",
                                snapshot=snapshot,
                            )
                        )
                    continue

                fingerprint = tuple(
                    (
                        s.selection_id,
                        str(s.best_back_price),
                        str(s.best_lay_price),
                        str(s.available_to_back),
                        str(s.available_to_lay),
                        str(getattr(s, "runner_status", "")),
                    )
                    for s in snapshot.selections
                )
                now_ts = time.time()
                if _market_last_fingerprint.get(market_id) != fingerprint:
                    _market_last_fingerprint[market_id] = fingerprint
                    _market_last_motion_ts[market_id] = now_ts
                elif market_no_movement_seconds > 0:
                    idle_for = now_ts - _market_last_motion_ts.get(market_id, now_ts)
                    if idle_for >= market_no_movement_seconds:
                        markets_to_retire.add(market_id)
                        if candidate_logger:
                            candidate_logger.log(
                                build_scan_record(
                                    market_id=market_id,
                                    event_name=event_name,
                                    has_snapshot=True,
                                    reason="retired_no_movement",
                                    snapshot=snapshot,
                                )
                            )
                        continue

                if learning_architect is not None and prediction_manager is not None:
                    try:
                        decision = learning_architect.evaluate_and_apply(prediction_manager)
                        if on_architect:
                            on_architect({
                                "ts": decision.ts,
                                "mode": decision.mode,
                                "applied": decision.applied,
                                "reason": decision.reason,
                                "proposals": decision.proposals,
                            })
                    except Exception as e:
                        logger.exception("Learning architect error: %s", e)

                back_opp = scan_snapshot(
                    snapshot,
                    event_name=event_name,
                    market_start=market_start,
                    max_stake_eur=effective_stake,
                )
                lay_opp = scan_snapshot_lay(
                    snapshot,
                    event_name=event_name,
                    market_start=market_start,
                    max_stake_eur=effective_stake,
                )
                opp = None
                if back_opp and lay_opp:
                    opp = back_opp if back_opp.net_profit_eur >= lay_opp.net_profit_eur else lay_opp
                else:
                    opp = back_opp or lay_opp

                if opp is None:
                    if candidate_logger:
                        candidate_logger.log(
                            build_scan_record(
                                market_id=market_id,
                                event_name=event_name,
                                has_snapshot=True,
                                reason="no_arb_after_filters",
                                snapshot=snapshot,
                            )
                        )
                    _prev_snapshots[market_id] = snapshot
                    continue

                features = build_feature_vector(
                    snapshot=snapshot,
                    opportunity=opp,
                    market_start=market_start,
                    previous_snapshot=_prev_snapshots.get(market_id),
                )
                scored = score_opportunity(opp, features, prediction_confidence=prediction_confidence)
                _prev_snapshots[market_id] = snapshot

                if on_opportunity:
                    try:
                        on_opportunity(opp, scored)
                    except Exception:
                        pass

                if scored.decision != "EXECUTE":
                    if candidate_logger:
                        candidate_logger.log(
                            build_scan_record(
                                market_id=market_id,
                                event_name=event_name,
                                has_snapshot=True,
                                reason=f"decision_{scored.decision.lower()}",
                                snapshot=snapshot,
                                opportunity=opp,
                                scored=scored,
                                features=features,
                                executed=False,
                            )
                        )
                    continue

                sized_opp = _sized_for_execution(opp, scored)
                _execute_opp(sized_opp, scored, on_trade, features=features)
            except Exception as e:
                logger.exception("Scan/execute error for %s: %s", market_id, e)

        if markets_to_retire:
            pending_market_ids = set(prediction_manager.pending_market_ids()) if prediction_manager is not None else set()
            removable = [mid for mid in markets_to_retire if mid not in pending_market_ids]
            if removable:
                removable_set = set(removable)
                market_ids[:] = [mid for mid in market_ids if mid not in removable_set]
                for mid in removable_set:
                    meta.pop(mid, None)
                    _prev_snapshots.pop(mid, None)
                    _market_last_fingerprint.pop(mid, None)
                    _market_last_motion_ts.pop(mid, None)
                    _market_missing_cycles.pop(mid, None)
                logger.info("Retired %d inactive markets; active universe=%d", len(removable_set), len(market_ids))

        # --- Cross-market scans (MATCH_ODDS vs DRAW_NO_BET) ---
        if config.CROSS_MARKET_ENABLED and meta:
            try:
                active_meta = {mid: meta[mid] for mid in market_ids if mid in meta}
                event_groups = group_by_event(active_meta)
                for event_id, event_market_ids in event_groups.items():
                    if not _running:
                        break
                    pairs = get_cross_market_pairs(event_market_ids, active_meta)
                    for market_id_a, market_id_b, pair_type in pairs:
                        snap_a = price_cache.get_prices(market_id_a)
                        snap_b = price_cache.get_prices(market_id_b)
                        if snap_a is None or snap_b is None:
                            continue
                        if str(getattr(snap_a, "market_status", "OPEN") or "OPEN").upper() != "OPEN":
                            continue
                        if str(getattr(snap_b, "market_status", "OPEN") or "OPEN").upper() != "OPEN":
                            continue
                        m_meta = active_meta.get(market_id_a, {})
                        event_name = m_meta.get("event_name", "")
                        market_start = _parse_market_start(m_meta.get("market_start"))
                        pair_key = f"{market_id_a}+{market_id_b}:{pair_type}"
                        if pair_key in _executed_cross_pairs:
                            continue

                        opp = None
                        feature_snapshot = snap_a
                        if pair_type == "mo_dnb":
                            if not config.CROSS_MARKET_MO_DNB_ENABLED:
                                continue
                            opp = scan_cross_market(
                                snap_a,
                                snap_b,
                                event_name=event_name,
                                market_start=market_start,
                                max_stake_eur=effective_stake,
                            )
                        elif pair_type == "mo_ou25":
                            if not config.CROSS_MARKET_MO_OU25_ENABLED:
                                continue
                            opp = scan_cross_market_ou25(
                                snap_a,
                                snap_b,
                                event_name=event_name,
                                market_start=market_start,
                                max_stake_eur=effective_stake,
                            )
                        elif pair_type == "mo_btts":
                            if not config.CROSS_MARKET_MO_BTTS_ENABLED:
                                continue
                            opp = scan_cross_market_btts(
                                snap_a,
                                snap_b,
                                event_name=event_name,
                                market_start=market_start,
                                max_stake_eur=effective_stake,
                            )
                        elif pair_type == "cs_mo":
                            if not config.CROSS_MARKET_CS_MO_ENABLED:
                                continue
                            opp = scan_cross_market_cs_mo(
                                snap_a,
                                snap_b,
                                event_name=event_name,
                                market_start=market_start,
                                max_stake_eur=effective_stake,
                            )
                        else:
                            continue

                        if opp is not None:
                            features = build_feature_vector(
                                snapshot=feature_snapshot,
                                opportunity=opp,
                                market_start=market_start,
                                previous_snapshot=_prev_snapshots.get(feature_snapshot.market_id),
                            )
                            prediction_confidence = _extract_best_prediction(
                                prediction_manager,
                                feature_snapshot,
                                market_start=market_start,
                            )
                            scored = score_opportunity(opp, features, prediction_confidence=prediction_confidence)
                            if on_opportunity:
                                try:
                                    on_opportunity(opp, scored)
                                except Exception:
                                    pass
                            if scored.decision != "EXECUTE":
                                if candidate_logger:
                                    candidate_logger.log(
                                        build_scan_record(
                                            market_id=opp.market_id,
                                            event_name=event_name,
                                            has_snapshot=True,
                                            reason=f"decision_{scored.decision.lower()}",
                                            snapshot=feature_snapshot,
                                            opportunity=opp,
                                            scored=scored,
                                            features=features,
                                            executed=False,
                                        )
                                    )
                                continue
                            _executed_cross_pairs.add(pair_key)
                            sized_opp = _sized_for_execution(opp, scored)
                            _execute_opp(sized_opp, scored, on_trade, features=features)
            except Exception as e:
                logger.exception("Cross-market scan error: %s", e)

        # Settle pending prediction positions for markets that are no longer in the
        # active scan universe (e.g. after restart with a different market set).
        if prediction_manager is not None:
            try:
                watched_market_ids = set(market_ids)
                pending_outside_scan = [
                    mid for mid in prediction_manager.pending_market_ids()
                    if mid not in watched_market_ids
                ]
                if pending_outside_scan:
                    settlement_payload = prediction_manager.process_pending_settlements(
                        pending_outside_scan,
                        price_cache=price_cache,
                    )
                    if on_prediction:
                        on_prediction(settlement_payload)
                    try:
                        alert_prediction_bet(settlement_payload)
                    except Exception:
                        pass
            except Exception as e:
                logger.exception("Prediction settlement watcher error: %s", e)

        await asyncio.sleep(scan_interval_seconds)


def main() -> None:
    """Login, get markets, run poller + scan loop. Paper mode only until gate passed."""
    if not config.PAPER_TRADING:
        logger.error("PAPER_TRADING must be true. Refusing to run.")
        return

    # Login
    try:
        client = create_and_login()
    except Exception as e:
        logger.exception("Betfair login failed: %s", e)
        return

    # When PAPER_TRADING=false, call init_live_executor(client) here before starting the loop.

    # Market list: env MARKET_IDS or broad multi-sport discovery
    market_ids_str = os.getenv("MARKET_IDS", "")
    market_ids = [m.strip() for m in market_ids_str.split(",") if m.strip()]
    market_metadata = {}
    runner_names = {}

    if not market_ids:
        try:
            market_ids, market_metadata, runner_names = discover_markets(
                client,
                max_total=config.SCAN_MAX_MARKETS,
                include_in_play=config.SCAN_INCLUDE_IN_PLAY,
            )
        except Exception as e:
            logger.exception("Market discovery failed: %s", e)
            client.logout()
            return

    if not market_ids:
        logger.warning("No markets to watch. Set MARKET_IDS in .env or try when markets are open.")
        client.logout()
        return

    logger.info("Watching %d markets: %s", len(market_ids), market_ids[:5])

    price_cache = PriceCache(max_age_seconds=config.STALE_PRICE_SECONDS)
    risk_manager = RiskManager(
        max_stake_eur=config.MAX_STAKE_EUR,
        daily_loss_limit_eur=config.DAILY_LOSS_LIMIT_EUR,
    )
    paper_executor = PaperExecutor(
        initial_balance_eur=config.INITIAL_BALANCE_EUR,
        state_path=config.PAPER_STATE_PATH,
        trades_log_path=config.PAPER_TRADES_LOG_PATH,
    )
    candidate_logger = CandidateLogger(config.CANDIDATE_LOG_DIR) if config.CANDIDATE_LOG_ENABLED else None
    prediction_manager = None
    info_arb_manager = BetfairInformationArbManager() if getattr(config, "BETFAIR_EXTERNAL_SIGNALS_ENABLED", True) else None
    learning_architect = LearningArchitect() if config.ARCHITECT_ENABLED else None
    qa_agent = LiveQAAgent() if config.QA_AGENT_ENABLED else None
    audit_agent = AuditAgent()
    clv_tracker = CLVTracker(config.CLV_LOG_DIR) if config.CLV_ENABLED else None
    if config.PREDICTION_ENABLED:
        model_kinds = [x.strip() for x in config.PREDICTION_MODEL_KINDS.split(",") if x.strip()]
        prediction_manager = MultiModelPredictionManager(
            model_kinds=model_kinds,
            initial_balance_eur=float(config.PREDICTION_INITIAL_BALANCE_EUR),
            stake_fraction=float(config.PREDICTION_STAKE_FRACTION),
            min_stake_eur=float(config.PREDICTION_MIN_STAKE_EUR),
            max_stake_eur=float(config.PREDICTION_MAX_STAKE_EUR),
            min_edge=float(config.PREDICTION_MIN_EDGE),
            min_liquidity_eur=float(config.PREDICTION_MIN_LIQUIDITY_EUR),
            model_dir=config.PREDICTION_MODEL_DIR,
            state_dir=config.PREDICTION_STATE_DIR,
            save_every=config.PREDICTION_SAVE_EVERY,
            clv_tracker=clv_tracker,
        )
    risk_manager.set_daily_pnl(paper_executor.daily_pnl)
    risk_manager.set_open_bets(paper_executor.open_bets)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _poller_metrics: Dict[str, object] = {}
    _last_poller_log_ts: float = 0.0

    async def _main() -> None:
        nonlocal _last_poller_log_ts
        order_monitor = OrderMonitor(client=client)

        async def _daily_summary_task() -> None:
            while _running:
                now = datetime.now(timezone.utc)
                next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                await asyncio.sleep(max(1.0, (next_midnight - now).total_seconds()))
                try:
                    alert_daily_summary(paper_executor, prediction_manager)
                except Exception:
                    pass

        async def _qa_task() -> None:
            if qa_agent is None:
                while _running:
                    await asyncio.sleep(60.0)
                return
            await qa_agent.run(
                is_running=lambda: _running,
                prediction_manager=prediction_manager,
                request_shutdown=_shutdown,
            )

        async def _audit_task() -> None:
            await audit_agent.run(
                is_running=lambda: _running,
                auto_apply=False,
            )

        def _on_poller_metrics(payload: Dict[str, object]) -> None:
            nonlocal _last_poller_log_ts
            _poller_metrics.update(payload or {})
            now_ts = datetime.now(timezone.utc).timestamp()
            if now_ts - _last_poller_log_ts >= 30.0:
                _last_poller_log_ts = now_ts
                logger.info(
                    "poller cycle: snapshots=%s books=%s batches=%s dur=%.2fs timeouts=%s errors=%s",
                    _poller_metrics.get("snapshots_set"),
                    _poller_metrics.get("books_received"),
                    _poller_metrics.get("batches_total"),
                    float(_poller_metrics.get("cycle_duration_sec", 0.0) or 0.0),
                    _poller_metrics.get("timeouts"),
                    _poller_metrics.get("errors"),
                )

        poller_task = asyncio.create_task(
            run_price_poller(
                client,
                market_ids,
                price_cache,
                interval_seconds=config.PRICE_POLL_INTERVAL_SECONDS,
                is_running=lambda: _running,
                runner_names=runner_names,
                on_metrics=_on_poller_metrics,
                extra_market_ids_provider=(
                    (lambda: prediction_manager.pending_market_ids()) if prediction_manager is not None else None
                ),
            )
        )
        loop_task = asyncio.create_task(
            run_loop(
                market_ids,
                price_cache,
                risk_manager,
                paper_executor,
                scan_interval_seconds=config.SCAN_INTERVAL_SECONDS,
                market_metadata=market_metadata,
                candidate_logger=candidate_logger,
                prediction_manager=prediction_manager,
                info_arb_manager=info_arb_manager,
                learning_architect=learning_architect,
                poller_metrics_provider=lambda: dict(_poller_metrics),
            )
        )
        summary_task = asyncio.create_task(_daily_summary_task())
        qa_task = asyncio.create_task(_qa_task())
        audit_task_handle = asyncio.create_task(_audit_task())
        await order_monitor.start()
        try:
            await asyncio.gather(poller_task, loop_task, summary_task, qa_task, audit_task_handle)
        except asyncio.CancelledError:
            pass
        finally:
            summary_task.cancel()
            qa_task.cancel()
            audit_task_handle.cancel()
            await order_monitor.stop()
            client.logout()
            logger.info("Shutdown complete.")

    def _signal_handler() -> None:
        _shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(_main())
    except KeyboardInterrupt:
        _shutdown()
    finally:
        loop.close()


if __name__ == "__main__":
    main()
