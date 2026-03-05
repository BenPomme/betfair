"""
Multi-model online prediction paper accounts.

Each model runs its own fake bankroll in parallel so performance can be compared
live in the dashboard.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple

from core.types import PriceSnapshot
from data.clv_tracker import CLVTracker
from strategy.features import build_market_microstructure
from strategy.predictive_model import PredictionExample, PureLogitModel, ResidualLogitModel

FEATURE_NAMES = [
    "spread_mean",
    "imbalance",
    "depth_total_eur",
    "price_velocity",
    "short_volatility",
    "time_to_start_sec",
    "in_play",
    "weighted_spread",
    "lay_back_ratio",
    "top_of_book_concentration",
    "selection_count",
    "volume_momentum",
    "back_lay_crossover",
    "overround_distance",
    "depth_ratio_top3",
    "price_range",
]

RECENT_LIFT_WINDOW = 200


@dataclass(frozen=True)
class _PendingBet:
    bet_id: str
    market_id: str
    selection_id: str
    selection_name: str
    event_name: str
    entry_odds: float
    stake: float
    base_prob: float
    predicted_prob: float
    features: Dict[str, float]


@dataclass(frozen=True)
class _LearningCandidate:
    market_id: str
    selection_id: str
    selection_name: str
    event_name: str
    entry_odds: float
    base_prob: float
    predicted_prob: float
    features: Dict[str, float]


class OnlinePredictionEngine:
    """
    One prediction model + one fake account.

    Supported model kinds:
    - implied_market: market-implied probability only (control)
    - residual_logit: online residual logistic model
    - pure_logit: online logistic model on microstructure features only
    """

    def __init__(
        self,
        model_id: str,
        model_kind: str,
        initial_balance_eur: float,
        stake_fraction: float,
        min_stake_eur: float,
        max_stake_eur: float,
        min_edge: float,
        min_liquidity_eur: float,
        model_path: str,
        save_every: int = 25,
        clv_tracker: Optional[CLVTracker] = None,
        state_path: Optional[str] = None,
    ):
        self.model_id = model_id
        self.model_kind = model_kind
        self.initial_balance = float(initial_balance_eur)
        self.balance = float(initial_balance_eur)
        self.stake_fraction = float(stake_fraction)
        self.min_stake = float(min_stake_eur)
        self.max_stake = float(max_stake_eur)
        self.min_edge = float(min_edge)
        self.min_liquidity = float(min_liquidity_eur)
        self.model_path = Path(model_path)
        self.save_every = max(1, int(save_every))

        self.total_bets = 0
        self.settled_bets = 0
        self.wins = 0
        self.losses = 0
        self.voids = 0
        self.resets = 0
        self.total_pnl = 0.0
        self.update_count = 0
        self.brier_sum = 0.0
        self.last_edge = 0.0
        self.last_prediction = 0.0
        self.learning_tracked = 0
        self.learning_settled = 0
        self.learning_model_updates = 0
        self.learning_brier_sum = 0.0
        self.baseline_brier_sum = 0.0
        self.learning_baseline_brier_sum = 0.0
        self._recent_brier_lift = deque(maxlen=RECENT_LIFT_WINDOW)
        self._recent_learning_brier_lift = deque(maxlen=RECENT_LIFT_WINDOW)

        self._pending: Dict[Tuple[str, str], _PendingBet] = {}
        self._learning_candidates: Dict[str, _LearningCandidate] = {}
        self._prev_snapshots: Dict[str, PriceSnapshot] = {}
        self._examples_log = Path(f"data/prediction/online_examples_{self.model_id}.jsonl")
        self._examples_log.parent.mkdir(parents=True, exist_ok=True)
        self._clv_tracker = clv_tracker
        self._state_path = Path(state_path) if state_path else None

        self.model = self._load_or_init_model()
        self._load_state()

    def has_pending_market(self, market_id: str) -> bool:
        if market_id in self._learning_candidates:
            return True
        return any(b.market_id == market_id for b in self._pending.values())

    def pending_market_ids(self) -> List[str]:
        mids = {b.market_id for b in self._pending.values()}
        mids.update(self._learning_candidates.keys())
        return sorted(mids)

    def _load_or_init_model(self):
        if self.model_kind == "implied_market":
            return None
        if self.model_kind == "residual_logit":
            if self.model_path.exists():
                try:
                    return ResidualLogitModel.load(str(self.model_path))
                except Exception:
                    pass
            return ResidualLogitModel(feature_names=FEATURE_NAMES)
        if self.model_kind == "pure_logit":
            if self.model_path.exists():
                try:
                    return PureLogitModel.load(str(self.model_path))
                except Exception:
                    pass
            return PureLogitModel(feature_names=FEATURE_NAMES)
        raise ValueError(f"Unsupported model_kind: {self.model_kind}")

    def _maybe_save_model(self) -> None:
        if self.model is None:
            return
        if self.update_count % self.save_every == 0:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.model_path))

    def _save_state(self) -> None:
        if self._state_path is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_id": self.model_id,
            "model_kind": self.model_kind,
            "balance": self.balance,
            "total_bets": self.total_bets,
            "settled_bets": self.settled_bets,
            "wins": self.wins,
            "losses": self.losses,
            "voids": self.voids,
            "resets": self.resets,
            "total_pnl": self.total_pnl,
            "update_count": self.update_count,
            "brier_sum": self.brier_sum,
            "last_edge": self.last_edge,
            "last_prediction": self.last_prediction,
            "learning_tracked": self.learning_tracked,
            "learning_settled": self.learning_settled,
            "learning_model_updates": self.learning_model_updates,
            "learning_brier_sum": self.learning_brier_sum,
            "baseline_brier_sum": self.baseline_brier_sum,
            "learning_baseline_brier_sum": self.learning_baseline_brier_sum,
            "recent_brier_lift": list(self._recent_brier_lift),
            "recent_learning_brier_lift": list(self._recent_learning_brier_lift),
            "stake_fraction": self.stake_fraction,
            "min_edge": self.min_edge,
            "pending": [
                {
                    "market_id": b.market_id,
                    "selection_id": b.selection_id,
                    "bet_id": b.bet_id,
                    "selection_name": b.selection_name,
                    "event_name": b.event_name,
                    "entry_odds": b.entry_odds,
                    "stake": b.stake,
                    "base_prob": b.base_prob,
                    "predicted_prob": b.predicted_prob,
                    "features": b.features,
                }
                for b in self._pending.values()
            ],
            "learning_candidates": [
                {
                    "market_id": c.market_id,
                    "selection_id": c.selection_id,
                    "selection_name": c.selection_name,
                    "event_name": c.event_name,
                    "entry_odds": c.entry_odds,
                    "base_prob": c.base_prob,
                    "predicted_prob": c.predicted_prob,
                    "features": c.features,
                }
                for c in self._learning_candidates.values()
            ],
        }
        self._state_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    def _load_state(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            self.balance = float(raw.get("balance", self.balance))
            self.total_bets = int(raw.get("total_bets", self.total_bets))
            self.settled_bets = int(raw.get("settled_bets", self.settled_bets))
            self.wins = int(raw.get("wins", self.wins))
            self.losses = int(raw.get("losses", self.losses))
            self.voids = int(raw.get("voids", self.voids))
            self.resets = int(raw.get("resets", self.resets))
            self.total_pnl = float(raw.get("total_pnl", self.total_pnl))
            self.update_count = int(raw.get("update_count", self.update_count))
            self.brier_sum = float(raw.get("brier_sum", self.brier_sum))
            self.last_edge = float(raw.get("last_edge", self.last_edge))
            self.last_prediction = float(raw.get("last_prediction", self.last_prediction))
            self.learning_tracked = int(raw.get("learning_tracked", self.learning_tracked))
            self.learning_settled = int(raw.get("learning_settled", self.learning_settled))
            self.learning_model_updates = int(raw.get("learning_model_updates", self.learning_model_updates))
            self.learning_brier_sum = float(raw.get("learning_brier_sum", self.learning_brier_sum))
            self.baseline_brier_sum = float(raw.get("baseline_brier_sum", self.baseline_brier_sum))
            self.learning_baseline_brier_sum = float(
                raw.get("learning_baseline_brier_sum", self.learning_baseline_brier_sum)
            )
            recent_lift = raw.get("recent_brier_lift", [])
            if isinstance(recent_lift, list):
                for v in recent_lift[-RECENT_LIFT_WINDOW:]:
                    try:
                        self._recent_brier_lift.append(float(v))
                    except Exception:
                        continue
            recent_learning_lift = raw.get("recent_learning_brier_lift", [])
            if isinstance(recent_learning_lift, list):
                for v in recent_learning_lift[-RECENT_LIFT_WINDOW:]:
                    try:
                        self._recent_learning_brier_lift.append(float(v))
                    except Exception:
                        continue
            self.stake_fraction = float(raw.get("stake_fraction", self.stake_fraction))
            self.min_edge = float(raw.get("min_edge", self.min_edge))
            pending = raw.get("pending", [])
            if isinstance(pending, list):
                for p in pending:
                    try:
                        bet = _PendingBet(
                            bet_id=str(p["bet_id"]),
                            market_id=str(p["market_id"]),
                            selection_id=str(p["selection_id"]),
                            selection_name=str(p.get("selection_name", "")),
                            event_name=str(p.get("event_name", "")),
                            entry_odds=float(p["entry_odds"]),
                            stake=float(p["stake"]),
                            base_prob=float(p["base_prob"]),
                            predicted_prob=float(p["predicted_prob"]),
                            features={k: float(v) for k, v in dict(p.get("features", {})).items()},
                        )
                        self._pending[(bet.market_id, bet.selection_id)] = bet
                    except Exception:
                        continue
            learning_candidates = raw.get("learning_candidates", [])
            if isinstance(learning_candidates, list):
                for c in learning_candidates:
                    try:
                        cand = _LearningCandidate(
                            market_id=str(c["market_id"]),
                            selection_id=str(c["selection_id"]),
                            selection_name=str(c.get("selection_name", "")),
                            event_name=str(c.get("event_name", "")),
                            entry_odds=float(c["entry_odds"]),
                            base_prob=float(c["base_prob"]),
                            predicted_prob=float(c["predicted_prob"]),
                            features={k: float(v) for k, v in dict(c.get("features", {})).items()},
                        )
                        self._learning_candidates[cand.market_id] = cand
                    except Exception:
                        continue
        except Exception:
            pass

    def _features_from_snapshot(
        self,
        snapshot: PriceSnapshot,
        market_start: Optional[datetime],
    ) -> Dict[str, float]:
        prev = self._prev_snapshots.get(snapshot.market_id)
        micro = build_market_microstructure(snapshot, market_start=market_start, previous_snapshot=prev)
        return {
            "spread_mean": float(micro.spread_mean),
            "imbalance": float(micro.imbalance),
            "depth_total_eur": float(micro.depth_total_eur),
            "price_velocity": float(micro.price_velocity),
            "short_volatility": float(micro.short_volatility),
            "time_to_start_sec": float(micro.time_to_start_sec),
            "in_play": 1.0 if micro.in_play else 0.0,
            "weighted_spread": float(micro.weighted_spread),
            "lay_back_ratio": float(micro.lay_back_ratio),
            "top_of_book_concentration": float(micro.top_of_book_concentration),
            "selection_count": float(micro.selection_count),
            "volume_momentum": float(micro.volume_momentum),
            "back_lay_crossover": float(micro.back_lay_crossover),
            "overround_distance": float(micro.overround_distance),
            "depth_ratio_top3": float(micro.depth_ratio_top3),
            "price_range": float(micro.price_range),
        }

    def _predict_prob(self, base_prob: float, features: Dict[str, float]) -> float:
        if self.model_kind == "implied_market":
            return max(1e-6, min(1.0 - 1e-6, base_prob))
        if self.model_kind == "residual_logit":
            return self.model.predict_proba(base_prob, features)
        return self.model.predict_proba(features)

    def _append_example(self, ex: PredictionExample) -> None:
        payload = {
            "timestamp": ex.timestamp,
            "base_prob": ex.base_prob,
            "odds": ex.odds,
            "label": ex.label,
            "model_id": self.model_id,
            "model_kind": self.model_kind,
            **ex.features,
        }
        with self._examples_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _update_model(self, ex: PredictionExample) -> None:
        if self.model_kind == "implied_market":
            return
        self.model.fit([ex], epochs=1, lr=0.03)
        self.update_count += 1
        self._maybe_save_model()
        self._save_state()

    def _stake_size(self) -> float:
        stake = self.balance * self.stake_fraction
        stake = max(self.min_stake, stake)
        stake = min(self.max_stake, stake, self.balance)
        return round(stake, 2)

    def _reset_if_bust(self) -> bool:
        if self.balance > 0:
            return False
        self.resets += 1
        self.balance = self.initial_balance
        self._pending.clear()
        self._save_state()
        return True

    def _select_learning_candidate(
        self,
        market_id: str,
        snapshot: PriceSnapshot,
        event_name: str,
        features: Dict[str, float],
    ) -> Optional[_LearningCandidate]:
        if str(getattr(snapshot, "market_status", "OPEN") or "OPEN") == "CLOSED":
            return None
        best = None
        for sel in snapshot.selections:
            odds = float(sel.best_back_price)
            liq = float(sel.available_to_back)
            if odds <= 1.01 or liq < self.min_liquidity:
                continue
            base_prob = 1.0 / odds
            pred_prob = self._predict_prob(base_prob, features)
            if best is None or liq > best["liq"]:
                best = {
                    "selection_id": sel.selection_id,
                    "selection_name": sel.name,
                    "entry_odds": odds,
                    "base_prob": base_prob,
                    "predicted_prob": pred_prob,
                    "liq": liq,
                }
        if best is None:
            return None
        return _LearningCandidate(
            market_id=market_id,
            selection_id=best["selection_id"],
            selection_name=best["selection_name"],
            event_name=event_name,
            entry_odds=best["entry_odds"],
            base_prob=best["base_prob"],
            predicted_prob=best["predicted_prob"],
            features=features,
        )

    def _maybe_track_learning_candidate(
        self,
        market_id: str,
        snapshot: PriceSnapshot,
        event_name: str,
        features: Dict[str, float],
    ) -> Optional[dict]:
        if market_id in self._learning_candidates:
            return None
        if any(b.market_id == market_id for b in self._pending.values()):
            return None
        cand = self._select_learning_candidate(market_id, snapshot, event_name, features)
        if cand is None:
            return None
        self._learning_candidates[market_id] = cand
        self.learning_tracked += 1
        self._save_state()
        return {
            "kind": "prediction_learning_track",
            "model_id": self.model_id,
            "market_id": market_id,
            "selection": cand.selection_name,
            "pred_prob": round(cand.predicted_prob, 5),
            "base_prob": round(cand.base_prob, 5),
        }

    def _settle_learning_candidate(self, snapshot: PriceSnapshot) -> List[dict]:
        events: List[dict] = []
        if str(getattr(snapshot, "market_status", "OPEN") or "OPEN") != "CLOSED":
            return events
        cand = self._learning_candidates.get(snapshot.market_id)
        if cand is None:
            return events
        by_id = {s.selection_id: s for s in snapshot.selections}
        sel = by_id.get(cand.selection_id)
        if sel is None:
            return events
        status = str(getattr(sel, "runner_status", "UNKNOWN") or "UNKNOWN").upper()
        if status == "WINNER":
            y = 1
        elif status == "LOSER":
            y = 0
        else:
            # Removed/void/unknown labels are not useful for supervised updates.
            if status in {"REMOVED", "REMOVED_VACANT", "VOID", "HIDDEN"}:
                self._learning_candidates.pop(snapshot.market_id, None)
                self._save_state()
            return events

        ex = PredictionExample(
            timestamp=datetime.now(timezone.utc).isoformat(),
            base_prob=cand.base_prob,
            odds=cand.entry_odds,
            label=y,
            features=cand.features,
        )
        self.learning_settled += 1
        model_brier = (cand.predicted_prob - y) ** 2
        baseline_brier = (cand.base_prob - y) ** 2
        self.learning_brier_sum += model_brier
        self.learning_baseline_brier_sum += baseline_brier
        self._recent_learning_brier_lift.append(baseline_brier - model_brier)
        if self.model_kind != "implied_market":
            self._update_model(ex)
            self.learning_model_updates += 1
        self._append_example(ex)
        self._learning_candidates.pop(snapshot.market_id, None)
        self._save_state()
        events.append(
            {
                "kind": "prediction_learning_settle",
                "model_id": self.model_id,
                "market_id": cand.market_id,
                "selection": cand.selection_name,
                "won": y == 1,
                "updated": self.model_kind != "implied_market",
                "pred_prob": round(cand.predicted_prob, 5),
                "base_prob": round(cand.base_prob, 5),
                "brier_lift": round(baseline_brier - model_brier, 6),
            }
        )
        return events

    def _settle_pending(self, snapshot: PriceSnapshot) -> List[dict]:
        events: List[dict] = []
        if str(getattr(snapshot, "market_status", "OPEN") or "OPEN") != "CLOSED":
            return events
        to_remove = []
        by_id = {s.selection_id: s for s in snapshot.selections}

        for key, bet in list(self._pending.items()):
            if bet.market_id != snapshot.market_id:
                continue
            sel = by_id.get(bet.selection_id)
            if sel is None:
                continue
            status = str(getattr(sel, "runner_status", "UNKNOWN") or "UNKNOWN").upper()
            if status == "WINNER":
                y = 1
            elif status == "LOSER":
                y = 0
            elif status in {"REMOVED", "REMOVED_VACANT", "VOID", "HIDDEN"}:
                self.voids += 1
                self.balance += bet.stake
                events.append(
                    {
                        "kind": "prediction_settle",
                        "model_id": self.model_id,
                        "market_id": bet.market_id,
                        "selection": bet.selection_name,
                        "won": False,
                        "void": True,
                        "pnl_eur": 0.0,
                        "balance_eur": round(self.balance, 2),
                        "reset": False,
                    }
                )
                to_remove.append(key)
                self._save_state()
                continue
            else:
                continue
            ex = PredictionExample(
                timestamp=datetime.now(timezone.utc).isoformat(),
                base_prob=bet.base_prob,
                odds=bet.entry_odds,
                label=y,
                features=bet.features,
            )
            self._update_model(ex)
            self._append_example(ex)

            model_brier = (bet.predicted_prob - y) ** 2
            baseline_brier = (bet.base_prob - y) ** 2
            self.brier_sum += model_brier
            self.baseline_brier_sum += baseline_brier
            self._recent_brier_lift.append(baseline_brier - model_brier)
            self.settled_bets += 1
            if y == 1:
                self.wins += 1
                payout = bet.stake * bet.entry_odds
                pnl = payout - bet.stake
                self.balance += payout
            else:
                self.losses += 1
                pnl = -bet.stake
            self.total_pnl += pnl
            busted = self._reset_if_bust()
            events.append(
                {
                    "kind": "prediction_settle",
                    "model_id": self.model_id,
                    "market_id": bet.market_id,
                    "selection": bet.selection_name,
                    "won": y == 1,
                    "void": False,
                    "pnl_eur": round(pnl, 4),
                    "balance_eur": round(self.balance, 2),
                    "reset": busted,
                    "brier_lift": round(baseline_brier - model_brier, 6),
                    "clv": self._clv_tracker.compute_clv(bet.bet_id) if self._clv_tracker else None,
                }
            )
            to_remove.append(key)
            self._save_state()

        for key in to_remove:
            self._pending.pop(key, None)
        if to_remove:
            self._save_state()
        return events

    def _open_bet(
        self,
        market_id: str,
        snapshot: PriceSnapshot,
        event_name: str,
        features: Dict[str, float],
    ) -> Optional[dict]:
        if str(getattr(snapshot, "market_status", "OPEN") or "OPEN") == "CLOSED":
            return None
        best = None
        for sel in snapshot.selections:
            odds = float(sel.best_back_price)
            liq = float(sel.available_to_back)
            if odds <= 1.01 or liq < self.min_liquidity:
                continue
            base_prob = 1.0 / odds
            pred_prob = self._predict_prob(base_prob, features)
            edge = pred_prob - base_prob
            if best is None or edge > best["edge"]:
                best = {
                    "selection_id": sel.selection_id,
                    "selection_name": sel.name,
                    "odds": odds,
                    "base_prob": base_prob,
                    "pred_prob": pred_prob,
                    "edge": edge,
                }

        if best is None or best["edge"] < self.min_edge:
            return None

        key = (market_id, best["selection_id"])
        if key in self._pending:
            return None

        stake = self._stake_size()
        if stake < self.min_stake or stake <= 0:
            return None

        self.balance -= stake
        self.total_bets += 1
        self.last_edge = best["edge"]
        self.last_prediction = best["pred_prob"]

        bet_id = f"{self.model_id}:{market_id}:{best['selection_id']}:{int(datetime.now(timezone.utc).timestamp())}"
        # If we had a shadow learning candidate for this market, remove it to avoid
        # double-labeling the same market for the same model.
        self._learning_candidates.pop(market_id, None)
        self._pending[key] = _PendingBet(
            bet_id=bet_id,
            market_id=market_id,
            selection_id=best["selection_id"],
            selection_name=best["selection_name"],
            event_name=event_name,
            entry_odds=best["odds"],
            stake=stake,
            base_prob=best["base_prob"],
            predicted_prob=best["pred_prob"],
            features=features,
        )
        if self._clv_tracker is not None:
            self._clv_tracker.record_entry(
                bet_id=bet_id,
                market_id=market_id,
                selection_id=best["selection_id"],
                entry_odds=best["odds"],
                entry_timestamp=datetime.now(timezone.utc).isoformat(),
            )
        self._save_state()
        return {
            "kind": "prediction_open",
            "model_id": self.model_id,
            "market_id": market_id,
            "event_name": event_name,
            "selection": best["selection_name"],
            "odds": round(best["odds"], 3),
            "edge": round(best["edge"], 5),
            "pred_prob": round(best["pred_prob"], 5),
            "stake_eur": stake,
            "balance_eur": round(self.balance, 2),
            "bet_id": bet_id,
        }

    def process_snapshot(
        self,
        market_id: str,
        snapshot: PriceSnapshot,
        event_name: str,
        market_start: Optional[datetime],
    ) -> Dict[str, object]:
        events: List[dict] = []
        if self._clv_tracker is not None:
            self._clv_tracker.record_closing_prices(market_id, snapshot)
        settle_events = self._settle_pending(snapshot)
        learning_events = self._settle_learning_candidate(snapshot)
        events.extend(settle_events)
        events.extend(learning_events)
        features = self._features_from_snapshot(snapshot, market_start)
        if not any(e.get("reset") for e in settle_events):
            opened = self._open_bet(market_id, snapshot, event_name, features)
            if opened:
                events.append(opened)
            else:
                tracked = self._maybe_track_learning_candidate(
                    market_id=market_id,
                    snapshot=snapshot,
                    event_name=event_name,
                    features=features,
                )
                if tracked:
                    events.append(tracked)
        self._prev_snapshots[market_id] = snapshot
        return {"events": events, "state": self.get_state()}

    def process_settlement_snapshot(
        self,
        market_id: str,
        snapshot: PriceSnapshot,
    ) -> Dict[str, object]:
        """
        Settle existing pending bets only, without opening new positions.
        Used for markets that dropped out of scan universe but still have open bets.
        """
        events: List[dict] = []
        if self._clv_tracker is not None:
            self._clv_tracker.record_closing_prices(market_id, snapshot)
        settle_events = self._settle_pending(snapshot)
        learning_events = self._settle_learning_candidate(snapshot)
        events.extend(settle_events)
        events.extend(learning_events)
        self._prev_snapshots[market_id] = snapshot
        return {"events": events, "state": self.get_state()}

    def get_state(self) -> Dict[str, object]:
        avg_brier = (self.brier_sum / self.settled_bets) if self.settled_bets > 0 else 0.0
        baseline_avg_brier = (
            self.baseline_brier_sum / self.settled_bets if self.settled_bets > 0 else 0.0
        )
        learning_avg_brier = (
            self.learning_brier_sum / self.learning_settled if self.learning_settled > 0 else 0.0
        )
        learning_baseline_avg_brier = (
            self.learning_baseline_brier_sum / self.learning_settled if self.learning_settled > 0 else 0.0
        )
        brier_lift = baseline_avg_brier - avg_brier
        learning_brier_lift = learning_baseline_avg_brier - learning_avg_brier
        win_rate = (self.wins / self.settled_bets) if self.settled_bets > 0 else 0.0
        roi = (self.total_pnl / self.initial_balance) if self.initial_balance > 0 else 0.0
        recent_brier_lift = (
            sum(self._recent_brier_lift) / len(self._recent_brier_lift)
            if len(self._recent_brier_lift) > 0 else 0.0
        )
        recent_learning_brier_lift = (
            sum(self._recent_learning_brier_lift) / len(self._recent_learning_brier_lift)
            if len(self._recent_learning_brier_lift) > 0 else 0.0
        )
        return {
            "model_id": self.model_id,
            "model_kind": self.model_kind,
            "enabled": True,
            "stake_fraction": self.stake_fraction,
            "min_edge": self.min_edge,
            "balance_eur": round(self.balance, 2),
            "initial_balance_eur": round(self.initial_balance, 2),
            "total_bets": self.total_bets,
            "settled_bets": self.settled_bets,
            "wins": self.wins,
            "losses": self.losses,
            "voids": self.voids,
            "win_rate": round(win_rate, 4),
            "roi_pct": round(roi * 100, 4),
            "total_pnl_eur": round(self.total_pnl, 4),
            "resets": self.resets,
            "open_positions": len(self._pending),
            "avg_brier": round(avg_brier, 6),
            "baseline_avg_brier": round(baseline_avg_brier, 6),
            "brier_lift_abs": round(brier_lift, 6),
            "model_updates": self.update_count,
            "learning_tracked": self.learning_tracked,
            "learning_settled": self.learning_settled,
            "learning_updates": self.learning_model_updates,
            "learning_avg_brier": round(learning_avg_brier, 6),
            "learning_baseline_avg_brier": round(learning_baseline_avg_brier, 6),
            "learning_brier_lift_abs": round(learning_brier_lift, 6),
            "learning_open_markets": len(self._learning_candidates),
            "recent_brier_lift": round(recent_brier_lift, 6),
            "recent_learning_brier_lift": round(recent_learning_brier_lift, 6),
            "model_path": str(self.model_path),
            "model_version": f"{self.model_kind}_online_v1",
            "last_edge": round(self.last_edge, 6),
            "last_prediction": round(self.last_prediction, 6),
        }


class MultiModelPredictionManager:
    """Runs multiple prediction engines in parallel and aggregates telemetry."""

    def __init__(
        self,
        model_kinds: List[str],
        initial_balance_eur: float,
        stake_fraction: float,
        min_stake_eur: float,
        max_stake_eur: float,
        min_edge: float,
        min_liquidity_eur: float,
        model_dir: str,
        save_every: int = 25,
        clv_tracker: Optional[CLVTracker] = None,
        state_dir: Optional[str] = None,
    ):
        self.engines: Dict[str, OnlinePredictionEngine] = {}
        model_root = Path(model_dir)
        model_root.mkdir(parents=True, exist_ok=True)
        state_root = Path(state_dir) if state_dir else None
        if state_root is not None:
            state_root.mkdir(parents=True, exist_ok=True)
        for idx, kind in enumerate(model_kinds):
            model_id = f"{kind}_{idx+1}"
            model_path = str(model_root / f"{model_id}.json")
            model_state_path = str(state_root / f"{model_id}.json") if state_root is not None else None
            self.engines[model_id] = OnlinePredictionEngine(
                model_id=model_id,
                model_kind=kind,
                initial_balance_eur=initial_balance_eur,
                stake_fraction=stake_fraction,
                min_stake_eur=min_stake_eur,
                max_stake_eur=max_stake_eur,
                min_edge=min_edge,
                min_liquidity_eur=min_liquidity_eur,
                model_path=model_path,
                save_every=save_every,
                clv_tracker=clv_tracker,
                state_path=model_state_path,
            )

    def process_snapshot(
        self,
        market_id: str,
        snapshot: PriceSnapshot,
        event_name: str,
        market_start: Optional[datetime],
    ) -> Dict[str, object]:
        events: List[dict] = []
        models: Dict[str, dict] = {}
        for model_id, engine in self.engines.items():
            out = engine.process_snapshot(market_id, snapshot, event_name, market_start)
            events.extend(out.get("events", []))
            models[model_id] = out.get("state", {})
        return {"events": events, "models": models}

    def initial_state(self) -> Dict[str, dict]:
        return {model_id: engine.get_state() for model_id, engine in self.engines.items()}

    def pending_market_ids(self) -> List[str]:
        pending: set[str] = set()
        for engine in self.engines.values():
            pending.update(engine.pending_market_ids())
        return sorted(pending)

    def process_pending_settlements(
        self,
        market_ids: List[str],
        price_cache,
    ) -> Dict[str, object]:
        """
        Run settlement-only updates for the provided market IDs.
        This is intentionally separate from process_snapshot to avoid opening
        additional bets on out-of-universe markets.
        """
        events: List[dict] = []
        models: Dict[str, dict] = {}
        for market_id in market_ids:
            snapshot = price_cache.get_prices(market_id)
            if snapshot is None:
                continue
            for model_id, engine in self.engines.items():
                if not engine.has_pending_market(market_id):
                    continue
                out = engine.process_settlement_snapshot(market_id, snapshot)
                events.extend(out.get("events", []))
                models[model_id] = out.get("state", {})
        if not models:
            models = self.initial_state()
        return {"events": events, "models": models}
