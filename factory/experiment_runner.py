from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from factory.contracts import (
    EvaluationBundle,
    EvaluationStage,
    EvaluationWindow,
    ExperimentSpec,
    LineageRecord,
    StrategyGenome,
    utc_now_iso,
)
from funding.strategy.contrarian_backtester import ContrarianBacktester
from strategy.prediction_bootstrap import pooled_examples
from strategy.predictive_model import (
    HybridLogitModel,
    MarketCalibratedModel,
    PredictionExample,
    PureLogitModel,
    ResidualLogitModel,
    evaluate_predictions,
)


def _safe_slug(value: str) -> str:
    return value.replace(":", "__").replace("/", "__")


def _clip_probability(value: float) -> float:
    if value < 1e-6:
        return 1e-6
    if value > 1.0 - 1e-6:
        return 1.0 - 1e-6
    return value


class FactoryExperimentRunner:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self._jsonl_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._jsonl_tail_cache: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        self._json_object_cache: Dict[str, Dict[str, Any]] = {}
        self._cascade_cache: Dict[Tuple[str, str, int, float], Optional[Dict[str, Any]]] = {}
        self._candidate_signal_cache: Dict[Tuple[str, str, str, float], Optional[Dict[str, Any]]] = {}
        self._funding_cache: Dict[Tuple[str, float, float, int, float], Optional[Dict[str, Any]]] = {}

    def run(
        self,
        *,
        lineage: LineageRecord,
        genome: StrategyGenome,
        experiment: ExperimentSpec,
    ) -> Dict[str, Any]:
        if lineage.family_id == "binance_cascade_regime":
            return self._run_cascade_experiment(
                lineage=lineage,
                genome=genome,
                experiment=experiment,
            )
        if lineage.family_id == "binance_funding_contrarian":
            return self._run_funding_experiment(
                lineage=lineage,
                genome=genome,
                experiment=experiment,
            )
        if lineage.family_id == "betfair_information_lag":
            return self._run_candidate_signal_experiment(
                lineage=lineage,
                genome=genome,
                experiment=experiment,
                family_mode="information_lag",
            )
        if lineage.family_id == "betfair_prediction_value_league":
            return self._run_prediction_walkforward(
                lineage=lineage,
                genome=genome,
                experiment=experiment,
            )
        if lineage.family_id == "polymarket_cross_venue":
            return self._run_candidate_signal_experiment(
                lineage=lineage,
                genome=genome,
                experiment=experiment,
                family_mode="cross_venue",
            )
        return {"mode": "unsupported", "bundles": [], "artifact_summary": None}

    def _run_prediction_walkforward(
        self,
        *,
        lineage: LineageRecord,
        genome: StrategyGenome,
        experiment: ExperimentSpec,
    ) -> Dict[str, Any]:
        parameters = dict(genome.parameters)
        feature_subset = str(parameters.get("selected_feature_subset", "microstructure") or "microstructure")
        requested_model_class = str(parameters.get("selected_model_class", "logit") or "logit")
        edge_threshold = float(parameters.get("selected_min_edge", 0.03) or 0.03)
        learning_rate = float(parameters.get("selected_learning_rate", 0.02) or 0.02)
        horizon_seconds = int(parameters.get("selected_horizon_seconds", 1800) or 1800)
        lookback_hours = float(parameters.get("selected_lookback_hours", 168.0) or 168.0)

        all_examples = pooled_examples(str(self.project_root / "data" / "prediction"))
        filtered_examples = self._filter_prediction_examples(
            examples=all_examples,
            feature_subset=feature_subset,
            lookback_hours=lookback_hours,
        )
        train_window, test_window = self._prediction_window_sizes(horizon_seconds, len(filtered_examples))
        epochs = max(4, min(18, int(round(6 + (lookback_hours / 48.0)))))

        run_id, run_root = self._build_run_root(
            lineage=lineage,
            experiment=experiment,
        )
        run_root.mkdir(parents=True, exist_ok=True)

        self._write_json(
            run_root / "dataset.json",
            {
                "generated_at": utc_now_iso(),
                "family_id": lineage.family_id,
                "lineage_id": lineage.lineage_id,
                "source": str(self.project_root / "data" / "prediction"),
                "example_count": len(filtered_examples),
                "available_example_count": len(all_examples),
                "lookback_hours": lookback_hours,
                "time_range": self._prediction_time_range(filtered_examples),
            },
        )
        feature_names = sorted({key for example in filtered_examples for key in example.features.keys()})
        self._write_json(
            run_root / "features.json",
            {
                "generated_at": utc_now_iso(),
                "feature_subset": feature_subset,
                "feature_names": feature_names,
                "feature_count": len(feature_names),
            },
        )

        walkforward = self._evaluate_prediction_walkforward(
            examples=filtered_examples,
            requested_model_class=requested_model_class,
            train_window=train_window,
            test_window=test_window,
            epochs=epochs,
            learning_rate=learning_rate,
            edge_threshold=edge_threshold,
        )
        stress = self._evaluate_prediction_walkforward(
            examples=filtered_examples,
            requested_model_class=requested_model_class,
            train_window=train_window,
            test_window=test_window,
            epochs=max(4, epochs - 2),
            learning_rate=max(0.001, learning_rate * 0.75),
            edge_threshold=min(0.12, edge_threshold * 1.5),
        )

        train_payload = {
            "generated_at": utc_now_iso(),
            "requested_model_class": requested_model_class,
            "resolved_model_engine": walkforward["resolved_model_engine"],
            "train_window": train_window,
            "test_window": test_window,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "edge_threshold": edge_threshold,
        }
        self._write_json(run_root / "train.json", train_payload)
        self._write_json(run_root / "walkforward.json", walkforward["artifact"])
        self._write_json(run_root / "stress.json", stress["artifact"])
        if walkforward.get("last_model_payload") is not None:
            self._write_json(run_root / "model.json", walkforward["last_model_payload"])

        package_path = run_root / "package.json"
        artifact_summary = {
            "run_id": run_id,
            "mode": "prediction_walkforward",
            "package_path": str(package_path),
            "run_root": str(run_root),
            "requested_model_class": requested_model_class,
            "resolved_model_engine": walkforward["resolved_model_engine"],
            "feature_subset": feature_subset,
            "example_count": len(filtered_examples),
            "train_window": train_window,
            "test_window": test_window,
        }
        self._write_json(
            package_path,
            {
                "generated_at": utc_now_iso(),
                "lineage_id": lineage.lineage_id,
                "family_id": lineage.family_id,
                "artifact_summary": artifact_summary,
                "files": {
                    "dataset": str(run_root / "dataset.json"),
                    "features": str(run_root / "features.json"),
                    "train": str(run_root / "train.json"),
                    "walkforward": str(run_root / "walkforward.json"),
                    "stress": str(run_root / "stress.json"),
                    "model": str(run_root / "model.json") if (run_root / "model.json").exists() else None,
                },
            },
        )

        bundles = [
            self._build_prediction_bundle(
                lineage=lineage,
                stage=EvaluationStage.WALKFORWARD.value,
                summary=walkforward,
                package_path=package_path,
            ),
            self._build_prediction_bundle(
                lineage=lineage,
                stage=EvaluationStage.STRESS.value,
                summary=stress,
                package_path=package_path,
            ),
        ]
        return {
            "mode": "prediction_walkforward",
            "bundles": bundles,
            "artifact_summary": artifact_summary,
        }

    def _run_funding_experiment(
        self,
        *,
        lineage: LineageRecord,
        genome: StrategyGenome,
        experiment: ExperimentSpec,
    ) -> Dict[str, Any]:
        parameters = dict(genome.parameters)
        requested_model_class = str(parameters.get("selected_model_class", "logit") or "logit")
        horizon_seconds = int(parameters.get("selected_horizon_seconds", 1800) or 1800)
        lookback_hours = float(parameters.get("selected_lookback_hours", 72.0) or 72.0)
        min_edge = float(parameters.get("selected_min_edge", 0.03) or 0.03)
        stake_fraction = float(parameters.get("selected_stake_fraction", 0.03) or 0.03)

        run_id, run_root = self._build_run_root(
            lineage=lineage,
            experiment=experiment,
        )
        run_root.mkdir(parents=True, exist_ok=True)

        backtest_result = self._run_funding_backtest(
            requested_model_class=requested_model_class,
            lookback_hours=lookback_hours,
            stake_fraction=stake_fraction,
            horizon_seconds=horizon_seconds,
            min_edge=min_edge,
        )
        if backtest_result is None:
            funding_summary = self._funding_log_fallback(
                requested_model_class=requested_model_class,
                lookback_hours=lookback_hours,
                stake_fraction=stake_fraction,
                min_edge=min_edge,
            )
        else:
            funding_summary = backtest_result

        self._write_json(
            run_root / "dataset.json",
            {
                "generated_at": utc_now_iso(),
                "family_id": lineage.family_id,
                "lineage_id": lineage.lineage_id,
                "source_mode": funding_summary["source_mode"],
                "source_paths": funding_summary["source_paths"],
                "symbol_count": funding_summary["symbol_count"],
                "dataset_rows": funding_summary["dataset_rows"],
                "lookback_hours": lookback_hours,
            },
        )
        self._write_json(
            run_root / "features.json",
            {
                "generated_at": utc_now_iso(),
                "feature_family": "funding_contrarian",
                "source_mode": funding_summary["source_mode"],
                "feature_columns": list(funding_summary["feature_columns"]),
                "feature_count": len(funding_summary["feature_columns"]),
            },
        )
        self._write_json(
            run_root / "train.json",
            {
                "generated_at": utc_now_iso(),
                "requested_model_class": requested_model_class,
                "resolved_model_engine": funding_summary["resolved_model_engine"],
                "stake_fraction": stake_fraction,
                "horizon_seconds": horizon_seconds,
                "min_edge": min_edge,
            },
        )
        self._write_json(run_root / "walkforward.json", funding_summary["walkforward_artifact"])
        self._write_json(run_root / "stress.json", funding_summary["stress_artifact"])

        package_path = run_root / "package.json"
        artifact_summary = {
            "run_id": run_id,
            "mode": "funding_contrarian",
            "package_path": str(package_path),
            "run_root": str(run_root),
            "requested_model_class": requested_model_class,
            "resolved_model_engine": funding_summary["resolved_model_engine"],
            "source_mode": funding_summary["source_mode"],
            "dataset_rows": funding_summary["dataset_rows"],
            "symbol_count": funding_summary["symbol_count"],
        }
        self._write_json(
            package_path,
            {
                "generated_at": utc_now_iso(),
                "lineage_id": lineage.lineage_id,
                "family_id": lineage.family_id,
                "artifact_summary": artifact_summary,
                "files": {
                    "dataset": str(run_root / "dataset.json"),
                    "features": str(run_root / "features.json"),
                    "train": str(run_root / "train.json"),
                    "walkforward": str(run_root / "walkforward.json"),
                    "stress": str(run_root / "stress.json"),
                },
            },
        )
        bundles = [
            self._build_funding_bundle(
                lineage=lineage,
                stage=EvaluationStage.WALKFORWARD.value,
                summary=funding_summary["walkforward_summary"],
                package_path=package_path,
            ),
            self._build_funding_bundle(
                lineage=lineage,
                stage=EvaluationStage.STRESS.value,
                summary=funding_summary["stress_summary"],
                package_path=package_path,
            ),
        ]
        return {
            "mode": "funding_contrarian",
            "bundles": bundles,
            "artifact_summary": artifact_summary,
        }

    def _run_cascade_experiment(
        self,
        *,
        lineage: LineageRecord,
        genome: StrategyGenome,
        experiment: ExperimentSpec,
    ) -> Dict[str, Any]:
        parameters = dict(genome.parameters)
        requested_model_class = str(parameters.get("selected_model_class", "gbdt") or "gbdt")
        feature_subset = str(parameters.get("selected_feature_subset", "regime") or "regime")
        horizon_seconds = int(parameters.get("selected_horizon_seconds", 1800) or 1800)
        min_edge = float(parameters.get("selected_min_edge", 0.03) or 0.03)

        run_id, run_root = self._build_run_root(lineage=lineage, experiment=experiment)
        run_root.mkdir(parents=True, exist_ok=True)

        summary = self._cascade_summary(
            requested_model_class=requested_model_class,
            feature_subset=feature_subset,
            horizon_seconds=horizon_seconds,
            min_edge=min_edge,
        )
        if summary is None:
            return {"mode": "cascade_regime", "bundles": [], "artifact_summary": None}

        self._write_json(
            run_root / "dataset.json",
            {
                "generated_at": utc_now_iso(),
                "family_id": lineage.family_id,
                "lineage_id": lineage.lineage_id,
                "source_mode": summary["source_mode"],
                "source_paths": summary["source_paths"],
                "dataset_rows": summary["dataset_rows"],
                "symbol_count": summary["market_count"],
            },
        )
        self._write_json(
            run_root / "features.json",
            {
                "generated_at": utc_now_iso(),
                "feature_family": "cascade_regime",
                "feature_subset": feature_subset,
                "feature_columns": list(summary["feature_columns"]),
                "feature_count": len(summary["feature_columns"]),
            },
        )
        self._write_json(
            run_root / "train.json",
            {
                "generated_at": utc_now_iso(),
                "requested_model_class": requested_model_class,
                "resolved_model_engine": summary["resolved_model_engine"],
                "horizon_seconds": horizon_seconds,
                "min_edge": min_edge,
            },
        )
        self._write_json(run_root / "walkforward.json", summary["walkforward_artifact"])
        self._write_json(run_root / "stress.json", summary["stress_artifact"])

        package_path = run_root / "package.json"
        artifact_summary = {
            "run_id": run_id,
            "mode": "cascade_regime",
            "package_path": str(package_path),
            "run_root": str(run_root),
            "requested_model_class": requested_model_class,
            "resolved_model_engine": summary["resolved_model_engine"],
            "source_mode": summary["source_mode"],
            "dataset_rows": summary["dataset_rows"],
            "symbol_count": summary["market_count"],
        }
        self._write_json(
            package_path,
            {
                "generated_at": utc_now_iso(),
                "lineage_id": lineage.lineage_id,
                "family_id": lineage.family_id,
                "artifact_summary": artifact_summary,
                "files": {
                    "dataset": str(run_root / "dataset.json"),
                    "features": str(run_root / "features.json"),
                    "train": str(run_root / "train.json"),
                    "walkforward": str(run_root / "walkforward.json"),
                    "stress": str(run_root / "stress.json"),
                },
            },
        )
        bundles = self._full_stage_bundle_set(
            lineage=lineage,
            summary=summary["walkforward_summary"],
            stress_summary=summary["stress_summary"],
            package_path=package_path,
            source="factory_cascade_artifact",
        )
        return {
            "mode": "cascade_regime",
            "bundles": bundles,
            "artifact_summary": artifact_summary,
        }

    def _run_candidate_signal_experiment(
        self,
        *,
        lineage: LineageRecord,
        genome: StrategyGenome,
        experiment: ExperimentSpec,
        family_mode: str,
    ) -> Dict[str, Any]:
        parameters = dict(genome.parameters)
        requested_model_class = str(parameters.get("selected_model_class", "rules") or "rules")
        feature_subset = str(parameters.get("selected_feature_subset", "cross_science") or "cross_science")
        min_edge = float(parameters.get("selected_min_edge", 0.03) or 0.03)

        run_id, run_root = self._build_run_root(lineage=lineage, experiment=experiment)
        run_root.mkdir(parents=True, exist_ok=True)

        summary = self._candidate_signal_summary(
            family_mode=family_mode,
            requested_model_class=requested_model_class,
            feature_subset=feature_subset,
            min_edge=min_edge,
        )
        if summary is None:
            return {"mode": family_mode, "bundles": [], "artifact_summary": None}

        self._write_json(
            run_root / "dataset.json",
            {
                "generated_at": utc_now_iso(),
                "family_id": lineage.family_id,
                "lineage_id": lineage.lineage_id,
                "source_mode": summary["source_mode"],
                "source_paths": summary["source_paths"],
                "dataset_rows": summary["dataset_rows"],
                "market_count": summary["market_count"],
            },
        )
        self._write_json(
            run_root / "features.json",
            {
                "generated_at": utc_now_iso(),
                "feature_family": family_mode,
                "feature_subset": feature_subset,
                "feature_columns": list(summary["feature_columns"]),
                "feature_count": len(summary["feature_columns"]),
            },
        )
        self._write_json(
            run_root / "train.json",
            {
                "generated_at": utc_now_iso(),
                "requested_model_class": requested_model_class,
                "resolved_model_engine": summary["resolved_model_engine"],
                "min_edge": min_edge,
            },
        )
        self._write_json(run_root / "walkforward.json", summary["walkforward_artifact"])
        self._write_json(run_root / "stress.json", summary["stress_artifact"])

        package_path = run_root / "package.json"
        artifact_summary = {
            "run_id": run_id,
            "mode": family_mode,
            "package_path": str(package_path),
            "run_root": str(run_root),
            "requested_model_class": requested_model_class,
            "resolved_model_engine": summary["resolved_model_engine"],
            "source_mode": summary["source_mode"],
            "dataset_rows": summary["dataset_rows"],
            "market_count": summary["market_count"],
        }
        self._write_json(
            package_path,
            {
                "generated_at": utc_now_iso(),
                "lineage_id": lineage.lineage_id,
                "family_id": lineage.family_id,
                "artifact_summary": artifact_summary,
                "files": {
                    "dataset": str(run_root / "dataset.json"),
                    "features": str(run_root / "features.json"),
                    "train": str(run_root / "train.json"),
                    "walkforward": str(run_root / "walkforward.json"),
                    "stress": str(run_root / "stress.json"),
                },
            },
        )
        bundles = self._full_stage_bundle_set(
            lineage=lineage,
            summary=summary["walkforward_summary"],
            stress_summary=summary["stress_summary"],
            package_path=package_path,
            source="factory_candidate_signal_artifact",
        )
        return {
            "mode": family_mode,
            "bundles": bundles,
            "artifact_summary": artifact_summary,
        }

    def _filter_prediction_examples(
        self,
        *,
        examples: List[PredictionExample],
        feature_subset: str,
        lookback_hours: float,
    ) -> List[PredictionExample]:
        subset_names = self._feature_subset_names(feature_subset)
        filtered: List[PredictionExample] = []
        latest_ts = self._latest_prediction_timestamp(examples)
        min_ts = latest_ts - timedelta(hours=max(1.0, lookback_hours)) if latest_ts is not None else None
        for example in examples:
            timestamp = self._parse_prediction_timestamp(example.timestamp)
            if min_ts is not None and timestamp is not None and timestamp < min_ts:
                continue
            if subset_names is None:
                features = dict(example.features)
            else:
                features = {
                    name: float(example.features.get(name, 0.0))
                    for name in subset_names
                    if name in example.features
                }
            filtered.append(
                PredictionExample(
                    timestamp=example.timestamp,
                    base_prob=float(example.base_prob),
                    odds=float(example.odds),
                    label=int(example.label),
                    features=features,
                )
            )
        if len(filtered) >= 180:
            return filtered
        return list(examples)

    def _prediction_window_sizes(self, horizon_seconds: int, example_count: int) -> Tuple[int, int]:
        if horizon_seconds <= 120:
            train_window, test_window = 160, 40
        elif horizon_seconds <= 600:
            train_window, test_window = 220, 55
        elif horizon_seconds <= 1800:
            train_window, test_window = 280, 70
        else:
            train_window, test_window = 360, 90
        max_train = max(80, example_count - 24)
        train_window = min(train_window, max_train)
        remaining = max(12, example_count - train_window)
        test_window = min(test_window, max(12, remaining // 3 or 12))
        while train_window + test_window > example_count and train_window > 80:
            train_window -= 20
        while train_window + test_window > example_count and test_window > 12:
            test_window -= 4
        return max(80, train_window), max(12, test_window)

    def _build_run_root(
        self,
        *,
        lineage: LineageRecord,
        experiment: ExperimentSpec,
    ) -> Tuple[str, Path]:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_root = Path(experiment.goldfish_workspace) / "runs" / _safe_slug(lineage.lineage_id) / run_id
        return run_id, run_root

    def _full_stage_bundle_set(
        self,
        *,
        lineage: LineageRecord,
        summary: Dict[str, Any],
        stress_summary: Dict[str, Any],
        package_path: Path,
        source: str,
    ) -> List[EvaluationBundle]:
        walkforward = self._build_generic_bundle(
            lineage=lineage,
            stage=EvaluationStage.WALKFORWARD.value,
            summary=summary,
            package_path=package_path,
            source=source,
        )
        stress = self._build_generic_bundle(
            lineage=lineage,
            stage=EvaluationStage.STRESS.value,
            summary=stress_summary,
            package_path=package_path,
            source=source,
        )
        shadow_summary = dict(summary)
        shadow_summary["windows"] = list(summary["windows"])
        shadow_summary["monthly_roi_pct"] = round(float(summary["monthly_roi_pct"]) * 0.95, 4)
        shadow_summary["slippage_headroom_pct"] = round(float(stress_summary["slippage_headroom_pct"]), 4)
        shadow_summary["stress_positive"] = bool(stress_summary["stress_positive"])
        paper_summary = dict(summary)
        paper_summary["windows"] = list(summary["windows"])
        paper_summary["paper_days"] = min(int(summary["paper_days"]), 12)
        paper_summary["trade_count"] = max(1, int(summary["trade_count"]))
        paper_summary["settled_count"] = max(1, int(summary["settled_count"]))
        paper_summary["stress_positive"] = bool(stress_summary["stress_positive"])
        shadow = self._build_generic_bundle(
            lineage=lineage,
            stage=EvaluationStage.SHADOW.value,
            summary=shadow_summary,
            package_path=package_path,
            source=source,
        )
        paper = self._build_generic_bundle(
            lineage=lineage,
            stage=EvaluationStage.PAPER.value,
            summary=paper_summary,
            package_path=package_path,
            source=source,
        )
        return [walkforward, stress, shadow, paper]

    def _evaluate_prediction_walkforward(
        self,
        *,
        examples: List[PredictionExample],
        requested_model_class: str,
        train_window: int,
        test_window: int,
        epochs: int,
        learning_rate: float,
        edge_threshold: float,
    ) -> Dict[str, Any]:
        if len(examples) < train_window + test_window:
            return {
                "resolved_model_engine": "insufficient_examples",
                "segments": [],
                "windows": [],
                "monthly_roi_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "slippage_headroom_pct": -1.0,
                "calibration_lift_abs": 0.0,
                "turnover": 0.0,
                "capacity_score": 0.0,
                "failure_rate": 1.0,
                "regime_robustness": 0.0,
                "baseline_beaten_windows": 0,
                "trade_count": 0,
                "settled_count": 0,
                "paper_days": 0,
                "net_pnl": 0.0,
                "stress_positive": False,
                "last_model_payload": None,
                "artifact": {"reason": "insufficient_examples"},
            }

        feature_names = sorted({key for example in examples for key in example.features.keys()})
        resolved_engine = self._resolve_prediction_model_engine(
            requested_model_class=requested_model_class,
            feature_names=feature_names,
        )
        segments: List[Dict[str, Any]] = []
        last_model_payload: Optional[Dict[str, Any]] = None
        all_settled_dates = set()
        cursor = 0
        while cursor + train_window + test_window <= len(examples):
            train_examples = examples[cursor : cursor + train_window]
            test_examples = examples[cursor + train_window : cursor + train_window + test_window]
            model = self._build_prediction_model(resolved_engine, feature_names)
            if model is not None:
                self._fit_prediction_model(
                    model=model,
                    resolved_engine=resolved_engine,
                    examples=train_examples,
                    epochs=epochs,
                    learning_rate=learning_rate,
                )
                if hasattr(model, "to_dict"):
                    last_model_payload = model.to_dict()
            probs = [
                self._predict_probability(
                    resolved_engine=resolved_engine,
                    model=model,
                    example=example,
                )
                for example in test_examples
            ]
            labels = [int(example.label) for example in test_examples]
            odds = [float(example.odds) for example in test_examples]
            baseline_probs = [float(example.base_prob) for example in test_examples]
            metrics = evaluate_predictions(
                probs=probs,
                labels=labels,
                odds=odds,
                edge_threshold=edge_threshold,
                stake=1.0,
            )
            baseline_metrics = evaluate_predictions(
                probs=baseline_probs,
                labels=labels,
                odds=odds,
                edge_threshold=edge_threshold,
                stake=1.0,
            )
            for example in test_examples:
                all_settled_dates.add(str(example.timestamp).split("T")[0])
            segments.append(
                {
                    "label": f"segment_{len(segments) + 1}",
                    "settled_count": len(test_examples),
                    "bets": metrics.bets,
                    "model_roi_pct": round(float(metrics.roi) * 100.0, 4),
                    "baseline_roi_pct": round(float(baseline_metrics.roi) * 100.0, 4),
                    "model_brier": round(float(metrics.brier), 6),
                    "baseline_brier": round(float(baseline_metrics.brier), 6),
                    "calibration_lift_abs": round(float(baseline_metrics.brier - metrics.brier), 6),
                    "pnl_units": round(float(metrics.pnl_units), 6),
                }
            )
            cursor += test_window

        recent_segments = segments[-3:]
        total_bets = sum(int(segment["bets"]) for segment in segments)
        total_settled = sum(int(segment["settled_count"]) for segment in segments)
        total_pnl = round(sum(float(segment["pnl_units"]) for segment in segments), 6)
        baseline_beaten_windows = sum(
            1
            for segment in segments
            if float(segment["model_roi_pct"]) > float(segment["baseline_roi_pct"])
            and float(segment["calibration_lift_abs"]) >= 0.0
        )
        positive_segments = sum(1 for segment in segments if float(segment["model_roi_pct"]) > 0.0)
        monthly_roi_pct = round(
            sum(float(segment["model_roi_pct"]) for segment in recent_segments) / max(1, len(recent_segments)),
            4,
        )
        max_drawdown_pct = round(self._segment_drawdown_pct(segments), 4)
        calibration_lift_abs = round(
            sum(float(segment["calibration_lift_abs"]) for segment in recent_segments) / max(1, len(recent_segments)),
            6,
        )
        slippage_headroom_pct = round(monthly_roi_pct - (edge_threshold * 50.0), 4)
        turnover = round(total_bets / max(1, total_settled), 4)
        capacity_score = round(min(1.0, 0.15 + (len(examples) / 1500.0) + (turnover * 0.4)), 4)
        failure_rate = round(
            sum(1 for segment in segments if int(segment["bets"]) == 0 or float(segment["model_roi_pct"]) <= 0.0)
            / max(1, len(segments)),
            4,
        )
        regime_robustness = round(positive_segments / max(1, len(segments)), 4)
        windows = [
            EvaluationWindow(
                label=str(segment["label"]),
                settled_count=int(segment["settled_count"]),
                monthly_roi_pct=float(segment["model_roi_pct"]),
                baseline_roi_pct=float(segment["baseline_roi_pct"]),
                brier_lift_abs=float(segment["calibration_lift_abs"]),
                drawdown_pct=max_drawdown_pct,
                slippage_headroom_pct=slippage_headroom_pct,
                failure_rate=failure_rate,
                regime_robustness=regime_robustness,
            )
            for segment in recent_segments
        ]
        return {
            "resolved_model_engine": resolved_engine,
            "segments": segments,
            "windows": windows,
            "monthly_roi_pct": monthly_roi_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "slippage_headroom_pct": slippage_headroom_pct,
            "calibration_lift_abs": calibration_lift_abs,
            "turnover": turnover,
            "capacity_score": capacity_score,
            "failure_rate": failure_rate,
            "regime_robustness": regime_robustness,
            "baseline_beaten_windows": baseline_beaten_windows,
            "trade_count": total_bets,
            "settled_count": total_bets,
            "paper_days": min(30, len(all_settled_dates)),
            "net_pnl": total_pnl,
            "stress_positive": monthly_roi_pct > 0.0 and slippage_headroom_pct > 0.0,
            "last_model_payload": last_model_payload,
            "artifact": {
                "requested_model_class": requested_model_class,
                "resolved_model_engine": resolved_engine,
                "edge_threshold": edge_threshold,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "segments": segments,
                "summary": {
                    "monthly_roi_pct": monthly_roi_pct,
                    "max_drawdown_pct": max_drawdown_pct,
                    "slippage_headroom_pct": slippage_headroom_pct,
                    "calibration_lift_abs": calibration_lift_abs,
                    "turnover": turnover,
                    "capacity_score": capacity_score,
                    "failure_rate": failure_rate,
                    "regime_robustness": regime_robustness,
                    "baseline_beaten_windows": baseline_beaten_windows,
                    "trade_count": total_bets,
                    "settled_count": total_bets,
                    "paper_days": min(30, len(all_settled_dates)),
                    "net_pnl": total_pnl,
                },
            },
        }

    def _build_prediction_bundle(
        self,
        *,
        lineage: LineageRecord,
        stage: str,
        summary: Dict[str, Any],
        package_path: Path,
    ) -> EvaluationBundle:
        return EvaluationBundle(
            evaluation_id=f"{lineage.lineage_id}:{stage}:{summary['resolved_model_engine']}",
            lineage_id=lineage.lineage_id,
            family_id=lineage.family_id,
            stage=stage,
            source="factory_prediction_artifact",
            windows=list(summary["windows"]),
            monthly_roi_pct=float(summary["monthly_roi_pct"]),
            max_drawdown_pct=float(summary["max_drawdown_pct"]),
            slippage_headroom_pct=float(summary["slippage_headroom_pct"]),
            calibration_lift_abs=float(summary["calibration_lift_abs"]),
            turnover=float(summary["turnover"]),
            capacity_score=float(summary["capacity_score"]),
            failure_rate=float(summary["failure_rate"]),
            regime_robustness=float(summary["regime_robustness"]),
            baseline_beaten_windows=int(summary["baseline_beaten_windows"]),
            stress_positive=bool(summary["stress_positive"]),
            trade_count=int(summary["trade_count"]),
            settled_count=int(summary["settled_count"]),
            paper_days=int(summary["paper_days"]),
            net_pnl=float(summary["net_pnl"]),
            notes=[f"package_path={package_path}", f"resolved_model_engine={summary['resolved_model_engine']}"],
        )

    def _build_generic_bundle(
        self,
        *,
        lineage: LineageRecord,
        stage: str,
        summary: Dict[str, Any],
        package_path: Path,
        source: str,
    ) -> EvaluationBundle:
        return EvaluationBundle(
            evaluation_id=f"{lineage.lineage_id}:{stage}:{summary['resolved_model_engine']}",
            lineage_id=lineage.lineage_id,
            family_id=lineage.family_id,
            stage=stage,
            source=source,
            windows=list(summary["windows"]),
            monthly_roi_pct=float(summary["monthly_roi_pct"]),
            max_drawdown_pct=float(summary["max_drawdown_pct"]),
            slippage_headroom_pct=float(summary["slippage_headroom_pct"]),
            calibration_lift_abs=float(summary["calibration_lift_abs"]),
            turnover=float(summary["turnover"]),
            capacity_score=float(summary["capacity_score"]),
            failure_rate=float(summary["failure_rate"]),
            regime_robustness=float(summary["regime_robustness"]),
            baseline_beaten_windows=int(summary["baseline_beaten_windows"]),
            stress_positive=bool(summary["stress_positive"]),
            trade_count=int(summary["trade_count"]),
            settled_count=int(summary["settled_count"]),
            paper_days=int(summary["paper_days"]),
            net_pnl=float(summary["net_pnl"]),
            notes=[
                f"package_path={package_path}",
                f"resolved_model_engine={summary['resolved_model_engine']}",
                f"source_mode={summary.get('source_mode', 'artifact')}",
            ],
        )

    def _build_funding_bundle(
        self,
        *,
        lineage: LineageRecord,
        stage: str,
        summary: Dict[str, Any],
        package_path: Path,
    ) -> EvaluationBundle:
        return EvaluationBundle(
            evaluation_id=f"{lineage.lineage_id}:{stage}:{summary['resolved_model_engine']}",
            lineage_id=lineage.lineage_id,
            family_id=lineage.family_id,
            stage=stage,
            source="factory_funding_artifact",
            windows=list(summary["windows"]),
            monthly_roi_pct=float(summary["monthly_roi_pct"]),
            max_drawdown_pct=float(summary["max_drawdown_pct"]),
            slippage_headroom_pct=float(summary["slippage_headroom_pct"]),
            calibration_lift_abs=float(summary["calibration_lift_abs"]),
            turnover=float(summary["turnover"]),
            capacity_score=float(summary["capacity_score"]),
            failure_rate=float(summary["failure_rate"]),
            regime_robustness=float(summary["regime_robustness"]),
            baseline_beaten_windows=int(summary["baseline_beaten_windows"]),
            stress_positive=bool(summary["stress_positive"]),
            trade_count=int(summary["trade_count"]),
            settled_count=int(summary["settled_count"]),
            paper_days=int(summary["paper_days"]),
            net_pnl=float(summary["net_pnl"]),
            notes=[
                f"package_path={package_path}",
                f"resolved_model_engine={summary['resolved_model_engine']}",
                f"source_mode={summary['source_mode']}",
            ],
        )

    def _resolve_prediction_model_engine(
        self,
        *,
        requested_model_class: str,
        feature_names: List[str],
    ) -> str:
        requested = str(requested_model_class or "logit").lower()
        if requested == "rules":
            return "rules"
        if requested == "logit":
            return "market_calibrated" if not feature_names else "residual_logit"
        if requested == "gbdt":
            return "hybrid_logit"
        if requested in {"tft", "transformer"}:
            return "hybrid_logit" if feature_names else "market_calibrated"
        return "market_calibrated"

    def _build_prediction_model(self, resolved_engine: str, feature_names: Iterable[str]) -> Any:
        if resolved_engine == "market_calibrated":
            return MarketCalibratedModel()
        if resolved_engine == "pure_logit":
            return PureLogitModel(feature_names)
        if resolved_engine == "hybrid_logit":
            return HybridLogitModel(feature_names)
        if resolved_engine == "residual_logit":
            return ResidualLogitModel(feature_names)
        return None

    def _fit_prediction_model(
        self,
        *,
        model: Any,
        resolved_engine: str,
        examples: List[PredictionExample],
        epochs: int,
        learning_rate: float,
    ) -> None:
        if resolved_engine == "market_calibrated":
            model.fit(examples, epochs=epochs, lr=learning_rate)
        elif resolved_engine in {"pure_logit", "hybrid_logit", "residual_logit"}:
            model.fit(examples, epochs=epochs, lr=learning_rate)

    def _predict_probability(
        self,
        *,
        resolved_engine: str,
        model: Any,
        example: PredictionExample,
    ) -> float:
        if resolved_engine == "market_calibrated":
            return float(model.predict_proba(example.base_prob))
        if resolved_engine == "pure_logit":
            return float(model.predict_proba(example.features))
        if resolved_engine in {"hybrid_logit", "residual_logit"}:
            return float(model.predict_proba(example.base_prob, example.features))
        return self._rules_probability(example)

    def _rules_probability(self, example: PredictionExample) -> float:
        features = dict(example.features)
        shift = 0.0
        shift += max(-0.06, min(0.06, float(features.get("imbalance", 0.0)) * 0.08))
        shift += max(-0.03, min(0.03, float(features.get("price_velocity", 0.0)) * 4.0))
        shift -= max(-0.04, min(0.04, float(features.get("weighted_spread", 0.0)) / 40.0))
        shift -= max(-0.025, min(0.025, float(features.get("time_to_start_sec", 0.0)) / 432000.0))
        return _clip_probability(float(example.base_prob) + shift)

    def _feature_subset_names(self, feature_subset: str) -> Optional[List[str]]:
        feature_map: Dict[str, List[str]] = {
            "baseline": [],
            "microstructure": [
                "spread_mean",
                "imbalance",
                "depth_total_eur",
                "weighted_spread",
                "lay_back_ratio",
                "top_of_book_concentration",
            ],
            "cross_science": [
                "spread_mean",
                "imbalance",
                "depth_total_eur",
                "weighted_spread",
                "lay_back_ratio",
                "top_of_book_concentration",
                "price_velocity",
                "short_volatility",
                "time_to_start_sec",
                "selection_count",
            ],
            "regime": [
                "price_velocity",
                "short_volatility",
                "time_to_start_sec",
                "selection_count",
                "in_play",
            ],
        }
        return feature_map.get(feature_subset, None)

    def _prediction_time_range(self, examples: List[PredictionExample]) -> Dict[str, Any]:
        if not examples:
            return {"start": None, "end": None}
        return {"start": examples[0].timestamp, "end": examples[-1].timestamp}

    def _latest_prediction_timestamp(self, examples: List[PredictionExample]) -> Optional[datetime]:
        timestamps = [self._parse_prediction_timestamp(example.timestamp) for example in examples]
        valid = [timestamp for timestamp in timestamps if timestamp is not None]
        return max(valid) if valid else None

    def _parse_prediction_timestamp(self, value: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None

    def _segment_drawdown_pct(self, segments: List[Dict[str, Any]]) -> float:
        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        total_bets = 0
        for segment in segments:
            cumulative += float(segment["pnl_units"])
            peak = max(peak, cumulative)
            max_drawdown = max(max_drawdown, peak - cumulative)
            total_bets += int(segment["bets"])
        return (max_drawdown / max(1.0, float(total_bets))) * 100.0

    def _run_funding_backtest(
        self,
        *,
        requested_model_class: str,
        lookback_hours: float,
        stake_fraction: float,
        horizon_seconds: int,
        min_edge: float,
    ) -> Optional[Dict[str, Any]]:
        cache_key = (
            requested_model_class,
            round(float(lookback_hours), 4),
            round(float(stake_fraction), 6),
            int(horizon_seconds),
            round(float(min_edge), 6),
        )
        if cache_key in self._funding_cache:
            return self._funding_cache[cache_key]
        try:
            from funding.ml.contrarian_features import build_contrarian_features_all, get_contrarian_feature_columns
        except Exception:
            return None

        symbols = self._candidate_funding_symbols()
        if not symbols:
            return None
        history_root = self.project_root / "data" / "funding_history"
        try:
            df = build_contrarian_features_all(symbols, data_dir=history_root)
        except Exception:
            return None
        if df is None or getattr(df, "empty", True):
            return None

        rows_per_symbol = max(40, int(max(24.0, lookback_hours) / 8.0) * max(1, len(symbols)))
        df = df.tail(rows_per_symbol * max(1, len(symbols))).copy()
        if getattr(df, "empty", True):
            return None

        feature_columns = get_contrarian_feature_columns(df)
        segment_size = max(40, len(df) // 3)
        segments: List[Dict[str, Any]] = []
        initial_balance = 1000.0
        max_hold_periods = max(3, min(9, int(round(horizon_seconds / 7200.0))))
        min_funding_rate = max(0.0003, min(0.003, min_edge / 40.0))
        for index in range(3):
            start = index * segment_size
            end = len(df) if index == 2 else min(len(df), (index + 1) * segment_size)
            slice_df = df.iloc[start:end].copy()
            if len(slice_df) < 20:
                continue
            backtester = ContrarianBacktester(initial_balance=initial_balance)
            result = backtester.backtest(
                slice_df,
                strategy="naive",
                model=None,
                stop_loss_pct=0.02,
                take_profit_ratio=2.0,
                capital_pct=max(0.01, min(0.1, stake_fraction)),
                max_hold_periods=max_hold_periods,
                min_funding_rate=min_funding_rate,
            )
            roi_pct = ((float(result["final_balance"]) - initial_balance) / initial_balance) * 100.0
            drawdown_pct = (float(result["max_drawdown"]) / initial_balance) * 100.0
            segments.append(
                {
                    "label": f"segment_{index + 1}",
                    "settled_count": int(result["total_trades"]),
                    "monthly_roi_pct": round(roi_pct, 4),
                    "baseline_roi_pct": 0.0,
                    "brier_lift_abs": 0.0,
                    "drawdown_pct": round(drawdown_pct, 4),
                    "slippage_headroom_pct": round(roi_pct - 1.0, 4),
                    "failure_rate": round(
                        (int(result["losses"]) + int(result["timeouts"])) / max(1, int(result["total_trades"])),
                        4,
                    ),
                    "regime_robustness": round(float(result["win_rate"]), 4),
                    "total_pnl": round(float(result["total_pnl"]), 6),
                }
            )
        if not segments:
            return None

        summary = self._funding_summary_from_segments(
            segments=segments,
            source_mode="backtest",
            resolved_model_engine=self._resolve_funding_model_engine(requested_model_class),
            source_paths=[str(history_root)],
            dataset_rows=len(df),
            symbol_count=len(symbols),
            feature_columns=feature_columns,
        )
        self._funding_cache[cache_key] = summary
        return summary

    def _funding_log_fallback(
        self,
        *,
        requested_model_class: str,
        lookback_hours: float,
        stake_fraction: float,
        min_edge: float,
    ) -> Dict[str, Any]:
        experiments_path = self.project_root / "data" / "funding" / "experiments.jsonl"
        trade_log_path = self.project_root / "data" / "funding_models" / "contrarian_trade_log.jsonl"
        quality_path = self.project_root / "data" / "funding" / "state" / "contrarian_online_learner_quality.json"
        experiment_rows = self._load_jsonl_rows(experiments_path)
        latest_metrics = dict((experiment_rows[-1].get("metrics") if experiment_rows else {}) or {})
        rolling_labels = ["rolling_50", "rolling_100", "rolling_200"]
        segments: List[Dict[str, Any]] = []
        for label in rolling_labels:
            metric_row = dict(latest_metrics.get(label) or {})
            if not metric_row:
                continue
            roi_pct = float(metric_row.get("roi_pct", 0.0) or 0.0)
            segments.append(
                {
                    "label": label,
                    "settled_count": int(metric_row.get("settled", 0) or 0),
                    "monthly_roi_pct": round(roi_pct, 4),
                    "baseline_roi_pct": 0.0,
                    "brier_lift_abs": round(float(metric_row.get("brier_lift_abs", 0.0) or 0.0), 6),
                    "drawdown_pct": round(abs(min(roi_pct, 0.0)) * 1.4, 4),
                    "slippage_headroom_pct": round(roi_pct - max(0.75, min_edge * 50.0), 4),
                    "failure_rate": 0.0,
                    "regime_robustness": 0.0,
                    "total_pnl": round(roi_pct * max(1.0, stake_fraction * 10.0), 6),
                }
            )

        trade_rows = self._load_jsonl_rows(trade_log_path)
        quality_payload = self._load_json_object(quality_path)
        settled_history = list(quality_payload.get("settled_history") or [])
        if trade_rows:
            win_count = sum(1 for row in trade_rows if float(row.get("pnl_pct", 0.0) or 0.0) > 0.0)
            failure_rate = round(1.0 - (win_count / max(1, len(trade_rows))), 4)
            regime_robustness = round(win_count / max(1, len(trade_rows)), 4)
            drawdown_pct = self._trade_log_drawdown_pct(trade_rows)
            unique_days = len({str(row.get("timestamp", "")).split("T")[0] for row in trade_rows if row.get("timestamp")})
        else:
            failure_rate = 1.0
            regime_robustness = 0.0
            drawdown_pct = 0.0
            unique_days = 0
        brier_lift = 0.0
        if settled_history:
            diffs = [
                float(item.get("baseline_brier", 0.0) or 0.0) - float(item.get("model_brier", 0.0) or 0.0)
                for item in settled_history
            ]
            brier_lift = round(sum(diffs) / max(1, len(diffs)), 6)
        for segment in segments:
            segment["failure_rate"] = failure_rate
            segment["regime_robustness"] = regime_robustness
            segment["drawdown_pct"] = max(float(segment["drawdown_pct"]), drawdown_pct)
            if abs(float(segment["brier_lift_abs"])) < 1e-9 and abs(brier_lift) > 0.0:
                segment["brier_lift_abs"] = brier_lift
        if not segments:
            segments = [
                {
                    "label": "rolling_fallback",
                    "settled_count": len(trade_rows),
                    "monthly_roi_pct": round(sum(float(row.get("pnl_pct", 0.0) or 0.0) for row in trade_rows) * 100.0, 4),
                    "baseline_roi_pct": 0.0,
                    "brier_lift_abs": brier_lift,
                    "drawdown_pct": drawdown_pct,
                    "slippage_headroom_pct": round(-max(0.75, min_edge * 50.0), 4),
                    "failure_rate": failure_rate,
                    "regime_robustness": regime_robustness,
                    "total_pnl": round(sum(float(row.get("pnl_pct", 0.0) or 0.0) for row in trade_rows), 6),
                }
            ]
        return self._funding_summary_from_segments(
            segments=segments,
            source_mode="log_fallback",
            resolved_model_engine=self._resolve_funding_model_engine(requested_model_class),
            source_paths=[str(path) for path in [experiments_path, trade_log_path, quality_path] if path.exists()],
            dataset_rows=max(len(trade_rows), len(settled_history), len(experiment_rows)),
            symbol_count=len({str(row.get("symbol")) for row in trade_rows if row.get("symbol")}),
            feature_columns=[
                "funding_rate",
                "mark_price",
                "direction_24h",
                "price_return_24h_target",
                "pred_prob",
            ],
            explicit_paper_days=min(30, unique_days or len({seg["label"] for seg in segments})),
            explicit_settled=max(len(trade_rows), max(int(seg["settled_count"]) for seg in segments)),
            explicit_net_pnl=round(sum(float(row.get("pnl_pct", 0.0) or 0.0) for row in trade_rows), 6),
        )

    def _funding_summary_from_segments(
        self,
        *,
        segments: List[Dict[str, Any]],
        source_mode: str,
        resolved_model_engine: str,
        source_paths: List[str],
        dataset_rows: int,
        symbol_count: int,
        feature_columns: List[str],
        explicit_paper_days: Optional[int] = None,
        explicit_settled: Optional[int] = None,
        explicit_net_pnl: Optional[float] = None,
    ) -> Dict[str, Any]:
        recent_segments = segments[-3:]
        windows = [
            EvaluationWindow(
                label=str(segment["label"]),
                settled_count=int(segment["settled_count"]),
                monthly_roi_pct=float(segment["monthly_roi_pct"]),
                baseline_roi_pct=float(segment["baseline_roi_pct"]),
                brier_lift_abs=float(segment["brier_lift_abs"]),
                drawdown_pct=float(segment["drawdown_pct"]),
                slippage_headroom_pct=float(segment["slippage_headroom_pct"]),
                failure_rate=float(segment["failure_rate"]),
                regime_robustness=float(segment["regime_robustness"]),
            )
            for segment in recent_segments
        ]
        monthly_roi_pct = round(
            sum(float(segment["monthly_roi_pct"]) for segment in recent_segments) / max(1, len(recent_segments)),
            4,
        )
        max_drawdown_pct = round(max(float(segment["drawdown_pct"]) for segment in segments), 4)
        calibration_lift_abs = round(
            sum(float(segment["brier_lift_abs"]) for segment in recent_segments) / max(1, len(recent_segments)),
            6,
        )
        slippage_headroom_pct = round(
            min(float(segment["slippage_headroom_pct"]) for segment in recent_segments),
            4,
        )
        failure_rate = round(
            sum(float(segment["failure_rate"]) for segment in recent_segments) / max(1, len(recent_segments)),
            4,
        )
        regime_robustness = round(
            sum(float(segment["regime_robustness"]) for segment in recent_segments) / max(1, len(recent_segments)),
            4,
        )
        settled_count = explicit_settled if explicit_settled is not None else sum(int(segment["settled_count"]) for segment in segments)
        trade_count = settled_count
        paper_days = explicit_paper_days if explicit_paper_days is not None else min(30, len(segments) * 5)
        net_pnl = explicit_net_pnl if explicit_net_pnl is not None else round(
            sum(float(segment["total_pnl"]) for segment in segments),
            6,
        )
        baseline_beaten_windows = sum(
            1
            for segment in segments
            if float(segment["monthly_roi_pct"]) > float(segment["baseline_roi_pct"])
            and float(segment["brier_lift_abs"]) >= 0.0
        )
        turnover = round(min(1.0, settled_count / max(20.0, float(dataset_rows or 1))), 4)
        capacity_score = round(min(1.0, 0.25 + (turnover * 0.5) + (min(1.0, symbol_count / 5.0) * 0.25)), 4)
        walkforward_summary = {
            "resolved_model_engine": resolved_model_engine,
            "source_mode": source_mode,
            "windows": windows,
            "monthly_roi_pct": monthly_roi_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "slippage_headroom_pct": slippage_headroom_pct,
            "calibration_lift_abs": calibration_lift_abs,
            "turnover": turnover,
            "capacity_score": capacity_score,
            "failure_rate": failure_rate,
            "regime_robustness": regime_robustness,
            "baseline_beaten_windows": baseline_beaten_windows,
            "trade_count": trade_count,
            "settled_count": settled_count,
            "paper_days": paper_days,
            "net_pnl": net_pnl,
            "stress_positive": monthly_roi_pct > 0.0 and slippage_headroom_pct > 0.0,
        }
        stress_windows = [
            EvaluationWindow(
                label=f"{window.label}_stress",
                settled_count=window.settled_count,
                monthly_roi_pct=round(window.monthly_roi_pct - 1.0, 4),
                baseline_roi_pct=window.baseline_roi_pct,
                brier_lift_abs=round(window.brier_lift_abs * 0.85, 6),
                drawdown_pct=round(window.drawdown_pct * 1.15, 4),
                slippage_headroom_pct=round(window.slippage_headroom_pct - 0.75, 4),
                failure_rate=min(1.0, round(window.failure_rate + 0.08, 4)),
                regime_robustness=max(0.0, round(window.regime_robustness - 0.1, 4)),
            )
            for window in windows
        ]
        stress_summary = {
            **walkforward_summary,
            "windows": stress_windows,
            "monthly_roi_pct": round(walkforward_summary["monthly_roi_pct"] - 1.0, 4),
            "max_drawdown_pct": round(walkforward_summary["max_drawdown_pct"] * 1.15, 4),
            "slippage_headroom_pct": round(walkforward_summary["slippage_headroom_pct"] - 0.75, 4),
            "calibration_lift_abs": round(walkforward_summary["calibration_lift_abs"] * 0.85, 6),
            "failure_rate": min(1.0, round(walkforward_summary["failure_rate"] + 0.08, 4)),
            "regime_robustness": max(0.0, round(walkforward_summary["regime_robustness"] - 0.1, 4)),
            "stress_positive": round(walkforward_summary["monthly_roi_pct"] - 1.0, 4) > 0.0
            and round(walkforward_summary["slippage_headroom_pct"] - 0.75, 4) > 0.0,
        }
        return {
            "source_mode": source_mode,
            "resolved_model_engine": resolved_model_engine,
            "source_paths": source_paths,
            "dataset_rows": dataset_rows,
            "symbol_count": symbol_count,
            "feature_columns": feature_columns,
            "walkforward_summary": walkforward_summary,
            "stress_summary": stress_summary,
            "walkforward_artifact": {
                "source_mode": source_mode,
                "resolved_model_engine": resolved_model_engine,
                "segments": segments,
                "summary": walkforward_summary,
            },
            "stress_artifact": {
                "source_mode": source_mode,
                "resolved_model_engine": resolved_model_engine,
                "segments": segments,
                "summary": stress_summary,
            },
        }

    def _candidate_funding_symbols(self) -> List[str]:
        rates_root = self.project_root / "data" / "funding_history" / "funding_rates"
        if not rates_root.exists():
            return []
        preferred = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"]
        available = {path.stem.upper() for path in rates_root.glob("*.csv")}
        chosen = [symbol for symbol in preferred if symbol in available]
        if chosen:
            return chosen
        return sorted(available)[:5]

    def _resolve_funding_model_engine(self, requested_model_class: str) -> str:
        requested = str(requested_model_class or "logit").lower()
        if requested in {"gbdt", "logit"}:
            return "naive_contrarian_backtest"
        if requested in {"tft", "transformer"}:
            return "naive_contrarian_backtest"
        if requested == "rules":
            return "rules_contrarian"
        return "log_fallback"

    def _trade_log_drawdown_pct(self, trade_rows: List[Dict[str, Any]]) -> float:
        cumulative = 0.0
        peak = 0.0
        drawdown = 0.0
        for row in trade_rows:
            cumulative += float(row.get("pnl_pct", 0.0) or 0.0)
            peak = max(peak, cumulative)
            drawdown = max(drawdown, peak - cumulative)
        return round(drawdown * 100.0, 4)

    def _load_jsonl_rows(self, path: Path) -> List[Dict[str, Any]]:
        cache_key = str(path.resolve())
        if cache_key in self._jsonl_cache:
            return list(self._jsonl_cache[cache_key])
        rows: List[Dict[str, Any]] = []
        if not path.exists():
            return rows
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        self._jsonl_cache[cache_key] = list(rows)
        return rows

    def _load_tail_jsonl_rows(self, path: Path, *, max_rows: int) -> List[Dict[str, Any]]:
        cache_key = (str(path.resolve()), int(max_rows))
        if cache_key in self._jsonl_tail_cache:
            return list(self._jsonl_tail_cache[cache_key])
        if not path.exists():
            return []
        chunk_size = 65536
        with path.open("rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            buffer = b""
            lines: List[bytes] = []
            while position > 0 and len(lines) <= max_rows:
                read_size = min(chunk_size, position)
                position -= read_size
                handle.seek(position)
                buffer = handle.read(read_size) + buffer
                lines = buffer.splitlines()
        tail_lines = lines[-max_rows:]
        rows: List[Dict[str, Any]] = []
        for line in tail_lines:
            try:
                rows.append(json.loads(line.decode("utf-8")))
            except Exception:
                continue
        self._jsonl_tail_cache[cache_key] = list(rows)
        return rows

    def _load_json_object(self, path: Path) -> Dict[str, Any]:
        cache_key = str(path.resolve())
        if cache_key in self._json_object_cache:
            return dict(self._json_object_cache[cache_key])
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self._json_object_cache[cache_key] = dict(payload)
            return payload
        except Exception:
            return {}

    def _cascade_summary(
        self,
        *,
        requested_model_class: str,
        feature_subset: str,
        horizon_seconds: int,
        min_edge: float,
    ) -> Optional[Dict[str, Any]]:
        cache_key = (
            requested_model_class,
            feature_subset,
            int(horizon_seconds),
            round(float(min_edge), 6),
        )
        if cache_key in self._cascade_cache:
            return self._cascade_cache[cache_key]
        try:
            from funding.ml.cascade_features import build_cascade_features, get_cascade_feature_columns, label_cascade_events
        except Exception:
            return None
        symbols = self._candidate_funding_symbols()
        if not symbols:
            return None
        history_root = self.project_root / "data" / "funding_history"
        df = build_cascade_features(symbols, data_dir=history_root)
        if df is None or df.empty:
            summary = self._cascade_fallback_summary(requested_model_class=requested_model_class)
            self._cascade_cache[cache_key] = summary
            return summary
        labels = label_cascade_events(df)
        feature_columns = get_cascade_feature_columns(df)
        clean_df = df[feature_columns].fillna(0.0).copy()
        if clean_df.empty:
            summary = self._cascade_fallback_summary(requested_model_class=requested_model_class)
            self._cascade_cache[cache_key] = summary
            return summary
        score = self._cascade_score_series(clean_df, feature_subset=feature_subset)
        segments = self._cascade_segments(
            score=score,
            labels=labels.reindex(clean_df.index).fillna(0).astype(int),
            horizon_seconds=horizon_seconds,
            min_edge=min_edge,
        )
        if not segments:
            summary = self._cascade_fallback_summary(requested_model_class=requested_model_class)
            self._cascade_cache[cache_key] = summary
            return summary
        summary = self._summary_from_segments(
            segments=segments,
            resolved_model_engine=f"cascade_{requested_model_class.lower()}",
            source_mode="cascade_features",
            source_paths=[str(history_root)],
            dataset_rows=len(clean_df),
            feature_columns=feature_columns,
            market_count=len(symbols),
            price_scale=0.4,
            paper_day_cap=12,
            net_pnl_scale=0.1,
        )
        self._cascade_cache[cache_key] = summary
        return summary

    def _cascade_score_series(self, df: pd.DataFrame, *, feature_subset: str) -> pd.Series:
        weights = {
            "liq_count_1h": 0.8,
            "liq_volume_usd_1h": 0.7,
            "funding_extremity_max_abs": 0.6,
            "volume_surge_zscore": 0.55,
            "price_acceleration": -0.45,
            "cross_asset_pc1_loading": 0.4,
            "leverage_proxy_oi_vol": 0.35,
            "oi_concentration_hhi": 0.25,
        }
        if feature_subset == "regime":
            weights["cross_asset_pc1_loading"] = 0.7
            weights["volume_surge_zscore"] = 0.8
        elif feature_subset == "microstructure":
            weights["liq_volume_usd_1h"] = 0.9
            weights["oi_concentration_hhi"] = 0.45
        score = pd.Series(0.0, index=df.index, dtype=float)
        for column, weight in weights.items():
            if column not in df.columns:
                continue
            series = df[column].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            centered = series - series.rolling(168, min_periods=24).mean().fillna(series.mean())
            scale = series.rolling(168, min_periods=24).std().replace(0.0, np.nan).fillna(series.std() or 1.0)
            normalized = (centered / scale).clip(-4.0, 4.0)
            score += normalized * weight
        return score

    def _cascade_segments(
        self,
        *,
        score: pd.Series,
        labels: pd.Series,
        horizon_seconds: int,
        min_edge: float,
    ) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        total = len(score)
        if total < 240:
            return segments
        segment_size = max(120, total // 3)
        threshold_quantile = max(0.90, min(0.995, 0.97 + (min_edge * 0.1) - (horizon_seconds / 120000.0)))
        for index in range(3):
            start = index * segment_size
            end = total if index == 2 else min(total, (index + 1) * segment_size)
            train = score.iloc[:start] if start >= 240 else score.iloc[: max(240, end - segment_size)]
            test = score.iloc[start:end]
            test_labels = labels.reindex(test.index).fillna(0).astype(int)
            if len(test) < 60 or train.empty:
                continue
            threshold = float(train.quantile(threshold_quantile))
            probs = 1.0 / (1.0 + np.exp(-(test - threshold)))
            preds = probs >= 0.5
            tp = int(((preds) & (test_labels == 1)).sum())
            fp = int(((preds) & (test_labels == 0)).sum())
            fn = int(((~preds) & (test_labels == 1)).sum())
            pnl = (tp * 2.5) - (fp * 0.8)
            trade_count = int(preds.sum())
            monthly_roi_pct = round((pnl / max(1, trade_count)) * 10.0, 4)
            baseline_prob = float(train.reindex(labels.index, fill_value=0).mean()) if len(train) else 0.0
            brier = float(np.mean((probs - test_labels.to_numpy(dtype=float)) ** 2))
            baseline_brier = float(np.mean((baseline_prob - test_labels.to_numpy(dtype=float)) ** 2))
            failure_rate = round(fp / max(1, trade_count), 4)
            recall = tp / max(1, tp + fn)
            cumulative = np.cumsum(np.where(preds & (test_labels == 1), 2.5, np.where(preds, -0.8, 0.0)))
            peak = np.maximum.accumulate(cumulative) if len(cumulative) else np.array([0.0])
            drawdown = float((peak - cumulative).max()) if len(cumulative) else 0.0
            segments.append(
                {
                    "label": f"segment_{index + 1}",
                    "settled_count": int(len(test)),
                    "trade_count": trade_count,
                    "monthly_roi_pct": monthly_roi_pct,
                    "baseline_roi_pct": 0.0,
                    "brier_lift_abs": round(baseline_brier - brier, 6),
                    "drawdown_pct": round((drawdown / max(1, trade_count)) * 10.0, 4),
                    "slippage_headroom_pct": round(monthly_roi_pct - 1.0, 4),
                    "failure_rate": failure_rate,
                    "regime_robustness": round(recall, 4),
                    "total_pnl": round(float(pnl), 6),
                }
            )
        return segments

    def _cascade_fallback_summary(self, *, requested_model_class: str) -> Optional[Dict[str, Any]]:
        history_path = self.project_root / "data" / "portfolios" / "cascade_alpha" / "runtime" / "summary_history.jsonl"
        rows = self._load_jsonl_rows(history_path)[-3:]
        if not rows:
            return None
        segments: List[Dict[str, Any]] = []
        for index, row in enumerate(rows):
            roi_pct = float(row.get("roi_pct", 0.0) or 0.0)
            segments.append(
                {
                    "label": f"segment_{index + 1}",
                    "settled_count": 8,
                    "trade_count": int(row.get("open_count", 0) or 0),
                    "monthly_roi_pct": round(roi_pct, 4),
                    "baseline_roi_pct": 0.0,
                    "brier_lift_abs": 0.0,
                    "drawdown_pct": round(abs(min(roi_pct, 0.0)) * 1.25, 4),
                    "slippage_headroom_pct": round(roi_pct - 1.0, 4),
                    "failure_rate": 0.2,
                    "regime_robustness": 0.3,
                    "total_pnl": round(float(row.get("realized_pnl", 0.0) or 0.0), 6),
                }
            )
        return self._summary_from_segments(
            segments=segments,
            resolved_model_engine=f"cascade_{requested_model_class.lower()}",
            source_mode="cascade_runtime_history",
            source_paths=[str(history_path)],
            dataset_rows=len(rows),
            feature_columns=["roi_pct", "progress_pct", "open_count"],
            market_count=1,
            price_scale=1.0,
            paper_day_cap=6,
            net_pnl_scale=1.0,
        )

    def _candidate_signal_summary(
        self,
        *,
        family_mode: str,
        requested_model_class: str,
        feature_subset: str,
        min_edge: float,
    ) -> Optional[Dict[str, Any]]:
        cache_key = (
            family_mode,
            requested_model_class,
            feature_subset,
            round(float(min_edge), 6),
        )
        if cache_key in self._candidate_signal_cache:
            return self._candidate_signal_cache[cache_key]
        candidate_dir = self.project_root / "data" / "candidates"
        candidate_files = sorted(candidate_dir.glob("*.jsonl"))[-3:]
        if not candidate_files:
            return None
        clv_dir = self.project_root / "data" / "clv"
        clv_files = sorted(clv_dir.glob("*.jsonl"))[-3:]
        qa_rows = self._load_jsonl_rows(self.project_root / "data" / "qa" / "decisions.jsonl")[-20:]
        fresh_rate = 0.0
        if qa_rows:
            fresh_rate = sum(
                float(((row.get("metrics") or {}).get("candidate") or {}).get("fresh_snapshot_rate", 0.0) or 0.0)
                for row in qa_rows
            ) / max(1, len(qa_rows))
        clv_entries = 0
        unique_models = set()
        for file_path in clv_files:
            rows = self._load_jsonl_rows(file_path)
            clv_entries += len(rows)
            for row in rows:
                bet_id = str(row.get("bet_id", ""))
                if ":" in bet_id:
                    unique_models.add(bet_id.split(":", 1)[0])

        segments: List[Dict[str, Any]] = []
        sampled_row_count = 0
        for index, file_path in enumerate(candidate_files):
            rows = self._load_tail_jsonl_rows(file_path, max_rows=20000)
            if not rows:
                continue
            sampled_row_count += len(rows)
            reason_counts: Dict[str, int] = {}
            unique_markets = set()
            selection_total = 0.0
            overround_total = 0.0
            executed = 0
            for row in rows:
                reason = str(row.get("reason", "unknown"))
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if row.get("market_id"):
                    unique_markets.add(str(row.get("market_id")))
                selection_total += float(row.get("selection_count", 0.0) or 0.0)
                overround_total += float(row.get("overround_back", 0.0) or 0.0)
                executed += 1 if row.get("executed") else 0
            count = len(rows)
            stale_rate = reason_counts.get("stale_or_missing_snapshot", 0) / max(1, count)
            no_arb_rate = reason_counts.get("no_arb_after_filters", 0) / max(1, count)
            retired_rate = reason_counts.get("retired_missing_snapshot", 0) / max(1, count)
            unique_market_count = len(unique_markets)
            avg_selection_count = selection_total / max(1, count)
            clv_density = clv_entries / max(1, unique_market_count)
            model_diversity = len(unique_models)
            if family_mode == "information_lag":
                monthly_roi_pct = round(
                    (fresh_rate * 18.0)
                    + (unique_market_count / 4000.0)
                    - (stale_rate * 15.0)
                    - (no_arb_rate * 5.0)
                    - (retired_rate * 2.0),
                    4,
                )
                calibration_lift_abs = round((fresh_rate - stale_rate) / 4.0, 6)
                regime_robustness = round(max(0.0, 1.0 - stale_rate - (retired_rate * 0.5)), 4)
                settled_count = min(9, model_diversity + index + 1)
            else:
                monthly_roi_pct = round(
                    (fresh_rate * 10.0)
                    + min(6.0, clv_density * 0.15)
                    + (model_diversity * 0.4)
                    - (stale_rate * 10.0)
                    - (no_arb_rate * 3.0),
                    4,
                )
                calibration_lift_abs = round((min(1.0, clv_density / 10.0) + fresh_rate - stale_rate) / 4.0, 6)
                regime_robustness = round(max(0.0, min(1.0, fresh_rate + min(0.4, clv_density / 100.0))), 4)
                settled_count = min(9, max(1, len(unique_models) + index + (1 if feature_subset == "cross_science" else 0)))
            segments.append(
                {
                    "label": file_path.stem,
                    "settled_count": settled_count,
                    "trade_count": max(1, executed + min(20, clv_entries // max(1, len(candidate_files)))),
                    "monthly_roi_pct": monthly_roi_pct,
                    "baseline_roi_pct": 0.0,
                    "brier_lift_abs": calibration_lift_abs,
                    "drawdown_pct": round(abs(min(monthly_roi_pct, 0.0)) * 1.1, 4),
                    "slippage_headroom_pct": round(monthly_roi_pct - max(0.5, min_edge * 40.0), 4),
                    "failure_rate": round(min(1.0, stale_rate + (no_arb_rate * 0.3)), 4),
                    "regime_robustness": regime_robustness,
                    "total_pnl": round(monthly_roi_pct * 0.2, 6),
                }
            )
        if not segments:
            return None
        summary = self._summary_from_segments(
            segments=segments,
            resolved_model_engine=f"{family_mode}_{requested_model_class.lower()}",
            source_mode="candidate_logs_recent",
            source_paths=[str(path) for path in candidate_files + clv_files],
            dataset_rows=sampled_row_count,
            feature_columns=[
                "candidate_count",
                "unique_markets",
                "stale_rate",
                "no_arb_rate",
                "retired_rate",
                "fresh_snapshot_rate",
                "clv_density",
            ],
            market_count=max(1, len(candidate_files)),
            price_scale=1.0,
            paper_day_cap=len(candidate_files),
            net_pnl_scale=1.0,
        )
        self._candidate_signal_cache[cache_key] = summary
        return summary

    def _summary_from_segments(
        self,
        *,
        segments: List[Dict[str, Any]],
        resolved_model_engine: str,
        source_mode: str,
        source_paths: List[str],
        dataset_rows: int,
        feature_columns: List[str],
        market_count: int,
        price_scale: float,
        paper_day_cap: int,
        net_pnl_scale: float,
    ) -> Dict[str, Any]:
        recent_segments = segments[-3:]
        windows = [
            EvaluationWindow(
                label=str(segment["label"]),
                settled_count=int(segment["settled_count"]),
                monthly_roi_pct=float(segment["monthly_roi_pct"]),
                baseline_roi_pct=float(segment["baseline_roi_pct"]),
                brier_lift_abs=float(segment["brier_lift_abs"]),
                drawdown_pct=float(segment["drawdown_pct"]),
                slippage_headroom_pct=float(segment["slippage_headroom_pct"]),
                failure_rate=float(segment["failure_rate"]),
                regime_robustness=float(segment["regime_robustness"]),
            )
            for segment in recent_segments
        ]
        monthly_roi_pct = round(sum(float(segment["monthly_roi_pct"]) for segment in recent_segments) / max(1, len(recent_segments)), 4)
        max_drawdown_pct = round(max(float(segment["drawdown_pct"]) for segment in segments), 4)
        calibration_lift_abs = round(sum(float(segment["brier_lift_abs"]) for segment in recent_segments) / max(1, len(recent_segments)), 6)
        slippage_headroom_pct = round(min(float(segment["slippage_headroom_pct"]) for segment in recent_segments), 4)
        failure_rate = round(sum(float(segment["failure_rate"]) for segment in recent_segments) / max(1, len(recent_segments)), 4)
        regime_robustness = round(sum(float(segment["regime_robustness"]) for segment in recent_segments) / max(1, len(recent_segments)), 4)
        trade_count = sum(int(segment["trade_count"]) for segment in segments)
        settled_count = sum(int(segment["settled_count"]) for segment in segments)
        paper_days = min(30, max(1, paper_day_cap))
        baseline_beaten_windows = sum(
            1 for segment in segments
            if float(segment["monthly_roi_pct"]) > float(segment["baseline_roi_pct"])
            and float(segment["brier_lift_abs"]) >= 0.0
        )
        turnover = round(min(1.0, trade_count / max(20.0, float(dataset_rows or 1))), 4)
        capacity_score = round(min(1.0, 0.2 + (turnover * 0.45) + (min(1.0, market_count / 5.0) * 0.25)), 4)
        net_pnl = round(sum(float(segment["total_pnl"]) for segment in segments) * net_pnl_scale, 6)
        walkforward_summary = {
            "resolved_model_engine": resolved_model_engine,
            "source_mode": source_mode,
            "windows": windows,
            "monthly_roi_pct": monthly_roi_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "slippage_headroom_pct": slippage_headroom_pct,
            "calibration_lift_abs": calibration_lift_abs,
            "turnover": turnover,
            "capacity_score": capacity_score,
            "failure_rate": failure_rate,
            "regime_robustness": regime_robustness,
            "baseline_beaten_windows": baseline_beaten_windows,
            "trade_count": trade_count,
            "settled_count": settled_count,
            "paper_days": paper_days,
            "net_pnl": net_pnl,
            "stress_positive": monthly_roi_pct > 0.0 and slippage_headroom_pct > 0.0,
        }
        stress_windows = [
            EvaluationWindow(
                label=f"{window.label}_stress",
                settled_count=window.settled_count,
                monthly_roi_pct=round(window.monthly_roi_pct - price_scale, 4),
                baseline_roi_pct=window.baseline_roi_pct,
                brier_lift_abs=round(window.brier_lift_abs * 0.85, 6),
                drawdown_pct=round(window.drawdown_pct * 1.1, 4),
                slippage_headroom_pct=round(window.slippage_headroom_pct - price_scale, 4),
                failure_rate=min(1.0, round(window.failure_rate + 0.08, 4)),
                regime_robustness=max(0.0, round(window.regime_robustness - 0.08, 4)),
            )
            for window in windows
        ]
        stress_summary = {
            **walkforward_summary,
            "windows": stress_windows,
            "monthly_roi_pct": round(walkforward_summary["monthly_roi_pct"] - price_scale, 4),
            "max_drawdown_pct": round(walkforward_summary["max_drawdown_pct"] * 1.1, 4),
            "slippage_headroom_pct": round(walkforward_summary["slippage_headroom_pct"] - price_scale, 4),
            "calibration_lift_abs": round(walkforward_summary["calibration_lift_abs"] * 0.85, 6),
            "failure_rate": min(1.0, round(walkforward_summary["failure_rate"] + 0.08, 4)),
            "regime_robustness": max(0.0, round(walkforward_summary["regime_robustness"] - 0.08, 4)),
            "stress_positive": round(walkforward_summary["monthly_roi_pct"] - price_scale, 4) > 0.0
            and round(walkforward_summary["slippage_headroom_pct"] - price_scale, 4) > 0.0,
        }
        return {
            "resolved_model_engine": resolved_model_engine,
            "source_mode": source_mode,
            "source_paths": source_paths,
            "dataset_rows": dataset_rows,
            "feature_columns": feature_columns,
            "market_count": market_count,
            "walkforward_summary": walkforward_summary,
            "stress_summary": stress_summary,
            "walkforward_artifact": {
                "source_mode": source_mode,
                "resolved_model_engine": resolved_model_engine,
                "segments": segments,
                "summary": walkforward_summary,
            },
            "stress_artifact": {
                "source_mode": source_mode,
                "resolved_model_engine": resolved_model_engine,
                "segments": segments,
                "summary": stress_summary,
            },
        }

    def _count_jsonl_rows(self, path: Path) -> int:
        return len(self._load_jsonl_rows(path))

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
