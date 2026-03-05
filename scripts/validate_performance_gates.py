#!/usr/bin/env python3
"""
Validate baseline-vs-ML performance gates from candidate logger JSONL files.

This is a replay/proxy evaluator for paper-phase gate checks:
- Opportunity->execution precision lift (target >= 25%)
- Expected profit/trade lift (target >= 15%)
- No increase in max drawdown
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import config


@dataclass
class TradeLike:
    ts: datetime
    proxy_profit: float
    threshold: float
    decision: str
    has_opportunity: bool


def _parse_ts(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


def _iter_records(input_dir: Path) -> Iterable[dict]:
    for path in sorted(input_dir.glob("*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
        except Exception:
            continue


def _build_trades(records: Iterable[dict]) -> List[TradeLike]:
    out: List[TradeLike] = []
    for r in records:
        # Keep only records where a scored opportunity existed.
        has_opp = r.get("net_profit_eur") is not None
        if not has_opp:
            continue
        decision = str(r.get("decision", "UNKNOWN"))
        expected = r.get("expected_net_profit_eur")
        net = r.get("net_profit_eur")
        fill = r.get("fill_prob")
        if expected is None:
            try:
                n = float(net)
            except Exception:
                continue
            try:
                fp = float(fill) if fill is not None else 0.5
            except Exception:
                fp = 0.5
            proxy_profit = n * fp
        else:
            try:
                proxy_profit = float(expected)
            except Exception:
                continue
        try:
            threshold = float(r.get("dynamic_threshold_eur", float(config.ML_BASE_DECISION_THRESHOLD_EUR)))
        except Exception:
            threshold = float(config.ML_BASE_DECISION_THRESHOLD_EUR)
        out.append(
            TradeLike(
                ts=_parse_ts(str(r.get("ts", ""))),
                proxy_profit=proxy_profit,
                threshold=threshold,
                decision=decision,
                has_opportunity=True,
            )
        )
    out.sort(key=lambda x: x.ts)
    return out


def _max_drawdown(pnls: List[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _metrics(trades: List[TradeLike], use_ml_policy: bool) -> Dict[str, float]:
    executed: List[TradeLike] = []
    for t in trades:
        if use_ml_policy:
            if t.decision == "EXECUTE":
                executed.append(t)
        else:
            # Baseline policy: execute every scored opportunity.
            if t.has_opportunity:
                executed.append(t)
    if not executed:
        return {
            "executed_count": 0.0,
            "precision": 0.0,
            "avg_profit_per_trade": 0.0,
            "total_profit": 0.0,
            "max_drawdown": 0.0,
        }
    successes = sum(1 for t in executed if t.proxy_profit >= t.threshold)
    pnl = [t.proxy_profit for t in executed]
    total_profit = sum(pnl)
    return {
        "executed_count": float(len(executed)),
        "precision": float(successes / len(executed)),
        "avg_profit_per_trade": float(total_profit / len(executed)),
        "total_profit": float(total_profit),
        "max_drawdown": float(_max_drawdown(pnl)),
    }


def _lift(new: float, base: float) -> float:
    if base == 0:
        return 0.0
    return ((new / base) - 1.0) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/candidates")
    parser.add_argument("--output-dir", default="data/reports/performance_gates")
    parser.add_argument("--min-samples", type=int, default=200)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input dir not found: {input_dir}")
        return 1

    trades = _build_trades(_iter_records(input_dir))
    if len(trades) < args.min_samples:
        print(f"Not enough samples: {len(trades)} < {args.min_samples}")
        return 1

    baseline = _metrics(trades, use_ml_policy=False)
    ml = _metrics(trades, use_ml_policy=True)
    precision_lift = _lift(ml["precision"], baseline["precision"])
    profit_lift = _lift(ml["avg_profit_per_trade"], baseline["avg_profit_per_trade"])
    drawdown_diff = _lift(ml["max_drawdown"], baseline["max_drawdown"]) if baseline["max_drawdown"] > 0 else 0.0

    gate_precision = precision_lift >= 25.0
    gate_profit = profit_lift >= 15.0
    gate_drawdown = ml["max_drawdown"] <= baseline["max_drawdown"]
    pass_all = gate_precision and gate_profit and gate_drawdown

    now = datetime.now(timezone.utc).isoformat()
    report = {
        "status": "pass" if pass_all else "fail",
        "pass_all": pass_all,
        "generated_at": now,
        "sample_size": len(trades),
        "gates": {
            "precision_lift_ge_25pct": gate_precision,
            "profit_per_trade_lift_ge_15pct": gate_profit,
            "drawdown_not_increased": gate_drawdown,
        },
        "baseline": baseline,
        "ml_gated": ml,
        "improvements": {
            "precision_lift_pct": round(precision_lift, 4),
            "profit_per_trade_lift_pct": round(profit_lift, 4),
            "drawdown_diff_pct": round(drawdown_diff, 4),
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stamped = out_dir / f"gate_report_{ts}.json"
    latest = out_dir / "latest.json"
    stamped.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Report written: {stamped}")
    print(f"Latest: {latest}")
    print(
        f"Status={report['status']} "
        f"precision_lift={precision_lift:.2f}% "
        f"profit_lift={profit_lift:.2f}% "
        f"drawdown_diff={drawdown_diff:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
