#!/usr/bin/env python3
"""
Compare two prediction experiment snapshots from data/prediction/experiments.jsonl.

Usage:
  python scripts/compare_prediction_runs.py --run-a residual_logit_2-200 --run-b residual_logit_2-400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _load(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    out: list[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _find(entries: list[Dict[str, Any]], run_id: str) -> Optional[Dict[str, Any]]:
    for e in entries:
        if str(e.get("run_id")) == run_id:
            return e
    return None


def _metric(entry: Dict[str, Any], path: list[str], default: float = 0.0) -> float:
    cur: Any = entry
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    try:
        return float(cur)
    except Exception:
        return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="data/prediction/experiments.jsonl")
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    args = parser.parse_args()

    entries = _load(Path(args.log))
    a = _find(entries, args.run_a)
    b = _find(entries, args.run_b)
    if a is None or b is None:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "run_not_found",
                    "run_a_found": a is not None,
                    "run_b_found": b is not None,
                },
                indent=2,
            )
        )
        return 1

    keys = [
        ("rolling_100_brier_lift", ["metrics", "rolling_100", "brier_lift_abs"]),
        ("rolling_200_brier_lift", ["metrics", "rolling_200", "brier_lift_abs"]),
        ("rolling_100_roi_pct", ["metrics", "rolling_100", "roi_pct"]),
        ("rolling_200_roi_pct", ["metrics", "rolling_200", "roi_pct"]),
    ]
    deltas = {}
    for label, p in keys:
        av = _metric(a, p)
        bv = _metric(b, p)
        deltas[label] = {"a": av, "b": bv, "delta_b_minus_a": round(bv - av, 6)}

    out = {
        "ok": True,
        "run_a": {"run_id": a.get("run_id"), "config_hash": a.get("config_hash"), "gate": a.get("gate")},
        "run_b": {"run_id": b.get("run_id"), "config_hash": b.get("config_hash"), "gate": b.get("gate")},
        "deltas": deltas,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
