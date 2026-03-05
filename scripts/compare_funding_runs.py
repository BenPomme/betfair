#!/usr/bin/env python3
"""
Compare two funding experiment snapshots from data/funding/experiments.jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _find(entries: List[Dict[str, Any]], run_id: str) -> Dict[str, Any]:
    for e in reversed(entries):
        if str(e.get("run_id")) == run_id:
            return e
    raise SystemExit(f"run_id not found: {run_id}")


def _get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return cur if cur is not None else default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="data/funding/experiments.jsonl")
    parser.add_argument("--a", required=True, help="Older/baseline run_id")
    parser.add_argument("--b", required=True, help="Newer/candidate run_id")
    args = parser.parse_args()

    entries = _load_entries(Path(args.log))
    a = _find(entries, args.a)
    b = _find(entries, args.b)

    fields = [
        ("rolling_100_brier_lift", ["metrics", "rolling_100", "brier_lift_abs"]),
        ("rolling_200_brier_lift", ["metrics", "rolling_200", "brier_lift_abs"]),
        ("rolling_100_roi_pct", ["metrics", "rolling_100", "roi_pct"]),
        ("rolling_200_roi_pct", ["metrics", "rolling_200", "roi_pct"]),
        ("strict_gate_pass", ["gate", "strict_gate_pass"]),
    ]

    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print("Config hashes:", _get(a, ["config_hash"], ""), "->", _get(b, ["config_hash"], ""))
    print("")
    for label, path in fields:
        av = _get(a, path)
        bv = _get(b, path)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            delta = bv - av
            print(f"{label:24} {av:>10.6f} -> {bv:>10.6f}   delta={delta:+.6f}")
        else:
            print(f"{label:24} {av} -> {bv}")


if __name__ == "__main__":
    main()
