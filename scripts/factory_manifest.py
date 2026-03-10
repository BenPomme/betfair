#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

try:
    from dotenv import load_dotenv

    load_dotenv(project_root / ".env", override=True)
except ImportError:
    pass

import config
from factory.registry import FactoryRegistry


def _registry() -> FactoryRegistry:
    root = Path(getattr(config, "FACTORY_ROOT", "data/factory"))
    if not root.is_absolute():
        root = project_root / root
    return FactoryRegistry(root)


def _manifest_row(manifest) -> dict:
    return {
        "manifest_id": manifest.manifest_id,
        "family_id": manifest.family_id,
        "lineage_id": manifest.lineage_id,
        "portfolio_targets": list(manifest.portfolio_targets),
        "status": manifest.status,
        "approved_stage": manifest.approved_stage,
        "approved_by": manifest.approved_by,
        "approved_at": manifest.approved_at,
        "live_loadable": manifest.is_live_loadable(),
        "artifact_refs": dict(manifest.artifact_refs),
    }


def _cmd_list(args: argparse.Namespace) -> int:
    manifests = _registry().manifests()
    if args.portfolio:
        manifests = [manifest for manifest in manifests if args.portfolio in manifest.portfolio_targets]
    if args.status:
        manifests = [manifest for manifest in manifests if manifest.status == args.status]
    print(json.dumps([_manifest_row(manifest) for manifest in manifests], indent=2))
    return 0


def _cmd_approve(args: argparse.Namespace) -> int:
    manifest = _registry().approve_manifest(args.manifest_id, approved_by=args.approved_by, note=args.note)
    if manifest is None:
        print(json.dumps({"ok": False, "error": "manifest_not_found", "manifest_id": args.manifest_id}, indent=2))
        return 1
    print(json.dumps({"ok": True, "manifest": _manifest_row(manifest)}, indent=2))
    return 0


def _cmd_reject(args: argparse.Namespace) -> int:
    manifest = _registry().reject_manifest(args.manifest_id, note=args.note)
    if manifest is None:
        print(json.dumps({"ok": False, "error": "manifest_not_found", "manifest_id": args.manifest_id}, indent=2))
        return 1
    print(json.dumps({"ok": True, "manifest": _manifest_row(manifest)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Operate research-factory manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List known manifests.")
    list_parser.add_argument("--portfolio", help="Filter manifests by portfolio target.")
    list_parser.add_argument("--status", help="Filter manifests by status.")
    list_parser.set_defaults(func=_cmd_list)

    approve_parser = subparsers.add_parser("approve", help="Approve a manifest for live loading.")
    approve_parser.add_argument("manifest_id")
    approve_parser.add_argument("--approved-by", required=True)
    approve_parser.add_argument("--note")
    approve_parser.set_defaults(func=_cmd_approve)

    reject_parser = subparsers.add_parser("reject", help="Reject a manifest.")
    reject_parser.add_argument("manifest_id")
    reject_parser.add_argument("--note")
    reject_parser.set_defaults(func=_cmd_reject)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
