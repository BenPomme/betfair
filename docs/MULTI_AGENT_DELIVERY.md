# Multi-Agent Delivery Blueprint (Betfair-Only Quant Upgrade)

This document defines how work is split across agent tiers with cost discipline.

## Tier Routing

- Tier 1 (cheap/local): `qwen2.5:32b`
  - Scaffolding, deterministic feature extraction boilerplate, dashboard UI wiring, test fixtures.
- Tier 2 (cheap/local stronger): `qwen3.5:27b` or `qwen3.5:35b-a3b`
  - Data pipelines, model training/eval scripts, integration wiring, replay analysis.
- Tier 3 (highest capability)
  - Any change under `/core` or `/execution`, decision gates, risk-sensitive review signoff.

## Lanes

1. Lane A (Tier 1):
   - Files: `strategy/features.py`, `data/candidate_logger.py`, unit tests.
   - Output: deterministic features + JSONL candidate dataset.
2. Lane B (Tier 2):
   - Files: training scripts in `scripts/` and model artifact plumbing in `strategy/model_inference.py`.
   - Output: walk-forward evaluation and model artifact format.
3. Lane C (Tier 2):
   - Files: `monitoring/engine.py`, `monitoring/templates/dashboard.html`.
   - Output: opportunity funnel and scoring telemetry in dashboard.
4. Lane D (Tier 3):
   - Files: `main.py`, `execution/executor.py`, `execution/paper_executor.py`, `/core` review.
   - Output: execution gating, adaptive policy metadata, safety invariants.

## Merge Protocol

1. Define/lock interfaces first:
   - `FeatureVector`
   - `ScoredOpportunity`
   - `score_opportunity(opportunity, features)`
2. No concurrent edits on the same file.
3. Integrate in this order: A + B + C, then D.
4. Tier 3 review is mandatory before enabling live paths.

## Acceptance Checklist

- Unit tests pass for features/inference and existing scanner/executor tests.
- Dashboard shows funnel counters: scanned, candidates, scored, deferred, executed.
- Candidate logs produced under `data/candidates/`.
- Paper mode only until acceptance gates in project plan are met.
