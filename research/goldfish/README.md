Goldfish sidecar for model reseeds and replay discipline.

Purpose:
- keep experiment provenance and replay decisions out of the live runners
- publish accepted research runs back into `data/research/goldfish/manifests`
- mirror finalized prediction/funding runs into the legacy JSONL feeds the dashboard already understands

Families:
- `betfair_prediction`
- `betfair_info_arb`
- `polymarket`
- `funding`

Typical flow:
1. Run a family pipeline in Goldfish.
2. Finalize the run.
3. Publish the artifact manifest and decision with `scripts/goldfish_publish_run.py`.

The live runners do not depend on Goldfish being present.
