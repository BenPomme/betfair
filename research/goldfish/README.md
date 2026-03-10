# Goldfish Sidecar Workspaces

These workspaces are research-only sidecars for the strategy factory.

- They package isolated experiment pipelines per strategy family.
- They are not part of the live trading hot path.
- Accepted artifacts flow one way from these workspaces back into the native runners through approved manifests.

Each workspace keeps a minimal `pipeline.yaml` and `STATE.md` scaffold so the factory can resume work, preserve provenance, and recover operator context after restarts.
