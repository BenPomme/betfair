# Roadmap

## Current MEV Status

The `mev_scout_sol` portfolio is not a real learning system yet.

What exists now:
- separate `mev_scout_sol` portfolio and runner
- basic Solana provider scaffold
- basic parser for `raydium`, `orca`, `jupiter` markers
- latency probe
- shadow opportunity accounting

What does not exist yet:
- real Solana transaction stream ingestion
- whale watchlist or robust whale classification
- event store with labeled outcomes
- model training loop
- ranking model for opportunities
- execution simulation with realistic landing / latency decay
- Jito / Yellowstone backed event feed

Current practical state:
- we are not following all whales
- we are effectively following zero whales until a real provider is wired
- the current provider returns no live events
- current MEV output is therefore research scaffolding only

### MEV Next Build Steps

1. Implement a real Solana provider:
   - `MEV_SCOUT_SOL_RPC_URL`
   - `MEV_SCOUT_SOL_WS_URL`
   - ideally `MEV_SCOUT_SOL_YELLOWSTONE_URL`
2. Build DEX-specific decoding for:
   - Raydium
   - Orca
   - Jupiter
3. Add whale detection:
   - wallet watchlists
   - USD-size thresholding
   - route-aware event parsing
4. Add shadow-labeling:
   - post-trade drift
   - latency-adjusted theoretical edge
   - venue / route attribution
5. Add a real learner:
   - opportunity ranker
   - expected edge model
   - latency decay model

## Discord Monitoring

Discord support is implemented in code but not configured.

To enable it locally, add to `.env`:

```env
DISCORD_ENABLED=true
DISCORD_WEBHOOK_URL=your_discord_webhook_url
DISCORD_DIGEST_ENABLED=true
DISCORD_DIGEST_INTERVAL_MINUTES=30
```

Optional related settings already supported:

```env
NOTIFICATIONS_ENABLED=true
NOTIFY_DEDUPE_WINDOW_SECONDS=600
```

What happens after configuration:
- immediate alerts for portfolio starts/stops/restarts
- degrade / recover alerts
- critical training failures
- periodic digest with portfolio-level P&L and readiness

What is still needed from the operator:
- a real Discord webhook URL

## Remote Work Architecture

Goal:
- this machine stays on and runs trading + learning
- another computer can be used to code against the same repo
- code changes made remotely can be applied on this machine

Recommended architecture:

1. Keep this machine as the execution host.
2. Put the repo on GitHub as the source of truth.
3. From another computer:
   - work in a normal clone of the repo
   - push commits/branches to GitHub
4. On this machine:
   - run a small deployment / sync loop or manual pull step
   - restart affected portfolios after validated changes

Important constraint:
- editing the same Git repo from another computer does not automatically make this computer execute those changes
- there must be a sync + deploy mechanism on this machine

### Recommended Deployment Modes

#### Option A: GitHub push -> this machine auto-deploys

Best long-term option.

Flow:
1. You work from any computer.
2. You push to `main` or a deployment branch.
3. This machine runs a watcher or scheduled task:
   - `git fetch`
   - compare local HEAD to remote
   - `git pull --ff-only`
   - restart selected portfolio processes
4. Command center reflects the updated code.

Pros:
- simple mental model
- works from any machine
- auditable through git history

Cons:
- needs careful deployment guardrails

#### Option B: Remote shell into this machine and use Codex there

Best for safety and full context.

Flow:
1. Remote into this machine with Tailscale + RDP, Parsec, or VS Code Remote/SSH.
2. Run Codex directly on this machine.
3. All edits happen where the processes are running.

Pros:
- no deployment drift
- same filesystem, same data, same runtime

Cons:
- depends on remote access quality

### Recommended Guardrails

Regardless of option:
- never auto-pull unreviewed code into running trading without a deploy gate
- deploy by portfolio where possible
- preserve runtime state and training artifacts
- restart only the affected portfolio runner
- keep command center separate from data directories

## Betfair Operator Prerequisite

For production-grade Betfair auth, place one of these into `certs/`:
- `client-2048.key` matching the existing `client-2048.crt`
- or a single valid `.pem`

Without this, Betfair can only use weaker fallback behavior and remains auth-blocked for serious operation.
