# Agentic POD + Dropshipping Research Notes

## Scope for this research pass
- Build a practical shortlist of connectors, agent frameworks, and ops tooling for a fully automated print-on-demand (POD) and dropshipping system.
- Check whether Luka repositories (`meerkat`, `elephant`, `goldfish`) can improve reliability, robustness, and cost efficiency.

## Environment limitation encountered
Direct outbound access to GitHub and `raw.githubusercontent.com` was blocked in this environment during this pass (HTTP CONNECT 403). Because of that, the external lists below are curated from known ecosystem options and should be validated before implementation.

## Connectors and integration layers to evaluate

### Commerce channels
- **Shopify Admin API / GraphQL**: catalog, orders, fulfillment, discounts, pricing, webhooks.
- **Etsy API**: listing sync, inventory updates, order ingestion.
- **Amazon Selling Partner API (SP-API)**: listings, feeds, orders, reports, ads-related data where available.
- **eBay Sell APIs**: listing + order endpoints for secondary channel expansion.
- **WooCommerce REST API**: optional self-hosted storefront channel.

### POD and supplier connectors
- **Printful API**: product templates, mockups, order submit/status, webhooks.
- **Printify API**: product/blueprint workflows, order routing, status tracking.
- **Gelato API**: global print network and shipping integration.
- **Gooten API**: alternate POD network for routing redundancy.
- **CJdropshipping / DSers / AutoDS connectors**: non-POD SKU expansion and sourcing automation.

### Logistics and shipping
- **Shippo / EasyPost / ShipStation APIs**: label creation, rate shopping, tracking normalization.
- **AfterShip / 17TRACK APIs**: customer-facing shipment tracking and status webhooks.
- **3PL connectors** (if hybrid inventory): Deliverr/Flexport/local 3PL API adapters.

### Payments, finance, and accounting
- **Stripe API**: direct payment/control for DTC flows.
- **PayPal APIs**: alternate checkout coverage.
- **QuickBooks/Xero APIs**: automated reconciliation and accounting close.
- **Tax engines (Avalara/TaxJar)**: sales tax/VAT automation.

### Marketing and growth
- **Meta Marketing API**: campaign creation, budget control, reporting.
- **Google Ads API**: search/shopping campaign ops.
- **TikTok Ads API**: creative-heavy paid social channel.
- **Klaviyo / Mailchimp APIs**: lifecycle email and retention automations.
- **Pinterest API**: catalog and discovery traffic.

### Customer support and trust
- **Gorgias / Zendesk / Freshdesk APIs**: ticketing automation.
- **Intercom API**: conversational support workflows.
- **Judge.me / Yotpo APIs**: review ingestion and trust loops.

### Data and monitoring connectors
- **Segment / RudderStack**: event routing from storefront to warehouse/tools.
- **BigQuery / Snowflake / Redshift connectors**: performance warehouse.
- **dbt + orchestration (Airflow/Prefect/Dagster)**: model pipelines.
- **Sentry / Datadog / OpenTelemetry**: reliability and incident observability.

## Agent frameworks and "skills" ecosystems to evaluate

### Multi-agent orchestration frameworks
- **LangGraph**: graph-based, stateful agent workflows with deterministic checkpoints.
- **AutoGen**: agent-to-agent collaboration patterns.
- **CrewAI**: role-based multi-agent task decomposition.
- **Semantic Kernel**: plugin-centric orchestration with planner support.
- **Haystack Agents**: retrieval + tool invocation in production pipelines.

### Tooling standards and reusable skills
- **Model Context Protocol (MCP)** servers: standardized tool connectors for APIs/internal services.
- **OpenAPI-generated tool adapters**: generate strongly-typed connectors from API schemas.
- **Policy/guardrail runtimes** (e.g., policy-as-code): enforce spend/compliance gates before action.

### Workflow and job reliability
- **Temporal**: durable workflows, retries, long-running process safety.
- **Prefect / Dagster**: data and ML pipeline orchestration.
- **Celery + Redis/RabbitMQ**: lightweight async worker model.
- **Kafka / NATS**: event bus backbone for decoupled agents.

### Retrieval and memory
- **Postgres + pgvector**: operational + semantic memory in one system.
- **Qdrant / Weaviate / Pinecone**: vector retrieval for historical campaign/playbook memory.
- **Feast** (optional): feature store for model-led scoring.

## Reliability/cost design patterns (important for agentic operations)
- **Route-by-confidence**: only auto-execute when confidence + unit economics thresholds pass.
- **Two-stage actions**: draft -> validate -> execute for listing, spend, and refunds.
- **Kill switch agents**: pause spend/listings when anomaly thresholds are breached.
- **Supplier redundancy**: keep 2+ POD providers per high-volume SKU class.
- **Adaptive creative budget caps**: limit spend until statistical significance is reached.
- **Unit economics guardrails**: include returns, chargebacks, platform fees, and defect allowances.
- **Policy-first publishing**: trademark/IP/content checks before listing/ad launch.

## Luka repository check: current findings

### What was verifiable locally
- This repo already includes a **Goldfish sidecar integration pattern** under `research/goldfish/` focused on replay discipline and model reseed provenance.
- The local doc indicates Goldfish is intentionally separated from live runners and used to publish accepted research artifacts back into manifests.
- `requirements-research.txt` includes `goldfish-ml>=0.2.0`, indicating a packaged dependency already expected in research environments.

### Potential value if Luka's wider stack is used
If `meerkat` and `elephant` follow similar conventions to `goldfish`, likely benefits would be:
- **More robust experiment lifecycle management** (clear finalize/publish gates).
- **Higher replayability and provenance** for decisions and model changes.
- **Operational separation** between live execution and research backtesting.
- **Cost control** through disciplined promotion of only accepted runs.

### Follow-up required
- Re-run this discovery when GitHub access is available and explicitly map:
  - architecture and APIs in `meerkat`, `elephant`, and upstream `goldfish`;
  - how they can backstop agent workflows (state, retries, model promotion);
  - migration/integration cost into this codebase.

## Suggested first implementation package (for the next step)
1. Create new folder `initiatives/agentic_pod/`.
2. Add a dedicated `AGENTS.md` for scope-specific rules, safety, and operating playbooks.
3. Implement connector contracts first (typed interfaces + webhooks + idempotency).
4. Add policy engine + risk guards before enabling autonomous execution.
5. Stand up minimal dashboards: margin, delivery SLA, policy incidents, spend anomalies.
