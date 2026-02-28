# Betfair API — Official sample code & references

Summary of the [Betfair Sample Code](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687537/Sample+Code) page and key sublinks. This project uses [betfairlightweight](https://github.com/betcode-org/betfair) (Python client library from the “Client Libraries” list).

---

## Betfair-developed sample code

All official samples follow the same workflow:

1. **Find** the next UK Horse Racing Win market  
2. **Get prices** for the market  
3. **Place a bet** on the market  
4. **Handle the error** when the bet fails (e.g. below minimum stake)

**Note:** These samples are kept simple for clarity. For production you should follow [Optimizing API Application Performance](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2699882/Optimizing+API+Application+Performance) (gzip, keep-alive, no `Expect: 100-Continue`).

| Language   | Docs | Repo |
|-----------|------|------|
| **Python** | [Python](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687059/Python) | [API-NG-sample-code/python](https://github.com/betfair/API-NG-sample-code/tree/master/python) |
| Java      | [Java](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687529/Java) | [API-NG-sample-code/java](https://github.com/betfair/API-NG-sample-code/tree/master/java) |
| Javascript | [Javascript](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687066/Javascript) | [API-NG-sample-code/javascript](https://github.com/betfair/API-NG-sample-code/tree/master/javascript) |
| C#        | [C#](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687074/C) | [API-NG-sample-code/cSharp](https://github.com/betfair/API-NG-sample-code/tree/master/cSharp) |
| PHP, Excel/VBA, Curl, Perl | See [Sample Code](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687537/Sample+Code) | Same repo, other folders |

---

## Python sample (official)

- **Repo:** [betfair/API-NG-sample-code — python](https://github.com/betfair/API-NG-sample-code/tree/master/python)
- **Files:** `ApiNgDemoJsonRpc.py` (Python 2.7), `ApiNgDemoJsonRpc-python3.py`, `ApiNgDemoRescript.py`
- **Flow:**
  - Call `listEventTypes` with empty filter to get event types.
  - Get “Horse Racing” event type ID, then `listMarketCatalogue` with `eventTypeIds`, `marketCountries: ["GB"]`, `marketTypeCodes: ["WIN"]`, `marketStartTime.from = now`, `sort: FIRST_TO_START`, `maxResults: 1`.
  - Get `marketId` and first runner `selectionId` from catalogue.
  - Call `listMarketBook` with `marketIds` and `priceProjection.priceData: ["EX_BEST_OFFERS"]` to read `availableToBack` / `availableToLay`.
  - Call `placeOrders` with `marketId`, `instructions` (selectionId, side BACK, orderType LIMIT, limitOrder size/price/persistenceType), then handle error (e.g. `instructionReports[0].errorCode`).
- **Endpoints:** JSON-RPC at `https://api.betfair.com/exchange/betting/json-rpc/v1`; Rescript at `https://api.betfair.com/rest/v1.0/<operationName>/`.
- **Headers:** `X-Application`, `X-Authentication` (session token), `content-type: application/json`.

---

## Client libraries (community)

| Language | Repo | Description |
|----------|------|-------------|
| **Python** | [betcode-org/betfair](https://github.com/betcode-org/betfair) | **Lightweight Python wrapper for API-NG (with streaming)** — used by this project as `betfairlightweight`. |
| **Python** | [betcode-org/flumine](https://github.com/betcode-org/flumine) | Betting/trading framework built on the same wrapper. |
| Java      | [joelpob/jbetfairng](https://github.com/joelpob/jbetfairng) | Client library for Java. |
| C#        | [joelpob/betfairng](https://github.com/joelpob/betfairng) | API-NG client for C#. |
| Node.js   | [AlgoTrader/betfair](https://github.com/AlgoTrader/betfair) | API-NG client for Node.js. |
| Others    | See [Sample Code — Client Libraries](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687537/Sample+Code) | PHP, Ruby, Perl, Excel, Scala, R, C++, Rust. |

---

## Exchange Stream API

Real-time market (and order) data over a **subscription-based** connection. Main doc: [Exchange Stream API](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687396/Exchange+Stream+API).

**Overview**

- **Protocol:** WebSocket. Subscribe by sending a **marketSubscriptionMessage** with a market filter (e.g. market IDs). You receive continuous updates (order book / ladder data) instead of polling REST.
- **Unsubscribe:** Change the subscription message (e.g. different `marketFilter`) to unsubscribe from current markets. Closed markets are auto-removed from the subscription.
- **Heartbeat:** Configurable `heartbeatMs`; server sends heartbeats when there is no market activity to keep the connection alive.
- **Reconnection:** On reconnect, resend your subscription and use `initialClk` / `clk`; you get a `RESUB_DELTA` to patch to the latest state. See [Market Streaming - how do I manage re-connections?](https://support.developer.betfair.com/hc/en-us/articles/360000391612-Market-Streaming-how-do-I-managed-re-connections).
- **Access:** Stream API access must be requested (e.g. [How do I get access to the Stream API?](https://support.developer.betfair.com/hc/en-us/articles/115003887871-How-do-I-get-access-to-the-Stream-API)). For web app keys, bearer token is required: [Stream API - Bearer Token Must Be Used for Web App Key](https://support.developer.betfair.com/hc/en-us/articles/360000391432-Stream-API-Bearer-Token-Must-Be-Used-for-Web-App-Key).

**Support / FAQ (Exchange Stream API)**

- [Exchange Stream API — Support section](https://support.developer.betfair.com/hc/en-us/sections/360000520652-Exchange-Stream-API): segmentation, OCM/MCM matching, heartbeats & conflation, reconnections, unsubscribe, void bets, runner changes (size=0), connection closed errors, “Infinity” size, NOT_AUTHORIZED, UNEXPECTED_ERROR, conflation messages, closed markets auto-removal, Web Socket support.
- [Market & Order Stream API - How does it work?](https://support.developer.betfair.com/hc/en-us/articles/360000402291-Market-Order-Stream-API-How-does-it-work)
- [Market Streaming - Heartbeat and Conflation](https://support.developer.betfair.com/hc/en-us/articles/360000402611-Market-Streaming-How-do-the-Heartbeat-and-Conflation-requests-work)
- [How do you unsubscribe from a market?](https://support.developer.betfair.com/hc/en-us/articles/360000391532-How-do-you-unsubscribe-from-a-market-using-the-Stream-API)

**Official sample code**

- [Stream API sample code](https://github.com/betfair/stream-api-sample-code) — C#, Java, Node.js.

**In this project**

- `data/betfair_stream.py`: WebSocket client using betfairlightweight streaming; subscribes to markets with `EX_ALL_OFFERS`, `EX_MARKET_DEF`; converts stream updates to `PriceSnapshot` and calls `on_price_update`; reconnects with exponential backoff + jitter. The dashboard uses REST polling (`data/price_poller.py`) by default; the stream client is available when Stream API access is enabled on your app key.

---

## Historical data

- Service: [historicdata.betfair.com](https://historicdata.betfair.com)
- [betcode-org/betfair](https://github.com/betcode-org/betfair) — parse/output historical data, backtesting, CSV.
- [Betfair Historical Data Processor](https://www.betfairhistoricdata.co.uk/) — convert downloaded files to CSV.
- [Competition & Event Mapping Data](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2686993/Additional+Information#CompetitionId&EventMappingData) (2018–2023).

---

## Performance (official)

[Optimizing API Application Performance](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2699882/Optimizing+API+Application+Performance):

- **Do not** send `Expect: 100-Continue` (can cause 417).
- **Do** send `Accept-Encoding: gzip, deflate` for compressed responses.
- **Do** use persistent connections (`Connection: keep-alive`).

The `requests` library (used by betfairlightweight) typically handles gzip and keep-alive by default.

---

## Tutorials (Betfair Australia / community)

- [Betfair API Python Tutorial](https://betfair-datascientists.github.io/tutorials/apiPythontutorial/)
- [How to Automate I: Understanding Flumine](https://betfair-datascientists.github.io/tutorials/How_to_Automate_1/)
- [Historical data — Json to Csv (Python)](https://betfair-datascientists.github.io/tutorials/jsonToCsvRevisited/)
- [Backtesting with Betfair JSON stream data](https://betfair-datascientists.github.io/historicData/backtestingRatingsTutorial/#complete-code)

---

## Mapping to this project

| We do | Official / library |
|-------|---------------------|
| Login (cert or interactive) | Non-Interactive login + Interactive login (see Login & Session Management in docs). |
| `list_market_catalogue` | Same operation; we use `market_filter(event_type_ids=..., market_countries=...)` or empty for all. |
| `list_market_book` | Same; we use `price_projection(price_data(ex_all_offers=True))` for ladder. |
| `place_orders` | Same; we use `PlaceInstruction` + `LimitOrder` (side BACK, persistence_type LAPSE). |
| Streaming | Optional; we have a stream client in `data/betfair_stream.py`; dashboard uses REST polling. |

All links above point to the [Betfair Exchange API Documentation](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/) or the stated GitHub repos.
