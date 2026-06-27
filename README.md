# Trade Assistant — AI Multi-Agent Trading Dashboard

A split-panel trading assistant with a live TradingView chart on the left and an AI-powered trade setup summary on the right. Analyse or execute trades using live data from Delta Exchange (crypto) and Fyers (NSE equities) — powered by OpenAI or Ollama via A2A-connected agents and MCP tool servers.

---

## Kronos Fine-Tune — XGBoost

`kronos_train.ipynb` trains an XGBoost classifier on 5 years of candle data.

**Features (28 total):**
- EMA 9/21/50/SMA200 ratios (scale-invariant)
- RSI(14), ATR % of price, body %
- Trend label, CPR position, distance to support/resistance
- Volume ratio vs 10-bar average
- 16 candlestick pattern one-hot flags

**Label:** `1` if `close[t+1] > close[t]`, else `0`

**Output:** `models/kronos_{SYMBOL}_{INTERVAL}.json`

Once trained, `KronosAgent` auto-loads the model on startup and uses the ML probability instead of the hand-coded score. Falls back to rule-based score if no model file exists.

**Training steps:**
1. Open `kronos_train.ipynb` — set `SYMBOL`, `INTERVAL`, `YEARS`
2. Run all cells — fetches history, extracts features, trains, evaluates, saves
3. Model auto-activates — no code change needed



`KronosAgent` (`backend/agents/kronos_agent.py`) is a **pure-Python, zero-LLM** technical analysis agent that runs before the LLM analysis and pre-computes:

| Indicator | Detail |
|---|---|
| EMA 9 / 21 / 50 / SMA 200 | Trend alignment |
| RSI (14) | Overbought / oversold zone |
| ATR (14) | Volatility measure |
| CPR (Central Pivot Range) | Pivot, BC, TC, R1/R2, S1/S2 |
| Support / Resistance | Swing-high / swing-low from last 20 bars |
| Candlestick patterns | Doji, Hammer, Engulfing, Morning/Evening Star, Marubozu, etc. |
| Volume signal | High / low / normal vs 10-bar average |
| Composite bias | Scored buy/sell/neutral signal combining all above |

The `AnalysisAgent` calls Kronos in parallel with the existing MCP fetches (candles + metrics + news). Kronos context is injected into the LLM prompt as a structured block — the LLM validates the setup rather than discovering it from raw OHLCV.

**Service:** `:8103`  
**Fallback:** runs in-process if `:8103` is unreachable  
**No LLM cost** — all pure Python math

---

## Candle Forecast Notebook

`candle_forecast.ipynb` — backtestable forecast vs reality:

1. Set `SYMBOL`, `INTERVAL`, `FORECAST_AT` (historical timestamp), `FORECAST_N`
2. Fetches history up to that point → runs Kronos TA → calls LLM for N-candle forecast
3. Fetches the real candles that followed
4. Scores: direction hit/miss, target hit, SL hit, per-candle close error %
5. Plots forecast vs reality side-by-side (dark theme)

---



| # | File | Fix |
|---|---|---|
| 1 | `backend/agents/trade_agent/` | Deleted — dead code, broken import, duplicates ExecutionAgent |
| 2 | `backend/agents/analysis_agent.py` | `asyncio` moved to top-level import; `_INTERVAL_MAP` uses lowercase `1d`; LLM now receives all 20 candles with period high/low/avg instead of 5 |
| 3 | `backend/mcp/connector.py` | `_SERVERS` dict moved inside `_get_server_cmd()` — resolved lazily so settings are fully loaded before first call |
| 4 | `backend/agents/orchestrator.py` | `_normalise()` no longer appends `USDT` to unknown symbols (NSE equities like `RBLBANK` were becoming `RBLBANKUSDT`); hard 55% confidence floor added in `_handle_execute()` before any order is sent |
| 5 | `backend/agents/execution_agent.py` | Deterministic position sizing: `qty = floor(risk_amount / abs(entry - sl))` at 1% risk on ₹1,00,000 — replaces LLM-hallucinated qty |
| 6 | `backend/a2a/client.py` | Retry logic: up to 2 retries with 1.5× exponential back-off on `ConnectError`, `TimeoutException`, `RemoteProtocolError` |



```bash
cd trade_assistant
pip install -r requirements.txt
cp .env.example .env   # add your API keys

# Start all services
python start.py
```

| URL | What opens |
|---|---|
| http://localhost:8100 | Dashboard (HTML frontend) |
| http://localhost:8100/docs | Swagger API docs |

---

## Start Commands

### All services (recommended)

```bash
python start.py
```

Starts 3 services in order:
1. **Analysis Agent** → `:8101`
2. **Execution Agent** → `:8102`
3. **FastAPI + HTML UI** → `:8100`

### Manual (separate terminals)

```bash
# Terminal 1 — Analysis Agent (A2A service, port 8101)
python -m services.analysis_agent_service --port 8101

# Terminal 2 — Execution Agent (A2A service, port 8102)
python -m services.execution_agent_service --port 8102

# Terminal 3 — FastAPI + HTML UI (port 8100)
uvicorn backend.main:app --host 0.0.0.0 --port 8100 --reload
# Open http://localhost:8100 in browser
```

### MCP Servers (spawned automatically — run manually for testing)

```bash
python -m mcp_servers.delta_server    # Delta Exchange tools (candles, orders)
python -m mcp_servers.fyers_server    # Fyers NSE/BSE tools (candles, orders)
python -m mcp_servers.tavily_server   # News search (Google RSS fallback)
```

### Verify all services are running

```bash
curl http://localhost:8100/health
curl http://localhost:8101/.well-known/agent.json
curl http://localhost:8102/.well-known/agent.json
```

Expected responses:
```json
{"status": "ok", "version": "1.0.0"}
{"name": "analysis_agent", "url": "http://0.0.0.0:8101", "capabilities": ["task"]}
{"name": "execution_agent", "url": "http://0.0.0.0:8102", "capabilities": ["task"]}
```

> **Note:** The gateway (:8100) has an in-process fallback — if :8101 is down **or times out**, analysis runs inside the main process automatically. A2A client timeout is 300 s. MCP calls (candles, metrics, news) run in parallel to reduce total latency. Per-tool MCP timeouts: fetch_candles 25 s, fetch_perpetual_metrics 15 s, search_news 15 s — on timeout, partial data is passed to the LLM and analysis still completes.

---

## Architecture

```
User (browser http://localhost:8100)
      │
      ▼ GET /  → serves frontend/index.html
      │ POST /chat  (no auth)
FastAPI Gateway :8100
      │
      ├── OrchestratorAgent (intent routing + LLM)
      │       │
      │       ├── A2A HTTP ──► Analysis Agent :8101
      │       │                   └── MCP stdio ──► delta/fyers/tavily servers
      │       │                   (fallback: runs in-process if :8101 is down)
      │       │
      │       ├── A2A HTTP ──► Execution Agent :8102
      │       │                   └── MCP stdio ──► delta/fyers servers
      │       │
      │       └── Tavily MCP ──► web search (research intent only)
      │
      └── Auth (JWT + SQLite sessions)
```

**Intent routing:**
| User message | Route |
|---|---|
| "Analyse RBLBANK for intraday" | → AnalysisAgent (:8101) |
| "Execute trade on BTCUSDT" | → AnalysisAgent then ExecutionAgent (:8102) |
| "What is Canara Bank YoY profit?" | → Tavily MCP web search → LLM |
| "Hello / general questions" | → LLM directly |

---

## Services & Ports

| Service | Port | File | Description |
|---|---|---|---|
| FastAPI + HTML UI | 8100 | `backend/main.py` | Chat, auth, Fyers token endpoints + serves `frontend/index.html` |
| Analysis Agent | 8101 | `services/analysis_agent_service.py` | Market data + LLM analysis |
| Execution Agent | 8102 | `services/execution_agent_service.py` | Trade plan + order placement |
| Delta MCP | stdio | `mcp_servers/delta_server.py` | Auto-spawned per call |
| Fyers MCP | stdio | `mcp_servers/fyers_server.py` | Auto-spawned per call |
| Tavily MCP | stdio | `mcp_servers/tavily_server.py` | Auto-spawned per call |

---

## Configuration (`.env`)

```bash
# LLM
OPENAI_API_KEY=sk-...
DEFAULT_LLM_PROVIDER=openai        # openai | ollama
OLLAMA_BASE_URL=http://localhost:11434

# Delta Exchange
DELTA_API_KEY=...
DELTA_API_SECRET=...
DELTA_REGION=global

# Fyers (token expires daily — see Token Refresh below)
FYERS_CLIENT_ID=...
FYERS_ACCESS_TOKEN=eyJ...
FYERS_SECRET_KEY=...               # needed for token generation

# Tavily (optional — falls back to Google RSS)
TAVILY_API_KEY=tvly-...

# LangSmith tracing (optional)
LANGCHAIN_TRACING_V2=false
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=trade-assistant

# Auth
SECRET_KEY=change-me-32-chars-min
```

---

## Fyers Token Refresh

Fyers access tokens **expire every day at midnight IST**. Refresh daily before trading.

**Step 1** — Get the login URL:
```bash
curl http://localhost:8100/fyers/login-url
```
Open the `login_url` in your browser, login, copy the `code=` from the redirect URL.

**Step 2** — Exchange code for token (auto-updates `.env`):
```bash
curl -X POST http://localhost:8100/fyers/generate-token \
  -H "Content-Type: application/json" \
  -d '{"auth_code": "PASTE_CODE_HERE"}'
```

**Check token status:**
```bash
curl http://localhost:8100/fyers/auth-status
# → {"authenticated": true, "expires_at": "2026-06-24 00:30 UTC", "expired": false}
```

---

## API Reference

### Chat & Sessions
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/register` | Create account `{"username":"...", "password":"..."}` |
| `POST` | `/auth/login` | Login → JWT token |
| `POST` | `/chat` | Send message → agent response |
| `POST` | `/sessions/new` | Start new chat session |
| `GET` | `/sessions` | List your sessions |
| `GET` | `/sessions/{id}/messages` | Load chat history + token count |

### Fyers Token
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/fyers/auth-status` | Check token validity + JWT expiry |
| `GET` | `/fyers/login-url` | Get OAuth login URL |
| `POST` | `/fyers/generate-token` | Exchange auth code → update .env |

### Meta
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/logs/download` | Download log file |
| `GET` | `/docs` | Swagger UI |

---

## Supported Symbols

**Delta Exchange (crypto):**
`BTCUSDT`, `ETHUSDT`, `XAUUSDT` (Gold), `SOLUSDT`, `BNBUSDT`

**Fyers (NSE equities):**
`NIFTY50`, `BANKNIFTY`, `RELIANCE`, `TCS`, `SBIN`, `HDFCBANK`, `ICICIBANK`,
`AXISBANK`, `CANBK` (Canara), `PNB`, `BANKBARODA`, `INFY`, `WIPRO`,
`TATAMOTORS`, `MARUTI`, `BAJAJFINANCE`, `SUNPHARMA`, `ONGC`, `NTPC`,
`LT`, `ADANIPORTS`, `RBLBANK`, `IDBI` and more

---

## UI Layout

The frontend is a pure HTML/CSS/JS single-page app served directly by FastAPI at `GET /` — no Streamlit, no login required.

| Column | Width | Content |
|---|---|---|
| Left | ~65% | TradingView widget (`tv.js`) — created once, symbol/TF updated via `setSymbol()`/`setResolution()` without destroying the iframe. Drawings sync to TV account (log in on tradingview.com in the same browser). |
| Right | ~35% | Symbol dropdown (30+ symbols: Crypto/Indices/Banking/IT/Large Cap), broker badge, Analyse/Execute buttons, trade summary cards. **Hide/Show Panel** in header. |

**Symbol → Broker auto-detection:**
- `BTC`, `ETHUSDT`, `SOLUSDT`, etc. → **Delta (Crypto)** badge → routes to Delta Exchange
- `RELIANCE`, `NIFTY`, `BANKNIFTY`, etc. → **Fyers (NSE)** badge → routes to Fyers

**Symbol → TradingView mapping:**
- `BTCUSDT` → `BINANCE:BTCUSDT`
- `RELIANCE` → `NSE:RELIANCE`
- `NIFTY` / `NIFTY50` → `NSE:NIFTY`
- `BANKNIFTY` → `NSE:BANKNIFTY`

**Trade Setup Summary cards** (shown after Analyse):
1. **Trade Setup** — trend (bull/bear colour coded), strength, price, confidence
2. **Levels** — entry zone, stop loss, R:R ratio, targets (T1/T2/T3)
3. **Key Levels** — support/resistance
4. **Analysis** — plain-text LLM summary
5. **Execution Plan** (after Execute) — action, qty, order type, paper/real mode

---

## Features

- **Pure HTML/CSS/JS frontend** — no framework, no login, served directly from FastAPI at port 8100
- **Auto broker routing** — symbol auto-detects Delta (crypto) vs Fyers (NSE) with a live badge
- **JWT auth** — login/register, sessions saved to SQLite
- **Token counter** — live tokens used + estimated OpenAI cost
- **LLM selection** — OpenAI gpt-4o-mini or Ollama per session
- **Paper / Real** trade mode toggle
- **4 intent types** — analyse, execute, research (web search), chat
- **In-process fallback** — works even if agent services are down
- **Decoupled agents** — LLM provider injected per request, no config coupling
- **LangSmith tracing** — trace all agent calls (set `LANGCHAIN_TRACING_V2=true`)

---

## Project Structure

```
trade_assistant/
├── start.py                         ← Start all 4 services
├── requirements.txt
├── .env.example
├── mcp_servers/
│   ├── delta_server.py              ← 8 tools: candles, metrics, orders
│   ├── fyers_server.py              ← 7 tools: candles, quotes, orders
│   └── tavily_server.py             ← 3 tools: news search
├── services/
│   ├── analysis_agent_service.py    ← A2A wrapper :8101
│   └── execution_agent_service.py   ← A2A wrapper :8102
├── backend/
│   ├── main.py                      ← FastAPI :8100
│   ├── config.py                    ← Settings (absolute .env path)
│   ├── protocol.py                  ← Task / TaskResponse / Artifact
│   ├── a2a/
│   │   ├── server.py                ← A2AServer (wraps agent as HTTP)
│   │   └── client.py                ← A2AClient (async + fallback)
│   ├── agents/
│   │   ├── orchestrator.py          ← Intent routing + A2A dispatch
│   │   ├── analysis_agent.py        ← MCP fetch + LLM analysis
│   │   └── execution_agent.py       ← LLM plan + MCP orders
│   ├── auth/store.py                ← SQLite users + JWT + sessions
│   ├── brokers/delta/               ← DeltaSource + DeltaExecutor
│   ├── brokers/fyers/               ← FyersSource
│   ├── llm/factory.py               ← LLMFactory (OpenAI / Ollama)
│   └── mcp/connector.py             ← call_mcp_tool() one-shot helper
├── frontend/
│   ├── index.html                   ← HTML/CSS/JS dashboard (no framework, no login)
│   └── app.py                       ← Legacy Streamlit UI (superseded)
├── tests/
│   └── test_trade_assistant.py      ← 10 tests
└── data/
    ├── users.db                     ← SQLite auth DB
    └── logs/trade_assistant.log     ← Rotating log file
```

---

## Running Tests

```bash
python -m pytest tests/ -v
# 10 tests — runs in ~9s
```
