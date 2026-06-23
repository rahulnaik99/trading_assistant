# Trade Assistant — AI Multi-Agent Trading Chat

A ChatGPT-style trading assistant that analyses markets, identifies demand/supply zones, and executes trades using live data from Delta Exchange and Fyers — powered by OpenAI or Ollama via A2A-connected agents and MCP tool servers.

---

## Quick Start

```bash
# 1. Clone and install
cd trade_assistant
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Edit .env — add OPENAI_API_KEY, DELTA_API_KEY, FYERS_CLIENT_ID etc.

# 3. Start all services
python start.py
```

Open **http://localhost:8501**

---

## What it can do

| Ask | What happens |
|---|---|
| *"Analyse RBLBANK for intraday"* | Fetches 15m Fyers data → S&D zones, BOS/CHoCH, LLM analysis |
| *"Demand/supply zones of BTCUSDT in 15m"* | Fetches Delta candles → ICT/SMC analysis |
| *"Swing trade setup for Reliance"* | 1h candles + perpetual metrics + news |
| *"Execute trade on ETH"* | Analysis → bracket order via Delta MCP (paper/real) |
| *"Canara Bank YoY profit?"* | Tavily web search → grounded LLM answer |
| *"What is the PE ratio of HDFC Bank?"* | Web search → current data, not stale training |

---

## Architecture

```
User (Streamlit :8501)
      │
      ▼ POST /chat
FastAPI Gateway :8100
      │
      ├── OrchestratorAgent (intent routing + LLM)
      │       │
      │       ├──► A2A HTTP → Analysis Agent Service :8101
      │       │       └── Fetches candles/metrics via MCP servers
      │       │           (falls back in-process if :8101 is down)
      │       │
      │       ├──► A2A HTTP → Execution Agent Service :8102
      │       │       └── Places bracket orders via MCP servers
      │       │
      │       └──► Tavily MCP → web search (research intent)
      │
      └── Auth (JWT + SQLite sessions)

MCP Servers (stdio subprocess per call):
  mcp_servers/delta_server.py   — candles, perp metrics, orders
  mcp_servers/fyers_server.py   — candles, quotes, orders
  mcp_servers/tavily_server.py  — news search + Google RSS fallback
```

---

## Services

| Service | Port | Start command |
|---|---|---|
| FastAPI Gateway | 8100 | `uvicorn backend.main:app --port 8100` |
| Analysis Agent | 8101 | `python -m services.analysis_agent_service` |
| Execution Agent | 8102 | `python -m services.execution_agent_service` |
| Streamlit UI | 8501 | `streamlit run frontend/app.py` |

**Start all at once:** `python start.py`

**Verify services are up:**
```bash
curl http://localhost:8100/health
curl http://localhost:8101/.well-known/agent.json
curl http://localhost:8102/.well-known/agent.json
```

---

## Configuration (`.env`)

```bash
# LLM
OPENAI_API_KEY=sk-...
DEFAULT_LLM_PROVIDER=openai       # openai | ollama
OLLAMA_BASE_URL=http://localhost:11434

# Delta Exchange (crypto)
DELTA_API_KEY=...
DELTA_API_SECRET=...
DELTA_REGION=global

# Fyers (NSE/BSE equities)
FYERS_CLIENT_ID=...               # e.g. OYTP8YTH2B-100
FYERS_ACCESS_TOKEN=eyJ...         # expires daily — see Token Refresh below
FYERS_SECRET_KEY=...              # needed for token generation

# Tavily (optional — falls back to Google RSS)
TAVILY_API_KEY=tvly-...

# LangSmith tracing (optional)
LANGCHAIN_TRACING_V2=false
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=trade-assistant

# Auth
SECRET_KEY=your-secret-key-32-chars-min
```

---

## Fyers Token Refresh

Fyers access tokens **expire every day at midnight IST**. Refresh daily before trading:

**Step 1** — Get the OAuth login URL:
```
GET http://localhost:8100/fyers/login-url
```
Open the `login_url` in your browser, login, copy the `code=` value from the redirect URL.

**Step 2** — Exchange for a new token (auto-updates `.env`):
```bash
curl -X POST http://localhost:8100/fyers/generate-token \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"auth_code": "PASTE_CODE_HERE"}'
```

**Check current token status:**
```
GET http://localhost:8100/fyers/auth-status
```
Returns: `{"authenticated": true/false, "expires_at": "2026-06-23 00:30 UTC", "expired": false}`

---

## Supported Symbols

**Delta Exchange (crypto):**
`BTCUSDT`, `ETHUSDT`, `XAUUSDT`, `SOLUSDT`, `BNBUSDT`

**Fyers (NSE equities):**
`NIFTY50`, `BANKNIFTY`, `RELIANCE`, `TCS`, `SBIN`, `HDFCBANK`, `ICICIBANK`,
`AXISBANK`, `CANBK` (Canara Bank), `PNB`, `BANKBARODA`, `INFY`, `WIPRO`,
`TATAMOTORS`, `MARUTI`, `BAJAJFINANCE`, `SUNPHARMA`, `ONGC`, `NTPC`,
`LT`, `ADANIPORTS`, `RBLBANK`, and more

---

## API Reference

### Core
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/register` | Create account |
| `POST` | `/auth/login` | Login → JWT token |
| `POST` | `/chat` | Send message → routed to agents |
| `POST` | `/sessions/new` | Create chat session |
| `GET` | `/sessions` | List your sessions |
| `GET` | `/sessions/{id}/messages` | Load chat history |

### Fyers
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/fyers/auth-status` | Check token validity + expiry |
| `GET` | `/fyers/login-url` | Get OAuth login URL |
| `POST` | `/fyers/generate-token` | Exchange auth code → token + update .env |

### Meta
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/logs/download` | Download log file |
| `GET` | `/docs` | Swagger UI |

---

## Features

- **ChatGPT-style UI** — session history, multiple conversations, dark theme
- **JWT auth** — login/register with username + password, sessions saved to SQLite
- **Live token counter** — shows tokens used + estimated cost per session
- **LLM selection** — OpenAI gpt-4o-mini or Ollama (llama3.2) per session
- **Paper / Real trade mode** — paper simulates, real places bracket orders
- **4 intent types** — analyse, execute, research (web search), chat
- **In-process fallback** — analysis works even when :8101 service is down
- **LangSmith tracing** — trace all agent calls (set `LANGCHAIN_TRACING_V2=true`)
- **Graceful auth errors** — clear message when Fyers token is expired
- **Decoupled agents** — LLM provider injected per-request, no config coupling

---

## Project Structure

```
trade_assistant/
├── start.py                     ← Start all services
├── requirements.txt
├── .env.example
├── mcp_servers/                 ← MCP tool servers (stdio)
│   ├── delta_server.py          (8 tools: candles, metrics, orders)
│   ├── fyers_server.py          (7 tools: candles, quotes, orders)
│   └── tavily_server.py         (3 tools: news search)
├── backend/
│   ├── main.py                  ← FastAPI :8100 + auth + fyers token endpoints
│   ├── config.py                ← Settings (absolute .env path)
│   ├── protocol.py              ← A2A Task / TaskResponse / Artifact
│   ├── a2a/
│   │   ├── server.py            ← A2AServer (wraps any agent as HTTP)
│   │   └── client.py            ← A2AClient (async httpx with fallback)
│   ├── agents/
│   │   ├── orchestrator.py      ← Intent routing + A2A dispatch + fallback
│   │   ├── analysis_agent.py    ← MCP data fetch + LLM analysis (stateless)
│   │   └── execution_agent.py   ← LLM execution plan + MCP orders (stateless)
│   ├── auth/store.py            ← SQLite users + JWT + sessions + messages
│   ├── brokers/
│   │   ├── delta/               ← DeltaSource + DeltaExecutor
│   │   └── fyers/               ← FyersSource
│   ├── llm/factory.py           ← LLMFactory (OpenAI / Ollama)
│   └── mcp/connector.py         ← call_mcp_tool() one-shot helper
├── services/
│   ├── analysis_agent_service.py  ← A2A service wrapper :8101
│   └── execution_agent_service.py ← A2A service wrapper :8102
├── frontend/
│   └── app.py                   ← Streamlit chat UI
├── tests/
│   └── test_trade_assistant.py  ← 10 tests (auth, agents, A2A, Delta live)
└── data/
    ├── users.db                 ← SQLite auth DB
    └── logs/trade_assistant.log ← Rotating log file
```

---

## Running Tests

```bash
python -m pytest tests/ -v
# 10 tests — runs in ~9s
```

Tests cover: config loading, auth flow, A2A protocol, LLM factory, MCP connector, AnalysisAgent, ExecutionAgent, OrchestratorAgent A2A routing, FastAPI HTTP stack, Delta API live.
