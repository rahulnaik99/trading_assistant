# Trade Assistant — AI Trading Chat

Multi-agent AI trading assistant with ChatGPT-style interface.

## Quick Start

```bash
# 1. Go to project
cd trade_assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and edit .env
cp .env.example .env
# Edit .env with your API keys

# 4. Start everything
python start.py

# Or start manually:
# Terminal 1 — Backend (port 8100)
uvicorn backend.main:app --host 0.0.0.0 --port 8100 --reload

# Terminal 2 — Frontend (port 8501)
streamlit run frontend/app.py --server.port 8501
```

**Open:** http://localhost:8501

## Architecture

```
User (Streamlit chat)
      ↓ POST /chat
FastAPI :8100  ←→  OrchestratorAgent
                      ├── AnalysisAgent ←→ Delta/Fyers/Tavily MCP
                      └── ExecutionAgent ←→ Delta/Fyers MCP
```

## Agents

| Agent | Role |
|---|---|
| **OrchestratorAgent** | Parses intent, routes to sub-agents, formats reply |
| **AnalysisAgent** | Fetches candles + metrics via MCP, calls LLM for analysis |
| **ExecutionAgent** | Builds execution plan, places bracket orders via MCP |

## Features
- Chat interface with session history (SQLite)
- Login/register with JWT auth
- Live token counter + cost estimate
- OpenAI or Ollama LLM selection
- Paper or Real trade mode
- LangSmith tracing (set LANGCHAIN_TRACING_V2=true)
- Supports: BTCUSDT, ETHUSDT, NSE:RELIANCE-EQ, NIFTY50, and more

## Example Queries
- *"Analyse Reliance for intraday trade"*
- *"What's the swing trade setup for BTCUSDT?"*
- *"Execute trade on ETH"*
- *"Longterm outlook for NIFTY50?"*
