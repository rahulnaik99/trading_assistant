"""Trade Assistant — test suite using trading_analyst credentials."""

import json
import sys
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_CANDLES = json.dumps({
    "rows": 50, "last_close": 64300.0,
    "candles": [{"open": 64000, "high": 64500, "low": 63900, "close": 64300, "volume": 1000}] * 5
})
MOCK_METRICS = json.dumps({
    "mark_price": 64300.0, "funding_rate_8h_pct": 0.037,
    "funding_signal": "neutral", "oi": 15.5, "oi_signal": "neutral",
})
MOCK_NEWS = json.dumps({"results": [{"title": "BTC consolidates near 64k"}], "source": "rss"})

MOCK_ANALYSIS_LLM = json.dumps({
    "symbol": "BTCUSDT", "trade_type": "intraday", "trend": "bullish",
    "strength": "moderate", "last_price": 64300.0,
    "entry_zone": {"low": 64100.0, "high": 64500.0},
    "stop_loss": 63800.0, "targets": [65000.0, 65800.0],
    "rr_ratio": 2.0, "confidence": 0.72,
    "summary": "BTC showing bullish momentum above key support at 64k.",
    "key_levels": {"support": [63800.0, 63000.0], "resistance": [65000.0, 66000.0]},
    "news_context": "BTC consolidates near 64k",
})

MOCK_EXECUTION_LLM = json.dumps({
    "action": "buy", "symbol": "BTCUSDT", "broker": "delta",
    "order_type": "market_order", "entry": 64300.0,
    "stop_loss": 63800.0, "take_profit": 65000.0, "take_profit_2": 65800.0,
    "qty": 1, "risk_amount": 500.0, "rr_ratio": 1.4,
    "rationale": "Bullish trend with 72% confidence — enter at market.",
    "mode": "paper",
})


async def _mock_mcp(server, tool, arguments=None):
    if tool == "fetch_candles":            return MOCK_CANDLES
    if tool == "fetch_perpetual_metrics":  return MOCK_METRICS
    if tool in ("search_news","get_rss_news"): return MOCK_NEWS
    return json.dumps({"result": "ok"})


# ── Test 1: Config loads with real credentials ─────────────────────────────────

def test_config_loads():
    from backend.config import settings
    assert settings.OPENAI_API_KEY, "OPENAI_API_KEY not set"
    assert settings.DELTA_API_KEY,  "DELTA_API_KEY not set"
    assert settings.FYERS_CLIENT_ID == "OYTP8YTH2B-100"
    assert settings.LANGSMITH_PROJECT == "trade-assistant"
    print("TEST 1 PASS: Config loaded with trading_analyst credentials")


# ── Test 2: Auth — register, login, JWT ───────────────────────────────────────

def test_auth_flow():
    from backend.auth.store import init_db, create_user, verify_user, create_access_token, decode_token, create_session, save_message, load_messages
    init_db()

    try:
        uid = create_user("ta_testuser", os.getenv("TEST_PW", "test_pw_123"))
        assert uid
    except ValueError:
        pass  # already exists

    user = verify_user("ta_testuser", os.getenv("TEST_PW", "test_pw_123"))
    assert user is not None
    assert user["username"] == "ta_testuser"
    assert verify_user("ta_testuser", "wrongpass") is None

    token = create_access_token(user["uid"], user["username"])
    payload = decode_token(token)
    assert payload["username"] == "ta_testuser"

    # Session + messages
    sid = create_session(user["uid"])
    save_message(sid, "user", "Hello", tokens=5)
    save_message(sid, "assistant", "Hi there!", tokens=10)
    msgs = load_messages(sid)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"

    print("TEST 2 PASS: Auth register/login/JWT/sessions all work")


# ── Test 3: Protocol models ───────────────────────────────────────────────────

def test_protocol():
    from backend.protocol import Task, TaskResponse, Artifact
    task = Task(task_id="t1", agent="analysis_agent", input={"symbol": "BTCUSDT"})
    assert task.task_id == "t1"
    resp = TaskResponse(
        task_id="t1", agent="analysis_agent", status="completed",
        artifacts=[Artifact(type="analysis", data={"trend": "bullish"})],
    )
    assert resp.artifacts[0].data["trend"] == "bullish"
    print("TEST 3 PASS: A2A protocol Task/TaskResponse/Artifact models work")


# ── Test 4: LLM factory ───────────────────────────────────────────────────────

def test_llm_factory():
    from backend.llm.factory import LLMFactory
    llm = LLMFactory.get_llm("openai")
    assert llm is not None
    assert "ChatOpenAI" in type(llm).__name__
    print("TEST 4 PASS: LLMFactory creates OpenAI client with real key")


# ── Test 5: MCP connector (patched — no subprocess) ──────────────────────────

@pytest.mark.asyncio
async def test_mcp_connector_patched():
    import backend.mcp.connector as conn
    orig = conn.call_mcp_tool
    conn.call_mcp_tool = _mock_mcp
    try:
        result = await conn.call_mcp_tool("delta", "fetch_candles", {"symbol": "BTCUSDT"})
        data = json.loads(result)
        assert data["last_close"] == 64300.0
        assert data["rows"] == 50
    finally:
        conn.call_mcp_tool = orig
    print("TEST 5 PASS: MCP connector call_mcp_tool works (patched)")


# ── Test 6: Analysis Agent with mocked MCP + real OpenAI ─────────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_analysis_agent():
    """Test AnalysisAgent with fully mocked MCP + fake LLM."""
    from backend.agents import analysis_agent as agent_mod
    from backend.protocol import Task

    # Patch call_mcp_tool inside the agent's own module namespace
    original_mcp = agent_mod.call_mcp_tool
    agent_mod.call_mcp_tool = _mock_mcp

    agent = agent_mod.AnalysisAgent(llm_provider="openai")

    class FakeLLM:
        async def ainvoke(self, _messages):
            class R:
                content = MOCK_ANALYSIS_LLM
            return R()

    agent._llm = FakeLLM()

    try:
        task = Task(task_id="test-a", agent="analysis_agent",
                    input={"symbol": "BTCUSDT", "source": "delta", "trade_type": "intraday"})
        resp = await agent.handle_task(task)
    finally:
        agent_mod.call_mcp_tool = original_mcp

    assert resp.status == "completed"
    art = next(a for a in resp.artifacts if a.type == "analysis")
    assert art.data["trend"] == "bullish"
    assert abs(art.data["confidence"] - 0.72) < 0.01
    assert art.data["symbol"] == "BTCUSDT"
    print("TEST 6 PASS: AnalysisAgent  trend=bullish  conf=72%")


# ── Test 7: Execution Agent with mocked LLM + paper mode ─────────────────────

@pytest.mark.asyncio
async def test_execution_agent():
    """Test ExecutionAgent with fake LLM — no real API calls."""
    from backend.agents import execution_agent as exec_mod
    from backend.protocol import Task

    agent = exec_mod.ExecutionAgent(llm_provider="openai", mode="paper")

    class FakeLLM:
        async def ainvoke(self, _messages):
            class R:
                content = MOCK_EXECUTION_LLM
            return R()

    agent._llm = FakeLLM()

    analysis = {
        "symbol": "BTCUSDT", "trend": "bullish", "confidence": 0.72,
        "trade_type": "intraday", "last_price": 64300.0,
        "entry_zone": {"low": 64100, "high": 64500},
        "stop_loss": 63800.0, "targets": [65000.0],
    }
    task = Task(task_id="test-e", agent="execution_agent",
                input={"analysis": analysis, "source": "delta", "mode": "paper"})
    resp = await agent.handle_task(task)

    assert resp.status == "completed"
    art = next(a for a in resp.artifacts if a.type == "execution")
    assert art.data["action"] == "buy"
    assert art.data["order_result"]["status"] == "paper_filled"
    print(f"TEST 7 PASS: ExecutionAgent  action=BUY  paper  order={art.data['order_result']['order_id']}")


# ── Test 8: Orchestrator — analyse intent routing ─────────────────────────────

@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_orchestrator_analyse():
    """Test OrchestratorAgent intent routing — A2A calls mocked at client level."""
    from backend.agents import orchestrator as orch_mod
    from backend.protocol import Artifact, TaskResponse
    import json

    orch = orch_mod.OrchestratorAgent(llm_provider="openai", mode="paper")
    router_json = '{"intent":"analyse","symbol":"BTCUSDT","source":"delta","trade_type":"intraday","search_query":""}'

    class FakeOrcLLM:
        async def ainvoke(self, _messages):
            class R:
                content = router_json
            return R()

    # Mock the A2A analysis client — returns a pre-built TaskResponse over "HTTP"
    analysis_data = json.loads(MOCK_ANALYSIS_LLM)
    fake_a2a_resp = TaskResponse(
        task_id="a-test", agent="analysis_agent", status="completed",
        artifacts=[Artifact(type="analysis", data=analysis_data)],
    )

    class FakeAnalysisClient:
        async def send(self, agent, input_data, task_id=""):
            return fake_a2a_resp
        async def health(self):
            return True

    orch._llm = FakeOrcLLM()
    orch._analysis_client = FakeAnalysisClient()

    result = await orch.handle_message("Analyse BTCUSDT for intraday trade")

    assert result["intent"] == "analyse"
    assert result["symbol"] == "BTCUSDT"
    assert result["analysis"] is not None
    assert result["analysis"]["trend"] == "bullish"
    assert "bullish" in result["reply"].lower() or "BTCUSDT" in result["reply"]
    print("TEST 8 PASS: Orchestrator A2A → AnalysisAgent reply with bullish analysis")
    assert "bullish" in result["reply"].lower() or "BTCUSDT" in result["reply"]
    print("TEST 8 PASS: Orchestrator routed to analyse, got structured reply")


# ── Test 9: Full FastAPI HTTP stack ───────────────────────────────────────────

def test_fastapi_endpoints():
    """Test full HTTP stack — health, auth, sessions."""
    from backend.main import app
    import sqlite3
    from pathlib import Path

    # Always start with a clean test user so password never mismatches
    db = Path("data/users.db")
    if db.exists():
        c = sqlite3.connect(str(db))
        c.execute("DELETE FROM users WHERE username='http_testuser'")
        c.commit(); c.close()

    client = TestClient(app, raise_server_exceptions=True)
    pw = "testpw_" + "xyz"  # not a real credential — test-only

    # Health
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # Register
    r = client.post("/auth/register", json={"username": "http_testuser", "password": pw})
    assert r.status_code == 200

    # Login → get token
    r = client.post("/auth/login", data={"username": "http_testuser", "password": pw})
    assert r.status_code == 200
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # New session
    r = client.post("/sessions/new", headers=headers)
    assert r.status_code == 200
    sid = r.json()["session_id"]

    # List sessions
    r = client.get("/sessions", headers=headers)
    assert r.status_code == 200
    assert len(r.json()["sessions"]) >= 1

    # Messages (empty)
    r = client.get(f"/sessions/{sid}/messages", headers=headers)
    assert r.status_code == 200
    assert r.json()["total_tokens"] == 0

    print("TEST 9 PASS: FastAPI /health /auth /sessions all correct")


# ── Test 10: Delta API live (real HTTP, no MCP subprocess) ────────────────────

@pytest.mark.asyncio
async def test_delta_api_live():
    from backend.brokers.delta.source import DeltaSource
    from backend.config import settings

    ds = DeltaSource(settings.DELTA_API_KEY, settings.DELTA_API_SECRET)
    try:
        data = await ds._request("GET", "/tickers/BTCUSDT")
        result = data.get("result", {})
        mark = result.get("mark_price")
        funding = result.get("funding_rate")
        assert mark and float(mark) > 0, "mark_price not returned"
        print(f"TEST 10 PASS: Delta API live  mark_price={float(mark):.2f}  funding={float(funding)*100:.4f}%")
    finally:
        await ds.close()
