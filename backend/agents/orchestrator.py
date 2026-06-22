"""Main Orchestrator Agent — routes user queries to Analysis or Execution agents via A2A HTTP."""

import json
import logging
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.a2a.client import A2AClient
from backend.config import settings
from backend.llm.factory import LLMFactory
from backend.mcp.connector import call_mcp_tool
from backend.protocol import Task

logger = logging.getLogger(__name__)

_ROUTER_SYSTEM = """You are a trading assistant dispatcher. Classify the user's intent into one of:

1. "analyse"  — user wants market analysis, price outlook, trade recommendation
2. "execute"  — user wants to place/execute a trade
3. "research" — user asks a factual question that needs web search:
                 company financials (profit, revenue, EPS, PE ratio, results),
                 economic data, news events, IPO info, FII/DII data,
                 regulatory changes, sector data, macroeconomic figures
4. "chat"     — simple greeting, general knowledge, or out-of-scope

Extract:
- symbol: e.g. "NSE:CANBK-EQ", "BTCUSDT", "NIFTY50" — normalise to exchange format
- source: "fyers" for Indian equities/indices, "delta" for crypto
- trade_type: "intraday", "swing", or "longterm"
- intent: "analyse", "execute", "research", or "chat"
- search_query: for research intent, the best web search query (empty string otherwise)

Respond ONLY with JSON:
{"intent": string, "symbol": string, "source": string, "trade_type": string, "search_query": string}"""

# ── Symbol maps ────────────────────────────────────────────────────────────────
_FYERS_MAP: dict[str, str] = {
    "reliance": "NSE:RELIANCE-EQ", "tcs": "NSE:TCS-EQ",
    "nifty": "NSE:NIFTY50-INDEX", "banknifty": "NSE:NIFTYBANK-INDEX",
    "sbin": "NSE:SBIN-EQ", "hdfcbank": "NSE:HDFCBANK-EQ",
    "infy": "NSE:INFY-EQ", "wipro": "NSE:WIPRO-EQ",
    "icicibank": "NSE:ICICIBANK-EQ", "axisbank": "NSE:AXISBANK-EQ",
    "kotak": "NSE:KOTAKBANK-EQ", "kotakbank": "NSE:KOTAKBANK-EQ",
    # Public sector banks
    "canara": "NSE:CANBK-EQ", "canarabank": "NSE:CANBK-EQ", "canbk": "NSE:CANBK-EQ",
    "pnb": "NSE:PNB-EQ", "punjabnational": "NSE:PNB-EQ",
    "bankbaroda": "NSE:BANKBARODA-EQ", "bob": "NSE:BANKBARODA-EQ",
    "unionbank": "NSE:UNIONBANK-EQ", "indianbank": "NSE:INDIANB-EQ",
    "boi": "NSE:BANKINDIA-EQ", "bankindia": "NSE:BANKINDIA-EQ",
    # Large caps
    "tatamotors": "NSE:TATAMOTORS-EQ", "tatasteel": "NSE:TATASTEEL-EQ",
    "maruti": "NSE:MARUTI-EQ", "bajajfinance": "NSE:BAJFINANCE-EQ",
    "bajajfinserv": "NSE:BAJAJFINSV-EQ", "hul": "NSE:HINDUNILVR-EQ",
    "hindunilvr": "NSE:HINDUNILVR-EQ", "asianpaint": "NSE:ASIANPAINT-EQ",
    "sunpharma": "NSE:SUNPHARMA-EQ", "drreddy": "NSE:DRREDDY-EQ",
    "ongc": "NSE:ONGC-EQ", "ntpc": "NSE:NTPC-EQ",
    "powergrid": "NSE:POWERGRID-EQ", "adaniports": "NSE:ADANIPORTS-EQ",
    "adanient": "NSE:ADANIENT-EQ", "lt": "NSE:LT-EQ", "larsen": "NSE:LT-EQ",
    "rblbank": "NSE:RBLBANK-EQ", "idbi": "NSE:IDBI-EQ",
}
_DELTA_MAP: dict[str, str] = {
    "btc": "BTCUSDT", "bitcoin": "BTCUSDT",
    "eth": "ETHUSDT", "ethereum": "ETHUSDT",
    "gold": "XAUUSDT", "xau": "XAUUSDT",
    "sol": "SOLUSDT", "bnb": "BNBUSDT",
}


def _normalise(raw: str) -> tuple[str, str]:
    """Return (normalised_symbol, source) from a raw symbol string."""
    s = raw.lower().strip()
    for k, v in _FYERS_MAP.items():
        if k in s:
            return v, "fyers"
    for k, v in _DELTA_MAP.items():
        if k in s:
            return v, "delta"
    sym = raw.upper().replace(" ", "")
    if not sym.endswith("USDT") and not sym.startswith("NSE:"):
        sym += "USDT"
    return sym, "delta"


def _build_result(intent: str, symbol: str, source: str, trade_type: str) -> dict[str, Any]:
    return {"intent": intent, "symbol": symbol, "source": source,
            "trade_type": trade_type, "analysis": None, "execution": None, "reply": ""}


def _parse_route(content: str) -> dict:
    try:
        cleaned = re.sub(r"```json|```", "", content).strip()
        return json.loads(cleaned)
    except Exception:
        return {"intent": "chat", "symbol": "BTCUSDT", "source": "delta",
                "trade_type": "intraday", "search_query": ""}


def _build_chat_messages(
    system_prompt: str, history: list[dict], user_message: str
) -> list:
    msgs = [SystemMessage(content=system_prompt)]
    for h in (history or [])[-6:]:
        cls = HumanMessage if h["role"] == "user" else AIMessage
        msgs.append(cls(content=h["content"]))
    msgs.append(HumanMessage(content=user_message))
    return msgs


# ── Orchestrator ───────────────────────────────────────────────────────────────

class OrchestratorAgent:
    """Entry point — routes user messages to Analysis, Execution, Research, or Chat.

    Analysis and Execution agents are called via A2A HTTP (separate processes):
      Analysis  → POST http://localhost:8101/a2a/task
      Execution → POST http://localhost:8102/a2a/task
    """

    def __init__(self, llm_provider: str = "openai", mode: str = "paper") -> None:
        self._llm_provider = llm_provider
        self._mode = mode
        self._analysis_client  = A2AClient(settings.ANALYSIS_AGENT_URL)
        self._execution_client = A2AClient(settings.EXECUTION_AGENT_URL)
        self._llm = None
        logger.info("OrchestratorAgent init  provider=%s  mode=%s  "
                    "analysis_url=%s  execution_url=%s",
                    llm_provider, mode,
                    settings.ANALYSIS_AGENT_URL, settings.EXECUTION_AGENT_URL)

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLMFactory.get_llm(self._llm_provider)
        return self._llm

    async def handle_message(
        self,
        user_message: str,
        history: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Route user message → sub-agent → formatted reply."""
        task_id = str(uuid.uuid4())[:8]
        logger.info("orchestrator msg  id=%s  %s", task_id, user_message[:80])

        route = await self._route(user_message)
        intent       = route.get("intent", "chat")
        symbol_raw   = route.get("symbol", "BTCUSDT")
        source       = route.get("source", "delta")
        trade_type   = route.get("trade_type", "intraday")
        search_query = route.get("search_query", "") or user_message

        symbol, src_inferred = _normalise(symbol_raw)
        if source not in ("delta", "fyers"):
            source = src_inferred

        logger.info("orchestrator route  intent=%s  symbol=%s  source=%s  type=%s",
                    intent, symbol, source, trade_type)

        result = _build_result(intent, symbol, source, trade_type)

        if intent == "analyse":
            result.update(await self._handle_analyse(task_id, symbol, source, trade_type))
        elif intent == "execute":
            result.update(await self._handle_execute(task_id, symbol, source, trade_type))
        elif intent == "research":
            result["reply"] = await self._handle_research(search_query, user_message, history)
        else:
            result["reply"] = await self._handle_chat(user_message, history)

        return result

    # ── Intent handlers ───────────────────────────────────────────────────────

    async def _route(self, user_message: str) -> dict:
        resp = await self.llm.ainvoke([
            SystemMessage(content=_ROUTER_SYSTEM),
            HumanMessage(content=user_message),
        ])
        return _parse_route(resp.content)

    async def _handle_analyse(
        self, task_id: str, symbol: str, source: str, trade_type: str
    ) -> dict[str, Any]:
        logger.info("► A2A → AnalysisAgent  url=%s  symbol=%s  source=%s  type=%s",
                    settings.ANALYSIS_AGENT_URL, symbol, source, trade_type)
        resp = await self._analysis_client.send(
            agent="analysis_agent",
            input_data={"symbol": symbol, "source": source, "trade_type": trade_type},
            task_id=f"a-{task_id}",
        )
        if resp.status != "completed":
            err = resp.error or "unknown error"
            logger.error("A2A AnalysisAgent failed  task_id=%s  error=%s", task_id, err)
            return {"reply": f"Analysis failed: {err}"}
        analysis = next((a.data for a in resp.artifacts if a.type == "analysis"), {})
        logger.info("A2A AnalysisAgent done  trend=%s  conf=%s",
                    analysis.get("trend"), analysis.get("confidence"))
        return {"analysis": analysis, "reply": self._format_analysis(analysis)}

    async def _handle_execute(
        self, task_id: str, symbol: str, source: str, trade_type: str
    ) -> dict[str, Any]:
        # Step 1: get fresh analysis via A2A
        logger.info("► A2A → AnalysisAgent (for execution)  symbol=%s", symbol)
        a_resp = await self._analysis_client.send(
            agent="analysis_agent",
            input_data={"symbol": symbol, "source": source, "trade_type": trade_type},
            task_id=f"a-{task_id}",
        )
        analysis = {}
        if a_resp.status == "completed":
            analysis = next((a.data for a in a_resp.artifacts if a.type == "analysis"), {})
            logger.info("A2A Analysis done  trend=%s", analysis.get("trend"))

        # Step 2: execute via A2A
        logger.info("► A2A → ExecutionAgent  url=%s  symbol=%s  mode=%s",
                    settings.EXECUTION_AGENT_URL, symbol, self._mode)
        e_resp = await self._execution_client.send(
            agent="execution_agent",
            input_data={"analysis": analysis, "source": source, "mode": self._mode},
            task_id=f"e-{task_id}",
        )
        if e_resp.status != "completed":
            err = e_resp.error or "unknown error"
            logger.error("A2A ExecutionAgent failed  task_id=%s  error=%s", task_id, err)
            return {"analysis": analysis,
                    "reply": f"Execution failed: {err}"}
        execution = next((a.data for a in e_resp.artifacts if a.type == "execution"), {})
        logger.info("A2A ExecutionAgent done  action=%s", execution.get("action"))
        return {"analysis": analysis, "execution": execution,
                "reply": self._format_execution(execution)}

    async def _handle_research(
        self, search_query: str, user_message: str, history: list[dict] | None
    ) -> str:
        logger.info("► Tavily MCP research  query=%s", search_query[:80])
        raw = await call_mcp_tool("tavily", "search_market",
                                  {"query": search_query, "max_results": 5})
        data = json.loads(raw)
        headlines = [r.get("title", "") for r in data.get("results", []) if r.get("title")]
        answer    = data.get("answer", "")
        logger.info("► Research done  sources=%d  answer_len=%d", len(headlines), len(answer))

        web_ctx = ""
        if answer:
            web_ctx = f"Web search answer: {answer}\n\n"
        if headlines:
            web_ctx += "Recent sources:\n" + "\n".join(f"• {h}" for h in headlines[:5])

        system = (
            "You are a knowledgeable trading and finance assistant covering Indian equities "
            "(NSE/BSE) and global crypto markets. Answer concisely and accurately.\n"
            "Use the web search results below as your primary source. "
            "Mention if data may be outdated."
            f"\n\n--- WEB SEARCH RESULTS ---\n{web_ctx}\n---"
        )
        msgs = _build_chat_messages(system, history or [], user_message)
        resp = await self.llm.ainvoke(msgs)
        return resp.content

    async def _handle_chat(self, user_message: str, history: list[dict] | None) -> str:
        logger.info("► Chat response")
        system = (
            "You are a helpful trading assistant covering Indian equities (NSE/BSE) "
            "and global crypto (Delta Exchange). Answer concisely."
        )
        msgs = _build_chat_messages(system, history or [], user_message)
        resp = await self.llm.ainvoke(msgs)
        return resp.content

    # ── Formatters ────────────────────────────────────────────────────────────

    def _format_analysis(self, a: dict) -> str:
        if not a:
            return "Could not generate analysis."
        sym   = a.get("symbol", "?")
        trend = a.get("trend", "?").upper()
        conf  = float(a.get("confidence", 0))
        summ  = a.get("summary", "")
        entry = a.get("entry_zone", {})
        sl    = a.get("stop_loss")
        tgts  = a.get("targets", [])
        rr    = a.get("rr_ratio", 0)
        news  = a.get("news_context", "")

        lines = [
            f"**{sym} — {a.get('trade_type','').upper()} Analysis**",
            f"Trend: **{trend}** | Confidence: **{conf:.0%}**",
            "",
            f"📊 {summ}" if summ else "",
            "",
            f"**Entry zone:** {entry.get('low','?')} – {entry.get('high','?')}" if entry else "",
            f"**Stop loss:** {sl}" if sl else "",
            f"**Targets:** {', '.join(str(t) for t in tgts)}" if tgts else "",
            f"**R:R:** {rr:.1f}:1" if rr else "",
            "",
            f"📰 *{news}*" if news and news != "No recent news" else "",
        ]
        return "\n".join(ln for ln in lines if ln is not None)

    def _format_execution(self, e: dict) -> str:
        if not e:
            return "Could not generate execution plan."
        action = str(e.get("action", "hold")).upper()
        sym    = e.get("symbol", "?")
        mode   = e.get("mode", "paper")
        order  = e.get("order_result", {})
        status = order.get("status", "?")

        lines = [f"**{sym} — Execution Plan ({mode.upper()})**", f"Action: **{action}**", ""]
        if action in ("BUY", "SELL"):
            lines += [
                f"Entry: **{e.get('entry')}**",
                f"Stop Loss: **{e.get('stop_loss')}**",
                f"Take Profit: **{e.get('take_profit')}**",
                f"Qty: **{e.get('qty', 1)}**  |  R:R: **{e.get('rr_ratio', 0):.1f}:1**",
                f"Risk: ₹{e.get('risk_amount', 0):,.0f}", "",
            ]
            if mode == "paper":
                lines.append(f"✅ *Paper trade (ID: {order.get('order_id','?')})*")
            elif status == "placed":
                lines.append(f"✅ *Live order placed (ID: {order.get('order_id','?')})*")
            else:
                lines.append(f"⚠️ Order status: {status}")
        else:
            lines.append(f"⏸ Holding — {e.get('rationale', 'No trade recommended')}")

        return "\n".join(ln for ln in lines if ln is not None)
