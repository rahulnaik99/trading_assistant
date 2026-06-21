"""Main Orchestrator Agent — routes user queries to Analysis or Execution agents."""

import json
import logging
import re
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.analysis_agent import AnalysisAgent
from backend.agents.execution_agent import ExecutionAgent
from backend.llm.factory import LLMFactory, traceable
from backend.protocol import Task, TaskResponse

logger = logging.getLogger(__name__)

_ROUTER_SYSTEM = """You are a trading assistant dispatcher. Classify the user's intent into one of:

1. "analyse"  — user wants market analysis, price outlook, trade recommendation
2. "execute"  — user wants to place/execute a trade
3. "chat"     — general question, greeting, or out-of-scope

Extract:
- symbol: e.g. "NSE:RELIANCE-EQ", "BTCUSDT", "NIFTY50" — normalise to exchange format
- source: "fyers" for Indian equities/indices, "delta" for crypto
- trade_type: "intraday", "swing", or "longterm"
- intent: "analyse", "execute", or "chat"

Respond ONLY with JSON:
{"intent": string, "symbol": string, "source": string, "trade_type": string}"""

# Symbol normalisation helpers
_FYERS_MAP = {
    "reliance": "NSE:RELIANCE-EQ", "tcs": "NSE:TCS-EQ",
    "nifty": "NSE:NIFTY50-INDEX", "banknifty": "NSE:NIFTYBANK-INDEX",
    "sbin": "NSE:SBIN-EQ", "hdfcbank": "NSE:HDFCBANK-EQ",
    "infy": "NSE:INFY-EQ", "wipro": "NSE:WIPRO-EQ",
    "icicibank": "NSE:ICICIBANK-EQ",
}
_DELTA_MAP = {
    "btc": "BTCUSDT", "bitcoin": "BTCUSDT",
    "eth": "ETHUSDT", "ethereum": "ETHUSDT",
    "gold": "XAUUSDT", "xau": "XAUUSDT",
    "sol": "SOLUSDT", "bnb": "BNBUSDT",
}


def _normalise(raw: str) -> tuple[str, str]:
    """Return (normalised_symbol, source)."""
    s = raw.lower().strip()
    for k, v in _FYERS_MAP.items():
        if k in s:
            return v, "fyers"
    for k, v in _DELTA_MAP.items():
        if k in s:
            return v, "delta"
    # Assume crypto by default if looks like a ticker
    sym = raw.upper().replace(" ", "")
    if not sym.endswith("USDT") and not sym.startswith("NSE:"):
        sym += "USDT"
    return sym, "delta"


class OrchestratorAgent:
    """Entry point: parses user message, routes to sub-agents, returns human reply."""

    def __init__(self, llm_provider: str = "openai", mode: str = "paper") -> None:
        self._llm_provider = llm_provider
        self._mode = mode
        self._analysis_agent = AnalysisAgent(llm_provider)
        self._execution_agent = ExecutionAgent(llm_provider, mode)
        self._llm = None
        logger.info("OrchestratorAgent: init  provider=%s  mode=%s", llm_provider, mode)

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLMFactory.get_llm(self._llm_provider)
        return self._llm

    @traceable(name="orchestrator.handle_message", run_type="chain")
    async def handle_message(
        self,
        user_message: str,
        history: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Process a user message and return {reply, intent, analysis, execution, tokens}.

        history: list of {role, content} dicts for conversation context.
        """
        task_id = str(uuid.uuid4())[:8]
        logger.info("orchestrator: message  id=%s  msg=%s", task_id, user_message[:80])

        # ── 1. Route intent ───────────────────────────────────────────────────
        logger.info("► Routing user intent")
        route_resp = await self.llm.ainvoke([
            SystemMessage(content=_ROUTER_SYSTEM),
            HumanMessage(content=user_message),
        ])
        route: dict = {}
        try:
            cleaned = re.sub(r"```json|```", "", route_resp.content).strip()
            route = json.loads(cleaned)
        except Exception:
            route = {"intent": "chat", "symbol": "BTCUSDT", "source": "delta", "trade_type": "intraday"}

        intent     = route.get("intent", "chat")
        symbol_raw = route.get("symbol", "BTCUSDT")
        source     = route.get("source", "delta")
        trade_type = route.get("trade_type", "intraday")

        # Normalise symbol
        symbol, source_inferred = _normalise(symbol_raw)
        if source == "delta" or source == "fyers":
            pass  # trust LLM routing
        else:
            source = source_inferred

        logger.info("orchestrator: route  intent=%s  symbol=%s  source=%s  type=%s",
                    intent, symbol, source, trade_type)

        result: dict[str, Any] = {
            "intent": intent,
            "symbol": symbol,
            "source": source,
            "trade_type": trade_type,
            "analysis": None,
            "execution": None,
            "reply": "",
        }

        # ── 2. Dispatch to sub-agents ─────────────────────────────────────────
        if intent == "analyse":
            logger.info("► Dispatching to AnalysisAgent via A2A")
            a_resp = await self._analysis_agent.handle_task(Task(
                task_id=f"a-{task_id}", agent="analysis_agent",
                input={"symbol": symbol, "source": source, "trade_type": trade_type},
            ))
            if a_resp.status == "completed":
                analysis = next((art.data for art in a_resp.artifacts if art.type == "analysis"), {})
                result["analysis"] = analysis
                result["reply"] = self._format_analysis(analysis)
            else:
                result["reply"] = f"Analysis failed: {a_resp.error or 'unknown error'}"

        elif intent == "execute":
            # First get fresh analysis
            logger.info("► Getting fresh analysis for execution")
            a_resp = await self._analysis_agent.handle_task(Task(
                task_id=f"a-{task_id}", agent="analysis_agent",
                input={"symbol": symbol, "source": source, "trade_type": trade_type},
            ))
            analysis = {}
            if a_resp.status == "completed":
                analysis = next((art.data for art in a_resp.artifacts if art.type == "analysis"), {})
                result["analysis"] = analysis

            # Then execute
            logger.info("► Dispatching to ExecutionAgent via A2A")
            e_resp = await self._execution_agent.handle_task(Task(
                task_id=f"e-{task_id}", agent="execution_agent",
                input={"analysis": analysis, "source": source, "mode": self._mode},
            ))
            if e_resp.status == "completed":
                execution = next((art.data for art in e_resp.artifacts if art.type == "execution"), {})
                result["execution"] = execution
                result["reply"] = self._format_execution(execution)
            else:
                result["reply"] = f"Execution failed: {e_resp.error or 'unknown error'}"

        else:
            # General chat — use LLM with trading context
            logger.info("► Chat response")
            msgs = [SystemMessage(content=(
                "You are a helpful trading assistant for Indian equities (NSE) and crypto (Delta Exchange). "
                "Answer concisely. If the user wants analysis or trade execution, tell them to ask specifically."
            ))]
            if history:
                from langchain_core.messages import AIMessage
                for h in history[-6:]:  # last 6 turns
                    if h["role"] == "user":
                        msgs.append(HumanMessage(content=h["content"]))
                    else:
                        msgs.append(AIMessage(content=h["content"]))
            msgs.append(HumanMessage(content=user_message))
            chat_resp = await self.llm.ainvoke(msgs)
            result["reply"] = chat_resp.content

        return result

    # ── Formatters ────────────────────────────────────────────────────────────

    def _format_analysis(self, a: dict) -> str:
        if not a:
            return "Could not generate analysis."
        trend  = a.get("trend", "?").upper()
        sym    = a.get("symbol", "?")
        conf   = float(a.get("confidence", 0))
        summ   = a.get("summary", "")
        entry  = a.get("entry_zone", {})
        sl     = a.get("stop_loss")
        tgts   = a.get("targets", [])
        rr     = a.get("rr_ratio", 0)
        news   = a.get("news_context", "")

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
        return "\n".join(l for l in lines if l is not None)

    def _format_execution(self, e: dict) -> str:
        if not e:
            return "Could not generate execution plan."
        action = str(e.get("action", "hold")).upper()
        sym    = e.get("symbol", "?")
        mode   = e.get("mode", "paper")
        order  = e.get("order_result", {})
        status = order.get("status", "?")

        lines = [
            f"**{sym} — Execution Plan ({mode.upper()})**",
            f"Action: **{action}**",
            "",
        ]
        if action in ("BUY", "SELL"):
            lines += [
                f"Entry: **{e.get('entry')}**",
                f"Stop Loss: **{e.get('stop_loss')}**",
                f"Take Profit: **{e.get('take_profit')}**",
                f"Qty: **{e.get('qty', 1)}**  |  R:R: **{e.get('rr_ratio', 0):.1f}:1**",
                f"Risk: ₹{e.get('risk_amount', 0):,.0f}",
                "",
            ]
            if mode == "paper":
                lines.append(f"✅ *Paper trade simulated (Order ID: {order.get('order_id','?')})*")
            elif status == "placed":
                lines.append(f"✅ *Live order placed (ID: {order.get('order_id','?')})*")
            else:
                lines.append(f"⚠️ Order status: {status}")
        else:
            lines.append(f"⏸ Holding — {e.get('rationale', 'No trade recommended')}")

        return "\n".join(l for l in lines if l is not None)
