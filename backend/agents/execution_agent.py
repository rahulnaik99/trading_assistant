"""Execution Agent — builds an execution plan and places orders via MCP."""

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.llm.factory import LLMFactory, traceable
from backend.mcp.connector import call_mcp_tool
from backend.protocol import Artifact, Task, TaskResponse

logger = logging.getLogger(__name__)

_SYSTEM = """You are a professional trade execution specialist focused on minimum risk and maximum profit.

Given a trade analysis, produce a precise execution plan with concrete order parameters.

Rules:
1. Risk per trade ≤ 1% of account (assume ₹100,000 / $1,000 account)
2. Always use bracket orders (entry + stop-loss + take-profit)
3. R:R must be ≥ 1.5:1
4. Use market order for entry unless analysis says otherwise
5. For intraday: exit before market close

Respond ONLY with valid JSON:
{
  "action": "buy|sell|hold",
  "symbol": string,
  "broker": "delta|fyers",
  "order_type": "market_order|limit_order",
  "entry": float,
  "stop_loss": float,
  "take_profit": float,
  "take_profit_2": float,
  "qty": int,
  "risk_amount": float,
  "rr_ratio": float,
  "rationale": string,
  "mode": "paper|real"
}"""


class ExecutionAgent:
    """Plans and executes trades using analysis from AnalysisAgent."""

    agent_name = "execution_agent"

    def __init__(self, llm_provider: str = "", mode: str = "paper") -> None:
        self._llm_provider = llm_provider
        self._mode = mode
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLMFactory.get_llm(self._llm_provider)
        return self._llm

    @traceable(name="execution_agent.handle_task", run_type="chain")
    async def handle_task(self, task: Task) -> TaskResponse:
        inp = task.input or {}
        analysis: dict = inp.get("analysis", {})
        source: str    = inp.get("source", "delta")
        mode: str      = inp.get("mode", self._mode)

        symbol    = analysis.get("symbol", inp.get("symbol", "BTCUSDT"))
        trend     = analysis.get("trend", "sideways")
        confidence= float(analysis.get("confidence", 0))

        logger.info("execution_agent START  symbol=%s  trend=%s  confidence=%.2f  mode=%s",
                    symbol, trend, confidence, mode)

        # ── 1. Build execution plan via LLM ──────────────────────────────────
        prompt = (
            f"Symbol: {symbol}\n"
            f"Trend: {trend.upper()}  Confidence: {confidence:.0%}\n"
            f"Trade type: {analysis.get('trade_type','intraday')}\n"
            f"Last price: {analysis.get('last_price', 'N/A')}\n"
            f"Entry zone: {analysis.get('entry_zone', {})}\n"
            f"Stop loss: {analysis.get('stop_loss', 'N/A')}\n"
            f"Targets: {analysis.get('targets', [])}\n"
            f"Key levels: {analysis.get('key_levels', {})}\n"
            f"News: {analysis.get('news_context', '')}\n"
            f"Broker: {source}\n"
            f"Mode: {mode}\n\n"
            f"Produce a precise execution plan. Action must be 'hold' if confidence < 0.55 "
            f"or trend is sideways."
        )

        logger.info("► Calling LLM for execution plan  symbol=%s", symbol)
        response = await self.llm.ainvoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()

        plan: dict[str, Any] = {}
        try:
            cleaned = re.sub(r"```json|```", "", raw).strip()
            plan = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    plan = json.loads(m.group())
                except Exception:
                    pass
            if not plan:
                plan = {"action": "hold", "rationale": "Could not parse plan", "symbol": symbol}

        plan["mode"] = mode
        plan["symbol"] = symbol

        action = str(plan.get("action", "hold")).lower()
        logger.info("execution_agent: plan  action=%s  entry=%s  sl=%s  tp=%s",
                    action, plan.get("entry"), plan.get("stop_loss"), plan.get("take_profit"))

        # ── 2. Execute via MCP if action is buy/sell and mode is real ─────────
        order_result: dict = {}
        if action in ("buy", "sell") and mode == "real":
            entry      = float(plan.get("entry", 0))
            stop_loss  = float(plan.get("stop_loss", 0))
            take_profit= float(plan.get("take_profit", 0))
            qty        = int(plan.get("qty", 1))

            if entry and stop_loss and take_profit:
                logger.info("► Placing REAL order via %s MCP  symbol=%s  side=%s  qty=%d",
                            source, symbol, action, qty)
                order_json = await call_mcp_tool(
                    server=source,
                    tool="place_order",
                    arguments={
                        "symbol": symbol, "side": action, "qty": qty,
                        "entry": entry, "stop_loss": stop_loss, "take_profit": take_profit,
                    },
                )
                order_result = json.loads(order_json)
                logger.info("► Order result: %s", order_result)
            else:
                logger.warning("execution_agent: incomplete levels — order skipped")
                order_result = {"status": "skipped", "reason": "Incomplete entry/SL/TP levels"}

        elif action in ("buy", "sell") and mode == "paper":
            order_result = {
                "status": "paper_filled",
                "order_id": f"PAPER-{task.task_id}",
                "symbol": symbol, "side": action,
                "entry": plan.get("entry"),
                "stop_loss": plan.get("stop_loss"),
                "take_profit": plan.get("take_profit"),
                "qty": plan.get("qty", 1),
            }
            logger.info("► Paper trade simulated  symbol=%s  side=%s", symbol, action)

        else:
            order_result = {"status": "hold", "reason": plan.get("rationale", "No trade")}

        result = {**plan, "order_result": order_result, "analysis_used": analysis}

        return TaskResponse(
            task_id=task.task_id, agent=self.agent_name, status="completed",
            artifacts=[Artifact(type="execution", data=result)],
        )
