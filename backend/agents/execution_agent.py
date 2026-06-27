"""Execution Agent — builds an execution plan and places orders via MCP.

Fully decoupled:
  - LLM provider injected via task.input["llm_provider"]
  - Mode (paper/real) injected via task.input["mode"]
  - No direct import of settings / config
"""

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.mcp.connector import call_mcp_tool
from backend.protocol import Artifact, Task, TaskResponse

logger = logging.getLogger(__name__)

_SYSTEM = """You are a professional trade execution specialist focused on minimum risk and maximum profit.

Given a trade analysis, produce a precise execution plan with concrete order parameters.

Rules:
1. Risk per trade <= 1% of account (assume 100,000 INR / $1,000 account)
2. Always use bracket orders (entry + stop-loss + take-profit)
3. R:R must be >= 1.5:1
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


def _get_llm(provider: str):
    """Lazy-import to keep execution_agent decoupled at module level."""
    from backend.llm.factory import LLMFactory
    return LLMFactory.get_llm(provider or "openai")


def _parse_plan(raw: str, symbol: str) -> dict[str, Any]:
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {"action": "hold", "rationale": "Could not parse plan", "symbol": symbol}


class ExecutionAgent:
    """Plans and executes trades.

    LLM provider and trade mode are resolved from task.input at call time,
    keeping this module independent of settings/config.
    """

    agent_name = "execution_agent"

    async def handle_task(self, task: Task) -> TaskResponse:
        inp          = task.input or {}
        analysis     = inp.get("analysis", {})
        source       = inp.get("source", "delta")
        mode         = inp.get("mode", "paper")
        llm_provider = inp.get("llm_provider", "openai")
        symbol       = analysis.get("symbol", inp.get("symbol", "BTCUSDT"))
        confidence   = float(analysis.get("confidence", 0))

        logger.info("execution_agent START  symbol=%s  conf=%.2f  mode=%s  llm=%s",
                    symbol, confidence, mode, llm_provider)

        plan         = await self._build_plan(analysis, source, mode, llm_provider)
        plan["mode"] = mode
        plan["symbol"] = symbol

        # Override LLM-hallucinated qty with a real risk-based calculation
        plan = self._apply_position_sizing(plan, analysis)

        action = str(plan.get("action", "hold")).lower()

        logger.info("execution_agent: action=%s  entry=%s  sl=%s  tp=%s  qty=%s",
                    action, plan.get("entry"), plan.get("stop_loss"),
                    plan.get("take_profit"), plan.get("qty"))

        order_result = await self._execute(plan, symbol, source, action, mode, task.task_id)
        result       = {**plan, "order_result": order_result, "analysis_used": analysis}

        return TaskResponse(
            task_id=task.task_id, agent=self.agent_name, status="completed",
            artifacts=[Artifact(type="execution", data=result)],
        )

    def _apply_position_sizing(self, plan: dict, analysis: dict) -> dict:
        """Replace LLM-hallucinated qty with a deterministic risk-based calculation.

        Uses 1% risk on a 100,000 INR / $1,000 account.
        qty = floor(risk_amount / abs(entry - stop_loss))
        Minimum qty is always 1.
        """
        try:
            entry = float(plan.get("entry") or analysis.get("entry_zone", {}).get("low") or 0)
            sl    = float(plan.get("stop_loss") or analysis.get("stop_loss") or 0)
            if entry > 0 and sl > 0 and entry != sl:
                account_size = 100_000.0
                risk_pct     = 0.01
                risk_amount  = account_size * risk_pct          # 1,000
                risk_per_unit = abs(entry - sl)
                qty = max(1, int(risk_amount / risk_per_unit))
                plan["qty"]         = qty
                plan["risk_amount"] = round(risk_per_unit * qty, 2)
                logger.info("position_sizing: entry=%.4f  sl=%.4f  risk/unit=%.4f  qty=%d",
                            entry, sl, risk_per_unit, qty)
        except Exception as exc:
            logger.warning("position_sizing failed, keeping LLM qty: %s", exc)
        return plan

    async def _build_plan(
        self, analysis: dict, source: str, mode: str, llm_provider: str
    ) -> dict[str, Any]:
        """Call LLM to produce an execution plan from the analysis."""
        symbol = analysis.get("symbol", "BTCUSDT")
        prompt = self._build_prompt(symbol, analysis, source, mode)
        logger.info("► Calling LLM for execution plan  provider=%s", llm_provider)
        llm      = _get_llm(llm_provider)
        response = await llm.ainvoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return _parse_plan(response.content.strip(), symbol)

    def _build_prompt(self, symbol: str, analysis: dict, source: str, mode: str) -> str:
        confidence = float(analysis.get("confidence", 0))
        trend      = analysis.get("trend", "sideways")
        return (
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

    async def _execute(
        self, plan: dict, symbol: str, source: str, action: str, mode: str, task_id: str
    ) -> dict:
        if action not in ("buy", "sell"):
            return {"status": "hold", "reason": plan.get("rationale", "No trade")}

        if mode == "paper":
            logger.info("► Paper trade simulated  symbol=%s  side=%s", symbol, action)
            return {
                "status": "paper_filled",
                "order_id": f"PAPER-{task_id}",
                "symbol": symbol, "side": action,
                "entry": plan.get("entry"),
                "stop_loss": plan.get("stop_loss"),
                "take_profit": plan.get("take_profit"),
                "qty": plan.get("qty", 1),
            }

        # Real order via MCP
        entry       = float(plan.get("entry", 0))
        stop_loss   = float(plan.get("stop_loss", 0))
        take_profit = float(plan.get("take_profit", 0))
        qty         = int(plan.get("qty", 1))

        if not (entry and stop_loss and take_profit):
            logger.warning("execution_agent: incomplete levels — order skipped")
            return {"status": "skipped", "reason": "Incomplete entry/SL/TP levels"}

        logger.info("► Placing REAL order via %s MCP  symbol=%s  side=%s  qty=%d",
                    source, symbol, action, qty)
        order_json = await call_mcp_tool(source, "place_order", {
            "symbol": symbol, "side": action, "qty": qty,
            "entry": entry, "stop_loss": stop_loss, "take_profit": take_profit,
        })
        result = json.loads(order_json)
        logger.info("► Order result: %s", result)
        return result
