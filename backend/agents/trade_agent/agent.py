"""Trade Agent — executes orders based on a trade_recommendation artifact."""

from __future__ import annotations

import logging
from typing import Any

from backend.a2a.protocol import Artifact, Task, TaskResponse
from backend.a2a.server import BaseA2AAgent

logger = logging.getLogger(__name__)


class TradeAgent(BaseA2AAgent):
    """Receives a recommendation and executes a bracket order via the correct broker.

    Task input expected keys:
        recommendation  : dict   — trade_recommendation artifact data
        broker          : str    — "fyers" | "delta"
        mode            : str    — "paper" | "real"
        symbol          : str    — instrument symbol
        qty             : int    — quantity / contract size
        confidence_min  : float  — skip trade if confidence < this (default 0.6)
    """

    agent_name = "trade_agent"

    async def handle_task(self, task: Task) -> TaskResponse:
        inp = task.input or {}
        rec: dict[str, Any] = inp.get("recommendation", {})
        broker: str = inp.get("broker", "fyers")
        mode: str = inp.get("mode", "paper")
        symbol: str = inp.get("symbol", "")
        qty: int = int(inp.get("qty", 1))
        confidence_min: float = float(inp.get("confidence_min", 0.6))

        # ── 1. Validate recommendation fields ────────────────────────────────
        action = str(rec.get("recommendation", "")).upper()
        direction = str(rec.get("direction", "")).lower()
        confidence = rec.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        best_entry = rec.get("best_entry") or {}
        stop_loss_rec = rec.get("stop_loss") or {}
        target_rec = rec.get("target") or {}

        entry = best_entry.get("level") if isinstance(best_entry, dict) else None
        sl = stop_loss_rec.get("level") if isinstance(stop_loss_rec, dict) else None
        target = target_rec.get("primary") if isinstance(target_rec, dict) else None

        # ── 2. Guards ─────────────────────────────────────────────────────────
        if "AVOID" in action or ("WAIT" in action and "TAKE_TRADE" not in action):
            return self._skip(task, f"LLM says {action} — no trade")

        if direction not in ("long", "short"):
            return self._skip(task, f"direction={direction!r} is not actionable")

        if confidence < confidence_min:
            return self._skip(
                task,
                f"confidence {confidence:.2f} < threshold {confidence_min:.2f}",
            )

        if entry is None or sl is None or target is None:
            return self._skip(task, "incomplete levels: entry/sl/target missing from recommendation")

        entry, sl, target = float(entry), float(sl), float(target)
        side = "buy" if direction == "long" else "sell"

        # ── 3. Paper mode ─────────────────────────────────────────────────────
        if mode == "paper":
            result = {
                "mode": "paper",
                "broker": broker,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "entry": entry,
                "sl": sl,
                "target": target,
                "order_id": f"PAPER-{task.task_id}",
                "status": "simulated",
                "confidence": confidence,
            }
            logger.info(
                "TradeAgent: PAPER trade simulated  symbol=%s side=%s entry=%.2f sl=%.2f target=%.2f",
                symbol, side, entry, sl, target,
            )
            return self._ok(task, result)

        # ── 4. Real mode ──────────────────────────────────────────────────────
        try:
            if broker == "fyers":
                result = await self._execute_fyers(inp, symbol, side, qty, entry, sl, target)
            elif broker == "delta":
                result = await self._execute_delta(inp, symbol, side, qty, entry, sl, target)
            else:
                return self._fail(task, f"Unknown broker: {broker!r}")
        except Exception as exc:
            logger.exception("TradeAgent: order placement failed")
            return self._fail(task, f"Order placement failed: {exc}")

        result.update({"mode": "real", "broker": broker, "confidence": confidence})
        return self._ok(task, result)

    # ── Broker dispatchers ────────────────────────────────────────────────────

    async def _execute_fyers(
        self, inp: dict, symbol: str, side: str, qty: int,
        entry: float, sl: float, target: float,
    ) -> dict[str, Any]:
        from backend.config import settings
        from backend.agents.trade_agent.fyers_executor import FyersExecutor

        client_id    = inp.get("client_id")    or settings.FYERS_CLIENT_ID
        access_token = inp.get("access_token") or settings.FYERS_ACCESS_TOKEN
        ex = FyersExecutor(client_id=client_id, access_token=access_token)

        try:
            resp = await ex.place_bracket_order(
                symbol=symbol, side=side, qty=qty,
                entry=entry, stop_loss=sl, target=target,
            )
        finally:
            await ex.close()

        order_id = resp.get("id") or resp.get("data", {}).get("id", "")
        return {
            "symbol": symbol, "side": side, "qty": qty,
            "entry": entry, "sl": sl, "target": target,
            "order_id": str(order_id),
            "status": "placed" if resp.get("s") == "ok" else "failed",
            "raw_response": resp,
        }

    async def _execute_delta(
        self, inp: dict, symbol: str, side: str, qty: int,
        entry: float, sl: float, target: float,
    ) -> dict[str, Any]:
        from backend.config import settings
        from backend.brokers.delta.executor import DeltaExecutor

        api_key = inp.get("api_key") or settings.DELTA_API_KEY
        api_secret = inp.get("api_secret") or settings.DELTA_API_SECRET
        region = inp.get("region") or settings.DELTA_REGION
        ex = DeltaExecutor(api_key=api_key, api_secret=api_secret, region=region)

        try:
            resp = await ex.place_bracket_order(
                symbol=symbol, side=side, qty=qty,
                entry=entry, stop_loss=sl, target=target,
            )
            result_data = resp.get("result", {}) or {}
            order_id = result_data.get("id", "")
            success = bool(resp.get("success"))
            return {
                "symbol": symbol, "side": side, "qty": qty,
                "entry": entry, "sl": sl, "target": target,
                "order_id": str(order_id),
                "status": "placed" if success else "failed",
                "raw_response": resp,
            }
        finally:
            await ex.close()

    # ── Response helpers ──────────────────────────────────────────────────────

    def _ok(self, task: Task, data: dict) -> TaskResponse:
        return TaskResponse(
            task_id=task.task_id, agent=self.agent_name, status="completed",
            artifacts=[Artifact(type="trade_result", data=data)],
        )

    def _skip(self, task: Task, reason: str) -> TaskResponse:
        logger.info("TradeAgent: skip — %s", reason)
        return TaskResponse(
            task_id=task.task_id, agent=self.agent_name, status="completed",
            artifacts=[Artifact(type="trade_result", data={"status": "skipped", "reason": reason})],
        )

    def _fail(self, task: Task, error: str) -> TaskResponse:
        return TaskResponse(
            task_id=task.task_id, agent=self.agent_name, status="failed",
            error=error, artifacts=[],
        )
