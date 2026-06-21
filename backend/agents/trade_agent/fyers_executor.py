"""Fyers Bracket Order executor for NSE equity trading — async interface."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class FyersExecutor:
    """Place and manage bracket orders on Fyers for NSE equity.

    All public methods are async to match DeltaExecutor's interface.
    The underlying fyers_apiv3 SDK calls are synchronous, so they run in
    a thread pool executor to avoid blocking the event loop.
    """

    def __init__(self, client_id: str, access_token: str) -> None:
        self.client_id    = client_id
        self.access_token = access_token
        self._fyers       = None

    def _client(self):
        """Lazy-init the Fyers SDK client (sync — called inside thread)."""
        if self._fyers is None:
            from fyers_apiv3 import fyersModel  # type: ignore[import]
            token = (
                self.access_token
                if ":" in self.access_token
                else f"{self.client_id}:{self.access_token}"
            )
            self._fyers = fyersModel.FyersModel(
                client_id=self.client_id,
                token=token,
                log_path="",
            )
        return self._fyers

    async def _run(self, fn, *args, **kwargs):
        """Run a sync SDK call in a thread pool to avoid blocking the loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    # ── Order placement ───────────────────────────────────────────────────────

    async def place_bracket_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry: float,
        stop_loss: float,
        target: float,
        order_type: int = 2,
    ) -> dict[str, Any]:
        """Place a Bracket Order with SL and target (async)."""
        sl_points  = round(abs(entry - stop_loss), 2)
        tp_points  = round(abs(target - entry), 2)
        fyers_side = 1 if side.lower() == "buy" else -1

        data = {
            "symbol":       symbol,
            "qty":          qty,
            "type":         order_type,
            "side":         fyers_side,
            "productType":  "BO",
            "limitPrice":   round(entry, 2) if order_type == 1 else 0,
            "stopPrice":    0,
            "validity":     "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss":     sl_points,
            "takeProfit":   tp_points,
        }

        logger.info(
            "FyersExecutor.place_bracket_order: symbol=%s side=%s qty=%d "
            "entry=%.2f sl_pts=%.2f tp_pts=%.2f",
            symbol, side, qty, entry, sl_points, tp_points,
        )
        response: dict = await self._run(self._client().place_order, data=data)
        logger.info("FyersExecutor: response=%s", response)
        return response

    async def place_order(self, data: dict[str, Any]) -> dict[str, Any]:
        """Raw SDK place_order wrapper (async)."""
        return await self._run(self._client().place_order, data=data)

    # ── Position / order queries ───────────────────────────────────────────────

    async def get_positions(self) -> list[dict[str, Any]]:
        """Return list of open positions from Fyers (async)."""
        resp = await self._run(self._client().positions)
        if resp.get("s") == "ok":
            return resp.get("netPositions", []) or []
        logger.warning("FyersExecutor.get_positions: unexpected response: %s", resp)
        return []

    async def get_orders(self) -> list[dict[str, Any]]:
        """Return today's orders (async)."""
        resp = await self._run(self._client().orderbook)
        if resp.get("s") == "ok":
            return resp.get("orderBook", []) or []
        return []

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an order by ID (async)."""
        resp = await self._run(self._client().cancel_order, data={"id": order_id})
        logger.info("FyersExecutor.cancel_order: id=%s response=%s", order_id, resp)
        return resp

    async def get_ltp(self, symbol: str) -> float | None:
        """Fetch last traded price for a symbol (async)."""
        resp = await self._run(self._client().quotes, data={"symbols": symbol})
        if resp.get("s") == "ok":
            quotes = resp.get("d", [])
            if quotes:
                return float(quotes[0].get("v", {}).get("lp", 0)) or None
        return None

    async def close(self) -> None:
        """No-op for API symmetry with DeltaExecutor."""
