"""Delta Exchange order executor for perpetual futures (BTCUSD, XAUUSD)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Symbols supported by this bot — mapped to Delta product symbols
SUPPORTED_SYMBOLS = {"BTCUSD", "BTCUSDT", "XAUUSD", "XAUUSDT"}


class DeltaExecutor:
    """Place and manage bracket orders on Delta Exchange perpetual futures.

    Reuses DeltaSource's authenticated _request() and _generate_signature()
    so we never duplicate the HMAC logic.
    """

    def __init__(self, api_key: str, api_secret: str, region: str = "global") -> None:
        # Reuse DeltaSource for HTTP + auth plumbing
        from backend.brokers.delta.source import DeltaSource
        self._src = DeltaSource(api_key=api_key, api_secret=api_secret, region=region)
        self._product_cache: dict[str, int] = {}  # symbol → product_id

    # ── Product ID resolution ─────────────────────────────────────────────────

    async def _get_product_id(self, symbol: str) -> int:
        """Resolve a symbol to Delta product_id (cached, paginated)."""
        symbol = symbol.upper().removesuffix(".P")
        if symbol in self._product_cache:
            return self._product_cache[symbol]

        # Filter by perpetual_futures — avoids paging through thousands of option contracts
        data = await self._src._request(
            "GET", "/products",
            params={"contract_types": "perpetual_futures", "page": 1, "page_size": 100}
        )
        products = data.get("result", [])
        for product in products:
            sym = (product.get("symbol") or "").upper()
            pid = product.get("id")
            if sym and pid:
                self._product_cache[sym] = int(pid)

        if symbol not in self._product_cache:
            raise ValueError(
                f"Symbol {symbol!r} not found in Delta products. "
                f"Check the symbol name at delta.exchange — e.g. BTCUSD, XAUUSDT. "
                f"Cached: {list(self._product_cache.keys())[:30]}"
            )
        return self._product_cache[symbol]

    # ── Order placement ───────────────────────────────────────────────────────

    async def place_bracket_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry: float,
        stop_loss: float,
        target: float,
        order_type: str = "market_order",
    ) -> dict[str, Any]:
        """Place a bracket order (entry + SL + TP) on Delta Exchange.

        Parameters
        ----------
        symbol     : str   Delta symbol, e.g. "BTCUSD", "XAUUSD"
        side       : str   "buy" or "sell"
        qty        : int   Contract size
        entry      : float Entry price (used for limit; ignored for market)
        stop_loss  : float Absolute SL price
        target     : float Absolute target price
        order_type : str   "market_order" | "limit_order"
        """
        symbol = symbol.upper().removesuffix(".P")
        product_id = await self._get_product_id(symbol)

        body: dict[str, Any] = {
            "product_id": product_id,
            "order_type": order_type,
            "side": side.lower(),
            "size": qty,
            "bracket_order": True,
            "bracket_stop_loss_price": str(round(stop_loss, 2)),
            "bracket_take_profit_price": str(round(target, 2)),
        }
        if order_type == "limit_order":
            body["limit_price"] = str(round(entry, 2))

        logger.info(
            "DeltaExecutor.place_bracket_order: symbol=%s side=%s qty=%d "
            "entry=%.2f sl=%.2f target=%.2f",
            symbol, side, qty, entry, stop_loss, target,
        )
        response = await self._src._request("POST", "/orders", body=body, authenticated=True)
        logger.info("DeltaExecutor: Delta response: %s", response)
        return response

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "market_order",
        limit_price: float = 0.0,
    ) -> dict[str, Any]:
        """Place a plain order (no bracket) — for manual/custom use."""
        symbol = symbol.upper().removesuffix(".P")
        product_id = await self._get_product_id(symbol)
        body: dict[str, Any] = {
            "product_id": product_id,
            "order_type": order_type,
            "side": side.lower(),
            "size": qty,
        }
        if order_type == "limit_order":
            body["limit_price"] = str(round(limit_price, 2))
        return await self._src._request("POST", "/orders", body=body, authenticated=True)

    # ── Position / order queries ───────────────────────────────────────────────

    async def get_positions(self) -> list[dict[str, Any]]:
        """Return list of open margined positions."""
        data = await self._src._request("GET", "/positions/margined", authenticated=True)
        return data.get("result", []) or []

    async def get_orders(self) -> list[dict[str, Any]]:
        """Return open orders."""
        data = await self._src._request("GET", "/orders", authenticated=True)
        return data.get("result", []) or []

    async def cancel_order(self, order_id: int | str) -> dict[str, Any]:
        """Cancel an order by ID."""
        resp = await self._src._request(
            "DELETE", f"/orders/{order_id}", authenticated=True
        )
        logger.info("DeltaExecutor.cancel_order: id=%s response=%s", order_id, resp)
        return resp

    async def get_ltp(self, symbol: str) -> float | None:
        """Fetch last traded price (mark price) for a symbol."""
        symbol = symbol.upper().removesuffix(".P")
        data = await self._src._request("GET", f"/tickers/{symbol}")
        result = data.get("result", {})
        try:
            return float(result.get("mark_price") or 0) or None
        except (TypeError, ValueError):
            return None

    async def close(self) -> None:
        """Close underlying HTTP session."""
        await self._src.close()
