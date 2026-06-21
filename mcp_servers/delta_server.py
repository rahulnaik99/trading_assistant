"""Delta Exchange MCP Server.

Exposes all Delta Exchange operations as MCP tools:

Tools:
  fetch_candles          — OHLCV candle data for any symbol/interval
  fetch_perpetual_metrics — Funding rate, OI, mark price for perpetuals
  get_mark_price         — Current mark price (LTP) for a symbol
  place_order            — Place bracket order (authenticated)
  place_market_order     — Place plain market order (authenticated)
  get_positions          — List open positions (authenticated)
  get_orders             — List open orders (authenticated)
  cancel_order           — Cancel an order by ID (authenticated)

Usage:
    python -m mcp_servers.delta_server

Connect via MCPClient:
    from backend.mcp.client import MCPClient
    client = MCPClient(command="python", args=["-m", "mcp_servers.delta_server"])
    await client.connect()
    result = await client.call_tool("fetch_candles", {
        "symbol": "BTCUSDT", "interval": "5m", "lookback_days": 3
    })
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# ── MCP server bootstrap ───────────────────────────────────────────────────────

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types as mcp_types
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

# ── Tool implementations ───────────────────────────────────────────────────────

async def _get_source():
    """Lazy-init DeltaSource with credentials from env / config."""
    from backend.brokers.delta.source import DeltaSource
    from backend.config import settings
    return DeltaSource(
        api_key=os.getenv("DELTA_API_KEY", settings.DELTA_API_KEY),
        api_secret=os.getenv("DELTA_API_SECRET", settings.DELTA_API_SECRET),
        region=os.getenv("DELTA_REGION", settings.DELTA_REGION),
    )


async def tool_fetch_candles(arguments: dict) -> str:
    """Fetch OHLCV candle data from Delta Exchange."""
    symbol       = arguments.get("symbol", "BTCUSDT")
    interval     = arguments.get("interval", "5m")
    lookback_days = int(arguments.get("lookback_days", 3))

    now     = datetime.now(timezone.utc)
    from_dt = now - timedelta(days=lookback_days)

    src = await _get_source()
    try:
        df = await src.fetch_ohlc(symbol=symbol, from_dt=from_dt, to_dt=now, interval=interval)
        if df.empty:
            return json.dumps({"error": f"No data for {symbol} {interval}", "rows": 0})
        rows = df.reset_index().tail(200).to_dict(orient="records")
        for r in rows:
            r["datetime"] = str(r["datetime"])
        return json.dumps({
            "symbol": symbol, "interval": interval,
            "rows": len(rows),
            "from": str(df.index[0]), "to": str(df.index[-1]),
            "last_close": float(df["close"].iloc[-1]),
            "candles": rows,
        })
    finally:
        await src.close()


async def tool_fetch_perpetual_metrics(arguments: dict) -> str:
    """Fetch funding rate, OI, mark price for a Delta perpetual."""
    symbol = arguments.get("symbol", "BTCUSDT")
    src = await _get_source()
    try:
        metrics = await src.fetch_perpetual_metrics(symbol)
        return json.dumps(metrics)
    finally:
        await src.close()


async def tool_get_mark_price(arguments: dict) -> str:
    """Get current mark price (LTP) for a Delta symbol."""
    symbol = arguments.get("symbol", "BTCUSDT")
    src = await _get_source()
    try:
        data = await src._request("GET", f"/tickers/{symbol.upper().removesuffix('.P')}")
        result = data.get("result", {}) or {}
        mark = result.get("mark_price")
        spot = result.get("spot_price")
        return json.dumps({
            "symbol": symbol,
            "mark_price": float(mark) if mark else None,
            "spot_price": float(spot) if spot else None,
        })
    finally:
        await src.close()


async def tool_place_order(arguments: dict) -> str:
    """Place a bracket order on Delta Exchange (authenticated)."""
    from backend.brokers.delta.executor import DeltaExecutor
    from backend.config import settings

    symbol     = arguments["symbol"]
    side       = arguments["side"]              # "buy" | "sell"
    qty        = int(arguments["qty"])
    entry      = float(arguments["entry"])
    stop_loss  = float(arguments["stop_loss"])
    take_profit= float(arguments["take_profit"])
    order_type = arguments.get("order_type", "market_order")

    ex = DeltaExecutor(
        api_key=os.getenv("DELTA_API_KEY", settings.DELTA_API_KEY),
        api_secret=os.getenv("DELTA_API_SECRET", settings.DELTA_API_SECRET),
        region=os.getenv("DELTA_REGION", settings.DELTA_REGION),
    )
    try:
        result = await ex.place_bracket_order(
            symbol=symbol, side=side, qty=qty,
            entry=entry, stop_loss=stop_loss, target=take_profit,
            order_type=order_type,
        )
        return json.dumps(result)
    finally:
        await ex.close()


async def tool_place_market_order(arguments: dict) -> str:
    """Place a plain market order on Delta Exchange (authenticated)."""
    from backend.brokers.delta.executor import DeltaExecutor
    from backend.config import settings

    symbol = arguments["symbol"]
    side   = arguments["side"]
    qty    = int(arguments["qty"])

    ex = DeltaExecutor(
        api_key=os.getenv("DELTA_API_KEY", settings.DELTA_API_KEY),
        api_secret=os.getenv("DELTA_API_SECRET", settings.DELTA_API_SECRET),
    )
    try:
        result = await ex.place_order(symbol=symbol, side=side, qty=qty)
        return json.dumps(result)
    finally:
        await ex.close()


async def tool_get_positions(arguments: dict) -> str:
    """Get open positions on Delta Exchange (authenticated)."""
    from backend.brokers.delta.executor import DeltaExecutor
    from backend.config import settings

    ex = DeltaExecutor(
        api_key=os.getenv("DELTA_API_KEY", settings.DELTA_API_KEY),
        api_secret=os.getenv("DELTA_API_SECRET", settings.DELTA_API_SECRET),
    )
    try:
        positions = await ex.get_positions()
        return json.dumps({"positions": positions, "count": len(positions)})
    finally:
        await ex.close()


async def tool_get_orders(arguments: dict) -> str:
    """Get open orders on Delta Exchange (authenticated)."""
    from backend.brokers.delta.executor import DeltaExecutor
    from backend.config import settings

    ex = DeltaExecutor(
        api_key=os.getenv("DELTA_API_KEY", settings.DELTA_API_KEY),
        api_secret=os.getenv("DELTA_API_SECRET", settings.DELTA_API_SECRET),
    )
    try:
        orders = await ex.get_orders()
        return json.dumps({"orders": orders, "count": len(orders)})
    finally:
        await ex.close()


async def tool_cancel_order(arguments: dict) -> str:
    """Cancel an open order on Delta Exchange (authenticated)."""
    from backend.brokers.delta.executor import DeltaExecutor
    from backend.config import settings

    order_id = arguments["order_id"]
    ex = DeltaExecutor(
        api_key=os.getenv("DELTA_API_KEY", settings.DELTA_API_KEY),
        api_secret=os.getenv("DELTA_API_SECRET", settings.DELTA_API_SECRET),
    )
    try:
        result = await ex.cancel_order(order_id)
        return json.dumps(result)
    finally:
        await ex.close()


# ── Tool registry ──────────────────────────────────────────────────────────────

TOOLS = [
    mcp_types.Tool(
        name="fetch_candles",
        description="Fetch OHLCV candle data from Delta Exchange for any symbol and interval.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol":        {"type": "string",  "description": "e.g. BTCUSDT, ETHUSDT, XAUUSDT"},
                "interval":      {"type": "string",  "description": "1m | 5m | 15m | 1h | 1D"},
                "lookback_days": {"type": "integer", "description": "Days of history (default 3)"},
            },
            "required": ["symbol"],
        },
    ),
    mcp_types.Tool(
        name="fetch_perpetual_metrics",
        description="Fetch funding rate, open interest, and mark price for a Delta perpetual futures symbol.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "e.g. BTCUSDT, ETHUSDT"},
            },
            "required": ["symbol"],
        },
    ),
    mcp_types.Tool(
        name="get_mark_price",
        description="Get the current mark price (LTP) and spot price for a Delta symbol.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "e.g. BTCUSDT"},
            },
            "required": ["symbol"],
        },
    ),
    mcp_types.Tool(
        name="place_order",
        description="Place a bracket order (entry + stop-loss + take-profit) on Delta Exchange. Requires valid API credentials.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol":      {"type": "string",  "description": "Delta symbol e.g. BTCUSDT"},
                "side":        {"type": "string",  "description": "buy | sell"},
                "qty":         {"type": "integer", "description": "Number of contracts"},
                "entry":       {"type": "number",  "description": "Entry price"},
                "stop_loss":   {"type": "number",  "description": "Stop-loss price"},
                "take_profit": {"type": "number",  "description": "Take-profit price"},
                "order_type":  {"type": "string",  "description": "market_order | limit_order (default market_order)"},
            },
            "required": ["symbol", "side", "qty", "entry", "stop_loss", "take_profit"],
        },
    ),
    mcp_types.Tool(
        name="place_market_order",
        description="Place a plain market order on Delta Exchange without bracket levels.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "side":   {"type": "string", "description": "buy | sell"},
                "qty":    {"type": "integer"},
            },
            "required": ["symbol", "side", "qty"],
        },
    ),
    mcp_types.Tool(
        name="get_positions",
        description="Get all open margined positions on Delta Exchange.",
        inputSchema={"type": "object", "properties": {}},
    ),
    mcp_types.Tool(
        name="get_orders",
        description="Get all open orders on Delta Exchange.",
        inputSchema={"type": "object", "properties": {}},
    ),
    mcp_types.Tool(
        name="cancel_order",
        description="Cancel an open order on Delta Exchange by its order ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID to cancel"},
            },
            "required": ["order_id"],
        },
    ),
]

TOOL_HANDLERS = {
    "fetch_candles":           tool_fetch_candles,
    "fetch_perpetual_metrics": tool_fetch_perpetual_metrics,
    "get_mark_price":          tool_get_mark_price,
    "place_order":             tool_place_order,
    "place_market_order":      tool_place_market_order,
    "get_positions":           tool_get_positions,
    "get_orders":              tool_get_orders,
    "cancel_order":            tool_cancel_order,
}


# ── MCP Server ─────────────────────────────────────────────────────────────────

async def main():
    if not _MCP_AVAILABLE:
        print("ERROR: mcp package not installed. Run: pip install mcp", file=sys.stderr)
        sys.exit(1)

    server = Server("delta-exchange")

    @server.list_tools()
    async def list_tools() -> list[mcp_types.Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
        handler = TOOL_HANDLERS.get(name)
        if not handler:
            raise ValueError(f"Unknown tool: {name!r}")
        try:
            result = await handler(arguments or {})
            return [mcp_types.TextContent(type="text", text=result)]
        except Exception as exc:
            logger.exception("delta_server: tool %s failed", name)
            return [mcp_types.TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    asyncio.run(main())
