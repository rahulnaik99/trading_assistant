"""Fyers MCP Server.

Exposes all Fyers NSE/BSE operations as MCP tools:

Tools:
  fetch_candles    — OHLCV candle data for any NSE/BSE symbol
  get_quote        — Current LTP, OHLC, volume for a symbol
  check_auth       — Verify Fyers access token validity
  place_order      — Place bracket order (authenticated)
  get_positions    — List open positions (authenticated)
  get_orders       — List today's orders (authenticated)
  cancel_order     — Cancel an order by ID (authenticated)

Usage:
    python -m mcp_servers.fyers_server

Connect via MCPClient:
    from backend.mcp.client import MCPClient
    client = MCPClient(command="python", args=["-m", "mcp_servers.fyers_server"])
    await client.connect()
    result = await client.call_tool("fetch_candles", {
        "symbol": "NSE:RELIANCE-EQ", "interval": "5m", "lookback_days": 5
    })
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types as mcp_types
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_fyers_source():
    from backend.agents.data_agent.sources.fyers_source import FyersSource
    from backend.config import settings
    return FyersSource(
        client_id=os.getenv("FYERS_CLIENT_ID", settings.FYERS_CLIENT_ID),
        access_token=os.getenv("FYERS_ACCESS_TOKEN", settings.FYERS_ACCESS_TOKEN),
    )


def _get_fyers_executor():
    from backend.agents.trade_agent.fyers_executor import FyersExecutor
    from backend.config import settings
    return FyersExecutor(
        client_id=os.getenv("FYERS_CLIENT_ID", settings.FYERS_CLIENT_ID),
        access_token=os.getenv("FYERS_ACCESS_TOKEN", settings.FYERS_ACCESS_TOKEN),
    )


# ── Tool implementations ───────────────────────────────────────────────────────

async def tool_fetch_candles(arguments: dict) -> str:
    """Fetch OHLCV candle data from Fyers."""
    symbol        = arguments.get("symbol", "NSE:NIFTY50-INDEX")
    interval      = arguments.get("interval", "5m")
    lookback_days = int(arguments.get("lookback_days", 5))

    now     = datetime.now(timezone.utc)
    from_dt = now - timedelta(days=lookback_days)

    src = _get_fyers_source()
    try:
        is_auth, msg = src.check_auth()
        if not is_auth:
            return json.dumps({"error": f"Fyers auth failed: {msg}"})

        df = src.fetch_ohlc(
            symbol=symbol, from_dt=from_dt, to_dt=now,
            interval=interval, access_token=src.access_token,
        )
        if df.empty:
            return json.dumps({"error": f"No data for {symbol} {interval}", "rows": 0})

        rows = df.reset_index().tail(200).to_dict(orient="records")
        for r in rows:
            r["datetime"] = str(r["datetime"])
        return json.dumps({
            "symbol": symbol, "interval": interval,
            "rows": len(rows),
            "last_close": float(df["close"].iloc[-1]),
            "candles": rows,
        })
    finally:
        pass  # FyersSource is synchronous, no close needed


async def tool_get_quote(arguments: dict) -> str:
    """Get current quote (LTP, OHLC, volume) for a Fyers symbol."""
    symbol = arguments.get("symbol", "NSE:NIFTY50-INDEX")
    ex = _get_fyers_executor()
    try:
        ltp = await ex.get_ltp(symbol)
        return json.dumps({"symbol": symbol, "ltp": ltp})
    finally:
        await ex.close()


async def tool_check_auth(arguments: dict) -> str:
    """Check if Fyers access token is valid."""
    src = _get_fyers_source()
    is_auth, msg = src.check_auth()
    return json.dumps({"authenticated": is_auth, "message": msg})


async def tool_place_order(arguments: dict) -> str:
    """Place a bracket order on Fyers (authenticated)."""
    symbol      = arguments["symbol"]
    side        = arguments["side"]       # "buy" | "sell"
    qty         = int(arguments["qty"])
    entry       = float(arguments["entry"])
    stop_loss   = float(arguments["stop_loss"])
    take_profit = float(arguments["take_profit"])

    ex = _get_fyers_executor()
    try:
        result = await ex.place_bracket_order(
            symbol=symbol, side=side, qty=qty,
            entry=entry, stop_loss=stop_loss, target=take_profit,
        )
        return json.dumps(result)
    finally:
        await ex.close()


async def tool_get_positions(arguments: dict) -> str:
    """Get open positions on Fyers (authenticated)."""
    ex = _get_fyers_executor()
    try:
        positions = await ex.get_positions()
        return json.dumps({"positions": positions, "count": len(positions)})
    finally:
        await ex.close()


async def tool_get_orders(arguments: dict) -> str:
    """Get today's orders on Fyers (authenticated)."""
    ex = _get_fyers_executor()
    try:
        orders = await ex.get_orders()
        return json.dumps({"orders": orders, "count": len(orders)})
    finally:
        await ex.close()


async def tool_cancel_order(arguments: dict) -> str:
    """Cancel an order on Fyers by order ID (authenticated)."""
    order_id = arguments["order_id"]
    ex = _get_fyers_executor()
    try:
        result = await ex.cancel_order(order_id)
        return json.dumps(result)
    finally:
        await ex.close()


# ── Tool registry ──────────────────────────────────────────────────────────────

TOOLS = [
    mcp_types.Tool(
        name="fetch_candles",
        description="Fetch OHLCV candle data from Fyers for NSE/BSE symbols.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol":        {"type": "string",  "description": "Fyers symbol e.g. NSE:RELIANCE-EQ, NSE:NIFTY50-INDEX"},
                "interval":      {"type": "string",  "description": "1m | 5m | 15m | 30m | 1h | 1D"},
                "lookback_days": {"type": "integer", "description": "Days of history (default 5)"},
            },
            "required": ["symbol"],
        },
    ),
    mcp_types.Tool(
        name="get_quote",
        description="Get the current LTP (last traded price) for a Fyers symbol.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Fyers symbol e.g. NSE:SBIN-EQ"},
            },
            "required": ["symbol"],
        },
    ),
    mcp_types.Tool(
        name="check_auth",
        description="Check if the Fyers access token is valid.",
        inputSchema={"type": "object", "properties": {}},
    ),
    mcp_types.Tool(
        name="place_order",
        description="Place a Fyers bracket order (BO) with entry, stop-loss and target. Requires valid access token.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol":      {"type": "string",  "description": "Fyers symbol e.g. NSE:RELIANCE-EQ"},
                "side":        {"type": "string",  "description": "buy | sell"},
                "qty":         {"type": "integer", "description": "Quantity"},
                "entry":       {"type": "number",  "description": "Entry price"},
                "stop_loss":   {"type": "number",  "description": "Stop-loss price"},
                "take_profit": {"type": "number",  "description": "Take-profit price"},
            },
            "required": ["symbol", "side", "qty", "entry", "stop_loss", "take_profit"],
        },
    ),
    mcp_types.Tool(
        name="get_positions",
        description="Get all open net positions on Fyers.",
        inputSchema={"type": "object", "properties": {}},
    ),
    mcp_types.Tool(
        name="get_orders",
        description="Get today's order book on Fyers.",
        inputSchema={"type": "object", "properties": {}},
    ),
    mcp_types.Tool(
        name="cancel_order",
        description="Cancel a Fyers order by its order ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
            },
            "required": ["order_id"],
        },
    ),
] if _MCP_AVAILABLE else []

TOOL_HANDLERS = {
    "fetch_candles": tool_fetch_candles,
    "get_quote":     tool_get_quote,
    "check_auth":    tool_check_auth,
    "place_order":   tool_place_order,
    "get_positions": tool_get_positions,
    "get_orders":    tool_get_orders,
    "cancel_order":  tool_cancel_order,
}


# ── MCP Server ─────────────────────────────────────────────────────────────────

async def main():
    if not _MCP_AVAILABLE:
        print("ERROR: mcp package not installed. Run: pip install mcp", file=sys.stderr)
        sys.exit(1)

    server = Server("fyers")

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
            logger.exception("fyers_server: tool %s failed", name)
            return [mcp_types.TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    asyncio.run(main())
