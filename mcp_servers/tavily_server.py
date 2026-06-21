"""Tavily MCP Server.

Exposes Tavily news/web search as MCP tools for market context.
Falls back to Google RSS if TAVILY_API_KEY is not set.

Tools:
  search_news       — Search recent news for a symbol/topic
  search_market     — General market/economic news search
  get_rss_news      — Google News RSS feed (no key required)

Usage:
    TAVILY_API_KEY=your-key python -m mcp_servers.tavily_server

Connect via MCPClient:
    from backend.mcp.client import MCPClient
    client = MCPClient(
        command="python",
        args=["-m", "mcp_servers.tavily_server"],
        env={"TAVILY_API_KEY": "your-key"}
    )
    await client.connect()
    result = await client.call_tool("search_news", {"query": "BTCUSDT price today"})
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys

logger = logging.getLogger(__name__)

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types as mcp_types
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


# ── Tool implementations ───────────────────────────────────────────────────────

async def tool_search_news(arguments: dict) -> str:
    """Search recent news using Tavily or Google RSS fallback."""
    query      = arguments.get("query", "BTC price")
    max_results= int(arguments.get("max_results", 5))
    days_back  = int(arguments.get("days_back", 3))

    tavily_key = os.getenv("TAVILY_API_KEY", "")

    if tavily_key:
        return await _tavily_search(query, max_results, days_back, tavily_key)
    else:
        return await _rss_search(query, max_results)


async def tool_search_market(arguments: dict) -> str:
    """Search for broader market/macro news."""
    query      = arguments.get("query", "crypto market outlook")
    max_results= int(arguments.get("max_results", 5))

    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        return await _tavily_search(query, max_results, days_back=7, api_key=tavily_key)
    else:
        return await _rss_search(query, max_results)


async def tool_get_rss_news(arguments: dict) -> str:
    """Fetch news from Google RSS (no API key required)."""
    symbol     = arguments.get("symbol", "BTCUSDT")
    max_results= int(arguments.get("max_results", 5))
    clean      = symbol.replace("NSE:", "").replace("-EQ", "").replace("USDT", "").upper()
    return await _rss_search(f"{clean} price", max_results)


async def _tavily_search(query: str, max_results: int, days_back: int, api_key: str) -> str:
    """Call Tavily REST search API."""
    import httpx

    url = "https://api.tavily.com/search"
    payload = {
        "api_key":     api_key,
        "query":       query,
        "max_results": max_results,
        "search_depth":"basic",
        "include_answer": True,
        "days":        days_back,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        headlines = [
            {"title": r.get("title", ""), "url": r.get("url", ""),
             "content": (r.get("content") or "")[:200]}
            for r in results[:max_results]
        ]
        answer = data.get("answer", "")
        return json.dumps({
            "query": query,
            "answer": answer,
            "results": headlines,
            "source": "tavily",
        })
    except Exception as exc:
        logger.warning("Tavily search failed: %s — falling back to RSS", exc)
        return await _rss_search(query, max_results)


async def _rss_search(query: str, max_results: int) -> str:
    """Fetch news from Google News RSS (free, no key)."""
    import httpx

    clean_query = query.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={clean_query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "TradingAnalyst/1.0"})
            if resp.status_code != 200:
                return json.dumps({"error": f"RSS returned {resp.status_code}", "results": []})

        titles = re.findall(r"<title><!\[CDATA\[(.+?)\]\]></title>", resp.text)
        if not titles:
            titles = re.findall(r"<title>(.+?)</title>", resp.text)

        headlines = [t.strip() for t in titles[1:max_results + 1] if t.strip()]
        return json.dumps({
            "query": query,
            "results": [{"title": h} for h in headlines],
            "source": "google_rss",
        })
    except Exception as exc:
        return json.dumps({"error": str(exc), "results": []})


# ── Tool registry ──────────────────────────────────────────────────────────────

TOOLS = [
    mcp_types.Tool(
        name="search_news",
        description="Search recent news for a trading symbol or topic. Uses Tavily if API key is set, Google RSS otherwise.",
        inputSchema={
            "type": "object",
            "properties": {
                "query":       {"type": "string",  "description": "Search query e.g. 'BTCUSDT price analysis' or 'RELIANCE earnings'"},
                "max_results": {"type": "integer", "description": "Max headlines to return (default 5)"},
                "days_back":   {"type": "integer", "description": "How many days back to search (default 3)"},
            },
            "required": ["query"],
        },
    ),
    mcp_types.Tool(
        name="search_market",
        description="Search for broader market context — macro news, Fed decisions, crypto trends.",
        inputSchema={
            "type": "object",
            "properties": {
                "query":       {"type": "string",  "description": "e.g. 'crypto market outlook this week'"},
                "max_results": {"type": "integer", "description": "Max results (default 5)"},
            },
            "required": ["query"],
        },
    ),
    mcp_types.Tool(
        name="get_rss_news",
        description="Get latest news for a trading symbol from Google News RSS. No API key required.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol":      {"type": "string",  "description": "Symbol e.g. BTCUSDT, NSE:RELIANCE-EQ"},
                "max_results": {"type": "integer", "description": "Max headlines (default 5)"},
            },
            "required": ["symbol"],
        },
    ),
] if _MCP_AVAILABLE else []

TOOL_HANDLERS = {
    "search_news":    tool_search_news,
    "search_market":  tool_search_market,
    "get_rss_news":   tool_get_rss_news,
}


# ── MCP Server ─────────────────────────────────────────────────────────────────

async def main():
    if not _MCP_AVAILABLE:
        print("ERROR: mcp package not installed. Run: pip install mcp", file=sys.stderr)
        sys.exit(1)

    server = Server("tavily-news")

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
            logger.exception("tavily_server: tool %s failed", name)
            return [mcp_types.TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    asyncio.run(main())
