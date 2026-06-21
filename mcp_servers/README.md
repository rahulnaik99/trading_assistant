"""Base MCP server utility.

Each MCP server in this package exposes a set of tools via the
Model Context Protocol (MCP) stdio transport.  The host process
(Claude, an agent, or another LLM) connects via stdin/stdout and
can discover + call tools using the standard MCP protocol.

Usage (run any server directly):
    python -m mcp_servers.delta_server
    python -m mcp_servers.fyers_server
    python -m mcp_servers.tavily_server

Connect from an agent:
    from backend.mcp.client import MCPClient
    client = MCPClient(command="python", args=["-m", "mcp_servers.delta_server"])
    await client.connect()
    result = await client.call_tool("fetch_candles", {"symbol": "BTCUSDT", "interval": "5m"})
    await client.close()
"""

MCP_SERVERS = {
    "delta":  {"module": "mcp_servers.delta_server",  "description": "Delta Exchange — candles, perpetual metrics, orders"},
    "fyers":  {"module": "mcp_servers.fyers_server",  "description": "Fyers — NSE/BSE candles, quotes, orders"},
    "tavily": {"module": "mcp_servers.tavily_server", "description": "Tavily — news search, market context"},
}
