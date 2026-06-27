"""MCP connector — one-shot helper to call any MCP server tool."""

import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Per-tool timeouts (seconds).
_TOOL_TIMEOUTS: dict[str, float] = {
    "fetch_candles":           25.0,
    "fetch_perpetual_metrics": 15.0,
    "get_quote":               15.0,
    "get_mark_price":          10.0,
    "search_news":             15.0,
    "search_market":           15.0,
}
_DEFAULT_TIMEOUT = 20.0


def _get_server_cmd(server: str) -> tuple[str, list[str]]:
    """Resolve MCP server command lazily so settings are fully loaded first."""
    from backend.config import settings
    _map = {
        "delta":  (settings.DELTA_MCP_COMMAND,  settings.DELTA_MCP_ARGS),
        "fyers":  (settings.FYERS_MCP_COMMAND,  settings.FYERS_MCP_ARGS),
        "tavily": (settings.TAVILY_MCP_COMMAND, settings.TAVILY_MCP_ARGS),
    }
    if server not in _map:
        raise KeyError(server)
    cmd, args_str = _map[server]
    return cmd, (args_str.split(",") if args_str else [])


async def call_mcp_tool(server: str, tool: str, arguments: dict[str, Any] | None = None) -> str:
    """Spawn an MCP server subprocess, call one tool, return text result.

    Falls back gracefully: returns JSON error string on failure (never raises).
    """
    try:
        from backend.mcp.client import MCPClient
    except ImportError:
        return json.dumps({"error": "mcp package not installed — pip install mcp"})

    try:
        cmd, args = _get_server_cmd(server)
    except KeyError:
        return json.dumps({"error": f"Unknown server {server!r} — use delta|fyers|tavily"})

    from backend.config import settings
    env = dict(os.environ)
    project_root = str(__import__("pathlib").Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    if server == "delta":
        env.update({
            "DELTA_API_KEY":    settings.DELTA_API_KEY,
            "DELTA_API_SECRET": settings.DELTA_API_SECRET,
            "DELTA_REGION":     settings.DELTA_REGION,
        })
    elif server == "fyers":
        env.update({
            "FYERS_CLIENT_ID":    settings.FYERS_CLIENT_ID,
            "FYERS_ACCESS_TOKEN": settings.FYERS_ACCESS_TOKEN,
        })
    elif server == "tavily":
        env["TAVILY_API_KEY"] = settings.TAVILY_API_KEY

    timeout = _TOOL_TIMEOUTS.get(tool, _DEFAULT_TIMEOUT)
    client  = MCPClient(command=cmd, args=args, env=env)
    try:
        logger.info("MCP call  server=%s  tool=%s  args=%s  timeout=%ss",
                    server, tool, list((arguments or {}).keys()), timeout)
        await client.connect()
        result = await asyncio.wait_for(
            client.call_tool(tool, arguments or {}),
            timeout=timeout,
        )
        logger.info("MCP done  server=%s  tool=%s  result_len=%d", server, tool, len(str(result)))
        return result
    except asyncio.TimeoutError:
        logger.warning("MCP timeout  server=%s  tool=%s  after=%ss", server, tool, timeout)
        return json.dumps({"error": f"{server}.{tool} timed out after {timeout}s"})
    except Exception as exc:
        logger.warning("MCP failed  server=%s  tool=%s  error=%s", server, tool, exc)
        return json.dumps({"error": str(exc)})
    finally:
        try:
            await client.close()
        except Exception:
            pass
