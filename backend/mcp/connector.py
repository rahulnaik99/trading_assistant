"""MCP connector — one-shot helper to call any MCP server tool."""

import json
import logging
import os
from typing import Any

from backend.config import settings

logger = logging.getLogger(__name__)

_SERVERS = {
    "delta":  (settings.DELTA_MCP_COMMAND,  settings.DELTA_MCP_ARGS),
    "fyers":  (settings.FYERS_MCP_COMMAND,  settings.FYERS_MCP_ARGS),
    "tavily": (settings.TAVILY_MCP_COMMAND, settings.TAVILY_MCP_ARGS),
}


async def call_mcp_tool(server: str, tool: str, arguments: dict[str, Any] | None = None) -> str:
    """Spawn an MCP server subprocess, call one tool, return text result.

    Falls back gracefully: returns JSON error string on failure (never raises).
    """
    try:
        from backend.mcp.client import MCPClient
    except ImportError:
        return json.dumps({"error": "mcp package not installed — pip install mcp"})

    if server not in _SERVERS:
        return json.dumps({"error": f"Unknown server {server!r} — use delta|fyers|tavily"})

    cmd, args_str = _SERVERS[server]
    args = args_str.split(",") if args_str else []

    env = dict(os.environ)
    # Ensure project root is on PYTHONPATH for MCP server subprocesses
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

    client = MCPClient(command=cmd, args=args, env=env)
    try:
        logger.info("MCP call  server=%s  tool=%s  args=%s", server, tool, list((arguments or {}).keys()))
        await client.connect()
        result = await client.call_tool(tool, arguments or {})
        logger.info("MCP done  server=%s  tool=%s  result_len=%d", server, tool, len(str(result)))
        return result
    except Exception as exc:
        logger.warning("MCP failed  server=%s  tool=%s  error=%s", server, tool, exc)
        return json.dumps({"error": str(exc)})
    finally:
        try:
            await client.close()
        except Exception:
            pass
