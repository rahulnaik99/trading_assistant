"""Generic MCP client for connecting to MCP servers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    StdioServerParameters = None  # type: ignore[assignment, misc]
    stdio_client = None  # type: ignore[assignment, misc]
    ClientSession = None  # type: ignore[assignment, misc]


class MCPClient:
    """Connects to an MCP server and calls tools."""

    def __init__(self, command: str, args: list[str] | None = None,
                 env: dict[str, str] | None = None) -> None:
        if not _MCP_AVAILABLE:
            raise ImportError("Install mcp: pip install mcp.")
        self.command = command
        self.args = args or []
        self.env = env
        self._session: ClientSession | None = None
        self._client = None
        self._null_fd = None

    async def connect(self) -> None:
        logger.info("MCPClient.connect: command=%s  args=%s", self.command, self.args)
        server_params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        self._null_fd = open(os.devnull, "w")
        self._client = stdio_client(server_params, errlog=self._null_fd)
        read, write = await self._client.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        logger.info("MCPClient.connect: connected")

    async def close(self) -> None:
        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
        finally:
            self._session = None
            try:
                if self._client:
                    await self._client.__aexit__(None, None, None)
            finally:
                self._client = None
                if self._null_fd:
                    self._null_fd.close()
                    self._null_fd = None

    async def list_tools(self) -> list[dict[str, Any]]:
        if not self._session:
            raise RuntimeError("Not connected. Call connect() first.")
        tools = await self._session.list_tools()
        return [{"name": t.name, "description": t.description} for t in tools.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        logger.info("MCPClient.call_tool: name=%s  args=%s", name, list(arguments.keys()))
        if not self._session:
            raise RuntimeError("Not connected. Call connect() first.")
        result = await self._session.call_tool(name, arguments=arguments)
        texts = [c.text for c in result.content if hasattr(c, "text")]
        return "\n".join(texts) if texts else str(result.content)

    async def call_tool_json(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.call_tool(name, arguments)
        try:
            return json.loads(result) if isinstance(result, str) else result
        except Exception:
            return {"raw": result}

    def is_auth_error(self, result: dict) -> bool:
        error = result.get("error", "")
        if isinstance(error, dict):
            error = error.get("code", "")
        return ("login" in str(error).lower() or "unauthorized" in str(error).lower()
                or "not authorised" in str(error).lower())
