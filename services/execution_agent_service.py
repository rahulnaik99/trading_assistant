"""Execution Agent Service — runs on port 8102 as a standalone A2A HTTP server.

Usage:
    python -m services.execution_agent_service
    python -m services.execution_agent_service --port 8102
"""

from __future__ import annotations

import argparse
import logging
import sys

from backend.a2a.server import A2AServer, BaseA2AAgent
from backend.agents.execution_agent import ExecutionAgent
from backend.config import settings
from backend.protocol import Task, TaskResponse

logger = logging.getLogger(__name__)

_SERVICE_PORT = 8102


class ExecutionAgentService(BaseA2AAgent):
    """A2A wrapper around ExecutionAgent."""

    agent_name = "execution_agent"

    def __init__(self, mode: str = "paper") -> None:
        super().__init__()
        self._agent = ExecutionAgent(mode=mode)
        logger.info("ExecutionAgentService ready  mode=%s", mode)

    async def handle_task(self, task: Task) -> TaskResponse:
        logger.info("execution_service: task_id=%s  input_keys=%s",
                    task.task_id, list(task.input.keys()))
        return await self._agent.handle_task(task)


def _make_app(port: int = _SERVICE_PORT, mode: str = "paper"):
    service = ExecutionAgentService(mode=mode)
    server  = A2AServer(agent=service, host="0.0.0.0", port=port)
    return server.create_app()


app = _make_app(mode=settings.TRADE_MODE if hasattr(settings, "TRADE_MODE") else "paper")


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Execution Agent A2A Service")
    parser.add_argument("--port", type=int, default=_SERVICE_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--mode", default="paper", choices=["paper", "real"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        stream=sys.stderr,
    )
    logger.info("Starting Execution Agent Service  host=%s  port=%d  mode=%s",
                args.host, args.port, args.mode)
    uvicorn.run(_make_app(args.port, args.mode), host=args.host, port=args.port)
