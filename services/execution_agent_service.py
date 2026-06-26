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
from backend.protocol import Task, TaskResponse

logger = logging.getLogger(__name__)

_SERVICE_PORT = 8102


class ExecutionAgentService(BaseA2AAgent):
    """A2A wrapper around ExecutionAgent.

    ExecutionAgent is stateless — mode and llm_provider are read from
    task.input at call time, so no constructor arguments are needed.
    """

    agent_name = "execution_agent"

    def __init__(self) -> None:
        super().__init__()
        self._agent = ExecutionAgent()
        logger.info("ExecutionAgentService ready")

    async def handle_task(self, task: Task) -> TaskResponse:
        logger.info("execution_service: task_id=%s  input_keys=%s",
                    task.task_id, list(task.input.keys()))
        return await self._agent.handle_task(task)


def _make_app(port: int = _SERVICE_PORT):
    service = ExecutionAgentService()
    server  = A2AServer(agent=service, host="0.0.0.0", port=port)
    return server.create_app()


app = _make_app()


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Execution Agent A2A Service")
    parser.add_argument("--port", type=int, default=_SERVICE_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        stream=sys.stderr,
    )
    logger.info("Starting Execution Agent Service  host=%s  port=%d", args.host, args.port)
    uvicorn.run(_make_app(args.port), host=args.host, port=args.port)
