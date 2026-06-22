"""Analysis Agent Service — runs on port 8101 as a standalone A2A HTTP server.

Usage:
    python -m services.analysis_agent_service
    python -m services.analysis_agent_service --port 8101

The orchestrator at :8100 calls this via:
    POST http://localhost:8101/a2a/task
"""

from __future__ import annotations

import argparse
import logging
import sys

from backend.a2a.server import A2AServer, BaseA2AAgent
from backend.agents.analysis_agent import AnalysisAgent
from backend.protocol import Task, TaskResponse

logger = logging.getLogger(__name__)

_SERVICE_PORT = 8101


class AnalysisAgentService(BaseA2AAgent):
    """A2A wrapper around AnalysisAgent."""

    agent_name = "analysis_agent"

    def __init__(self) -> None:
        super().__init__()
        self._agent = AnalysisAgent()
        logger.info("AnalysisAgentService ready")

    async def handle_task(self, task: Task) -> TaskResponse:
        logger.info("analysis_service: task_id=%s  input_keys=%s",
                    task.task_id, list(task.input.keys()))
        return await self._agent.handle_task(task)


def _make_app(port: int = _SERVICE_PORT):
    service = AnalysisAgentService()
    server  = A2AServer(agent=service, host="0.0.0.0", port=port)
    return server.create_app()


# Expose app for uvicorn --reload
app = _make_app()


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Analysis Agent A2A Service")
    parser.add_argument("--port", type=int, default=_SERVICE_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        stream=sys.stderr,
    )
    logger.info("Starting Analysis Agent Service  host=%s  port=%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)
