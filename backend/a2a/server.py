"""A2A Server — wraps any agent as a FastAPI HTTP endpoint.

POST /a2a/task         → runs agent.handle_task(), returns TaskResponse
GET  /a2a/task/{id}    → poll stored result
GET  /.well-known/agent.json → agent discovery card
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException

from backend.protocol import Artifact, Task, TaskResponse

logger = logging.getLogger(__name__)


class BaseA2AAgent:
    """Abstract base — subclass and implement handle_task."""
    agent_name: str = "base"

    async def handle_task(self, task: Task) -> TaskResponse:
        raise NotImplementedError


class A2AServer:
    """Exposes a single agent over HTTP via the A2A protocol."""

    def __init__(self, agent: BaseA2AAgent, host: str = "0.0.0.0", port: int = 8101) -> None:
        self.agent = agent
        self.host  = host
        self.port  = port
        self._store: dict[str, TaskResponse] = {}
        self._app: FastAPI | None = None

    def create_app(self) -> FastAPI:
        app = FastAPI(title=f"A2A — {self.agent.agent_name}")

        @app.post("/a2a/task", response_model=TaskResponse)
        async def submit(task: Task) -> TaskResponse:
            logger.info("A2A  agent=%s  task_id=%s", self.agent.agent_name, task.task_id)
            try:
                resp = await self.agent.handle_task(task)
            except Exception as exc:
                logger.exception("A2A agent error  task_id=%s", task.task_id)
                resp = TaskResponse(
                    task_id=task.task_id, agent=self.agent.agent_name,
                    status="failed", error=str(exc),
                )
            self._store[task.task_id] = resp
            logger.info("A2A done  agent=%s  status=%s", self.agent.agent_name, resp.status)
            return resp

        @app.get("/a2a/task/{task_id}", response_model=TaskResponse)
        async def poll(task_id: str) -> TaskResponse:
            if task_id not in self._store:
                raise HTTPException(status_code=404, detail="Task not found")
            return self._store[task_id]

        @app.get("/.well-known/agent.json")
        async def card() -> dict[str, Any]:
            return {
                "name": self.agent.agent_name,
                "url":  f"http://{self.host}:{self.port}",
                "capabilities": ["task"],
            }

        self._app = app
        return app

    def run(self) -> None:
        import uvicorn
        if not self._app:
            self._app = self.create_app()
        logger.info("A2A server starting  agent=%s  port=%d", self.agent.agent_name, self.port)
        uvicorn.run(self._app, host=self.host, port=self.port)
