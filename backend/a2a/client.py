"""A2A Client — sends Task to a remote agent over HTTP and returns TaskResponse."""

from __future__ import annotations

import logging
import uuid

import httpx

from backend.protocol import Task, TaskResponse

logger = logging.getLogger(__name__)


class A2AClient:
    """HTTP client for calling a remote A2A agent."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    async def send(self, agent: str, input_data: dict, task_id: str = "") -> TaskResponse:
        """POST /a2a/task and return the response."""
        task = Task(
            task_id=task_id or str(uuid.uuid4())[:8],
            agent=agent,
            input=input_data,
        )
        logger.info("A2AClient → %s  agent=%s  task_id=%s", self.base_url, agent, task.task_id)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/a2a/task",
                    json=task.model_dump(mode="json"),
                )
                resp.raise_for_status()
                result = TaskResponse(**resp.json())
                logger.info("A2AClient ← agent=%s  status=%s", agent, result.status)
                return result
            except httpx.ConnectError as exc:
                logger.error("A2AClient: cannot reach %s — %s", self.base_url, exc)
                return TaskResponse(
                    task_id=task.task_id, agent=agent, status="failed",
                    error=f"Agent service unreachable at {self.base_url}. Is it running? ({exc})",
                )
            except httpx.TimeoutException as exc:
                logger.error("A2AClient: timeout waiting for %s — %s", self.base_url, exc)
                return TaskResponse(
                    task_id=task.task_id, agent=agent, status="failed",
                    error=f"Agent service unreachable at {self.base_url} (timeout after {self.timeout}s). ({exc})",
                )
            except httpx.HTTPStatusError as exc:
                logger.error("A2AClient: HTTP %d from %s", exc.response.status_code, self.base_url)
                return TaskResponse(
                    task_id=task.task_id, agent=agent, status="failed",
                    error=f"Agent returned HTTP {exc.response.status_code}: {exc.response.text[:200]}",
                )

    async def health(self) -> bool:
        """Returns True if the agent service is reachable."""
        async with httpx.AsyncClient(timeout=3.0) as client:
            try:
                r = await client.get(f"{self.base_url}/.well-known/agent.json")
                return r.status_code == 200
            except Exception:
                return False
