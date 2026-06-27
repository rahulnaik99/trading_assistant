"""A2A Client — sends Task to a remote agent over HTTP and returns TaskResponse."""

from __future__ import annotations

import asyncio
import logging
import uuid

import httpx

from backend.protocol import Task, TaskResponse

logger = logging.getLogger(__name__)

_RETRYABLE = (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError)


class A2AClient:
    """HTTP client for calling a remote A2A agent with retry."""

    def __init__(self, base_url: str, timeout: float = 300.0, retries: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self.retries  = retries

    async def send(self, agent: str, input_data: dict, task_id: str = "") -> TaskResponse:
        """POST /a2a/task with up to self.retries retries on transient errors."""
        task = Task(
            task_id=task_id or str(uuid.uuid4())[:8],
            agent=agent,
            input=input_data,
        )
        logger.info("A2AClient → %s  agent=%s  task_id=%s", self.base_url, agent, task.task_id)

        last_err: str = ""
        for attempt in range(1, self.retries + 2):  # attempts = retries + 1
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                try:
                    resp = await client.post(
                        f"{self.base_url}/a2a/task",
                        json=task.model_dump(mode="json"),
                    )
                    resp.raise_for_status()
                    result = TaskResponse(**resp.json())
                    logger.info("A2AClient ← agent=%s  status=%s  attempt=%d",
                                agent, result.status, attempt)
                    return result

                except _RETRYABLE as exc:
                    last_err = str(exc)
                    if attempt <= self.retries:
                        wait = 1.5 ** (attempt - 1)   # 1s, 1.5s back-off
                        logger.warning("A2AClient: transient error (attempt %d/%d) — retrying in %.1fs  %s",
                                       attempt, self.retries + 1, wait, exc)
                        await asyncio.sleep(wait)
                    else:
                        logger.error("A2AClient: all retries exhausted  %s → %s", self.base_url, exc)

                except httpx.HTTPStatusError as exc:
                    logger.error("A2AClient: HTTP %d from %s", exc.response.status_code, self.base_url)
                    return TaskResponse(
                        task_id=task.task_id, agent=agent, status="failed",
                        error=f"Agent returned HTTP {exc.response.status_code}: {exc.response.text[:200]}",
                    )

        return TaskResponse(
            task_id=task.task_id, agent=agent, status="failed",
            error=f"Agent service unreachable at {self.base_url} after {self.retries + 1} attempts. ({last_err})",
        )

    async def health(self) -> bool:
        """Returns True if the agent service is reachable."""
        async with httpx.AsyncClient(timeout=3.0) as client:
            try:
                r = await client.get(f"{self.base_url}/.well-known/agent.json")
                return r.status_code == 200
            except Exception:
                return False

