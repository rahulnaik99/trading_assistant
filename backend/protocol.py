"""A2A protocol — Task / TaskResponse / Artifact shared by all agents."""

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    type: str
    data: dict


class Task(BaseModel):
    task_id: str
    agent: str
    input: dict = Field(default_factory=dict)


class TaskResponse(BaseModel):
    task_id: str
    agent: str
    status: str  # completed | failed
    artifacts: list[Artifact] = Field(default_factory=list)
    error: str | None = None
