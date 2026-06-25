from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str


class TrainRequest(BaseModel):
    episodes: int = 1
    render: bool = False


class TrainResponse(BaseModel):
    accepted: bool
    message: str
    episodes: int


class GameStatusResponse(BaseModel):
    mode: str
    score: int
    best_score: int
    epsilon: float
    training_steps: int
    memory_size: int
