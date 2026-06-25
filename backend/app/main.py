from __future__ import annotations

from fastapi import FastAPI, HTTPException

from game.config import Config
from game.service import SnakeGameService

from .schemas import GameStatusResponse, HealthResponse, TrainRequest, TrainResponse

app = FastAPI(title="Neural Snake API", version="0.1.0")
config = Config()
service = SnakeGameService(config)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="neural-snake-backend")


@app.get("/game/status", response_model=GameStatusResponse)
def game_status() -> GameStatusResponse:
    state = service.snapshot()
    return GameStatusResponse(
        mode=state["mode"],
        score=state["score"],
        best_score=state["best_score"],
        epsilon=state["epsilon"],
        training_steps=state["training_steps"],
        memory_size=state["memory_size"],
    )


@app.post("/train", response_model=TrainResponse)
def train(payload: TrainRequest) -> TrainResponse:
    if payload.episodes < 1:
        raise HTTPException(status_code=400, detail="episodes must be at least 1")
    service.train(payload.episodes, render=payload.render)
    return TrainResponse(accepted=True, message="Training started", episodes=payload.episodes)


@app.post("/game/reset", response_model=HealthResponse)
def reset_game() -> HealthResponse:
    service.reset()
    return HealthResponse(status="reset", service="neural-snake-backend")
