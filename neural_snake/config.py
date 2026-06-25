from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    GRID_SIZE: int = 24
    CELL_SIZE: int = 26
    PANEL_WIDTH: int = 320
    BOARD_PADDING: int = 20

    FPS: int = 12
    TRAIN_FPS: int = 30

    BG_COLOR: tuple[int, int, int] = (18, 18, 24)
    PANEL_BG_COLOR: tuple[int, int, int] = (24, 28, 38)
    GRID_COLOR: tuple[int, int, int] = (28, 32, 42)
    TEXT_COLOR: tuple[int, int, int] = (235, 239, 247)
    MUTED_TEXT_COLOR: tuple[int, int, int] = (155, 162, 180)
    ACCENT_COLOR: tuple[int, int, int] = (90, 180, 255)
    DANGER_COLOR: tuple[int, int, int] = (255, 92, 92)
    FOOD_COLOR: tuple[int, int, int] = (255, 180, 70)
    SNAKE_HEAD_COLOR: tuple[int, int, int] = (102, 255, 178)
    SNAKE_BODY_COLOR: tuple[int, int, int] = (56, 189, 120)
    SNAKE_TAIL_COLOR: tuple[int, int, int] = (29, 110, 73)

    REPLAY_SIZE: int = 20_000
    BATCH_SIZE: int = 128
    GAMMA: float = 0.98
    LR: float = 0.0007
    EPSILON_START: float = 1.0
    EPSILON_MIN: float = 0.02
    EPSILON_DECAY: float = 0.9994
    TARGET_SYNC_EVERY: int = 300
    TRAIN_START_SIZE: int = 700
    MAX_STEPS_PER_EPISODE: int = 700

    SAVE_PATH: str = "neural_snake_model.npz"
