from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .config import Config

Direction = Literal["UP", "RIGHT", "DOWN", "LEFT"]

DIRECTION_ORDER: list[Direction] = ["RIGHT", "DOWN", "LEFT", "UP"]
DIRECTION_VECTORS: dict[Direction, tuple[int, int]] = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}
KEY_DIRECTION_MAP = {
    "UP": "UP",
    "W": "UP",
    "RIGHT": "RIGHT",
    "D": "RIGHT",
    "DOWN": "DOWN",
    "S": "DOWN",
    "LEFT": "LEFT",
    "A": "LEFT",
}


@dataclass
class EpisodeResult:
    score: int = 0
    steps: int = 0
    reward: float = 0.0
    done: bool = False


@dataclass
class SnakeEnvironment:
    config: Config
    mode: Literal["train", "human"] = "train"
    snake: list[tuple[int, int]] = field(default_factory=list)
    direction: Direction = "RIGHT"
    pending_direction: Direction = "RIGHT"
    food: tuple[int, int] = (0, 0)
    score: int = 0
    total_steps: int = 0
    episode_steps: int = 0
    episode_reward: float = 0.0
    game_over: bool = False
    paused: bool = False
    last_distance: float = 0.0
    episode_index: int = 0
    history: list[EpisodeResult] = field(default_factory=list)
    best_episode_score: int = 0
    mean_episode_score: float = 0.0
    mean_episode_reward: float = 0.0

    def __post_init__(self) -> None:
        self.grid_size = self.config.GRID_SIZE
        self.board_px = self.grid_size * self.config.CELL_SIZE
        self.window_width = self.board_px + self.config.PANEL_WIDTH + self.config.BOARD_PADDING * 2
        self.window_height = self.board_px + self.config.BOARD_PADDING * 2
        self.board_origin = (self.config.BOARD_PADDING, self.config.BOARD_PADDING)
        self.reset_episode()

    @property
    def state_size(self) -> int:
        return 21

    @property
    def action_size(self) -> int:
        return 3

    def reset_episode(self) -> None:
        center = self.grid_size // 2
        self.snake = [(center, center), (center - 1, center), (center - 2, center)]
        self.direction = "RIGHT"
        self.pending_direction = "RIGHT"
        self.score = 0
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.game_over = False
        self.paused = False
        self.spawn_food()
        self.last_distance = self.distance_to_food(self.snake[0])
        self.episode_index += 1

    def spawn_food(self) -> None:
        available = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.snake
        ]
        if not available:
            self.food = (-1, -1)
            return
        self.food = available[int(np.random.randint(len(available)))]

    def distance_to_food(self, head: tuple[int, int]) -> float:
        return float(abs(head[0] - self.food[0]) + abs(head[1] - self.food[1]))

    def turn_from_action(self, action: int) -> Direction:
        index = DIRECTION_ORDER.index(self.direction)
        if action == 0:
            return self.direction
        if action == 1:
            return DIRECTION_ORDER[(index + 1) % 4]
        return DIRECTION_ORDER[(index - 1) % 4]

    def is_reverse(self, candidate: Direction, current: Direction) -> bool:
        return DIRECTION_VECTORS[candidate] == tuple(-v for v in DIRECTION_VECTORS[current])

    def apply_pending_direction(self) -> None:
        if not self.is_reverse(self.pending_direction, self.direction):
            self.direction = self.pending_direction

    def set_direction_from_key(self, key: int) -> None:
        import pygame

        key_map = {
            pygame.K_UP: "UP",
            pygame.K_w: "UP",
            pygame.K_RIGHT: "RIGHT",
            pygame.K_d: "RIGHT",
            pygame.K_DOWN: "DOWN",
            pygame.K_s: "DOWN",
            pygame.K_LEFT: "LEFT",
            pygame.K_a: "LEFT",
        }
        direction = key_map.get(key)
        if direction is None or self.is_reverse(direction, self.direction):
            return
        self.pending_direction = direction

    def free_space_ratio(self) -> float:
        occupied = set(self.snake)
        total = self.grid_size * self.grid_size
        return max(0.0, 1.0 - len(occupied) / total)

    def is_collision(self, point: tuple[int, int]) -> bool:
        x, y = point
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        return point in self.snake[1:]

    def get_state(self) -> np.ndarray:
        head_x, head_y = self.snake[0]
        dir_x, dir_y = DIRECTION_VECTORS[self.direction]
        point_straight = (head_x + dir_x, head_y + dir_y)
        point_right = (head_x + dir_y, head_y - dir_x)
        point_left = (head_x - dir_y, head_y + dir_x)

        danger_straight = self.is_collision(point_straight)
        danger_right = self.is_collision(point_right)
        danger_left = self.is_collision(point_left)

        food_dx = (self.food[0] - head_x) / max(1, self.grid_size - 1)
        food_dy = (self.food[1] - head_y) / max(1, self.grid_size - 1)
        head_x_norm = head_x / max(1, self.grid_size - 1)
        head_y_norm = head_y / max(1, self.grid_size - 1)
        length_norm = len(self.snake) / float(self.grid_size * self.grid_size)
        free_space_norm = self.free_space_ratio()

        distance_left = head_x / max(1, self.grid_size - 1)
        distance_right = (self.grid_size - 1 - head_x) / max(1, self.grid_size - 1)
        distance_up = head_y / max(1, self.grid_size - 1)
        distance_down = (self.grid_size - 1 - head_y) / max(1, self.grid_size - 1)

        direction_one_hot = [
            1.0 if self.direction == "UP" else 0.0,
            1.0 if self.direction == "RIGHT" else 0.0,
            1.0 if self.direction == "DOWN" else 0.0,
            1.0 if self.direction == "LEFT" else 0.0,
        ]

        food_direction = [
            1.0 if food_dx < 0 else 0.0,
            1.0 if food_dx > 0 else 0.0,
            1.0 if food_dy < 0 else 0.0,
            1.0 if food_dy > 0 else 0.0,
        ]

        return np.array(
            [
                float(danger_straight),
                float(danger_right),
                float(danger_left),
                *direction_one_hot,
                *food_direction,
                food_dx,
                food_dy,
                head_x_norm,
                head_y_norm,
                length_norm,
                free_space_norm,
                distance_left,
                distance_right,
                distance_up,
                distance_down,
            ],
            dtype=np.float32,
        )

    def step(self, action: int) -> EpisodeResult:
        if self.game_over:
            return EpisodeResult(score=self.score, steps=self.episode_steps, reward=0.0, done=True)

        self.apply_pending_direction()
        self.direction = self.turn_from_action(action)
        self.pending_direction = self.direction
        dir_x, dir_y = DIRECTION_VECTORS[self.direction]
        head_x, head_y = self.snake[0]
        next_head = (head_x + dir_x, head_y + dir_y)

        self.episode_steps += 1
        self.total_steps += 1

        if self.is_collision(next_head) or self.episode_steps > self.config.MAX_STEPS_PER_EPISODE:
            self.game_over = True
            self.episode_reward -= 10.0
            result = EpisodeResult(score=self.score, steps=self.episode_steps, reward=-10.0, done=True)
            self.history.append(result)
            return result

        self.snake.insert(0, next_head)
        reward = -0.02
        distance_now = self.distance_to_food(next_head)
        reward += 0.08 if distance_now < self.last_distance else -0.02
        self.last_distance = distance_now

        if next_head == self.food:
            self.score += 1
            reward += 12.0
            self.spawn_food()
            self.last_distance = self.distance_to_food(self.snake[0])
        else:
            self.snake.pop()

        self.episode_reward += reward
        return EpisodeResult(score=self.score, steps=self.episode_steps, reward=reward, done=False)

    def update_human(self) -> EpisodeResult:
        if self.game_over:
            return EpisodeResult(score=self.score, steps=self.episode_steps, reward=0.0, done=True)
        self.apply_pending_direction()
        return self.step(0)

    def update_stats(self) -> None:
        if not self.history:
            return
        recent_scores = [episode.score for episode in self.history[-20:]]
        recent_rewards = [episode.reward for episode in self.history[-20:]]
        self.best_episode_score = max(recent_scores, default=0)
        self.mean_episode_score = float(sum(recent_scores) / len(recent_scores)) if recent_scores else 0.0
        self.mean_episode_reward = float(sum(recent_rewards) / len(recent_rewards)) if recent_rewards else 0.0
