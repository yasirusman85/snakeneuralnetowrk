from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pygame

from .ai import SnakeAgent
from .config import Config

Direction = Literal["UP", "RIGHT", "DOWN", "LEFT"]
Action = Literal[0, 1, 2]


DIRECTION_ORDER: list[Direction] = ["RIGHT", "DOWN", "LEFT", "UP"]
DIRECTION_VECTORS: dict[Direction, tuple[int, int]] = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}
KEY_DIRECTION_MAP = {
    pygame.K_UP: "UP",
    pygame.K_w: "UP",
    pygame.K_RIGHT: "RIGHT",
    pygame.K_d: "RIGHT",
    pygame.K_DOWN: "DOWN",
    pygame.K_s: "DOWN",
    pygame.K_LEFT: "LEFT",
    pygame.K_a: "LEFT",
}


@dataclass
class EpisodeResult:
    score: int = 0
    steps: int = 0
    reward: float = 0.0
    done: bool = False


@dataclass
class SnakeGame:
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

    def __post_init__(self) -> None:
        self.grid_size = self.config.GRID_SIZE
        self.board_px = self.grid_size * self.config.CELL_SIZE
        self.window_width = self.board_px + self.config.PANEL_WIDTH + self.config.BOARD_PADDING * 2
        self.window_height = self.board_px + self.config.BOARD_PADDING * 2
        self.board_origin = (self.config.BOARD_PADDING, self.config.BOARD_PADDING)
        self.reset_episode()

    @property
    def state_size(self) -> int:
        return 19

    @property
    def action_size(self) -> int:
        return 3

    def reset_episode(self) -> None:
        center = self.grid_size // 2
        self.snake = [
            (center, center),
            (center - 1, center),
            (center - 2, center),
        ]
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

    def set_direction_from_key(self, key: int) -> None:
        direction = KEY_DIRECTION_MAP.get(key)
        if direction is None:
            return
        if self.is_reverse(direction, self.direction):
            return
        self.pending_direction = direction

    @staticmethod
    def is_reverse(candidate: Direction, current: Direction) -> bool:
        return DIRECTION_VECTORS[candidate] == tuple(-v for v in DIRECTION_VECTORS[current])

    def apply_pending_direction(self) -> None:
        if not self.is_reverse(self.pending_direction, self.direction):
            self.direction = self.pending_direction

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

        state = np.array(
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
        return state

    def free_space_ratio(self) -> float:
        occupied = set(self.snake)
        total = self.grid_size * self.grid_size
        return max(0.0, 1.0 - len(occupied) / total)

    def is_collision(self, point: tuple[int, int]) -> bool:
        x, y = point
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        if point in self.snake[1:]:
            return True
        return False

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
            result = EpisodeResult(
                score=self.score,
                steps=self.episode_steps,
                reward=-10.0,
                done=True,
            )
            self.history.append(result)
            return result

        self.snake.insert(0, next_head)
        reward = -0.02
        distance_now = self.distance_to_food(next_head)

        if distance_now < self.last_distance:
            reward += 0.08
        else:
            reward -= 0.02
        self.last_distance = distance_now

        if next_head == self.food:
            self.score += 1
            reward += 12.0
            self.spawn_food()
            self.last_distance = self.distance_to_food(self.snake[0])
        else:
            self.snake.pop()

        self.episode_reward += reward
        result = EpisodeResult(score=self.score, steps=self.episode_steps, reward=reward, done=False)
        return result

    def update_human(self) -> EpisodeResult:
        if self.game_over:
            return EpisodeResult(score=self.score, steps=self.episode_steps, reward=0.0, done=True)
        self.apply_pending_direction()
        action = 0
        return self.step(action)

    def render(self, screen: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font, agent: SnakeAgent) -> None:
        screen.fill(self.config.BG_COLOR)
        board_rect = pygame.Rect(*self.board_origin, self.board_px, self.board_px)
        panel_rect = pygame.Rect(self.board_px + self.config.BOARD_PADDING * 2, 0, self.config.PANEL_WIDTH, self.window_height)
        pygame.draw.rect(screen, self.config.PANEL_BG_COLOR, panel_rect)
        pygame.draw.rect(screen, (42, 48, 62), board_rect, border_radius=12)

        self._draw_grid(screen)
        self._draw_food(screen)
        self._draw_snake(screen)
        self._draw_panel(screen, font, small_font, agent, panel_rect)

    def _draw_grid(self, screen: pygame.Surface) -> None:
        ox, oy = self.board_origin
        cell = self.config.CELL_SIZE
        for x in range(self.grid_size + 1):
            px = ox + x * cell
            pygame.draw.line(screen, self.config.GRID_COLOR, (px, oy), (px, oy + self.board_px), 1)
        for y in range(self.grid_size + 1):
            py = oy + y * cell
            pygame.draw.line(screen, self.config.GRID_COLOR, (ox, py), (ox + self.board_px, py), 1)

    def _cell_rect(self, point: tuple[int, int], inset: int = 3) -> pygame.Rect:
        ox, oy = self.board_origin
        x, y = point
        size = self.config.CELL_SIZE
        return pygame.Rect(ox + x * size + inset, oy + y * size + inset, size - inset * 2, size - inset * 2)

    def _draw_snake(self, screen: pygame.Surface) -> None:
        if not self.snake:
            return
        for index, segment in enumerate(reversed(self.snake)):
            alpha_index = len(self.snake) - 1 - index
            if alpha_index == 0:
                color = self.config.SNAKE_HEAD_COLOR
            elif alpha_index == len(self.snake) - 1:
                color = self.config.SNAKE_TAIL_COLOR
            else:
                blend = alpha_index / max(1, len(self.snake) - 1)
                color = tuple(
                    int(self.config.SNAKE_BODY_COLOR[i] * blend + self.config.SNAKE_HEAD_COLOR[i] * (1.0 - blend))
                    for i in range(3)
                )
            rect = self._cell_rect(segment, inset=2 if alpha_index == 0 else 4)
            pygame.draw.rect(screen, color, rect, border_radius=8)
            if alpha_index == 0:
                pygame.draw.rect(screen, (245, 255, 248), rect, width=2, border_radius=8)

    def _draw_food(self, screen: pygame.Surface) -> None:
        if self.food == (-1, -1):
            return
        rect = self._cell_rect(self.food, inset=5)
        pygame.draw.ellipse(screen, self.config.FOOD_COLOR, rect)
        glow = rect.inflate(8, 8)
        pygame.draw.ellipse(screen, (255, 220, 140), glow, width=2)

    def _draw_panel(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        small_font: pygame.font.Font,
        agent: SnakeAgent,
        panel_rect: pygame.Rect,
    ) -> None:
        x = panel_rect.x + 18
        y = 24
        lines = [
            ("NEURAL SNAKE", font, self.config.ACCENT_COLOR),
            (f"Mode: {self.mode.upper()}", small_font, self.config.TEXT_COLOR),
            (f"Episode: {agent.stats.episode}", small_font, self.config.TEXT_COLOR),
            (f"Score: {self.score}", small_font, self.config.TEXT_COLOR),
            (f"Best: {agent.stats.best_score}", small_font, self.config.TEXT_COLOR),
            (f"Steps: {self.episode_steps}", small_font, self.config.TEXT_COLOR),
            (f"Reward: {self.episode_reward:.2f}", small_font, self.config.TEXT_COLOR),
            (f"Epsilon: {agent.stats.epsilon:.3f}", small_font, self.config.TEXT_COLOR),
            (f"Memory: {len(agent.memory)}", small_font, self.config.TEXT_COLOR),
            (f"Loss: {agent.stats.mean_loss:.5f}", small_font, self.config.TEXT_COLOR),
            ("", small_font, self.config.TEXT_COLOR),
            ("Controls", font, self.config.ACCENT_COLOR),
            ("Arrows / WASD: move", small_font, self.config.MUTED_TEXT_COLOR),
            ("T: toggle training", small_font, self.config.MUTED_TEXT_COLOR),
            ("SPACE: pause", small_font, self.config.MUTED_TEXT_COLOR),
            ("S: save model", small_font, self.config.MUTED_TEXT_COLOR),
            ("L: load model", small_font, self.config.MUTED_TEXT_COLOR),
            ("R: reset episode", small_font, self.config.MUTED_TEXT_COLOR),
        ]
        for text, current_font, color in lines:
            if not text:
                y += 8
                continue
            surface = current_font.render(text, True, color)
            screen.blit(surface, (x, y))
            y += surface.get_height() + 8

        self._draw_progress(screen, x, panel_rect.bottom - 92, panel_rect.width - 36)

        if self.game_over:
            overlay = pygame.Surface((self.board_px, self.board_px), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, self.board_origin)
            banner = font.render("GAME OVER", True, self.config.DANGER_COLOR)
            hint = small_font.render("Press R to restart", True, self.config.TEXT_COLOR)
            screen.blit(banner, (self.board_origin[0] + self.board_px // 2 - banner.get_width() // 2, self.board_origin[1] + 120))
            screen.blit(hint, (self.board_origin[0] + self.board_px // 2 - hint.get_width() // 2, self.board_origin[1] + 170))

    def _draw_progress(self, screen: pygame.Surface, x: int, y: int, width: int) -> None:
        bar_height = 16
        outer = pygame.Rect(x, y, width, bar_height)
        inner = outer.inflate(-4, -4)
        pygame.draw.rect(screen, (56, 64, 82), outer, border_radius=8)
        fill_width = int(inner.width * max(0.0, min(1.0, 1.0 - self.config.EPSILON_MIN if self.mode == "train" else 0.0)))
        pygame.draw.rect(screen, self.config.ACCENT_COLOR, pygame.Rect(inner.x, inner.y, fill_width, inner.height), border_radius=6)
        label = f"Training intensity: {fill_width}/{inner.width}"
        text = pygame.font.SysFont("arial", 16).render(label, True, self.config.MUTED_TEXT_COLOR)
        screen.blit(text, (x, y - 24))

    def maybe_restart(self) -> None:
        if self.game_over and self.mode == "train":
            self.history.append(EpisodeResult(score=self.score, steps=self.episode_steps, reward=self.episode_reward, done=True))
            self.update_stats()
            self.reset_episode()

    def update_stats(self) -> None:
        if self.score > 0:
            pass
