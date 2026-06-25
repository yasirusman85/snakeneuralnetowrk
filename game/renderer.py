from __future__ import annotations

import pygame

from .agent import SnakeAgent
from .environment import SnakeEnvironment


class SnakeRenderer:
    def __init__(self, environment: SnakeEnvironment) -> None:
        self.environment = environment

    def render(self, screen: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font, agent: SnakeAgent) -> None:
        env = self.environment
        screen.fill(env.config.BG_COLOR)
        board_rect = pygame.Rect(*env.board_origin, env.board_px, env.board_px)
        panel_rect = pygame.Rect(env.board_px + env.config.BOARD_PADDING * 2, 0, env.config.PANEL_WIDTH, env.window_height)
        pygame.draw.rect(screen, env.config.PANEL_BG_COLOR, panel_rect)
        pygame.draw.rect(screen, (42, 48, 62), board_rect, border_radius=12)

        self._draw_grid(screen)
        self._draw_food(screen)
        self._draw_snake(screen)
        self._draw_panel(screen, font, small_font, agent, panel_rect)

    def _draw_grid(self, screen: pygame.Surface) -> None:
        env = self.environment
        ox, oy = env.board_origin
        cell = env.config.CELL_SIZE
        for x in range(env.grid_size + 1):
            px = ox + x * cell
            pygame.draw.line(screen, env.config.GRID_COLOR, (px, oy), (px, oy + env.board_px), 1)
        for y in range(env.grid_size + 1):
            py = oy + y * cell
            pygame.draw.line(screen, env.config.GRID_COLOR, (ox, py), (ox + env.board_px, py), 1)

    def _cell_rect(self, point: tuple[int, int], inset: int = 3) -> pygame.Rect:
        env = self.environment
        ox, oy = env.board_origin
        x, y = point
        size = env.config.CELL_SIZE
        return pygame.Rect(ox + x * size + inset, oy + y * size + inset, size - inset * 2, size - inset * 2)

    def _draw_snake(self, screen: pygame.Surface) -> None:
        env = self.environment
        if not env.snake:
            return
        for index, segment in enumerate(reversed(env.snake)):
            alpha_index = len(env.snake) - 1 - index
            if alpha_index == 0:
                color = env.config.SNAKE_HEAD_COLOR
            elif alpha_index == len(env.snake) - 1:
                color = env.config.SNAKE_TAIL_COLOR
            else:
                blend = alpha_index / max(1, len(env.snake) - 1)
                color = tuple(
                    int(env.config.SNAKE_BODY_COLOR[i] * blend + env.config.SNAKE_HEAD_COLOR[i] * (1.0 - blend))
                    for i in range(3)
                )
            rect = self._cell_rect(segment, inset=2 if alpha_index == 0 else 4)
            pygame.draw.rect(screen, color, rect, border_radius=8)
            if alpha_index == 0:
                pygame.draw.rect(screen, (245, 255, 248), rect, width=2, border_radius=8)

    def _draw_food(self, screen: pygame.Surface) -> None:
        env = self.environment
        if env.food == (-1, -1):
            return
        rect = self._cell_rect(env.food, inset=5)
        pygame.draw.ellipse(screen, env.config.FOOD_COLOR, rect)
        pygame.draw.ellipse(screen, (255, 220, 140), rect.inflate(8, 8), width=2)

    def _draw_panel(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        small_font: pygame.font.Font,
        agent: SnakeAgent,
        panel_rect: pygame.Rect,
    ) -> None:
        env = self.environment
        x = panel_rect.x + 18
        y = 24
        lines = [
            ("NEURAL SNAKE", font, env.config.ACCENT_COLOR),
            (f"Mode: {env.mode.upper()}", small_font, env.config.TEXT_COLOR),
            (f"Episode: {agent.stats.episode}", small_font, env.config.TEXT_COLOR),
            (f"Score: {env.score}", small_font, env.config.TEXT_COLOR),
            (f"Best: {agent.stats.best_score}", small_font, env.config.TEXT_COLOR),
            (f"Steps: {env.episode_steps}", small_font, env.config.TEXT_COLOR),
            (f"Reward: {env.episode_reward:.2f}", small_font, env.config.TEXT_COLOR),
            (f"Epsilon: {agent.stats.epsilon:.3f}", small_font, env.config.TEXT_COLOR),
            (f"Memory: {len(agent.memory)}", small_font, env.config.TEXT_COLOR),
            (f"Loss: {agent.stats.mean_loss:.5f}", small_font, env.config.TEXT_COLOR),
            ("", small_font, env.config.TEXT_COLOR),
            ("Controls", font, env.config.ACCENT_COLOR),
            ("Arrows / WASD: move", small_font, env.config.MUTED_TEXT_COLOR),
            ("T: toggle training", small_font, env.config.MUTED_TEXT_COLOR),
            ("SPACE: pause", small_font, env.config.MUTED_TEXT_COLOR),
            ("S: save model", small_font, env.config.MUTED_TEXT_COLOR),
            ("L: load model", small_font, env.config.MUTED_TEXT_COLOR),
            ("R: reset episode", small_font, env.config.MUTED_TEXT_COLOR),
        ]
        for text, current_font, color in lines:
            if not text:
                y += 8
                continue
            surface = current_font.render(text, True, color)
            screen.blit(surface, (x, y))
            y += surface.get_height() + 8

        self._draw_progress(screen, x, panel_rect.bottom - 92, panel_rect.width - 36, agent)

        if env.game_over:
            overlay = pygame.Surface((env.board_px, env.board_px), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, env.board_origin)
            banner = font.render("GAME OVER", True, env.config.DANGER_COLOR)
            hint = small_font.render("Press R to restart", True, env.config.TEXT_COLOR)
            screen.blit(banner, (env.board_origin[0] + env.board_px // 2 - banner.get_width() // 2, env.board_origin[1] + 120))
            screen.blit(hint, (env.board_origin[0] + env.board_px // 2 - hint.get_width() // 2, env.board_origin[1] + 170))

    def _draw_progress(self, screen: pygame.Surface, x: int, y: int, width: int, agent: SnakeAgent) -> None:
        env = self.environment
        bar_height = 16
        outer = pygame.Rect(x, y, width, bar_height)
        inner = outer.inflate(-4, -4)
        pygame.draw.rect(screen, (56, 64, 82), outer, border_radius=8)
        progress = (env.config.EPSILON_START - agent.stats.epsilon) / max(1e-6, env.config.EPSILON_START - env.config.EPSILON_MIN) if env.mode == "train" else 0.0
        progress = max(0.0, min(1.0, progress))
        pygame.draw.rect(screen, env.config.ACCENT_COLOR, pygame.Rect(inner.x, inner.y, int(inner.width * progress), inner.height), border_radius=6)
        text = pygame.font.SysFont("arial", 16).render(f"Training progress: {int(progress * 100)}%", True, env.config.MUTED_TEXT_COLOR)
        screen.blit(text, (x, y - 24))
