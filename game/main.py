from __future__ import annotations

import sys

import pygame

from .config import Config
from .service import SnakeGameService
from .renderer import SnakeRenderer


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Neural Snake")

    config = Config()
    service = SnakeGameService(config)
    if service.renderer is None:
        service.renderer = SnakeRenderer(service.environment)
    env = service.environment
    screen = pygame.display.set_mode((env.window_width, env.window_height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arialblack", 28)
    small_font = pygame.font.SysFont("arial", 18)

    running = True
    mode = "train"  # "train" or "test"
    paused = False

    while running:
        fps = config.TRAIN_FPS if mode == "train" else config.FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_t:
                    # Cycle: train <-> test
                    mode = "test" if mode == "train" else "train"
                    service.reset()
                elif event.key == pygame.K_r:
                    service.reset()
                elif event.key == pygame.K_s:
                    service.save()
                elif event.key == pygame.K_l:
                    service.load()

        if not paused:
            if mode == "train":
                service.step_training()
            elif mode == "test":
                if not service.environment.game_over:
                    service.step_test()

        service.renderer.render(screen, font, small_font, service.agent)
        pygame.display.set_caption(
            f"Neural Snake | {'TRAIN' if training_mode else 'HUMAN'} | Score {service.environment.score} | Best {service.agent.stats.best_score}"
        )
        pygame.display.flip()
        clock.tick(fps)

    service.save()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
