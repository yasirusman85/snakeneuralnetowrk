from __future__ import annotations

import sys

import pygame

from neural_snake.ai import SnakeAgent
from neural_snake.config import Config
from neural_snake.game import EpisodeResult, SnakeGame


def build_window(game: SnakeGame) -> pygame.Surface:
    return pygame.display.set_mode((game.window_width, game.window_height))


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Neural Snake")
    config = Config()
    game = SnakeGame(config=config, mode="train")
    agent = SnakeAgent(game.state_size, game.action_size, config)

    screen = build_window(game)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arialblack", 28)
    small_font = pygame.font.SysFont("arial", 18)
    running = True
    training_mode = True
    paused = False

    agent.load(config.SAVE_PATH)

    while running:
        fps = config.TRAIN_FPS if training_mode else config.FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_t:
                    training_mode = not training_mode
                    game.mode = "train" if training_mode else "human"
                    game.paused = False
                elif event.key == pygame.K_r:
                    game.reset_episode()
                elif event.key == pygame.K_s:
                    agent.save(config.SAVE_PATH)
                elif event.key == pygame.K_l:
                    agent.load(config.SAVE_PATH)
                elif not training_mode:
                    game.set_direction_from_key(event.key)

        if not paused:
            if training_mode:
                state = game.get_state()
                action = agent.act(state, training=True)
                result = game.step(action)
                next_state = game.get_state()
                agent.remember(state, action, result.reward, next_state, result.done)
                agent.train_step()

                previous_best = agent.stats.best_score
                agent.stats.total_steps += 1
                agent.stats.total_reward += result.reward
                agent.stats.last_score = game.score
                agent.stats.best_score = max(previous_best, game.score)
                if result.done:
                    agent.stats.episode += 1
                    if game.score > previous_best:
                        agent.save(config.SAVE_PATH)
                    game.history.append(
                        EpisodeResult(
                            score=game.score,
                            steps=game.episode_steps,
                            reward=game.episode_reward,
                            done=True,
                        )
                    )
                    game.update_stats()
                    game.reset_episode()
            else:
                if not game.game_over:
                    game.update_human()
                    agent.stats.last_score = game.score
                    agent.stats.best_score = max(agent.stats.best_score, game.score)

        game.render(screen, font, small_font, agent)
        title = f"Neural Snake | {'TRAIN' if training_mode else 'HUMAN'} | Score {game.score} | Best {agent.stats.best_score}"
        pygame.display.set_caption(title)
        pygame.display.flip()
        clock.tick(fps)

    agent.save(config.SAVE_PATH)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
