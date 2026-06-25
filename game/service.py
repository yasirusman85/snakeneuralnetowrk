from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agent import SnakeAgent
from .config import Config
from .environment import EpisodeResult, SnakeEnvironment


@dataclass
class SnakeGameService:
    config: Config

    def __post_init__(self) -> None:
        self.environment = SnakeEnvironment(self.config, mode="train")
        self.agent = SnakeAgent(self.environment.state_size, self.environment.action_size, self.config)
        self.renderer: Any = None
        try:
            from .renderer import SnakeRenderer

            self.renderer = SnakeRenderer(self.environment)
        except Exception:
            self.renderer = None
        self.agent.load(self.config.SAVE_PATH)

    def snapshot(self) -> dict[str, object]:
        env = self.environment
        return {
            "mode": env.mode,
            "score": env.score,
            "best_score": self.agent.stats.best_score,
            "epsilon": self.agent.stats.epsilon,
            "training_steps": self.agent.training_steps,
            "memory_size": len(self.agent.memory),
        }

    def reset(self) -> None:
        self.environment.reset_episode()

    def step_training(self) -> EpisodeResult:
        env = self.environment
        state = env.get_state()
        action = self.agent.act(state, training=True)
        result = env.step(action)
        next_state = env.get_state()
        self.agent.remember(state, action, result.reward, next_state, result.done)
        self.agent.train_step()
        self.agent.stats.total_steps += 1
        self.agent.stats.total_reward += result.reward
        self.agent.stats.last_score = env.score
        self.agent.stats.best_score = max(self.agent.stats.best_score, env.score)
        if result.done:
            self.agent.stats.episode += 1
            self.environment.history.append(EpisodeResult(score=env.score, steps=env.episode_steps, reward=env.episode_reward, done=True))
            self.environment.update_stats()
            self.environment.reset_episode()
        return result

    def train(self, episodes: int, render: bool = False) -> None:
        _ = render
        for _episode in range(episodes):
            while not self.environment.game_over:
                self.step_training()
            self.environment.reset_episode()

    def step_test(self) -> EpisodeResult:
        """Run one step in greedy (test) mode: no learning, no exploration."""
        env = self.environment
        state = env.get_state()
        action = self.agent.act(state, training=False)
        result = env.step(action)
        if result.done:
            self.agent.stats.episode += 1
            self.agent.stats.best_score = max(self.agent.stats.best_score, env.score)
        return result

    def save(self) -> None:
        self.agent.save(self.config.SAVE_PATH)

    def load(self) -> bool:
        return self.agent.load(self.config.SAVE_PATH)
