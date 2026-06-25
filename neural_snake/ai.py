from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, NamedTuple

import numpy as np


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        indices = np.random.choice(len(self.memory), size=batch_size, replace=False)
        return [self.memory[index] for index in indices]


class AdamState(NamedTuple):
    m: list[np.ndarray]
    v: list[np.ndarray]
    t: int


class MLPQNetwork:
    def __init__(self, layer_sizes: Iterable[int], seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.layer_sizes = list(layer_sizes)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.adam_state = AdamState(m=[], v=[], t=0)
        self._build()

    def _build(self) -> None:
        self.weights.clear()
        self.biases.clear()
        adam_m: list[np.ndarray] = []
        adam_v: list[np.ndarray] = []
        for input_size, output_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = np.sqrt(6.0 / (input_size + output_size))
            weight = self.rng.uniform(-limit, limit, size=(input_size, output_size)).astype(np.float32)
            bias = np.zeros(output_size, dtype=np.float32)
            self.weights.append(weight)
            self.biases.append(bias)
            adam_m.append(np.zeros_like(weight))
            adam_m.append(np.zeros_like(bias))
            adam_v.append(np.zeros_like(weight))
            adam_v.append(np.zeros_like(bias))
        self.adam_state = AdamState(m=adam_m, v=adam_v, t=0)

    def clone(self) -> "MLPQNetwork":
        other = MLPQNetwork(self.layer_sizes)
        other.weights = [w.copy() for w in self.weights]
        other.biases = [b.copy() for b in self.biases]
        other.adam_state = AdamState(
            m=[m.copy() for m in self.adam_state.m],
            v=[v.copy() for v in self.adam_state.v],
            t=self.adam_state.t,
        )
        return other

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [x]
        pre_activations: list[np.ndarray] = []
        current = x
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = current @ weight + bias
            pre_activations.append(z)
            if index < len(self.weights) - 1:
                current = np.maximum(z, 0.0)
            else:
                current = z
            activations.append(current)
        return activations, pre_activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        activations, _ = self.forward(x.astype(np.float32))
        return activations[-1]

    def train_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
    ) -> float:
        activations, pre_activations = self.forward(states)
        predictions = activations[-1]
        batch_size = states.shape[0]

        chosen_q = predictions[np.arange(batch_size), actions]
        errors = chosen_q - targets
        loss = float(np.mean(errors ** 2))

        grad_output = np.zeros_like(predictions)
        grad_output[np.arange(batch_size), actions] = (2.0 / batch_size) * errors

        self.adam_state = AdamState(
            m=[m.copy() for m in self.adam_state.m],
            v=[v.copy() for v in self.adam_state.v],
            t=self.adam_state.t + 1,
        )
        time_step = self.adam_state.t

        grad = grad_output
        grads_w: list[np.ndarray] = []
        grads_b: list[np.ndarray] = []

        for layer_index in reversed(range(len(self.weights))):
            activation_prev = activations[layer_index]
            grads_w.insert(0, activation_prev.T @ grad)
            grads_b.insert(0, np.sum(grad, axis=0))

            if layer_index > 0:
                grad = grad @ self.weights[layer_index].T
                grad = grad * (pre_activations[layer_index - 1] > 0)

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        updated_m = list(self.adam_state.m)
        updated_v = list(self.adam_state.v)

        for layer_index, (grad_w, grad_b) in enumerate(zip(grads_w, grads_b)):
            weight_m_index = layer_index * 2
            bias_m_index = weight_m_index + 1

            updated_m[weight_m_index] = beta1 * updated_m[weight_m_index] + (1.0 - beta1) * grad_w
            updated_v[weight_m_index] = beta2 * updated_v[weight_m_index] + (1.0 - beta2) * (grad_w ** 2)

            updated_m[bias_m_index] = beta1 * updated_m[bias_m_index] + (1.0 - beta1) * grad_b
            updated_v[bias_m_index] = beta2 * updated_v[bias_m_index] + (1.0 - beta2) * (grad_b ** 2)

            m_hat_w = updated_m[weight_m_index] / (1.0 - beta1 ** time_step)
            v_hat_w = updated_v[weight_m_index] / (1.0 - beta2 ** time_step)
            m_hat_b = updated_m[bias_m_index] / (1.0 - beta1 ** time_step)
            v_hat_b = updated_v[bias_m_index] / (1.0 - beta2 ** time_step)

            self.weights[layer_index] -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + eps)
            self.biases[layer_index] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + eps)

        self.adam_state = AdamState(m=updated_m, v=updated_v, t=time_step)
        return loss

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {f"w{index}": weight for index, weight in enumerate(self.weights)}
        data.update({f"b{index}": bias for index, bias in enumerate(self.biases)})
        data["layer_sizes"] = np.array(self.layer_sizes, dtype=np.int32)
        np.savez_compressed(path, **data)

    def load(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        with np.load(path, allow_pickle=False) as data:
            layer_sizes = data["layer_sizes"].astype(int).tolist()
            if layer_sizes != self.layer_sizes:
                return False
            self.weights = [data[f"w{index}"].astype(np.float32) for index in range(len(self.layer_sizes) - 1)]
            self.biases = [data[f"b{index}"].astype(np.float32) for index in range(len(self.layer_sizes) - 1)]
        return True


@dataclass
class TrainingStats:
    episode: int = 0
    best_score: int = 0
    last_score: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    mean_loss: float = 0.0
    epsilon: float = 1.0


class SnakeAgent:
    def __init__(self, state_size: int, action_size: int, config) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.policy_net = MLPQNetwork([state_size, 128, 128, action_size])
        self.target_net = self.policy_net.clone()
        self.memory = ReplayMemory(config.REPLAY_SIZE)
        self.stats = TrainingStats(epsilon=config.EPSILON_START)
        self.training_steps = 0
        self.best_model_path = config.SAVE_PATH

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.stats.epsilon:
            return int(np.random.randint(self.action_size))
        q_values = self.policy_net.predict(state.astype(np.float32))[0]
        return int(np.argmax(q_values))

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.push(Transition(state, action, reward, next_state, done))

    def train_step(self) -> float | None:
        if len(self.memory) < self.config.TRAIN_START_SIZE:
            return None
        batch_size = min(self.config.BATCH_SIZE, len(self.memory))
        batch = self.memory.sample(batch_size)

        states = np.vstack([item.state for item in batch]).astype(np.float32)
        actions = np.array([item.action for item in batch], dtype=np.int64)
        rewards = np.array([item.reward for item in batch], dtype=np.float32)
        next_states = np.vstack([item.next_state for item in batch]).astype(np.float32)
        dones = np.array([item.done for item in batch], dtype=np.bool_)

        current_q = self.policy_net.predict(states)
        next_policy_q = self.policy_net.predict(next_states)
        next_target_q = self.target_net.predict(next_states)
        best_next_actions = np.argmax(next_policy_q, axis=1)
        target_values = rewards + (1.0 - dones.astype(np.float32)) * self.config.GAMMA * next_target_q[
            np.arange(batch_size), best_next_actions
        ]

        loss = self.policy_net.train_batch(states, actions, target_values, self.config.LR)
        self.training_steps += 1
        self.stats.mean_loss = 0.95 * self.stats.mean_loss + 0.05 * loss if self.stats.mean_loss else loss
        self.stats.epsilon = max(
            self.config.EPSILON_MIN,
            self.stats.epsilon * self.config.EPSILON_DECAY,
        )

        if self.training_steps % self.config.TARGET_SYNC_EVERY == 0:
            self.sync_target_network()
        return loss

    def sync_target_network(self) -> None:
        self.target_net = self.policy_net.clone()

    def save(self, path: str | Path | None = None) -> None:
        self.policy_net.save(path or self.best_model_path)

    def load(self, path: str | Path | None = None) -> bool:
        loaded = self.policy_net.load(path or self.best_model_path)
        if loaded:
            self.sync_target_network()
        return loaded
