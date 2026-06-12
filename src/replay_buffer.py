"""Experience replay buffer for DQN.

DQN does not train directly from only the latest transition. Instead, it stores
many past transitions and randomly samples a small batch during training.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transition:
    """One interaction record: state, action, reward, next_state, done."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """A fixed-size buffer that stores and samples DQN transitions."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: np.ndarray | list[float],
        action: int,
        reward: float,
        next_state: np.ndarray | list[float],
        done: bool,
    ) -> None:
        """Store one transition in the buffer."""
        transition = Transition(
            state=np.asarray(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32),
            done=bool(done),
        )
        self.buffer.append(transition)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Randomly sample a batch and return arrays for DQN training."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self.buffer):
            raise ValueError("batch_size cannot be larger than current buffer size")

        batch = random.sample(self.buffer, batch_size)

        states = np.asarray([transition.state for transition in batch], dtype=np.float32)
        actions = np.asarray([transition.action for transition in batch], dtype=np.int64)
        rewards = np.asarray([transition.reward for transition in batch], dtype=np.float32)
        next_states = np.asarray(
            [transition.next_state for transition in batch], dtype=np.float32
        )
        dones = np.asarray([transition.done for transition in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones
