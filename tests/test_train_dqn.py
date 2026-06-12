import unittest

import numpy as np
import torch
from torch import nn

from src.train_dqn import compute_dqn_loss, select_action


class FixedQNetwork(nn.Module):
    def __init__(self, q_values: list[float]) -> None:
        super().__init__()
        self.q_values = torch.tensor(q_values, dtype=torch.float32)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.q_values.repeat(states.shape[0], 1)


class SequenceQNetwork(nn.Module):
    def __init__(self, q_values: list[list[float]]) -> None:
        super().__init__()
        self.q_values = torch.tensor(q_values, dtype=torch.float32)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.q_values[: states.shape[0]]


class CallSequenceQNetwork(nn.Module):
    def __init__(self, q_values_by_call: list[list[float]]) -> None:
        super().__init__()
        self.q_values_by_call = [
            torch.tensor(q_values, dtype=torch.float32)
            for q_values in q_values_by_call
        ]
        self.call_index = 0

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        q_values = self.q_values_by_call[self.call_index]
        self.call_index += 1
        return q_values.repeat(states.shape[0], 1)


class TrainDqnTest(unittest.TestCase):
    def test_select_action_uses_greedy_action_when_epsilon_is_zero(self) -> None:
        policy_net = FixedQNetwork([0.2, 1.5])
        state = np.array([0.0, 0.1, -0.2, 0.3], dtype=np.float32)

        action = select_action(
            policy_net=policy_net,
            state=state,
            epsilon=0.0,
            action_dim=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(action, 1)

    def test_select_action_random_action_stays_inside_action_space(self) -> None:
        policy_net = FixedQNetwork([0.2, 1.5])
        state = np.zeros(4, dtype=np.float32)

        actions = {
            select_action(policy_net, state, epsilon=1.0, action_dim=2, device=torch.device("cpu"))
            for _ in range(20)
        }

        self.assertTrue(actions.issubset({0, 1}))

    def test_compute_dqn_loss_returns_scalar_loss(self) -> None:
        policy_net = FixedQNetwork([1.0, 2.0])
        target_net = FixedQNetwork([0.5, 1.5])
        batch = (
            np.zeros((2, 4), dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
            np.array([1.0, 1.0], dtype=np.float32),
            np.ones((2, 4), dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        )

        loss = compute_dqn_loss(
            policy_net=policy_net,
            target_net=target_net,
            batch=batch,
            gamma=0.99,
            device=torch.device("cpu"),
        )

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_double_dqn_uses_policy_network_to_select_next_action(self) -> None:
        policy_net = CallSequenceQNetwork([[0.0, 0.0], [1.0, 3.0]])
        target_net = SequenceQNetwork([[10.0, 1.0]])
        batch = (
            np.zeros((1, 4), dtype=np.float32),
            np.array([0], dtype=np.int64),
            np.array([1.0], dtype=np.float32),
            np.ones((1, 4), dtype=np.float32),
            np.array([0.0], dtype=np.float32),
        )

        loss = compute_dqn_loss(
            policy_net=policy_net,
            target_net=target_net,
            batch=batch,
            gamma=1.0,
            device=torch.device("cpu"),
            double_dqn=True,
        )

        self.assertAlmostEqual(loss.item(), 1.5, places=5)


if __name__ == "__main__":
    unittest.main()
