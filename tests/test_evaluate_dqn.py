import unittest

import numpy as np
import torch
from torch import nn

from src.test_dqn import select_greedy_action


class FixedQNetwork(nn.Module):
    def __init__(self, q_values: list[float]) -> None:
        super().__init__()
        self.q_values = torch.tensor(q_values, dtype=torch.float32)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.q_values.repeat(states.shape[0], 1)


class EvaluateDqnTest(unittest.TestCase):
    def test_select_greedy_action_returns_action_with_largest_q_value(self) -> None:
        policy_net = FixedQNetwork([0.1, 2.0])
        state = np.array([0.0, 0.1, -0.1, 0.2], dtype=np.float32)

        action = select_greedy_action(
            policy_net=policy_net,
            state=state,
            device=torch.device("cpu"),
        )

        self.assertEqual(action, 1)


if __name__ == "__main__":
    unittest.main()
