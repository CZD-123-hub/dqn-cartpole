import unittest

import torch

from src.model import DuelingQNetwork, QNetwork


class QNetworkTest(unittest.TestCase):
    def test_forward_returns_q_values_for_each_action(self) -> None:
        model = QNetwork(state_dim=4, action_dim=2)
        states = torch.zeros((3, 4), dtype=torch.float32)

        q_values = model(states)

        self.assertEqual(q_values.shape, (3, 2))

    def test_forward_accepts_single_state_batch(self) -> None:
        model = QNetwork(state_dim=4, action_dim=2)
        state = torch.tensor([[0.0, 0.1, -0.2, 0.3]], dtype=torch.float32)

        q_values = model(state)

        self.assertEqual(q_values.shape, (1, 2))
        self.assertEqual(q_values.dtype, torch.float32)


class DuelingQNetworkTest(unittest.TestCase):
    def test_forward_returns_q_values_for_each_action(self) -> None:
        model = DuelingQNetwork(state_dim=4, action_dim=2)
        states = torch.zeros((3, 4), dtype=torch.float32)

        q_values = model(states)

        self.assertEqual(q_values.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
