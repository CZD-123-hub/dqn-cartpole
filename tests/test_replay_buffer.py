import unittest

import numpy as np

from src.replay_buffer import ReplayBuffer


class ReplayBufferTest(unittest.TestCase):
    def test_keeps_only_latest_transitions_when_capacity_is_full(self) -> None:
        buffer = ReplayBuffer(capacity=2)

        buffer.push([0, 0, 0, 0], 0, 1.0, [1, 1, 1, 1], False)
        buffer.push([1, 1, 1, 1], 1, 1.0, [2, 2, 2, 2], False)
        buffer.push([2, 2, 2, 2], 0, 1.0, [3, 3, 3, 3], True)

        self.assertEqual(len(buffer), 2)
        states, actions, rewards, next_states, dones = buffer.sample(batch_size=2)

        self.assertEqual(states.shape, (2, 4))
        self.assertNotIn([0.0, 0.0, 0.0, 0.0], states.tolist())
        self.assertEqual(actions.shape, (2,))
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(next_states.shape, (2, 4))
        self.assertEqual(dones.shape, (2,))

    def test_sample_returns_numpy_arrays_for_training_batch(self) -> None:
        buffer = ReplayBuffer(capacity=10)

        for index in range(4):
            state = np.array([index, index + 1, index + 2, index + 3], dtype=np.float32)
            next_state = state + 1
            buffer.push(state, index % 2, 1.0, next_state, index == 3)

        states, actions, rewards, next_states, dones = buffer.sample(batch_size=3)

        self.assertEqual(states.dtype, np.float32)
        self.assertEqual(actions.dtype, np.int64)
        self.assertEqual(rewards.dtype, np.float32)
        self.assertEqual(next_states.dtype, np.float32)
        self.assertEqual(dones.dtype, np.float32)
        self.assertEqual(states.shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
