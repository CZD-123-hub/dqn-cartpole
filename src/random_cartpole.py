"""Run CartPole-v1 with a random policy.

This is the first runnable step of the project. It helps verify that
Gymnasium is installed correctly and that we understand the basic
environment interaction loop:

    state -> action -> next_state, reward, done
"""

from __future__ import annotations

import gymnasium as gym


def run_random_policy(episodes: int = 5, max_steps: int = 500) -> None:
    """Run a random policy on CartPole and print episode rewards."""
    env = gym.make("CartPole-v1")

    try:
        for episode in range(1, episodes + 1):
            state, info = env.reset()
            total_reward = 0.0

            for step in range(1, max_steps + 1):
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                state = next_state
                done = terminated or truncated

                if done:
                    break

            print(
                f"Episode {episode}: reward={total_reward:.1f}, "
                f"steps={step}, final_state={state}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    run_random_policy()
