"""Evaluate a trained DQN agent on CartPole-v1."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

try:
    from src.model import DuelingQNetwork, QNetwork
except ImportError:
    from model import DuelingQNetwork, QNetwork


def select_greedy_action(
    policy_net: nn.Module,
    state: np.ndarray,
    device: torch.device,
) -> int:
    """Select the action with the largest predicted Q-value."""
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())


def load_model(model_path: Path, device: torch.device, dueling: bool = False) -> nn.Module:
    """Load a trained CartPole QNetwork from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train first with src/train_dqn.py."
        )

    network_cls = DuelingQNetwork if dueling else QNetwork
    model = network_cls(state_dim=4, action_dim=2).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_dqn(
    model_path: Path = Path("models/dqn_cartpole.pth"),
    episodes: int = 10,
    max_steps: int = 500,
    seed: int = 123,
    render: bool = False,
    dueling: bool = False,
) -> list[float]:
    """Evaluate the trained model and return episode rewards."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = load_model(model_path, device, dueling=dueling)
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    rewards: list[float] = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            total_reward = 0.0

            for _ in range(max_steps):
                action = select_greedy_action(policy_net, state, device)
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            rewards.append(total_reward)
            print(f"Test episode {episode + 1:03d} | reward={total_reward:6.1f}")
    finally:
        env.close()

    average_reward = sum(rewards) / len(rewards)
    print(f"Average reward over {episodes} episodes: {average_reward:.1f}")
    return rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN on CartPole-v1")
    parser.add_argument("--model-path", type=Path, default=Path("models/dqn_cartpole.pth"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--dueling", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_dqn(
        model_path=args.model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        render=args.render,
        dueling=args.dueling,
    )
