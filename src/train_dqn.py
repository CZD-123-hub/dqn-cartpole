"""Train a basic DQN agent on CartPole-v1."""

from __future__ import annotations

import argparse
import copy
import csv
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

try:
    from src.model import QNetwork
    from src.replay_buffer import ReplayBuffer
except ImportError:
    from model import QNetwork
    from replay_buffer import ReplayBuffer


def set_seed(seed: int) -> None:
    """Set random seeds for repeatable small experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_epsilon(
    episode: int,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
) -> float:
    """Return exponentially decayed epsilon for epsilon-greedy exploration."""
    return max(epsilon_end, epsilon_start * (epsilon_decay**episode))


def select_action(
    policy_net: nn.Module,
    state: np.ndarray,
    epsilon: float,
    action_dim: int,
    device: torch.device,
) -> int:
    """Choose an action using epsilon-greedy exploration."""
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())


def compute_dqn_loss(
    policy_net: nn.Module,
    target_net: nn.Module,
    batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute the DQN Bellman loss for one sampled batch."""
    states, actions, rewards, next_states, dones = batch

    states_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
    actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=device)
    dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device)

    current_q_values = policy_net(states_tensor).gather(
        dim=1, index=actions_tensor.unsqueeze(1)
    )
    current_q_values = current_q_values.squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states_tensor).max(dim=1).values
        target_q_values = rewards_tensor + gamma * next_q_values * (1.0 - dones_tensor)

    return nn.SmoothL1Loss()(current_q_values, target_q_values)


def save_rewards(rewards: list[float], output_path: Path) -> None:
    """Save episode rewards to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward"])
        for episode, reward in enumerate(rewards, start=1):
            writer.writerow([episode, reward])


def plot_rewards(rewards: list[float], output_path: Path) -> None:
    """Plot and save the reward curve."""
    import os

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib_config_dir = output_path.parent / ".matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir.resolve()))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.5))
    plt.plot(rewards, label="Episode reward")
    if len(rewards) >= 10:
        moving_average = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        plt.plot(
            range(9, len(rewards)),
            moving_average,
            label="10-episode moving average",
        )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training Reward on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_dqn(
    episodes: int = 300,
    max_steps: int = 500,
    batch_size: int = 64,
    gamma: float = 0.99,
    learning_rate: float = 1e-3,
    buffer_capacity: int = 10000,
    target_update_freq: int = 200,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    solve_score: float = 475.0,
    solve_window: int = 20,
    log_interval: int = 1,
    seed: int = 42,
    model_path: Path = Path("models/dqn_cartpole.pth"),
    rewards_path: Path = Path("outputs/rewards.csv"),
    plot_path: Path = Path("outputs/reward_curve.png"),
) -> list[float]:
    """Train a basic DQN agent and return episode rewards."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)

    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    policy_net = QNetwork(state_dim=state_dim, action_dim=action_dim).to(device)
    target_net = QNetwork(state_dim=state_dim, action_dim=action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    episode_rewards: list[float] = []
    global_step = 0
    best_average_reward = float("-inf")
    best_state_dict = copy.deepcopy(policy_net.state_dict())

    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            epsilon = get_epsilon(
                episode=episode,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
            )

            for _ in range(max_steps):
                action = select_action(policy_net, state, epsilon, action_dim, device)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                global_step += 1

                if len(replay_buffer) >= batch_size:
                    batch = replay_buffer.sample(batch_size)
                    loss = compute_dqn_loss(
                        policy_net=policy_net,
                        target_net=target_net,
                        batch=batch,
                        gamma=gamma,
                        device=device,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
                    optimizer.step()

                if global_step % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                if done:
                    break

            episode_rewards.append(total_reward)
            recent_average_reward = (
                sum(episode_rewards[-solve_window:]) / solve_window
                if len(episode_rewards) >= solve_window
                else total_reward
            )
            if recent_average_reward > best_average_reward:
                best_average_reward = recent_average_reward
                best_state_dict = copy.deepcopy(policy_net.state_dict())

            should_log = (
                log_interval > 0
                and ((episode + 1) % log_interval == 0 or episode == 0)
            )
            if should_log:
                print(
                    f"Episode {episode + 1:03d} | "
                    f"reward={total_reward:6.1f} | "
                    f"avg={recent_average_reward:6.1f} | "
                    f"epsilon={epsilon:.3f}"
                )
            if len(episode_rewards) >= solve_window and recent_average_reward >= solve_score:
                print(
                    f"Solved with {solve_window}-episode average reward "
                    f"{recent_average_reward:.1f}"
                )
                break
    finally:
        env.close()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state_dict, model_path)
    save_rewards(episode_rewards, rewards_path)
    plot_rewards(episode_rewards, plot_path)
    return episode_rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on CartPole-v1")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--buffer-capacity", type=int, default=10000)
    parser.add_argument("--target-update-freq", type=int, default=200)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--solve-score", type=float, default=475.0)
    parser.add_argument("--solve-window", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=Path("models/dqn_cartpole.pth"))
    parser.add_argument("--rewards-path", type=Path, default=Path("outputs/rewards.csv"))
    parser.add_argument("--plot-path", type=Path, default=Path("outputs/reward_curve.png"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dqn(
        episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        buffer_capacity=args.buffer_capacity,
        target_update_freq=args.target_update_freq,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        solve_score=args.solve_score,
        solve_window=args.solve_window,
        log_interval=args.log_interval,
        seed=args.seed,
        model_path=args.model_path,
        rewards_path=args.rewards_path,
        plot_path=args.plot_path,
    )
