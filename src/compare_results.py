"""Compare reward curves from multiple DQN experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


EXPERIMENTS = [
    ("Basic DQN", Path("outputs/rewards.csv")),
    ("Slow epsilon decay", Path("outputs/experiments/slow_epsilon_rewards.csv")),
    ("Low learning rate", Path("outputs/experiments/low_lr_rewards.csv")),
    ("Double + Dueling DQN", Path("outputs/experiments/double_dueling_rewards.csv")),
]


def load_rewards(path: Path) -> list[float]:
    """Load episode rewards from a CSV file."""
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [float(row["reward"]) for row in reader]


def moving_average(values: list[float], window: int) -> np.ndarray:
    """Return moving average values."""
    if len(values) < window:
        return np.asarray(values, dtype=np.float32)
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def summarize_rewards(name: str, rewards: list[float], window: int) -> dict[str, str]:
    """Create one summary row for an experiment."""
    best_window_average = max(
        sum(rewards[index - window : index]) / window
        for index in range(window, len(rewards) + 1)
    ) if len(rewards) >= window else sum(rewards) / len(rewards)
    last_window_average = (
        sum(rewards[-window:]) / window if len(rewards) >= window else sum(rewards) / len(rewards)
    )

    return {
        "experiment": name,
        "episodes": str(len(rewards)),
        "max_reward": f"{max(rewards):.2f}",
        f"best_{window}_episode_avg": f"{best_window_average:.2f}",
        f"last_{window}_episode_avg": f"{last_window_average:.2f}",
        "episodes_ge_200": str(sum(reward >= 200 for reward in rewards)),
        "episodes_eq_500": str(sum(reward >= 500 for reward in rewards)),
    }


def save_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    """Save comparison summary rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(
    experiment_rewards: list[tuple[str, list[float]]],
    output_path: Path,
    window: int,
) -> None:
    """Plot moving-average reward curves for multiple experiments."""
    import os

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib_config_dir = output_path.parent / ".matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir.resolve()))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5.5))
    for name, rewards in experiment_rewards:
        smoothed = moving_average(rewards, window)
        start_episode = window if len(rewards) >= window else 1
        x_values = range(start_episode, start_episode + len(smoothed))
        plt.plot(x_values, smoothed, label=name, linewidth=2)

    plt.axhline(200, color="gray", linestyle="--", linewidth=1, label="Target: 200")
    plt.axhline(500, color="black", linestyle=":", linewidth=1, label="Max: 500")
    plt.xlabel("Episode")
    plt.ylabel(f"{window}-episode moving average reward")
    plt.title("DQN CartPole Reward Curve Comparison")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compare_results(
    experiments: list[tuple[str, Path]] = EXPERIMENTS,
    summary_path: Path = Path("outputs/comparison_summary.csv"),
    plot_path: Path = Path("outputs/comparison_curves.png"),
    window: int = 20,
) -> list[dict[str, str]]:
    """Create summary CSV and comparison curve plot."""
    experiment_rewards: list[tuple[str, list[float]]] = []
    for name, path in experiments:
        if not path.exists():
            raise FileNotFoundError(f"Missing reward file for {name}: {path}")
        experiment_rewards.append((name, load_rewards(path)))

    rows = [
        summarize_rewards(name=name, rewards=rewards, window=window)
        for name, rewards in experiment_rewards
    ]
    save_summary(rows, summary_path)
    plot_comparison(experiment_rewards, plot_path, window=window)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DQN experiment rewards")
    parser.add_argument("--summary-path", type=Path, default=Path("outputs/comparison_summary.csv"))
    parser.add_argument("--plot-path", type=Path, default=Path("outputs/comparison_curves.png"))
    parser.add_argument("--window", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary_rows = compare_results(
        summary_path=args.summary_path,
        plot_path=args.plot_path,
        window=args.window,
    )
    for row in summary_rows:
        print(row)
