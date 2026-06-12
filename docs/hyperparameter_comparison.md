# DQN-CartPole 超参数对比实验

日期：2026 年 6 月 12 日

## 一、实验目的

本次对比实验主要观察两个超参数对 DQN-CartPole 训练效果的影响：

- epsilon 衰减速度
- 学习率

对比目标不是穷尽所有组合，而是用少量实验理解这些参数对训练稳定性和收敛速度的影响。

## 二、实验设置

所有实验均使用同一个基础 DQN 结构：

```text
Linear(4 -> 128)
ReLU
Linear(128 -> 128)
ReLU
Linear(128 -> 2)
```

除特别说明外，其他主要参数保持一致：

- gamma：0.99
- batch size：64
- Replay Buffer 容量：10000
- Target Network 同步频率：100 step
- epsilon 最小值：0.05
- 随机种子：42

## 三、实验结果汇总

结果汇总表已保存到：

- `outputs/comparison_summary.csv`

训练曲线对比图已保存到：

- `outputs/comparison_curves.png`

| 实验 | 训练参数 | 训练轮数 | 最高 reward | 最后 20 轮平均 reward | 训练中 >=200 次数 | 10 轮测试平均 reward |
|---|---|---:|---:|---:|---:|---:|
| 推荐参数 | `lr=1e-3, epsilon_decay=0.985` | 335 | 500.0 | 476.15 | 80 | 500.0 |
| epsilon 衰减较慢 | `lr=1e-3, epsilon_decay=0.995` | 300 | 157.0 | 38.70 | 0 | 41.6 |
| 学习率较低 | `lr=1e-4, epsilon_decay=0.985` | 300 | 173.0 | 9.30 | 0 | 9.4 |

## 四、实验分析

### 1. epsilon 衰减速度的影响

使用 `epsilon_decay=0.995` 时，探索率下降较慢。训练到 300 轮时，epsilon 仍约为 0.223，智能体仍有较高概率随机行动。

这种设置有利于持续探索，但在 CartPole 这种较简单的任务中，过长时间的高随机探索会影响策略稳定形成。本次实验中，该设置最高 reward 只有 157，测试平均 reward 为 41.6，没有达到 200 步目标。

相比之下，`epsilon_decay=0.985` 能更快从探索转向利用。推荐参数在第 335 轮时达到最近 20 轮平均 reward 476.15，并在测试中稳定达到 500 分。

### 2. 学习率的影响

使用 `learning_rate=1e-4` 时，训练明显变慢。虽然较低学习率通常有助于稳定更新，但在本任务中，300 轮内网络参数更新幅度不足，策略没有有效学成。

该实验最后 20 轮平均 reward 只有 9.30，测试平均 reward 也只有 9.4，基本接近未学习策略。这说明在当前网络结构和训练轮数下，`1e-4` 学习率偏低。

### 3. 保存最佳模型的重要性

DQN 训练过程中 reward 会出现波动。之前只保存最后一轮模型时，即使训练中曾出现较高 reward，最终测试效果也可能不好。

现在训练脚本改为保存最近若干轮平均 reward 最好的模型，可以避免后期训练退化导致最终模型表现变差。这一点对 DQN 这种不稳定算法尤其重要。

## 五、结论

本次实验说明：

- DQN-CartPole 对 epsilon 衰减速度比较敏感。
- 对当前实现而言，`epsilon_decay=0.985` 明显优于 `0.995`。
- 学习率 `1e-4` 在 300 轮训练内偏低，学习速度不足。
- 当前较合适的基础参数是 `learning_rate=1e-3, epsilon_decay=0.985, target_update_freq=100`。

后续如果继续扩展，可以进一步尝试：

- `learning_rate=5e-4`
- 不同 `target_update_freq`
- 多随机种子重复实验

## 六、Double DQN + Dueling DQN 加分实验

在基础 DQN 完成后，继续实现了两个加分项：

- Double DQN：使用 policy network 选择下一状态动作，再使用 target network 评估该动作的 Q 值，从而降低普通 DQN 中 `max` 操作带来的 Q 值过估计问题。
- Dueling DQN：将网络拆成 value stream 和 advantage stream，再组合得到每个动作的 Q 值，使模型可以分别估计状态价值和动作优势。

训练命令：

```powershell
python src/train_dqn.py --episodes 500 --epsilon-decay 0.985 --target-update-freq 100 --solve-score 475 --solve-window 20 --double-dqn --dueling --log-interval 25 --model-path models/experiments/double_dueling.pth --rewards-path outputs/experiments/double_dueling_rewards.csv --plot-path outputs/experiments/double_dueling_curve.png
```

测试命令：

```powershell
python src/test_dqn.py --model-path models/experiments/double_dueling.pth --dueling --episodes 10
```

实验结果：

| 实验 | 训练轮数 | 最高 reward | 最佳 20 轮平均 reward | 训练中 >=200 次数 | 训练中 500 分次数 | 10 轮测试平均 reward |
|---|---:|---:|---:|---:|---:|---:|
| Double DQN + Dueling DQN | 500 | 500.0 | 425.60 | 138 | 60 | 500.0 |

该加分版本虽然训练曲线仍然存在波动，但保存的最佳模型在 10 轮测试中全部达到 500 分，说明加分模型同样可以完成 CartPole-v1 任务。
