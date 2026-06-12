# DQN-CartPole 实验记录

日期：2026 年 6 月 12 日

## 一、实验目标

本实验目标是使用基础 DQN 算法训练智能体解决 Gymnasium 中的 `CartPole-v1` 任务。目标效果是训练后的智能体能够稳定保持杆子平衡 200 步以上。

## 二、实验环境

- 操作系统：Windows
- Python：3.12.10
- IDE：PyCharm
- 深度学习框架：PyTorch CPU
- 强化学习环境：Gymnasium
- 项目路径：`D:\dqn-cartpole`

## 三、模型结构

当前 QNetwork 使用简单多层感知机：

```text
Linear(4 -> 128)
ReLU
Linear(128 -> 128)
ReLU
Linear(128 -> 2)
```

输入为 CartPole 的 4 维状态，输出为两个动作对应的 Q 值。

## 四、训练设置

训练命令：

```powershell
python src/train_dqn.py --episodes 500 --epsilon-decay 0.985 --target-update-freq 100 --solve-score 475 --solve-window 20
```

主要参数：

- 最大训练轮数：500 episodes
- 折扣因子：`gamma = 0.99`
- 学习率：`1e-3`
- batch size：64
- Replay Buffer 容量：10000
- Target Network 同步频率：100 step
- epsilon 初始值：1.0
- epsilon 最小值：0.05
- epsilon 衰减率：0.985
- 提前停止条件：最近 20 轮平均 reward 达到 475

## 五、训练结果

训练在第 335 个 episode 提前停止。

关键结果：

- 最高单轮 reward：500
- 最近 20 轮平均 reward：476.15
- reward 达到 200 以上的 episode 数：80
- reward 达到 500 的 episode 数：43

训练输出文件：

- `models/dqn_cartpole.pth`
- `outputs/rewards.csv`
- `outputs/reward_curve.png`

## 六、测试结果

测试命令：

```powershell
python src/test_dqn.py --episodes 10
```

测试结果：

```text
Test episode 001 | reward= 500.0
Test episode 002 | reward= 500.0
Test episode 003 | reward= 500.0
Test episode 004 | reward= 500.0
Test episode 005 | reward= 500.0
Test episode 006 | reward= 500.0
Test episode 007 | reward= 500.0
Test episode 008 | reward= 500.0
Test episode 009 | reward= 500.0
Test episode 010 | reward= 500.0
Average reward over 10 episodes: 500.0
```

测试结果说明，当前训练得到的模型已经能够稳定完成 CartPole-v1 任务，并超过“持续平衡 200 步以上”的项目目标。

## 七、阶段性分析

第一次使用默认参数训练 300 episodes 时，训练过程中曾出现超过 200 的 episode，但最终保存的是最后一轮模型，测试平均 reward 只有 86.5，说明 DQN 训练存在不稳定问题。

之后对训练脚本进行了两点改进：

1. 暴露 epsilon 衰减参数，方便调节探索率。
2. 保存最近若干轮平均 reward 最好的模型，而不是简单保存最后一轮模型。

改进后，使用更快的 epsilon 衰减和更频繁的 Target Network 同步，训练效果明显提升。最终模型在 10 轮测试中全部达到 500 分。

## 八、后续可改进方向

已经进一步完成了一个小规模超参数对比实验，记录见：

- `docs/hyperparameter_comparison.md`

对比结论：

- `learning_rate=1e-3, epsilon_decay=0.985` 效果最好，测试平均 reward 达到 500.0。
- `epsilon_decay=0.995` 衰减较慢，300 轮训练后测试平均 reward 只有 41.6。
- `learning_rate=1e-4` 学习速度太慢，300 轮训练后测试平均 reward 只有 9.4。

后续可以继续补充以下内容作为加分项：

- 对比不同学习率对训练曲线的影响。
- 对比不同 epsilon 衰减速度对探索和收敛的影响。
- 实现 Double DQN，降低 Q 值过估计问题。
- 实现 Dueling DQN，拆分 state value 和 action advantage。
- 多次随机种子实验，观察训练稳定性。
