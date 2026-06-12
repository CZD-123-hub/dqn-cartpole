# DQN CartPole

研0练手项目：使用 Deep Q-Network 解决 Gymnasium 的 `CartPole-v1` 问题。

## 当前进度

- [x] 创建基础项目结构
- [x] 跑通随机策略 `src/random_cartpole.py`
- [x] 实现经验回放池 `src/replay_buffer.py`
- [x] 实现 Q 网络 `src/model.py`
- [x] 实现 DQN 训练 `src/train_dqn.py`
- [x] 实现模型测试 `src/test_dqn.py`
- [x] 绘制 reward 曲线并整理实验记录

## 运行随机策略

在 PyCharm 终端中确认已经进入虚拟环境：

```powershell
(.venv) PS D:\dqn-cartpole>
```

然后运行：

```powershell
python src/random_cartpole.py
```

随机策略不会真正学习，只用于确认 CartPole 环境可以正常 reset、step 和 close。

## 运行 Replay Buffer 测试

```powershell
python -m unittest tests.test_replay_buffer
```

Replay Buffer 用来保存智能体和环境交互产生的经验：

```text
(state, action, reward, next_state, done)
```

DQN 训练时会从 Replay Buffer 中随机抽取一批经验，减少连续样本之间的相关性，让神经网络训练更稳定。

## 运行 DQN 训练

先用少量 episode 做快速验证：

```powershell
python src/train_dqn.py --episodes 3 --max-steps 20 --batch-size 4 --target-update-freq 10
```

正式训练可以使用默认参数：

```powershell
python src/train_dqn.py
```

当前推荐训练命令：

```powershell
python src/train_dqn.py --episodes 500 --epsilon-decay 0.985 --target-update-freq 100 --solve-score 475 --solve-window 20
```

训练脚本会输出：

- `models/dqn_cartpole.pth`：训练后的模型参数
- `outputs/rewards.csv`：每轮 episode 的 reward
- `outputs/reward_curve.png`：reward 曲线图

## 测试训练后的模型

```powershell
python src/test_dqn.py --episodes 10
```

当前训练模型在 10 轮测试中平均 reward 为 `500.0`。

## 实验记录

- `docs/experiment_record.md`：基础 DQN 训练和测试记录
- `docs/hyperparameter_comparison.md`：学习率和 epsilon 衰减对比实验
- `outputs/comparison_summary.csv`：多组实验结果汇总表
- `outputs/comparison_curves.png`：多组实验 reward 曲线对比图

生成对比图和汇总表：

```powershell
python src/compare_results.py
```

## 加分项

训练 Double DQN + Dueling DQN：

```powershell
python src/train_dqn.py --episodes 500 --epsilon-decay 0.985 --target-update-freq 100 --double-dqn --dueling --model-path models/experiments/double_dueling.pth --rewards-path outputs/experiments/double_dueling_rewards.csv --plot-path outputs/experiments/double_dueling_curve.png
```

测试 Dueling 模型时需要加上 `--dueling`：

```powershell
python src/test_dqn.py --model-path models/experiments/double_dueling.pth --dueling --episodes 10
```

如果想显示 CartPole 窗口，可以加上 `--render`：

```powershell
python src/test_dqn.py --episodes 3 --render
```
