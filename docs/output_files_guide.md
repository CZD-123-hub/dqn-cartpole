# outputs 目录文件说明

日期：2026 年 6 月 12 日

`outputs/` 目录用于保存训练过程产生的实验结果，包括 reward 原始数据、reward 曲线图、不同实验的对比图和汇总表。

## 一、根目录文件

### 1. `outputs/rewards.csv`

含义：基础 DQN 推荐参数训练时，每个 episode 对应的 reward。

来源命令：

```powershell
python src/train_dqn.py --episodes 500 --epsilon-decay 0.985 --target-update-freq 100 --solve-score 475 --solve-window 20
```

文件格式：

```text
episode,reward
1,8.0
2,22.0
...
```

关键结果：

- 训练轮数：335
- 最高 reward：500.0
- 最后 20 轮平均 reward：476.15
- reward >= 200 的 episode 数：80
- reward = 500 的 episode 数：43

用途：这是基础 DQN 的主要训练数据，可以用 Excel 打开，也可以用于重新绘制训练曲线。

### 2. `outputs/reward_curve.png`

含义：基础 DQN 推荐参数训练过程的 reward 曲线。

图中内容：

- 横轴：episode
- 纵轴：reward
- 蓝色曲线：每一轮 episode 的 reward
- 平滑曲线：10 轮滑动平均 reward

用途：这是报告中最重要的单模型训练曲线，用来展示基础 DQN 从随机策略逐渐学到高 reward 策略的过程。

建议报告说明：

> 从 reward 曲线可以看出，随着训练进行，智能体逐渐学会保持杆子平衡。最终训练提前停止时，最近 20 轮平均 reward 达到 476.15，已经超过项目要求的 200 步目标。

### 3. `outputs/comparison_summary.csv`

含义：多组实验结果汇总表。

包含实验：

- Basic DQN
- Slow epsilon decay
- Low learning rate
- Double + Dueling DQN

字段说明：

- `experiment`：实验名称
- `episodes`：训练轮数
- `max_reward`：训练中的最高 reward
- `best_20_episode_avg`：最佳 20 轮平均 reward
- `last_20_episode_avg`：最后 20 轮平均 reward
- `episodes_ge_200`：reward 大于等于 200 的 episode 数
- `episodes_eq_500`：reward 达到 500 的 episode 数

当前内容：

```text
Basic DQN,335,500.00,476.15,476.15,80,43
Slow epsilon decay,300,157.00,42.40,38.70,0,0
Low learning rate,300,173.00,48.55,9.30,0,0
Double + Dueling DQN,500,500.00,425.60,88.65,138,60
```

用途：适合直接放进实验报告或答辩 PPT，用来对比不同参数和算法版本的效果。

### 4. `outputs/comparison_curves.png`

含义：多组实验 reward 曲线对比图。

图中内容：

- Basic DQN
- Slow epsilon decay
- Low learning rate
- Double + Dueling DQN
- 200 分目标线
- 500 分满分线

曲线使用 20 轮滑动平均 reward，因此比原始 reward 更平滑，更适合展示总体趋势。

用途：这是最适合放在“超参数对比 / 加分实验”部分的图。它能直观看出推荐参数和加分模型明显优于慢 epsilon 衰减、低学习率两组实验。

建议报告说明：

> 对比图显示，较慢 epsilon 衰减和较低学习率都没有达到 200 分目标；推荐参数的基础 DQN 和 Double + Dueling DQN 均能学到较好策略，其中基础 DQN 最后 20 轮平均 reward 达到 476.15。

## 二、experiments 子目录文件

`outputs/experiments/` 保存不同实验分组的 reward 数据和曲线图。

### 1. `outputs/experiments/slow_epsilon_rewards.csv`

含义：epsilon 衰减较慢实验的 reward 数据。

训练设置：

```text
learning_rate = 1e-3
epsilon_decay = 0.995
```

关键结果：

- 训练轮数：300
- 最高 reward：157.0
- 最后 20 轮平均 reward：38.70
- reward >= 200 的 episode 数：0
- reward = 500 的 episode 数：0

结论：epsilon 衰减太慢时，智能体在较长时间内仍保持较高随机探索概率，策略难以稳定形成。

### 2. `outputs/experiments/slow_epsilon_curve.png`

含义：epsilon 衰减较慢实验的训练曲线。

用途：用于说明探索率衰减过慢会影响收敛效果。该曲线可以和基础 DQN 曲线对比。

### 3. `outputs/experiments/low_lr_rewards.csv`

含义：低学习率实验的 reward 数据。

训练设置：

```text
learning_rate = 1e-4
epsilon_decay = 0.985
```

关键结果：

- 训练轮数：300
- 最高 reward：173.0
- 最后 20 轮平均 reward：9.30
- reward >= 200 的 episode 数：0
- reward = 500 的 episode 数：0

结论：学习率过低时，网络参数更新太慢，300 轮内没有学到有效策略。

### 4. `outputs/experiments/low_lr_curve.png`

含义：低学习率实验的训练曲线。

用途：用于展示学习率过低导致训练速度慢、后期表现接近随机策略。

### 5. `outputs/experiments/double_dueling_rewards.csv`

含义：Double DQN + Dueling DQN 加分实验的 reward 数据。

训练设置：

```text
double_dqn = True
dueling = True
learning_rate = 1e-3
epsilon_decay = 0.985
target_update_freq = 100
```

关键结果：

- 训练轮数：500
- 最高 reward：500.0
- 最佳 20 轮平均 reward：425.60
- 最后 20 轮平均 reward：88.65
- reward >= 200 的 episode 数：138
- reward = 500 的 episode 数：60

结论：Double + Dueling 版本能够学到高分策略，保存的最佳模型在 10 轮测试中平均 reward 为 500.0。不过训练曲线仍然有波动，说明强化学习训练稳定性仍然需要关注。

### 6. `outputs/experiments/double_dueling_curve.png`

含义：Double DQN + Dueling DQN 的训练曲线。

用途：用于展示加分模型的训练过程。可以和基础 DQN 曲线一起对比，说明加分版本也能完成任务。

### 7. `outputs/experiments/smoke_double_dueling_rewards.csv`

含义：Double + Dueling 功能的短训练验证数据。

训练设置：

```text
episodes = 3
max_steps = 20
batch_size = 4
```

关键结果：

- 训练轮数：3
- 最高 reward：19.0
- 最后 3 轮平均 reward：14.67

用途：这是功能 smoke test 产物，只用于确认代码能运行，不用于正式实验分析。

### 8. `outputs/experiments/smoke_double_dueling_curve.png`

含义：Double + Dueling 短训练验证曲线。

用途：仅用于开发验证，不建议放入正式报告。

## 三、Matplotlib 缓存目录

### `outputs/.matplotlib/`

### `outputs/experiments/.matplotlib/`

含义：Matplotlib 生成图片时产生的字体缓存目录。

用途：不属于实验结果，不需要放入报告，也不需要上传 GitHub。项目 `.gitignore` 已经忽略这些目录。

## 四、报告中推荐使用哪些文件

建议正式报告中重点使用：

1. `outputs/reward_curve.png`
2. `outputs/comparison_curves.png`
3. `outputs/comparison_summary.csv`
4. `outputs/experiments/double_dueling_curve.png`

不建议正式报告使用：

1. `outputs/experiments/smoke_double_dueling_rewards.csv`
2. `outputs/experiments/smoke_double_dueling_curve.png`
3. `.matplotlib/` 缓存目录

## 五、可以如何描述最终结果

可以在报告中这样总结：

> 本项目完成了基础 DQN、Double DQN 和 Dueling DQN 的实现，并对学习率和 epsilon 衰减速度进行了对比实验。基础 DQN 推荐参数训练在第 335 个 episode 提前停止，最近 20 轮平均 reward 达到 476.15，测试 10 轮平均 reward 达到 500.0。Double + Dueling DQN 的最佳模型同样在 10 轮测试中达到平均 reward 500.0，说明智能体已经能够稳定完成 CartPole-v1 任务。
