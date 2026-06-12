# DQN-CartPole 项目进度总结与后续工作安排

日期：2026 年 6 月 12 日

## 一、当前总体进度

当前项目已经完成前期准备和基础模块搭建，整体处于“DQN 主体训练前”的阶段。

已经完成的工作主要包括：

- 阅读并整理实验室研0练手项目要求。
- 明确当前项目选择为 `DQN-CartPole`。
- 完成 Windows + PyCharm + Python 3.12.10 + PyTorch CPU + Gymnasium 的开发环境配置。
- 创建基础项目结构。
- 跑通 `CartPole-v1` 随机策略。
- 实现 Replay Buffer 经验回放池。
- 为 Replay Buffer 编写并通过单元测试。
- 实现 QNetwork 网络模型，并通过输入输出 shape 测试。
- 实现基础 DQN 训练脚本，并通过 3 个 episode 的短训练验证。
- 实现模型测试脚本，并能加载模型评估平均 reward。
- 完成一次正式训练，最佳模型 10 轮测试平均 reward 达到 500.0。
- 生成 reward 曲线，并整理实验记录。
- 整理学习报告、论文读书笔记和项目资料。

当前尚未完成的核心部分包括：

- 长轮次 DQN 正式训练。
- 最终 reward 曲线分析。
- 多组超参数对比实验。
- Double DQN、Dueling DQN 等加分项。

## 二、当前项目文件状态

当前项目主要目录如下：

```text
dqn-cartpole/
├── docs/
├── models/
├── outputs/
├── src/
│   ├── random_cartpole.py
│   ├── replay_buffer.py
│   ├── model.py
│   ├── train_dqn.py
│   └── test_dqn.py
├── tests/
├── README.md
├── requirements.txt
└── .gitignore
```

### 1. 已完成代码

`src/random_cartpole.py`

该文件用于运行 CartPole 随机策略，主要作用是验证 Gymnasium 环境可以正常创建、重置、交互和关闭。随机策略不会学习，但可以帮助理解强化学习基本交互流程：

```text
state -> action -> next_state, reward, done
```

`src/replay_buffer.py`

该文件实现经验回放池，用于保存智能体与环境交互产生的经验：

```text
(state, action, reward, next_state, done)
```

Replay Buffer 是后续 DQN 训练的重要基础。训练时不会只使用最新经验，而是从历史经验中随机采样 batch，从而降低连续样本之间的相关性，使训练更稳定。

`tests/test_replay_buffer.py`

该文件测试 Replay Buffer 的基础行为，包括：

- 容量满后是否保留较新的经验。
- 采样结果是否返回适合训练的 NumPy 数组。

当前测试命令：

```powershell
python -m unittest discover
```

当前测试结果：2 个测试通过。

### 2. 仍为占位的代码

`src/model.py`

该文件已经实现 QNetwork。当前采用最小 DQN 版本的多层感知机结构，输入为 CartPole 的 4 维状态，输出为 2 个动作对应的 Q 值。

`src/train_dqn.py`

该文件已经实现基础 DQN 训练循环，包括 epsilon-greedy 动作选择、Replay Buffer 采样、Bellman 目标计算、Target Network 同步、模型保存、reward CSV 保存和 reward 曲线绘制。当前已经通过 3 个 episode 的短训练验证。

`src/test_dqn.py`

该文件已经实现模型评估功能，可以加载 `models/dqn_cartpole.pth`，使用贪心策略选择 Q 值最大的动作，并输出多轮测试的平均 reward。

## 三、当前学习与文档进度

已经整理的文档包括：

- `docs/source_notes.md`：实验室项目资料阅读备注。
- `docs/paper_notes.md`：两篇论文读书笔记。
- `docs/learning_report.md`：学习报告 Markdown 版。
- `docs/learning_report.docx`：学习报告 Word 版。
- `docs/learning_report_polished.docx`：排版润色后的学习报告 Word 版。

基础知识学习进度：

- 已学习《机器学习》（西瓜书）的基础内容。
- 已学习《深度学习》（花书）的部分内容。
- 已学习 GitHub / Git 基础代码管理。
- 已阅读并整理两篇论文笔记：Transformer 和 BERT。

## 四、后续工作总目标

后续工作的核心目标是完成一个最小可运行、结果可展示的 DQN-CartPole 项目。

到 7 月初前，建议至少完成以下结果：

- 能够训练 DQN 智能体。
- 训练后的智能体在 `CartPole-v1` 中能够稳定达到 200 步以上。
- 能够保存训练好的模型。
- 能够加载模型进行测试。
- 能够生成 reward 曲线图。
- 能够整理一份最终实验记录，包括方法、参数、结果和问题分析。

## 五、后续工作安排

当前日期是 2026 年 6 月 12 日，距离 7 月初还有大约 2 到 3 周。建议按“先跑通，再优化，再整理”的顺序推进。

### 第一阶段：完成 DQN 最小闭环

时间建议：6 月 12 日到 6 月 16 日

目标：让项目从基础模块进入可训练状态。

具体任务：

1. 实现 `src/model.py` 中的 QNetwork。
2. QNetwork 输入为 4 维 CartPole 状态。
3. QNetwork 输出为 2 个动作对应的 Q 值。
4. 编写 QNetwork 的简单测试，确认输入输出 shape 正确。
5. 在 `src/train_dqn.py` 中搭建训练脚本框架。
6. 实现 epsilon-greedy 动作选择。
7. 接入 Replay Buffer。
8. 完成基本 Q 值更新逻辑。

阶段完成标志：

- `model.py` 不再是占位文件。
- `train_dqn.py` 可以启动训练。
- 训练过程中能够打印每轮 episode reward。

### 第二阶段：加入 Target Network 和模型保存

时间建议：6 月 17 日到 6 月 20 日

目标：提高训练稳定性，并让训练结果可以保存。

具体任务：

1. 在训练脚本中加入 policy network 和 target network。
2. 每隔固定步数同步 target network 参数。
3. 使用 Bellman 方程计算目标 Q 值。
4. 使用 MSELoss 或 SmoothL1Loss 训练 QNetwork。
5. 训练完成后保存模型到 `models/dqn_cartpole.pth`。
6. 保存每个 episode 的 reward 到 `outputs/rewards.csv`。

阶段完成标志：

- 训练过程 reward 有上升趋势。
- 模型文件可以保存到 `models/`。
- reward 数据可以保存到 `outputs/`。

### 第三阶段：测试模型与绘制曲线

时间建议：6 月 21 日到 6 月 24 日

目标：让项目具备可展示结果。

具体任务：

1. 实现 `src/test_dqn.py`。
2. 加载 `models/dqn_cartpole.pth`。
3. 使用贪心策略选择 Q 值最大的动作。
4. 测试训练后智能体的平均 reward。
5. 绘制训练 reward 曲线并保存到 `outputs/reward_curve.png`。
6. 观察是否达到稳定 200 步以上。

阶段完成标志：

- 可以通过命令测试模型效果。
- 可以得到 reward 曲线图片。
- 能够说明模型是否达到 200 步以上目标。

### 第四阶段：调参与结果分析

时间建议：6 月 25 日到 6 月 28 日

目标：提高结果稳定性，并形成实验分析。

具体任务：

1. 调整学习率，例如 `1e-3`、`5e-4`、`1e-4`。
2. 调整 epsilon 衰减速度。
3. 调整 batch size 和 replay buffer capacity。
4. 对比不同参数下 reward 曲线变化。
5. 记录训练失败或波动较大的情况。
6. 总结哪些参数对训练影响明显。

阶段完成标志：

- 至少完成 2 到 3 组参数对比。
- 有对应 reward 曲线或结果记录。
- 能够在实验报告中解释训练效果。

### 第五阶段：整理最终材料

时间建议：6 月 29 日到 7 月 1 日

目标：形成可以提交或汇报的完整材料。

具体任务：

1. 更新 `README.md`。
2. 补充运行命令和项目结构说明。
3. 整理最终实验记录。
4. 将 reward 曲线和测试结果加入报告。
5. 检查所有脚本是否能在 Windows + PyCharm + CPU 环境下运行。
6. 视情况上传 GitHub。

阶段完成标志：

- README 清楚说明如何训练和测试。
- 报告中有最终实验结果。
- 项目代码、文档、模型输出结构完整。

## 六、建议优先级

后续推进时，优先级建议如下：

1. 最高优先级：跑通基础 DQN 训练闭环。
2. 第二优先级：保存模型、测试模型、绘制 reward 曲线。
3. 第三优先级：整理 README 和实验记录。
4. 第四优先级：调参分析。
5. 加分优先级：Double DQN、Dueling DQN。

如果时间紧张，建议先确保基础 DQN 能跑通，并有 reward 曲线和测试结果。Double DQN、Dueling DQN 可以作为加分项，不要放在基础版本之前。

## 七、下一步立即要做的任务

下一步建议做 2 到 3 组超参数对比实验，例如学习率和 epsilon 衰减速度对训练效果的影响。

当前 QNetwork 已采用以下结构：

```text
Linear(4 -> 128)
ReLU
Linear(128 -> 128)
ReLU
Linear(128 -> 2)
```

已经写了简单测试，确认：

- 输入 shape 为 `(batch_size, 4)`。
- 输出 shape 为 `(batch_size, 2)`。
- 网络可以正常进行前向传播。

当前已经把 QNetwork、Replay Buffer、Gymnasium 环境和测试脚本连接起来，并完成正式训练。模型在 10 轮测试中平均 reward 达到 500.0，已经超过持续平衡 200 步以上的目标。

## 八、阶段性总结

当前项目基础已经搭好，环境也已经验证通过，Replay Buffer 也有测试保障。接下来的关键不是再扩展文档，而是尽快把 DQN 的训练闭环跑起来。

建议接下来保持“小步快跑”的节奏：

1. 先实现 QNetwork。
2. 再跑通最小训练循环。
3. 然后补 Target Network 和保存逻辑。
4. 最后做曲线、测试和报告整理。

这样可以避免一开始写过于复杂的代码，也能保证每一步都有可验证结果。
