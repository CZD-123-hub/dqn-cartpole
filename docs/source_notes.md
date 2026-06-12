# 项目资料阅读备注

## 已保存的原始资料

- `docs/机器学习要览（202604）.docx`
- `docs/研0练手项目.pdf`

## 抽取出的文本

- `docs/machine_learning_overview_extracted.txt`
- `docs/practice_project_extracted.txt`

## 和当前 DQN-CartPole 项目有关的要求

当前项目选择的是 `DQN-CartPole`。

任务要求：

- 使用深度 Q 网络，也就是 DQN，解决 OpenAI Gym / Gymnasium 中的 `CartPole-v1` 问题。
- 训练智能体控制小车，让杆子尽量保持平衡。
- 目标效果是训练后的智能体可以稳定运行，例如持续平衡 200 步以上。
- 需要可视化训练过程中的奖励曲线和策略效果。

加分项：

- 尝试 Double DQN、Dueling DQN 等算法变体。
- 调整网络结构以提升性能。
- 分析学习率、epsilon-greedy 探索率等超参数对结果的影响。

## 关于数据集

DQN-CartPole 不需要额外下载静态数据集。训练数据来自智能体与 `CartPole-v1` 环境交互时产生的经验：

```text
(state, action, reward, next_state, done)
```

这些经验会存入 Replay Buffer，然后随机采样用于 DQN 训练。

PDF 中出现的数据集链接包括：

- `https://huggingface.co/datasets/XiangPan/waimai_10k`
- `http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`

它们分别对应外卖评论情感分类和 CIFAR-10 图像分类两个备选项目，不是当前 DQN-CartPole 项目的必需数据集。
