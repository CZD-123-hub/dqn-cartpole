# 必读论文读书笔记

本笔记选择实验室必读论文中的两篇基础论文进行阅读：

1. Ashish Vaswani 等，Attention Is All You Need，2017  
   原文：https://arxiv.org/abs/1706.03762
2. Jacob Devlin 等，BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding，2018  
   原文：https://arxiv.org/abs/1810.04805

## 论文一：Attention Is All You Need

### 1. 基本信息

- 题目：Attention Is All You Need
- 作者：Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
- 时间：2017
- 研究方向：自然语言处理、序列建模、机器翻译
- 核心贡献：提出 Transformer 架构，用注意力机制替代传统 RNN/CNN 序列建模结构。

### 2. 研究背景

在 Transformer 出现之前，机器翻译等序列到序列任务主要依赖 RNN、LSTM、GRU 或 CNN。RNN 类模型按时间步顺序处理序列，天然适合处理文本顺序，但难以并行，长距离依赖建模也比较困难。CNN 可以提高并行性，但需要堆叠多层才能扩大感受野。

这篇论文提出一个关键问题：是否可以完全不依赖循环结构和卷积结构，仅使用注意力机制完成序列建模？

论文给出的答案是 Transformer。

### 3. 核心方法

Transformer 采用 Encoder-Decoder 结构。

Encoder 负责把输入序列编码成上下文表示，Decoder 根据编码结果逐步生成输出序列。与传统 Seq2Seq 不同，Transformer 的核心计算单元不是 RNN，而是 self-attention 和前馈神经网络。

#### Self-Attention

Self-Attention 的作用是让序列中每个 token 都能关注同一序列中的其他 token。每个输入会被映射成三个向量：

- Query：当前 token 想查找什么信息。
- Key：当前 token 能提供什么索引信息。
- Value：当前 token 实际携带的信息内容。

注意力权重由 Query 和 Key 的相似度决定，再用这些权重对 Value 加权求和。这样模型可以根据上下文动态判断哪些词更重要。

#### Multi-Head Attention

单个注意力头只能从一个表示空间学习关系。Multi-Head Attention 使用多个注意力头并行学习不同角度的关系，例如语法关系、指代关系、长距离依赖等。最后再把多个头的结果拼接起来。

#### Positional Encoding

因为 Transformer 没有 RNN 的时间顺序结构，所以需要额外加入位置信息。论文使用正弦和余弦函数构造位置编码，让模型知道 token 在序列中的相对或绝对位置。

### 4. 实验结果

论文在 WMT 2014 英德翻译和英法翻译任务上取得了很强的结果，同时训练速度优于当时常见的循环或卷积模型。论文展示出 Transformer 不仅效果好，而且更容易并行训练。

### 5. 个人理解

我认为这篇论文最重要的地方不是单纯提升了机器翻译指标，而是改变了深度学习处理序列数据的基本方式。过去处理文本常常默认要按顺序读入，而 Transformer 表明，只要设计好注意力机制和位置编码，模型可以一次性看到整个序列，并学习 token 之间的依赖关系。

从学习角度看，Transformer 的思想可以理解为：模型不再固定按照时间步传递信息，而是让每个位置自己判断应该关注哪些位置。这种机制后来成为 BERT、GPT 等大模型的基础。

### 6. 与当前项目的联系

当前 DQN-CartPole 项目属于强化学习，和 Transformer 的 NLP 任务不同。但两者都体现了深度学习中的一个共同思想：用神经网络学习复杂映射关系。

在 DQN 中，神经网络学习的是：

```text
state -> action value
```

在 Transformer 中，神经网络学习的是：

```text
token sequence -> contextual representation
```

虽然任务不同，但都需要理解网络结构、损失函数、训练稳定性和泛化能力。

## 论文二：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

### 1. 基本信息

- 题目：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- 作者：Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- 时间：2018
- 研究方向：自然语言处理、预训练语言模型
- 核心贡献：提出 BERT，通过双向 Transformer 编码器进行大规模预训练，再在下游任务上微调。

### 2. 研究背景

BERT 之前，很多 NLP 模型已经开始使用预训练词向量或语言模型。传统词向量如 Word2Vec、GloVe 可以表示词语语义，但同一个词在不同上下文中通常只有一个静态向量。后来的 ELMo、GPT 等方法引入上下文表示，但仍存在方向性或任务适配方面的限制。

BERT 的目标是学习深层双向语言表示，让模型在每一层都能同时利用左侧和右侧上下文信息。

### 3. 核心方法

BERT 基于 Transformer Encoder，不使用 Decoder。它先在大规模无标注文本上预训练，然后针对具体任务增加很少的任务层进行微调。

#### Masked Language Model

如果直接用普通语言模型训练，模型通常只能从左到右或从右到左预测，难以做到真正双向。BERT 使用 Masked Language Model：随机遮盖输入中的部分 token，让模型根据上下文预测被遮盖的词。

这个任务迫使模型同时利用左右上下文，从而学习双向表示。

#### Next Sentence Prediction

BERT 还设计了 Next Sentence Prediction 任务，让模型判断两个句子是否是连续上下文。这个设计主要服务于问答、自然语言推理等需要理解句子关系的任务。

#### Fine-Tuning

预训练完成后，BERT 可以在下游任务上微调。对于分类任务，可以使用 `[CLS]` 位置的输出接一个分类层；对于问答任务，可以预测答案片段的开始和结束位置。

### 4. 实验结果

BERT 在多个 NLP 任务上取得了当时非常强的效果，包括 GLUE、MultiNLI、SQuAD 等任务。论文的重要意义在于证明了“预训练 + 微调”范式在 NLP 中非常有效。

### 5. 个人理解

我认为 BERT 的核心价值在于把 NLP 模型训练拆成两个阶段：

1. 先用大量无标注文本学习通用语言表示。
2. 再用较少标注数据适配具体任务。

这种方式对于实际研究和应用都很重要，因为很多任务的标注数据有限，而无标注文本相对容易获得。BERT 说明了大规模预训练可以显著提升下游任务效果。

从结构上看，BERT 继承了 Transformer 的 self-attention 能力，但更关注“理解”任务，而不是生成任务。它适合文本分类、句子匹配、问答、命名实体识别等任务。

### 6. 与当前项目的联系

当前 DQN-CartPole 项目不是 NLP 项目，但阅读 BERT 有助于理解现代深度学习中的迁移学习和预训练思想。

对于我目前的阶段，BERT 带来的启发主要有三点：

- 深度学习模型的结构设计会显著影响效果。
- 大规模数据和训练目标设计同样重要。
- 很多复杂模型都可以拆成基础模块逐步理解，例如 Embedding、Attention、前馈网络、损失函数和优化器。

DQN 项目也可以采用类似的拆解思路：先理解环境交互，再实现 Replay Buffer，再实现 QNetwork，最后完成训练和评估。

## 总结

这两篇论文虽然主要属于 NLP 方向，但对理解深度学习发展非常重要。Transformer 提供了强大的序列建模结构，BERT 则展示了预训练语言模型在自然语言理解任务中的效果。

通过阅读这两篇论文，我对以下内容有了初步理解：

- 注意力机制如何建模序列内部依赖。
- Transformer 为什么能替代 RNN 处理序列任务。
- BERT 为什么采用双向 Transformer Encoder。
- 预训练和微调为什么能提升下游任务表现。
- 深度学习项目可以通过“模块拆解”的方式逐步学习和实现。
