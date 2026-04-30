---
title: 'LoRA: Low-Rank Adaptation of Large Language Models'
authors:
  - Edward Hu
  - Yelong Shen
  - Phillip Wallis
  - Zeyuan Allen-Zhu
  - Yuanzhi Li
  - Shean Wang
  - Lu Wang
  - Weizhu Chen
institutions: Microsoft Corporation
aliases:
  - LoRA
  - LoRA论文
  - 低秩适配
date: '2026-04-30'
publish_date: 2021-06
venue: ICLR 2022
tags:
  - 论文
  - NLP
  - 参数高效微调
  - 低秩分解
  - 微调
url: 'https://arxiv.org/abs/2106.09685'
code: 已开源（https://github.com/microsoft/LoRA）
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：本文提出了 LoRA（Low-Rank Adaptation），一种参数高效的大语言模型微调方法——冻结预训练权重，在每个 Transformer 层注入可训练的低秩分解矩阵（W + BA），将 GPT-3 175B 的可训练参数减少了 10,000 倍，GPU 内存需求减少 3 倍，同时保持甚至超越全参数微调的性能，且无推理延迟。

---

## Intro

### Motivation

大语言模型（如 GPT-3 175B）的微调面临严重的工程挑战：
1. **存储成本**：每个下游任务需要存储一份完整的 175B 参数副本
2. **训练成本**：全参数微调需要计算所有参数的梯度和优化器状态
3. **部署困难**：为每个客户/任务部署独立的全量模型不可行

已有的参数高效方法（Adapter、Prefix Tuning 等）存在问题：
- **Adapter 层**：引入推理延迟（需要在 Transformer 层中串行处理额外的 adapter 层）
- **Prefix Tuning**：难以优化，且占用序列长度减少可用上下文

### 核心洞察

受 Aghajanyan 等人（2020）"预训练模型具有低内在维度（intrinsic dimension）"的启发，作者假设：**模型适配过程中的权重更新也具有低"内在秩（intrinsic rank）"**。

这意味着：W (d x k) 的权重更新 deltaW 可以用一个低秩分解 BA 来近似，其中 B (d x r), A (r x k), r << min(d, k)。

### 贡献

1. **提出 LoRA**：通过低秩分解矩阵间接训练权重更新，冻结预训练权重
2. **GPT-3 175B 上减少 10,000 倍可训练参数**：从 175B 降至 ~17.5M
3. **无推理延迟**：训练后将 BA 与 W 合并（W' = W + BA），推理时零额外开销
4. **性能不降反升**：在 RoBERTa、DeBERTa、GPT-2、GPT-3 上达到或超越全参数微调
5. **经验性秩分析**：对权重更新的低秩特性进行了实证研究

---

## Method 核心方法

![](https://arxiv.org/html/2106.09685/x1.png)

*Figure 1: LoRA 的重参数化方案。冻结预训练权重 W，仅训练低秩分解矩阵 A 和 B。推理时 BA 可合并到 W 中（W' = W + BA），零额外推理延迟。*

### 1. 低秩参数化更新矩阵

对于预训练权重 W_0 (d x k)，LoRA 将微调后的权重表示为：

$$W = W_0 + \Delta W = W_0 + BA$$

其中：
- B 矩阵 (d x r)，A 矩阵 (r x k)，r << min(d, k)
- W_0 冻结，不接收梯度更新
- B 和 A 是可训练参数
- A 用随机高斯初始化，B 初始化为零（确保初始时 deltaW = 0）

前向传播：

$$h = W_0x + BAx = W_0x + B(Ax)$$

### 2. LoRA 应用的权重范围

作者主要将 LoRA 应用于 Transformer 的注意力层中的 **Q（Query）和 V（Value）投影矩阵**，有时也包括 K。FFN 层的权重通常不应用 LoRA。

### 3. Scaling Factor

实际训练中在 deltaW 前加一个缩放因子 alpha/r：

$$W = W_0 + \frac{\alpha}{r} \cdot BA$$

这个 alpha/r 的作用是让不同 rank 下的训练超参数（如学习率）可以通用，无需针对不同 r 重新调参。

### 4. 推理时合并

训练完成后，可以将 BA 合并到原始权重 W_0 中：

$$W' = W_0 + BA$$

推理时直接使用 W'，无需额外计算任何 LoRA 层的输出——零额外延迟。

### 5. 多任务存储

一个预训练模型 + 多个 LoRA 模块（每个 ~0.01% 参数）即可支持多个下游任务，存储和切换开销极低。

| 方法 | 可训练参数（GPT-3 175B） | 推理延迟 | 模型质量 |
| --- | --- | --- | --- |
| Full Fine-Tuning | 175B | 无 | 基线 |
| Adapter Layers | ~1-10% of 175B | +额外 | 接近 |
| Prefix Tuning | ~0.01% | 无额外 | 略低 |
| **LoRA** | **~0.01% (17.5M)** | **无** | **全参数同等或更好** |

---

## 实验/评估/结果

![](https://arxiv.org/html/2106.09685/x2.png)

*Figure 2: GPT-3 175B 上的验证精度与可训练参数量的关系。LoRA 以极少的可训练参数（约 17.5M）在 WikiSQL 和 MNLI 上达到或超越全参数微调，且远优于 Adapter 和 Prefix Tuning 等方法。*

### GPT-3 175B（核心结果）

在 E2E NLG Challenge、WebNLG、DART 三个数据集上：
- LoRA (r=4 或 r=16) 在大多数任务上**达到或超越**全参数微调
- 可训练参数减少 10,000 倍
- GPU 内存需求降低 3 倍（因为无需存储 175B 参数的优化器状态）
- 训练吞吐量提升约 25%

### RoBERTa / DeBERTa（GLUE）

在 GLUE 基准上，LoRA 在 RoBERTa base/large 和 DeBERTa XXL 上均达到全参数微调水平或更好：
- 在 MRPC、CoLA、STS-B 等任务上甚至略优于全参数微调
- 尤其在小数据集（如 MRPC 仅 3.6K 样本）上优势明显——LoRA 的低秩约束可能起到了正则化效果

### GPT-2（NLG 任务）

在 E2E NLG Challenge 上：
- LoRA 在 BLEU、NIST、METEOR、ROUGE-L、CIDEr 等指标上与全参数微调相当
- 多个 LoRA 变体（不同 r 值）在不同指标上各有所长

### 秩 r 的影响

- r=1 已经有效（对 GPT-3 175B 来说，r=1 即可）
- 增加 r 不一定带来持续改进，r=4 或 r=16 通常最优
- 不同权重矩阵（W_q vs W_v vs W_k vs W_o）的 LoRA 应用效果：
  - 仅 W_q 效果中等
  - 仅 W_q + W_v 效果很好
  - 加入更多矩阵（W_q + W_k + W_v + W_o）可能过拟合（尤其小数据集）

### 权重更新低秩性的实证验证

通过对不同 r 下 LoRA 和全参数微调的更新矩阵做 SVD 分析，论文证明了：
- 全参数微调的 deltaW 确实具有低秩性（前几个奇异向量贡献了大部分能量）
- LoRA 学到的低秩子空间与全参数微调的顶层奇异向量高度重叠
- 这解释了为什么 LoRA 能在极低秩下（r=1 或 4）保持性能

---

## 结论

LoRA 证明了语言模型适配过程中的权重更新具有低秩性质，并且这种低秩性质可以被高效地利用。通过冻结预训练权重并注入低秩分解矩阵，LoRA 在保持模型质量的同时大幅降低了存储、训练和部署成本。零推理延迟的设计使其特别适用于生产环境。LoRA 已成为当前最广泛使用的 LLM 参数高效微调方法。

---

## 思考

### 优点

1. **实用主义的最佳范例**：LoRA 的设计哲学极其务实——不追求极致压缩，而在"无推理延迟"和"与全参数微调同等性能"之间找到了完美的平衡。这个 trade-off 选对了，是它被广泛采用的根本原因。

2. **数学洞察与工程效率的统一**：权重更新低秩性的洞察（inspired by intrinsic dimensionality theory）直接转化为工程上的低秩分解设计。从理论假设到工程实现再到实证验证（SVD 分析），逻辑链完整且可复现。

3. **合并式推理的精妙设计**：训练后直接合并 W + BA -> W' 是一个简短的数学操作，却永久消除了推理延迟问题。这个设计比 Adapter 的外部模块多了这一额外操作，却是"有无延迟"的根本区别。

4. **跨模型规模的统一适用性**：从 RoBERTa（100M）到 GPT-3（175B），LoRA 在所有规模上都有效。且 r 很小（1-16）就能饱和性能，说明低秩性是一个相当普遍的性质。

5. **生态影响力**：HuggingFace PEFT 库对 LoRA 的集成、开源社区（LLaMA + LoRA / QLoRA 等）的广泛采用，使 LoRA 成为 fine-tuning 的标准操作。

### 缺点与局限

1. **低秩假设的非普适性**：LoRA 成功的前提是"模型适配的更新有低秩性"，但这个前提在哪些任务/哪些规模/哪些阶段上成立，论文并未充分讨论。某些任务可能需要高秩更新（如需要学习全新知识/领域/语言），此时 LoRA 可能表现不及全参数微调。

2. **r 的选择缺乏指导**：论文说 r=1 在 GPT-3 上已经有效，r=4 或 r=16 通常最好，但缺乏自动确定最优秩的方法论。

3. **仅应用于注意力权重**：论文主要将 LoRA 应用于 Q 和 V 投影矩阵。FFN 层可能也包含重要的适配信息，但在大规模应用中被忽略了。后续有 DoRA、AdaLoRA 等工作尝试改进这一点。

4. **多头注意力的 LoRA 应用未标准化**：多个注意力头共用一个 LoRA 还是每个头独立用？不同论文和实现采取了不同做法，缺乏标准化。

### 与已有 Wiki 的连接

- 关联概念：[[LoRA（低秩适配）]]、[[参数高效微调（PEFT）]]、[[Intrinsic Dimension]]、[[低秩分解]]
- 关联论文：[[AI阅读笔记/GPT-3]]（LoRA 的最大 motivator——GPT-3 175B 全参数微调不可行）
- 关联比较：LoRA vs Adapter vs Prefix Tuning vs Full Fine-Tuning
- 后续演进：QLoRA（量化 + LoRA）、DoRA、AdaLoRA、LoRA+ 等
- 关联问题：[[Wiki/Questions/LoRA 的低秩假设在什么情况下会失效]]
