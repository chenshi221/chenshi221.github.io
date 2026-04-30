---
title: 'Training Compute-Optimal Large Language Models'
authors:
  - Jordan Hoffmann
  - Sebastian Borgeaud
  - Arthur Mensch
  - Elena Buchatskaya
  - Trevor Cai
  - Eliza Rutherford
  - Diego de Las Casas
  - Lisa Anne Hendricks
  - Johannes Welbl
  - Aidan Clark
  - Tom Hennigan
  - Eric Noland
  - Katie Millican
  - George van den Driessche
  - Bogdan Damoc
  - Aurelia Guy
  - Simon Osindero
  - Karen Simonyan
  - Erich Elsen
  - Jack W. Rae
  - Oriol Vinyals
  - Laurent Sifre
institutions: DeepMind
aliases:
  - Chinchilla
  - 缩放定律
  - Chinchilla论文
  - Compute-Optimal
  - Chinchilla 缩放定律
date: '2026-04-30'
publish_date: 2022-03
venue: NeurIPS 2022
tags:
  - 论文
  - 大语言模型
  - Scaling Law
  - 计算最优
  - 预训练
url: 'https://arxiv.org/abs/2203.15556'
code: 模型权重未公开
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：本文通过三种互补方法系统研究了"给定计算预算下如何分配模型大小和数据量"的问题，修正了 Kaplan 等人（2020）的缩放定律，提出 Chinchilla 缩放定律——模型参数和训练 token 数应等比例缩放（而非之前认为的模型应比数据增长更快），并据此训练了 70B 的 Chinchilla，以远少于 GPT-3（175B）的参数在 1.4T tokens 上训练达到了更优的性能。

---

## Intro

### Motivation

大语言模型存在一个核心权衡：给定固定的计算预算（FLOPs），应该把预算更多地分配给更大的模型，还是更多的训练数据？Kaplan 等人（2020）的结论是，模型大小应比数据增长得更快（N_opt proportional to C^0.73, D_opt proportional to C^0.27）。但这一结论意味着——按此规律，大多数大模型（GPT-3 175B 训练 300B tokens、Gopher 280B 训练 300B tokens、MT-NLG 530B 训练 ~300B tokens）实际上都是"过度参数化而数据不足"的。

本文重新审视了这个问题，发现 Kaplan 等人的分析存在方法论问题（如固定了学习率调度、未充分遍历模型规模），并提出了修正的缩放定律。

### 贡献

1. **修正了 LLM 的缩放定律**：N_opt 正比于 C^0.5，D_opt 正比于 C^0.5（模型大小和训练 token 应等比例增长）
2. **三种方法的交叉验证**：Envelope of Training Curves、IsoFLOP Profiles、Parametric Loss Fitting，结论高度一致
3. **Chinchilla 模型**：70B 参数训练 1.4T tokens，在所有评估的下游任务上全面超越 Gopher-280B（4 倍参数）、GPT-3 175B、MT-NLG 530B
4. **确立"计算最优"训练范式**：直接影响了 LLaMA、LLaMA-2 等后续高效 LLM 的训练策略

---

![](https://arxiv.org/html/2203.15556/x1.png)

*Figure 1: 缩放定律预测对比。三种方法均预测当前大模型应显著更小、同时在更多数据上训练更久。蓝色曲线为本文化计算最优前沿，与 Kaplan 等人（2020）的预测形成对比。*

## Method 核心方法

### 三种互补分析方法

#### 方法 1: Envelope of Training Curves

在不同规模的模型上训练不同量的数据，观察 loss 曲线。将所有训练曲线的下包络（envelope）提取出来，推导在任意给定计算量下所能达到的最小 loss，进而反推最优的 N 和 D。

#### 方法 2: IsoFLOP Profiles

固定一组计算量（FLOPs），对每个计算量，变化模型大小 N 和数据量 D（使得 C approximate 6ND），训练后得到 loss。寻找每个 IsoFLOP 下的最优 N（使 loss 最小的参数量）。结果发现 N_opt 正比于 C^a，其中 a 约等于 0.5。

![](https://arxiv.org/html/2203.15556/x3.png)

*Figure 3: IsoFLOP 曲线。在不同计算预算下变化模型大小，loss 呈现清晰的 U 型谷底——证明对于给定的 FLOP 预算，存在一个最优模型大小。*

#### 方法 3: Parametric Loss Fitting

将 loss 建模为模型大小 N 和数据量 D 的参数化函数：

$$L(N, D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$$

通过拟合训练数据推断最优的 E、A、B、alpha、beta。拟合结果也支持 N_opt 正比于 C^0.5。

### 三种方法的一致性

三种不同的方法得出高度一致的结论：模型参数和训练 token 应等比例增长。这个结论与 Kaplan（2020）认为"模型增长应快于数据增长"的结论根本不同。差异的根源在于 Kaplan 等人固定了学习率衰减策略（对所有模型大小使用相同数量的 step），而本文对每个模型大小使用了独立的 cosine decay。

### Chinchilla 模型规格

- 参数量：70B
- 训练 token 数：1.4T（~4.6 倍于 Gopher）
- 架构：与 Gopher 相同，仅调整层数/宽度使其达到 70B
- 训练硬件：2048 TPUv4
- 相比 Gopher：参数 4 倍小，数据 4 倍多

| 模型 | 参数量 | 训练 tokens | 总计算量 |
| --- | --- | --- | --- |
| GPT-3 | 175B | ~300B | ~3.1e23 |
| Gopher | 280B | 300B | ~5.0e23 |
| MT-NLG | 530B | ~300B | ~9.5e23 |
| **Chinchilla** | **70B** | **1.4T** | **~5.9e23** |

---

## 实验/评估/结果

### 语言建模

- Chinchilla-70B 在所有评估数据集上（The Pile 各子集）的 perplexity 均超越 Gopher-280B

![](https://arxiv.org/html/2203.15556/x5.png)

*Figure 5: The Pile 各子集评估。Chinchilla 在所有子集上都实现了 bits-per-byte（bpb）改善（即下降），全方位超越 Gopher。*
- 在多数数据集上甚至超越了参数 5.6 倍大但训练不足的 MT-NLG-530B

### MMLU（大规模多任务语言理解）

| 模型 | 参数量 | MMLU |
| --- | --- | --- |
| GPT-3 | 175B | 43.9% |
| Gopher | 280B | 60.0% |
| Chinchilla | 70B | **67.5%** |

Chinchilla 的 MMLU 比 Gopher 高 7.5%，比 GPT-3 高 23.6%。

### 常识推理（HellaSwag、PIQA、WinoGrande 等）

- Chinchilla-70B 在所有常识推理 benchmark 上超越 Gopher-280B
- 在 WinoGrande 上尤其显著

### 阅读理解与 QA

- RACE-h：超越 Gopher
- Natural Questions：Chinchilla 在多个评估指标上显著提升

### BIG-Bench

- 在 56 个 BIG-Bench 任务上的平均性能：Chinchilla 全面超越 Gopher

### 还有意外发现：不同方法的收敛曲线

论文通过回溯不同 checkpoint 的 benchmark 性能，发现不同能力涌现的速度不同：常识推理、阅读理解等在中等规模已基本收敛，而数学推理、代码生成等需要更大规模才能展现。

---

## 结论

Chinchilla 缩放定律对 LLM 训练有根本性的指导意义：**在给定计算预算下，模型参数量和训练 token 数应等比例增长**。这一结论直接影响了后续 LLM 的设计哲学——从 GPT-3 时代的"越大越好"（oversized model + insufficient data），转向了"在合适规模上投入更多数据"（LLaMA 路线）。70B 的 Chinchilla 在所有基准上全面超越 280B 的 Gopher，证明了"数据比参数更重要"。

---

## 思考

### 优点

1. **方法论极其严谨**：三种互补方法（Envelope、IsoFLOP、Parametric Fit）交叉验证结论，这种严谨性让结论非常可信。

2. **修正了广泛使用的 Kaplan 定律**：Kaplan 等人（2020）的结论被接受为"标准 wisdom"数年之久。本文通过更仔细的实验设计指出了其中固定学习率调度的方法论缺陷，属于教科书级的"通过更好的实验设计推翻广泛接受结论"。

3. **深刻影响了 LLM 的实际训练策略**：Chinchilla 定律直接指导了 LLaMA 系列（LLaMA-13B 训练 1T tokens）、LLaMA-2（70B 训练 2T tokens）、Mistral 等后续高效 LLM 的设计。可以说，这篇论文改变了 LLM 领域的工程实践。

4. **对能力涌现时间的记录**：论文中关于不同能力在训练过程中涌现时间不同的观察，虽然有初步定性，但对理解"预训练到底要多久"很有参考价值。

### 缺点与局限

1. **Chinchilla 定律的适用范围未充分讨论**：论文主要在小模型（最大到 ~10B 参数）上进行 scaling law 拟合，然后外推到 70B。这个外推是否在更大的规模（如 1T 参数）上仍然有效？后续研究对此有争议。

2. **Law 是 empirical 的，不是 theoretical 的**：论文推导的是拟合曲线的形式，而不是对"为什么"的解释。从计算理论上讲，等比例缩放是否是最优的仍有争论。

3. **数据多样性 vs 数据量**：Chinchilla 定律只讨论了"多少 tokens"，没有区分数据质量。实际中，高质量数据可能比多量低质数据更有效（phi 系列模型证明了这一点）。即在考虑"数据质量"这个维度时，Chinchilla 定律可能需要修正。

4. **训练成本比较缺乏全面的标准化**：Chinchilla-70B 训练了 1.4T tokens，计算量与 Gopher-280B（300B tokens）接近。但在推理时，70B 比 280B 更小更快——这个推理效率的提升论文有提到但未充分量化。

### 与已有 Wiki 的连接

- 关联概念：[[Scaling Law]]、[[计算最优]]、[[预训练效率]]
- 关联论文：[[AI阅读笔记/GPT-3]]（Chinchilla 修正的原始参考）、[[AI阅读笔记/LLaMA]]（Chinchilla 定律的典型应用）
- 关联矛盾：Chinchilla 定律 vs Kaplan 定律（数据量增长率不同）；Chinchilla 定律 vs "质量优先"路线（phi 系列、指令数据优先）
- 后续讨论：LLaMA-2（2T tokens）、LLaMA-3（15T tokens）的训练数据量远超 Chinchilla 最优预测，说明"超过 Chinchilla 最优点的继续训练"可能仍有好处
