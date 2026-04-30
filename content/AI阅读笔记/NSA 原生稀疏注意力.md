---
title: "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
authors:
  - Jingyang Yuan
  - Huazuo Gao
  - Damai Dai
  - Junyu Luo
  - Liang Zhao
  - Zhengyan Zhang
  - Zhenda Xie
  - Y. X. Wei
  - Lean Wang
  - Zhiping Xiao
  - Yuqing Wang
  - Chong Ruan
  - Ming Zhang
  - Wenfeng Liang
  - Wangding Zeng
institutions: DeepSeek-AI, Peking University, University of Washington
aliases:
  - NSA
  - 原生稀疏注意力
date: 2026-04-30
publish_date: 2025-02
venue: 'arXiv:2502.11089'
tags:
  - 论文
  - 稀疏注意力
  - 长上下文
  - 高效推理
  - DeepSeek
  - FlashAttention
  - KV-Cache
url: 'https://arxiv.org/abs/2502.11089'
code: ''
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：NSA（Native Sparse Attention）提出了一种从预训练阶段就原生支持稀疏计算的注意力机制，通过三层并行注意力路径（粗粒度压缩 + 细粒度选择 + 滑动窗口）和硬件对齐的 Triton 内核设计，在 27B 参数的 MoE 模型上实现了匹配或超越 Full Attention 的性能，同时训练加速 9 倍、解码加速 11.6 倍（64k 上下文）。

---

## Intro

### Motivation

长上下文建模是下一代 LLM 的核心能力，但标准注意力复杂度随序列长度平方增长。64k 序列解码时，softmax 注意力计算占总延迟的 70-80%。现有稀疏注意力方法存在两个致命缺陷：

1. **推理加速是幻觉**：很多方法只在特定阶段（prefill 或 decode）稀疏，另一阶段仍接近 Full Attention 成本；且不适应 MQA/GQA 等现代高效解码架构
2. **不支持训练**：多数稀疏方法仅用于推理阶段，包含不可微操作（如 k-means 聚类、SimHash），无法端到端训练。即使理论上可训练的方法，因 token 级选择的非连续内存访问，也无法高效反向传播

### 核心思想

NSA 从预训练开始就使用稀疏注意力，让模型在训练中学会最优的稀疏模式，而非事后对 Full Attention 模型做剪枝。

### 贡献

1. 提出三层并行注意力路径：**Token Compression**（粗粒度全局信息）+ **Token Selection**（细粒度关键 token）+ **Sliding Window**（局部上下文）
2. 硬件对齐的 **GQA-group-centric** Triton 内核设计，最大化 Tensor Core 利用率和内存访问效率
3. 实现端到端原生训练，在 27B 参数 MoE 模型上验证：通用 benchmark 超越 Full Attention，LongBench 平均分最高，AIME 推理评估中显著优于 Full Attention

---

## Method 核心方法

### 三层注意力路径

NSA 将原始 KV 对替换为三组更紧凑的表示 KV 对，并行计算后通过可学习的门控机制聚合：

$\mathbf{o}^{*}_{t} = \sum_{c \in \{\text{cmp},\text{slc},\text{win}\}} g_{t}^{c} \cdot \text{Attn}(\mathbf{q}_{t}, \tilde{K}_{t}^{c}, \tilde{V}_{t}^{c})$

![](https://arxiv.org/html/2502.11089/x2.png)

*Figure 2: NSA 架构总览。三条并行注意力路径：压缩注意力（粗粒度全局）、选择注意力（细粒度关键 token 块）、滑动窗口注意力（局部上下文），通过门控机制聚合。*

**1. Token Compression（粗粒度压缩）**

将连续 key/value 块通过可学习的 MLP + 位置编码压缩为块级表示：

$$\tilde{K}^{\text{cmp}}_{t} = \{\varphi(\mathbf{k}_{id+1:id+l}) \mid 0 \leq i \leq \lfloor\frac{t-l}{d}\rfloor\}$$

Block length $l=32$，stride $d=16$（重叠以减轻信息碎片化）。压缩表示捕获高层语义，极大减少注意力计算量。

**2. Token Selection（细粒度选择）**

基于块重要性分数选择 top-n 个 KV 块（blockwise selection），而非选择单个 token。核心创新：
- 复用压缩注意力的中间注意力分数来计算块级重要性分数，几乎无额外开销
- 在 GQA/MQA 架构下，同一 group 内所有 query head 共享块选择，确保 KV cache 加载一致
- 选择 $n=16$ 个块（含固定的 1 个初始块和 2 个局部块），每个选择块 $l'=64$ tokens

**3. Sliding Window（滑动窗口）**

维护最近 $w=512$ 个 token 的局部注意力。关键设计是**独立的 KV 投影**——三个分支使用各自的 key/value 权重，防止局部信息"shortcut"掉全局模式的学习。

### 硬件对齐的内核设计

不同于 FlashAttention 按时间连续 query 块加载，NSA 内核按 **GQA group** 组织：
- 对序列每个位置，加载整个 GQA group 内所有 query head（共享相同稀疏 KV 块）
- 在内循环中顺序加载连续 KV 块到 SRAM
- 外循环使用 Triton grid scheduler 调度

这实现了：(1) 通过 group 共享消除冗余 KV 传输；(2) 平衡 GPU SM 间的计算负载

---

## 实验/评估/结果

### 预训练设置

- 27B 总参数 / 3B 激活参数，GQA（4 groups, 64 heads）+ MoE（DeepSeekMoE, 72 路由专家 + 2 共享专家）
- 训练 270B tokens @ 8k 长度，后续继续训练和 SFT @ 32k 长度（YaRN）

### 通用 Benchmark

| Metric | Full Attention | NSA |
|--------|---------------|-----|
| MMLU | 0.567 | 0.565 |
| MMLU-PRO | 0.279 | **0.286** |
| GSM8K | 0.486 | **0.520** |
| DROP | 0.503 | **0.545** |
| HumanEval | 0.335 | **0.348** |
| **Average** | 0.443 | **0.456** |

NSA 在 7/9 指标上超越 Full Attention，尤其在推理相关任务（DROP +0.042, GSM8K +0.034）上优势明显。

### 长上下文评估（LongBench）

| Model | Avg. Score |
|-------|-----------|
| H2O | 0.303 |
| InfLLM | 0.383 |
| Quest | 0.392 |
| Exact-Top | 0.423 |
| Full Attention | 0.437 |
| **NSA** | **0.469** |

在多跳 QA（HPQ +0.087, 2Wiki +0.051）和代码理解（LCC +0.069）上优势尤为突出。Needle-in-a-Haystack 64k 测试中达到完美准确率。

### 思维链推理（AIME）

蒸馏 DeepSeek-R1 推理轨迹进行 SFT 后：

| Generation Limit | Full Attention-R | NSA-R |
|-----------------|-----------------|-------|
| 8192 tokens | 0.046 | **0.121** |
| 16384 tokens | 0.092 | **0.146** |

NSA-R 在两种上下文长度下均显著优于 Full Attention-R，证明原生稀疏注意力的预训练模式有利于捕获长程逻辑依赖。

### 效率分析

在 8-GPU A100 系统上：
- **训练加速**（64k 序列）：Forward 9.0x / Backward 6.0x（vs FlashAttention-2）
- **解码加速**（64k 序列）：预期 11.6x（内存访问量成比例减少）

---

## 结论

NSA 通过三层并行注意力路径（压缩+选择+滑动窗口）和硬件对齐的内核设计，首次实现了在性能不降的前提下，从预训练到推理全生命周期的高效稀疏注意力。NSA 证明了"原生的训练时稀疏"优于"事后的推理时稀疏"。

---

## 思考

### 优点

1. **全生命周期覆盖**：NSA 是目前少有的同时覆盖预训练、forward、backward、prefill 和 decoding 的稀疏注意力方案。大多数方法只聚焦推理阶段的一环，NSA 的"原生训练"理念是真正的范式转变。

2. **块级选择的硬件对齐**：选择连续 KV 块而非单个 token，这条设计与 FlashAttention 的块状计算哲学一致，使得 NSA 能直接利用 Tensor Core 的高吞吐。这比 token 级别的稀疏选择（如 HashAttention）在实现上优雅得多。

3. **压缩注意力的"免费"重要性分数**：复用压缩注意力中间结果来估计选择块重要性，这个设计简洁且高效——不引入额外计算开销，又保证足够的信号质量。

4. **独立 KV 投影防止 shortcut learning**：为三个分支设置独立 KV 权重，这是一个细致的训练动力学考量——防止局部窗口"走捷径"主导训练，迫使压缩和选择分支也学会有效特征。

5. **实验的全面性**：覆盖通用 benchmark、长上下文、CoT 推理三个维度，证明 NSA 不是只在特定场景有效的 trick。

### 缺点与局限性

1. **超参数较多且需人工设定**：block size $l=32$、stride $d=16$、selection block $l'=64$、selection count $n=16$、window $w=512$——这些参数对性能可能敏感，但论文缺乏系统的消融实验。

2. **训练成本未充分讨论**：虽然 NSA 注意力本身加速了，但压缩 MLP、门控 MLP、三组独立 KV 投影都增加了额外参数和计算。论文只对比了 NSA 注意力 vs Full Attention 注意力的速度，没有给出端到端训练的总成本对比。

3. **与 Full Attention 的性能差距有限**：虽然 NSA 在平均分上略超 Full Attention（0.456 vs 0.443），但边际优势叠加大量超参数调优后，这个优势是否稳健仍需更多验证。

4. **仅验证了 MoE 架构**：实验基于 MoE 模型，虽然声称内核与 GQA/MQA 兼容，但未在纯 dense 模型上验证。MoE 本身的稀疏性可能与注意力稀疏性存在交互效应。

5. **长序列上限未知**：64k 是实验的最长序列，对于 128k-1M 级别的超长上下文，三层分支的 token 预算分配策略可能需要重新设计。
