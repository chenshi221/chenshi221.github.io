---
title: "DeepSeek-V3 Technical Report"
authors:
  - DeepSeek-AI
institutions: DeepSeek-AI
aliases:
  - DeepSeek-V3
  - DeepSeekV3
date: '2026-04-30'
publish_date: 2024-12
venue: 'arXiv:2412.19437'
tags:
  - 论文
  - LLM
  - MoE
  - FP8训练
  - MTP
  - 负载均衡
url: 'https://arxiv.org/abs/2412.19437'
code: https://github.com/deepseek-ai/DeepSeek-V3
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：DeepSeek-V3 是 671B MoE 模型（37B 激活），在继承 V2 的 MLA + DeepSeekMoE 架构基础上，首次引入 auxiliary-loss-free 负载均衡、Multi-Token Prediction (MTP) 训练目标、FP8 混合精度训练和 DualPipe 流水线并行，仅用 2.788M H800 GPU 小时（约 $5.6M）完成 14.8T tokens 预训练，性能比肩 GPT-4o 和 Claude-3.5-Sonnet。

![](https://arxiv.org/html/2412.19437v2/x1.png)

*Figure 1: DeepSeek-V3 与各模型的 benchmark 性能对比。*

---

## Intro

### Motivation

在 DeepSeek-V2 已验证 MLA + DeepSeekMoE 的基础上，进一步扩大模型规模（671B/37B 激活），同时需要解决三个关键问题：负载均衡的辅助损失会损害性能、MoE 跨节点通信瓶颈、以及如何进一步提升训练效率。

### 贡献

1. **Auxiliary-loss-free 负载均衡**：通过动态 bias 调整替代辅助损失，消除负载均衡对模型性能的负面影响
2. **Multi-Token Prediction (MTP)**：序列预测多个 future tokens，增强模型性能并支持 speculative decoding 加速推理
3. **FP8 混合精度训练**：首次在大规模模型上验证 FP8 训练的可行性
4. **DualPipe 算法**：实现计算-通信近乎完全重叠
5. **极低训练成本**：14.8T tokens 预训练仅需 2.664M H800 GPU 小时（$5.3M）

---

## Method 核心方法

![](https://arxiv.org/html/2412.19437/x2.png)

*Figure 2: DeepSeek-V3 基础架构。继承 V2 的 MLA + DeepSeekMoE 设计，新增 auxiliary-loss-free 负载均衡策略。*

### 1. 架构创新

#### Auxiliary-loss-free 负载均衡

核心思想：为每个 expert 引入一个 bias 项 $b_i$，仅影响路由（Top-K 选择），不影响 gating value（乘以 FFN 输出的权重）。

- 训练过程中动态监控每个 expert 的负载
- 超载则 $b_i \leftarrow b_i - \gamma$，欠载则 $b_i \leftarrow b_i + \gamma$
- 同时保留一个极小的序列级平衡损失（$\alpha$ 非常小）防止极端不平衡
- **结果**：负载均衡的同时性能优于纯辅助损失方案
- 配合 node-limited routing（每个 token 最多发往 M 个节点）实现近乎完全的计算-通信重叠

#### Multi-Token Prediction (MTP)

与 [Gloeckle et al.] 的并行预测不同，DeepSeek-V3 的 MTP 是**序列化**预测：
- 使用 $D$ 个顺序 MTP 模块，每个包含一个 Transformer block 和一个投影矩阵
- 第 $k$ 个模块组合第 $(k-1)$ 深度的表示和 $(i+k)$ 位置 token 的 embedding
- 保持完整因果链，每个深度共享 embedding 层和 output head
- 同时可作 speculative decoding 加速推理

![](https://arxiv.org/html/2412.19437/x3.png)

*Figure 3: Multi-Token Prediction (MTP) 实现。顺序预测多个 future tokens，保持完整因果链，与并行预测方案不同。*

### 2. 训练基础设施

**FP8 混合精度训练**：
- 首次在大规模模型上验证 FP8 训练
- 大部分计算密集型操作（GEMM）使用 FP8
- 对精度敏感的操作保留 BF16/FP32

**DualPipe 流水线并行**：
- 相比 1F1B 有更少的 pipeline bubble
- 通过计算-通信重叠隐藏大部分通信开销
- 配合高效跨节点 all-to-all 通信 kernel，充分利用 IB 和 NVLink

**内存优化**：
- 无需使用昂贵的 Tensor Parallelism
- ZeRO-1 数据并行

### 3. 训练策略

- 预训练：14.8T tokens
- 两阶段上下文扩展：32K → 128K
- 后训练：SFT + RL，从 DeepSeek-R1 蒸馏推理能力
- 整个训练过程**零不可恢复的 loss spike，零回滚**

---

## 实验/评估/结果

### 关键数据

| 阶段 | H800 GPU 小时 | 成本（$2/GPU/h） |
|------|-------------|-----------------|
| 预训练 | 2,664K | $5.328M |
| 上下文扩展 | 119K | $0.238M |
| 后训练 | 5K | $0.01M |
| **总计** | **2,788K** | **$5.576M** |

### 性能

- **MMLU**：88.5，超过所有开源模型，比肩 GPT-4o
- **MATH-500**：超过 o1-preview
- **LiveCodeBench**：代码竞赛 benchmark 上排名第一
- **中文 SimpleQA**：超过 GPT-4o 和 Claude-3.5-Sonnet
- **训练稳定性**：完整训练过程零不可恢复 loss spike

---

## 结论

DeepSeek-V3 证明，通过算法-框架-硬件的协同设计，可以用极低的成本（$5.6M）训练出比肩顶级闭源模型的开源模型。Auxiliary-loss-free 负载均衡和 MTP 是轻量但有效的架构改进，FP8 训练和 DualPipe 是工程效率的关键。

---

## 思考

### 优点

1. **极致的经济性**：$5.6M 训练 671B 模型是行业标杆级效率，证明了 MoE + FP8 + DualPipe 路线的可行性
2. **Auxiliary-loss-free 负载均衡简洁优雅**：用动态 bias 替代辅助损失，既简单又有效，是 MoE 训练的重要方法论贡献
3. **训练稳定性令人惊叹**：零不可恢复 loss spike，在如此大规模的 MoE 训练中极为罕见
4. **从 R1 蒸馏推理能力的策略聪明**：避免了在 V3 上直接做长 CoT RL 的成本

### 缺点

1. **MTP 的性能增益不够显著**：论文没有给出 MTP 具体的 ablation 数据，其贡献相对有限
2. **后训练部分的描述较简略**：从 R1 蒸馏的具体方法、SFT/RL 的细节披露不足
3. **与 GPT-4o 在事实性上的差距**：英文 SimpleQA 上仍有明显差距
4. **推理部署成本的完整分析缺失**：虽然训练成本极低，但 671B 模型的推理部署（需要多节点）的实际成本未充分讨论
5. **模型目前仅支持文本**：不具备多模态能力
