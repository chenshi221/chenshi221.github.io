---
title: The Llama 3 Herd of Models
authors:
  - Llama Team
institutions: 'AI @ Meta'
aliases:
  - Llama 3
  - Llama 3论文
  - Llama 3.1
date: '2026-04-30'
publish_date: 2024-07
venue: 'arXiv:2407.21783'
tags:
  - 论文
  - 大语言模型
  - 开源
  - 多语言
  - Tool Use
  - Llama
url: 'https://arxiv.org/abs/2407.21783'
code: 已开源（模型权重公开发布）
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：本文介绍了 Llama 3 系列模型，包括 8B、70B 和 405B 参数量级，原生支持多语言、代码、推理和工具使用，最大模型在多项任务上与 GPT-4 性能相当。论文特别强调了数据、规模、复杂度管理三个核心杠杆，并采用标准 Dense Transformer 架构配合 SFT + Rejection Sampling + DPO 的简洁后训练方案。

---

## Intro

### Motivation

在 Llama 和 Llama 2 成功的基础上，Meta 团队的目标是构建一个更大、更强、能力更全面的开源基础模型，在保持开源可复现的前提下，与 GPT-4 等级别的闭源模型竞争。

### 核心设计哲学

Llama 3 有三个核心杠杆：

1. **数据（Data）**：相比 Llama 2 的 1.8T tokens，Llama 3 使用了约 15T 的多语言 tokens 进行预训练，大幅提升了数量和质量。预训练数据的处理和筛选流水线更细致，后训练阶段发展了更严格的质量保证和过滤方法。

2. **规模（Scale）**：旗舰 405B 模型使用 3.8e25 FLOPs 进行预训练，几乎是 Llama 2 最大版本训练计算量的 50 倍。模型在 15.6T 文本 tokens 上训练。根据 scaling law，405B 是近似计算最优的模型大小。同时，较小模型（8B、70B）的训练量远超计算最优——这种"过度训练"使它们在相同推理预算下性能更好。

3. **复杂度管理（Managing Complexity）**：刻意选择标准 Dense Transformer（而非 MoE）以保证训练稳定性；后训练采用 SFT + Rejection Sampling + DPO，而非更复杂的 RL 算法，降低调参难度。

### 贡献

1. **Llama 3 系列**：8B、70B、405B 三个规模，多语言、长上下文、工具使用
2. **405B 与 GPT-4 持平**：在大量基准上 405B Instruct 与 GPT-4 性能相当
3. **小模型的最佳性能**：8B 和 70B 在同参数量级中领先
4. **多模态能力**：通过组合式方法（compositional approach）将图像、视频和语音能力集成
5. **安全改进**：Llama Guard 3 输入输出安全模型
6. **全面的开源**：预训练和后训练 405B 版本均公开发布

---

## Method 核心方法

### 1. 模型架构

- 标准 Dense Transformer（decoder-only），刻意不采用 MoE

![](https://arxiv.org/html/2407.21783/x1.png)

*Figure 1: Llama 3 的整体架构与训练流程。Llama 3 是标准的 Transformer 语言模型，通过预测下一个 token 进行训练。*

- 上下文窗口：最高 128K tokens
- 在 Llama 2 架构基础上做了微小调整

### 2. 预训练

- 数据规模：约 15T 多语言 tokens（Llama 2 的 8 倍+）

![](https://arxiv.org/html/2407.21783/x2.png)

*Figure 2: 缩放定律 IsoFLOP 曲线。在 6x10^18 到 10^22 FLOPs 的计算预算下，loss 随模型大小呈 U 型，证明 405B 是近似计算最优的模型规模。*

- 405B 模型总计算量：3.8e25 FLOPs
- 遵循 scaling law，405B 为近似计算最优规模
- 小模型（8B、70B）训练远超计算最优的数据量——"过度训练"策略

### 3. 后训练

采用简洁的流程，避免复杂 RL 算法的不稳定性：

- **Supervised Fine-Tuning (SFT)**：在高质量指令数据上监督微调
- **Rejection Sampling (RS)**：对每个 prompt 生成多个回答，用奖励模型/评判工具筛选最佳答案，用于进一步训练
- **Direct Preference Optimization (DPO)**：直接基于偏好数据优化模型，无需显式训练奖励模型

关键选择：放弃 PPO（InstructGPT 方法），选用 DPO——更简单、更稳定、更容易规模化。

### 4. 工具使用与推理

Llama 3 原生支持工具调用（function calling），模型在预训练和后训练阶段都被训练来理解和执行工具使用。

### 5. 多模态组合式集成

Llama 3 通过组合式方法（compositional approach）集成多模态能力：
- 在语言模型之上接入视觉编码器、语音编码器/解码器
- 图像、视频和语音识别任务上表现竞，接近 SOTA
- 多模态版本仍在开发中，暂未广泛发布

---

## 实验/评估/结果

### 关键基准（基于 clipping 中的 Table 2，数值以论文为准）

Llama 3 405B Instruct 在以下基准上与 GPT-4 竞争：
- MMLU、HumanEval、GSM-8K 等核心 benchmark
- 在多数任务上与 GPT-4 持平或接近
- 8B 和 70B 版本分别在同等参数量级中领先同类模型

### 人类评估

广泛的 A/B 人类偏好评估确认 Llama 3 405B 与 GPT-4 在多数使用场景下表现相当。

### 安全与有用性平衡

Llama 3 在 helpfulness 和 harmlessness 之间达到了比 Llama 2 更好的平衡。

---

## 结论

Llama 3 证明了在开源路线上，通过数据规模、模型规模和复杂度管理的正确平衡，可以构建与 GPT-4 竞争的大语言模型。标准的 Dense Transformer 架构在 405B 规模上仍然有效，SFT+RS+DPO 的简洁后训练方案足以实现优秀的指令遵循和对齐性能。

---

## 思考

### 优点

1. **开源的重大推进**：405B 的完全开源是里程碑式的。它证明了开源社区在不需要闭源"魔法"的情况下也能达到 GPT-4 水平。

2. **复杂度管理哲学的正确性**：不做 MoE、不用复杂 RL（放弃 PPO 采用 DPO），选择更可预测、更稳定的技术路线。这种"在简单方案上做大"的思路在工程上非常明智。

3. **"过度训练"小模型的策略**：8B 和 70B 的训练远超计算最优——这让它们在推理时性价比极高。这是一个实用主义的洞见：scaling law 说的是训练阶段的最优，但推理阶段的成本也很重要。

4. **后训练流程简化**：SFT + Rejection Sampling + DPO 比 PPO-based RLHF 简单得多，但仍达到了优秀效果。这对开源社区的技术选择有重要指导意义。

### 缺点与局限

1. **Clipping 不完整**：本笔记基于的 clipping 仅 54 行（被截断），缺少大部分技术细节（如架构细节、具体的训练超参数、消融实验等），很多分析无法深入。

2. **多模态仍在开发**：虽然论文提到了图像/视频/语音的能力集成，但并未广泛发布，实际可用性有待验证。

3. **缺乏 Chinchilla 级别的方法论创新**：Llama 3 本质上是对已知技术的大规模执行和开源投入，不像 Chinchilla 那样带来了对 scaling law 的根本性修正。

4. **405B 的推理成本**：即使开源，405B 参数模型的推理需要大量 GPU 资源（8 块 A100/H100），对绝大多数个人和中小机构来说仍然不可行。

### 与已有 Wiki 的连接

- 关联概念：[[Scaling Law]]、[[DPO]]、[[Rejection Sampling]]、[[Tool Use]]
- 关联论文：[[AI阅读笔记/LLaMA]]（Llama 系列创刊作）、[[AI阅读笔记/Chinchilla 缩放定律]]（训练 15T tokens 远超 Chinchilla 最优）、[[AI阅读笔记/GPT-4]]（主要竞品）
- 关联实体：[[Llama 3]]
- 后续演进：Llama 4 的进展
