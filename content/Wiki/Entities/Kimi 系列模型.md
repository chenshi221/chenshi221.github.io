---
aliases:
  - Kimi 模型系列
  - Moonshot AI 模型
confidence: high
created: '2026-04-30'
sources:
  - '[[Wiki/Sources/Kimi k1.5 强化学习规模化]]'
  - '[[Wiki/Sources/Kimi K2 开放 Agent 智能]]'
  - '[[Wiki/Sources/Kimi K2.5 视觉 Agent 智能]]'
  - '[[Wiki/Sources/Kimi-VL 技术报告]]'
status: active
tags:
  - Kimi
  - Moonshot
  - LLM
  - MoE
  - 多模态
  - Agent
type: entity
updated: '2026-04-30'
---
# Kimi 系列模型

## 简介

Kimi 是 Moonshot AI（月之暗面）推出的大模型系列，以长上下文 RL、MoE 高效架构、Agent 数据合成和联合多模态优化为主要特色。从 k1.5 的推理 RL 到 K2.5 的多模态 Agent，形成了一条清晰的能力递进路线。

## 模型演进

### Kimi k1.5

- 核心目标：将 RL 训练上下文扩展到 128K，实现长链推理的强化学习。
- 关键创新：
  - **Long context RL（128K）**：支持多轮长链推理轨迹。
  - **Partial rollouts**：提高长轨迹 RL 训练效率。
  - **Online mirror descent**：比 PPO 更稳定的策略优化。
  - **Length penalty**：控制推理输出长度。
  - **Curriculum/prioritized sampling**：逐步提升训练难度。
  - **Long2short（四种方法）**：model merging、shortest rejection sampling、DPO、long2short RL——将长链推理能力压缩到短输出。
- 多模态：同时将 RL 训练扩展到视觉-文本任务。
- 来源：[[Wiki/Sources/Kimi k1.5 强化学习规模化]]

### Kimi K2

- 参数：1.04T 总参数 / 32B 激活参数（MoE）。
- 训练数据：15.5T tokens。
- 核心创新：
  - **MuonClip 优化器**：Muon + QK-Clip 梯度裁剪，解决 MoE 训练中的 loss spike 问题。
  - **Sparsity scaling law**：在给定 FLOPs 预算下，稀疏度 48（4.8% 激活比例）为最优。
  - **大规模 Agent 数据合成**：四步管线——定义 tool specs -> 生成 agent 描述 -> 生成任务 -> 合成交互轨迹。
  - **RLVR + self-critique rubric reward**：无需人工标注的奖励机制。
- 定位：首个明确以 Agent 能力为核心的 Kimi 大模型。
- 来源：[[Wiki/Sources/Kimi K2 开放 Agent 智能]]

### Kimi K2.5

- 核心目标：联合文本-视觉优化的多模态 Agent。
- 关键创新：
  - **Zero-vision SFT**：SFT 阶段先不提供视觉，防止视觉训练导致文本能力退化，再引入视觉。
  - **Agent Swarm / PARL**：多 Agent 并行执行 + 反思循环，解决复杂多步任务。
  - **MoonViT-3D**：原生分辨率视觉编码器，结合 NaViT dynamic packing。
  - **Toggle token-efficient RL**：选择性激活视觉 tokens 进行 RL，降低训练开销。
  - **DEP（Decoupled Encoder Process）**：编码-解码分离处理，提升推理效率。
  - **Joint multimodal RL**：文本和视觉同时进行 RL 训练。
- 定位：从纯文本 Agent（K2）扩展到多模态 Agent（K2.5）。
- 来源：[[Wiki/Sources/Kimi K2.5 视觉 Agent 智能]]

### Kimi-VL

- 参数：2.8B 激活参数 MoE VLM。
- 核心创新：
  - **MoonViT**：原生分辨率视觉编码器，支持 NaViT packing。
  - **128K 上下文**：支持长文档、多页 PDF 等多帧输入。
  - **Kimi-VL-Thinking**：叠加 long-CoT SFT + RL 的推理变体。
- 定位：极小参数量达到 SOTA VLM 性能，证明小 MoE VLM 的潜力。
- 来源：[[Wiki/Sources/Kimi-VL 技术报告]]

## 核心技术贡献一览

| 贡献 | 首次引入 | 说明 |
|------|----------|------|
| Long context RL (128K) | k1.5 | 超长上下文强化学习 |
| Partial rollouts | k1.5 | 长轨迹 RL 效率优化 |
| Online mirror descent | k1.5 | 稳定策略优化 |
| Long2short | k1.5 | 四种蒸馏方法 |
| MuonClip (QK-Clip) | K2 | MoE 训练稳定优化器 |
| Sparsity scaling law | K2 | 稀疏度优化理论 |
| Agent 数据合成管线 | K2 | 四步自动化 Agent 数据生成 |
| RLVR + self-critique | K2 | 免人工标注奖励 |
| Zero-vision SFT | K2.5 | 防止多模态训练退化 |
| Agent Swarm / PARL | K2.5 | 多 Agent 并行协作 |
| MoonViT-3D | K2.5 | 3D 视觉编码器 |
| Toggle RL | K2.5 | Token 高效视觉 RL |
| DEP | K2.5 | 编码器解耦 |
| MoonViT | Kimi-VL | 原生分辨率 ViT |
| VL-Thinking | Kimi-VL | VL 推理变体 |

## 设计哲学

1. **RL 驱动**：从 k1.5 到 K2.5，RL 始终是核心训练方法，从文本 RL 扩展到多模态 RL。
2. **Agent 原生**：K2 开始将 Agent 能力内化到模型训练，K2.5 进一步扩展到多模态 Agent。
3. **效率优先**：MuonClip、partial rollouts、Toggle RL、DEP 均围绕训练和推理效率。
4. **多层次**：从纯文本推理（k1.5）-> Agent 推理（K2）-> 多模态 Agent（K2.5）-> 高效 VLM（VL），形成完整的能力矩阵。
5. **理论驱动**：Sparsity scaling law 体现了对架构设计的理论化思考。

## 批判性评估

### 最大优势：理论化思维

Kimi 是所有国产大模型团队中最"学院派"的。sparsity scaling law 不是简单的"多试几个稀疏度看哪个好"，而是建立了一个理论框架——在给定 FLOPs 预算下，稀疏度的最优值是多少。MuonClip 也不是简单的"加个梯度裁剪"，而是分析了 MoE 训练中 QK attention 导致 loss spike 的根因。这种"不满足于 work，还要知道 why"的态度，是长期技术积累的基础。

### 与 DeepSeek 的对比：谁更"原创"？

Kimi 和 DeepSeek 在 MoE + RL + Agent 三位一体的框架下走了非常相似的路。但深层差异是：DeepSeek 的原创更"显性"（新架构组件、新训练算法都有明确的首次引入时间），Kimi 的原创更"隐性"（在同一框架下做了不同的算法选择和理论分析）。R1-Zero 的纯 RL 涌现是"谁都能看懂"的创新，sparsity scaling law 是"需要仔细读才能发现价值"的创新。

### 开源策略的代价

Kimi 在三条线中开源度最低。技术报告详细但模型权重不完全开放。这在短期是保护商业优势，但长期可能导致社区生态落后——开发者更愿意围绕 DeepSeek 和 Qwen 建设工具和 benchmark。Agent 能力本身就是高度依赖"生态"的（工具定义、环境适配），如果社区不在 Kimi 上建设，Agent 能力的实际可用性会被削弱。

### K2.5 的 zero-vision SFT：一个反直觉的设计选择

"训练多模态 Agent 时先不给视觉"——这听起来很奇怪。但它揭示了一个真实问题：视觉 token 的信息密度远低于文本 token（一张图几千个 token，但携带的信息可能用几十个文本 token 就能描述）。如果模型在训练早期就依赖视觉 token，它可能学会"看图"而不是"理解任务"。Zero-vision SFT 强制模型先学会 Agent 的核心能力（规划、工具使用），再加入视觉。这是一个聪明的工程决策，但它也暗示**当前的训练方法无法让模型同时学习文本 Agent 和视觉感知**——这是一个值得深入研究的问题。

### VL-Thinking 的潜力

Kimi-VL-Thinking（2.8B 激活 MoE VLM + long-CoT RL）是目前已知最小的"推理 VLM"。如果这个极小模型在视觉推理上能与 GPT-4o 竞争，那它代表的"小 MoE + 推理 RL"路线可能比 Dense 大 VLM 路线更高效。但性能数据来自 Kimi 自己，需要第三方验证。
