---
aliases:
  - DeepSeek 模型系列
confidence: high
created: '2026-04-30'
sources:
  - '[[Wiki/Sources/DeepSeek LLM 开源语言模型与长期主义]]'
  - '[[Wiki/Sources/DeepSeek-V2 技术报告]]'
  - '[[Wiki/Sources/DeepSeek-Coder-V2 代码智能]]'
  - '[[Wiki/Sources/DeepSeek-V3 技术报告]]'
  - '[[Wiki/Sources/DeepSeek-V3.2 开源大模型前沿]]'
  - '[[Wiki/Sources/DeepSeek-R1 强化学习推理]]'
status: active
tags:
  - DeepSeek
  - LLM
  - MoE
  - 推理模型
  - Agent
type: entity
updated: '2026-04-30'
---
# DeepSeek 系列模型

## 简介

DeepSeek（深度求索）是幻方量化旗下的大模型团队，以开源、高效、低成本著称。其模型系列从 scaling law 研究出发，逐步建立起 MLA-MoE 高效架构、GRPO 推理 RL、Agent 能力等完整技术栈。

## 模型演进

### DeepSeek LLM（首代，2024.01）

- 参数：7B / 67B
- 架构：LLaMA-like dense 架构
- 数据：2T tokens
- 贡献：提出以 non-embedding FLOPs/token（M）为核心的 scaling law，67B 超越 LLaMA-2 70B。
- 来源：[[Wiki/Sources/DeepSeek LLM 开源语言模型与长期主义]]

### DeepSeek-V2（2024.05）—— **MLA 首次引入的关键里程碑**

- 参数：236B 总参数 / 21B 激活参数（MoE）
- 训练数据：8.1T tokens 高质量多源语料
- 上下文：128K tokens
- 核心创新（两项架构级突破）：
  - **MLA（Multi-head Latent Attention）**：将 KV cache 压缩为低维潜在向量后再展开，**KV cache 减少 93.3%**。这是 DeepSeek 整个技术栈中最重要的原创贡献之一——大幅降低推理显存需求，为后续 V3/R1 的高效推理奠定基础。
  - **DeepSeekMoE**：共享专家 + 细粒度路由专家的双层设计，支持更灵活的专家分配。
- 效率对比（vs DeepSeek 67B dense）：
  - 性能显著更强，训练成本降低 42.5%
  - KV cache 减少 93.3%，生成吞吐量提升 5.76 倍
- 训练流程：预训练 → SFT → RL（GRPO 早期版本）
- Chat 版本在开源模型中达到顶级水平
- 来源：[[Wiki/Sources/DeepSeek-V2 技术报告]]

### DeepSeek-Coder-V2（2024.06）—— 代码智能特化

- 基于 DeepSeek-V2 中间 checkpoint，额外预训练 6T tokens 代码/数学数据
- 架构不变（MLA + DeepSeekMoE，236B 总/21B 激活）
- 核心提升：
  - 支持编程语言从 86 种扩展到 **338 种**
  - 上下文从 16K 扩展到 **128K**
  - 代码和数学任务达到 **GPT4-Turbo 级别**，开源模型中首次对标闭源代码模型
  - 同时保持通用语言能力
- 对标闭源：在 HumanEval、MBPP、LiveCodeBench、GSM8K、MATH 上超越 Claude 3 Opus 和 Gemini 1.5 Pro
- 来源：[[Wiki/Sources/DeepSeek-Coder-V2 代码智能]]

### DeepSeek-V3（2024.12）

- 参数：671B 总参数 / 37B 激活参数（MoE）
- 训练数据：14.8T tokens
- 训练成本：仅 \$5.576M（H800 GPU 小时）
- 核心创新：
  - **DeepSeekMoE**：辅助损失自由（auxiliary-loss-free）的负载均衡策略
  - **MLA**：Multi-head Latent Attention，高效推理
  - **FP8 混合精度训练**：首次在超大规模 MoE 上验证 FP8
  - **DualPipe**：计算-通信重叠流水线并行
  - **MTP**：Multi-Token Prediction 训练目标
- 来源：[[Wiki/Sources/DeepSeek-V3 技术报告]]

### DeepSeek-V3.1

- 在 V3 基础上进行持续预训练，增强基础能力。
- 具体改进细节参见 V3.2 报告。

### DeepSeek-V3.2（2025）

- **DSA（DeepSeek Sparse Attention）**：引入 lightning indexer + fine-grained token selection，实现高效稀疏注意力。
- **GRPO 稳定性策略**：无偏 KL 散度（unbiased KL）、off-policy sequence masking、keep routing、keep sampling mask。
- **Thinking-in-tool-use**：在工具调用过程中保持思维链，不多轮交替导致上下文膨胀。
- **大规模 Agent 任务合成**：覆盖 1800+ 环境、85000+ prompts。
- **Speciale 变体**：专注数学推理，获得 IMO/IOI 金牌。
- 来源：[[Wiki/Sources/DeepSeek-V3.2 开源大模型前沿]]

### DeepSeek-R1（2025.01）

- **R1-Zero**：首个纯 RL（无 SFT）训练出推理能力的实验。使用 GRPO + 基于规则的奖励（无 neural reward model），观察到 "aha moment"——模型在 RL 训练中自发产生反思和重新评估行为。
- **R1**：完整产品化版本，包含四阶段管线：
  1. 冷启动 SFT（数千条高质量推理样本）
  2. 推理导向 RL（GRPO + 语言一致性奖励）
  3. 拒绝采样 + 多域 SFT（约 800K 样本，涵盖写作、QA、翻译等）
  4. 全场景 RL（有用性 + 无害性对齐）
- **蒸馏模型**：从 R1 蒸馏出 6 个小型 dense 模型（1.5B-70B），蒸馏优于直接在小模型上做 RL。
- **失败尝试记录**：PRM（过程奖励模型）有 reward hacking 问题、MCTS（蒙特卡洛树搜索）在 token 级搜索空间过大。
- 来源：[[Wiki/Sources/DeepSeek-R1 强化学习推理]]

## 核心技术贡献一览

| 贡献 | 首次引入 | 说明 |
|------|----------|------|
| Scaling law (M) | DeepSeek LLM | 以 non-embedding FLOPs/token 为核心指标 |
| MLA | DeepSeek-V2 | Multi-head Latent Attention，KV cache 降低 93.3% |
| DeepSeekMoE | DeepSeek-V2/V3 | 辅助损失自由负载均衡 |
| Coder 特化 | DeepSeek-Coder-V2 | 338 语言代码模型，GPT4-Turbo 级，开源首次对标闭源 |
| FP8 训练 | DeepSeek-V3 | 大规模验证 |
| DualPipe | DeepSeek-V3 | 计算-通信重叠 |
| MTP | DeepSeek-V3 | Multi-Token Prediction |
| GRPO | DeepSeek-R1 | Group Relative Policy Optimization |
| R1-Zero 涌现 | DeepSeek-R1 | 纯 RL 推理涌现 |
| DSA | DeepSeek-V3.2 | DeepSeek Sparse Attention |
| Agent 合成 | DeepSeek-V3.2 | 1800+ 环境大规模合成 |

## 设计哲学

1. **效率优先**：MLA、MoE、FP8、DualPipe 均围绕降低训练和推理成本。
2. **开源驱动**：所有模型均开源，推动社区发展。Coder-V2 的 338 语言覆盖体现了开源对长尾场景的价值。
3. **RL 优先于 SFT**：R1-Zero 证明 RL 可以独立于 SFT 激发推理能力。
4. **Agent 原生**：V3.2 的 thinking-in-tool-use 和 agent 合成表明将 Agent 能力内化到模型训练中。
5. **代码特化分支**：Coder-V2 展示了在通用基座上通过继续预训练获得专项能力的可行性——338 种语言、GPT4-Turbo 级性能，同时保持通用能力不退化。

## 批判性评估

### 最大优势：完整的"全栈"创新

DeepSeek 是三家国产大模型中最"自洽"的一家。从基础架构（MLA）到训练策略（aux-loss-free MoE, FP8, DualPipe）到推理方法（GRPO, R1-Zero 涌现）到 Agent（V3.2 合成），形成了一个不需要外部依赖的技术闭环。这种自洽性使得每个组件的优化可以惠及其他组件——MLA 降低了推理开销，使得 MoE 的稀疏架构更可行；GRPO 的稳定性策略可以反哺预训练。

### 最大风险：复杂度诅咒

自洽性的另一面是复杂度。当你的架构栈有 5 个原创组件（MLA + DeepSeekMoE + FP8 + DualPipe + GRPO），任何一个出问题都很难 debug——因为没有现成的参考。这在长期可能形成"维护负担"：新成员需要理解整个自研栈才能贡献，限制了社区贡献的深度。

### R1-Zero 的浪漫与实用

R1-Zero 是 DeepSeek 最具科学价值的贡献——纯 RL 推理涌现是对"推理需要人类教"这一假设的根本挑战。但它被过度浪漫化了。"aha moment" 听起来像科幻，实际上可能是 RL 训练的统计必然——GRPO 奖励"正确的长推理"，模型自然会学会探索更长的推理链。把"涌现"换成"强化学习梯度引导的行为变化"，虽然不那么激动人心，但更接近事实。

### 蒸馏策略的隐藏洞见

R1 蒸馏出的 6 个小模型表现优于直接在小型基座模型上做 RL——这个结论被三家独立验证（Kimi long2short, Qwen3 strong-to-weak），可靠性很高。但它有一个隐藏的推论：**如果你的目标是得到一个高质量小模型，训练一个大 teacher 然后蒸馏，可能比直接训练一个小模型更高效。** 这改变了"模型效率"的计算方式——小模型的训练成本要加上 teacher 的训练成本。

### 开源策略的双刃剑

DeepSeek 完全开源（权重 + 技术报告细节），赢得了巨大的社区声誉。但这也意味着竞争对手可以研究它的每个技术细节并吸收。Kimi 的技术报告详细但权重不完全开源，Qwen3 的权重开源但技术报告细节最少——三家在"开放到什么程度"上做出了不同选择。DeepSeek 的选择最大胆，也最容易被"学习"。
