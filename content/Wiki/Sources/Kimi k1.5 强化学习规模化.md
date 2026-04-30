---
type: source
status: processed
source_file: '[[Clippings/Kimi k1.5 Scaling Reinforcement Learning with LLMs.md]]'
title: 'Kimi k1.5: Scaling Reinforcement Learning with LLMs'
site: Kimi Blog
author: Moonshot AI
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - RL
  - 推理模型
  - Kimi
  - long2short
  - 多模态
aliases:
  - Kimi k1.5
---
# Kimi k1.5 强化学习规模化

## 核心结论

- [[Kimi k1.5]] 提出将 RL 训练上下文扩展到 128K tokens，通过长上下文实现更复杂推理链路的强化学习。
- 核心创新包括 partial rollouts（提高长轨迹训练效率）、online mirror descent 策略优化、以及 length penalty 控制推理长度。
- 提出了 long2short 方法（模型合并、最短拒绝采样、DPO、long2short RL），将长链推理能力压缩到短输出中，实现效率与效果的平衡。
- 同时将 RL 训练扩展到多模态（文本+视觉），证明强化学习范式在视觉推理上同样有效。

## 关键事实

- 上下文长度扩展至 128K tokens，支持多轮长链推理的 RL 训练。
- 策略优化采用 online mirror descent，相比于 PPO 更稳定。
- curriculum sampling 和 prioritized sampling 控制训练数据难度，逐步提升。
- long2short 四种方法：model merging、shortest rejection sampling、DPO、long2short RL。
- 在数学和编程 benchmarks 上达到 SOTA 水平。

## 方法或论证路径

- 长上下文 RL 训练（128K）带来明显性能提升，但需要 partial rollouts 来提高效率。
- Online mirror descent 比传统 PPO 更稳定，尤其在大规模训练中。
- Long2short 管线验证了长链推理能力可以高效蒸馏到短输出模式。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/国产大模型演进]]、[[Wiki/Concepts/推理模型与强化学习]]
- 与 [[Wiki/Sources/DeepSeek-R1 强化学习推理]] 对照：Kimi 更侧重上下文扩展和 long2short，DeepSeek-R1 更侧重纯 RL 推理涌现。
- 支持：RL 是推理模型训练的有效范式，多模型/多团队独立验证。

## 可能的矛盾或待核实点

- Long2short 方法的泛化性：是否所有推理能力都能通过蒸馏压缩？
- 与 DeepSeek-R1 的不同训练策略（GRPO vs online mirror descent）对比效果待深入分析。

## 后续问题

- long2short 中的四种方法各自贡献度如何？能否量化分析？
- 128K 极限是否已经足够，还是有进一步扩展空间？
