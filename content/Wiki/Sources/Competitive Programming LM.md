---
type: source
status: processed
source_file: '[[Clippings/Competitive Programming with Large Reasoning Models.md]]'
title: Competitive Programming with Large Reasoning Models
site: arXiv
author: OpenAI
published: 2025-02
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - reasoning-model
  - competitive-programming
  - RL
  - o3
aliases:
  - o3 reasoning model
---
# Competitive Programming with Large Reasoning Models

## 核心结论

- OpenAI 展示了大语言模型的强化学习（RL）训练可显著提升复杂编程和推理任务性能。
- 对比了两条路线：o1-ioi（针对 IOI 2024 的手工推理策略）和 o3（通用推理模型）。o1-ioi 在 IOI 2024 现场达到 49 分位，放松条件下可获金牌。
- 关键发现：**通用 RL 扩展（o3）超越了手工领域特定策略（o1-ioi）**，o3 在无需领域特化策略和放松约束的情况下即获 IOI 金牌，CodeForces 评级与精英人类选手相当。

## 关键事实

- 作者来自 OpenAI。2025 年 2 月（arXiv:2502.06807）。
- o1-ioi：使用手工设计的测试时推理策略（包括暴力枚举、回溯搜索、贪心+验证等），专门为 IOI 竞赛优化。
- o3：通过大规模强化学习训练的通用推理模型，无领域特化设计。
- 结果对比：o1-ioi（有手调策略）在 IOI 2024 达 49 分位 / 放松约束下获金牌；o3（无领域策略）直接获 IOI 金牌 + CodeForces elite 级评级。
- 附带发现：o1 在 CodeForces 上的性能系统性地低于 o3，表明 RL 扩展仍是性能提升的主导因素。

## 方法亮点

- 两类推理增强的清晰对比：领域工程 vs 通用扩展，结果明确指向后者是更稳健的路径。
- 以 IOI 竞赛为严格推理 benchmark，验证了 RL 对复杂组合推理的提升效果。
- 与 CoT/ToT/GoT 不同，本文的"推理"来自训练阶段的 RL 训练，而非仅靠推理时的提示策略。

## 与现有 Wiki 的关系

- 是从"推理提示"到"推理模型"的关键转折点，见 [[Wiki/Topics/推理增强方法]]。
- 与 [[Wiki/Sources/Chain-of-Thought]]、[[Wiki/Sources/Tree of Thoughts]]、[[Wiki/Sources/Graph of Thoughts]] 构成"从 prompt 到 RL 训练"的完整演进。

## 后续问题

- o3 的 RL 训练中隐含了怎样的推理策略？与显式的 ToT/GoT 关系如何？
- 推理能力增强是否必然以训练成本大幅提升为代价？
