---
type: source
status: processed
source_file: >-
  [[Clippings/DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via
  Reinforcement Learning]]
title: >-
  DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement
  Learning
site: arXiv
author: DeepSeek-AI
published: 2025-01
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - deepseek
  - rl
  - reasoning
  - grpo
  - distillation
aliases: []
---

# DeepSeek-R1: 强化学习激励推理能力

## 核心结论

- 首次开源验证：无需 SFT 冷启动，直接对 base model 应用大规模 RL 即可显著激发推理能力（DeepSeek-R1-Zero 在 AIME 2024 从 15.6% 提升到 71.0%）。
- DeepSeek-R1-Zero 在 RL 过程中自然涌现自我验证、反思、长 CoT 等高级推理行为，出现 "aha moment"。
- DeepSeek-R1 通过少量冷启动数据 + 多阶段训练（推理 RL + 拒绝采样 SFT + 全场景 RL），达到与 OpenAI-o1-1217 相当的推理性能。
- 蒸馏比小型模型的 RL 训练更高效：DeepSeek-R1-Distill-Qwen-32B 在 AIME 达 72.6%，远超同等算力下的小模型 RL 训练。

## 关键事实

- 基础模型：DeepSeek-V3-Base，671B MoE，37B 激活。
- RL 算法：GRPO（Group Relative Policy Optimization），无需 critic 模型。
- 奖励设计：rule-based reward（准确度 + 格式），不使用过程奖励模型（PRM），避免 reward hacking。
- 冷启动数据：数千条长 CoT 数据，包含可读格式 + 摘要。
- 蒸馏：使用 800K 样本微调 Qwen 和 Llama 系列，1.5B / 7B / 8B / 14B / 32B / 70B 六个规模。
- 尝试的失败方案：PRM（细粒度步骤定义难、reward hacking）、MCTS（搜索空间过大、价值模型难训练）。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Entities/DeepSeek 系列模型]]、[[Wiki/Concepts/推理模型与强化学习]]、[[Wiki/Topics/国产大模型演进]]
- DeepSeek-R1 与 [[Wiki/Sources/Kimi k1.5 强化学习规模化]] 同为国产 RL 推理路线的核心论文，两者在方法上有显著对比价值。
- 对 OpenAI o1 的开源对标，是 RL 推理范式的重要里程碑。

## 可能的矛盾或待核实点

- R1-Zero 可读性和语言混合问题严重，冷启动是否是必须的折中？
- 蒸馏 vs RL 的结论（蒸馏更优）在不同基座模型和更大规模 RL 下是否仍然成立？

## 后续问题

- 长 CoT 如何扩展到软件工程和 function calling 任务？
- 如何进一步提升 RL 训练时的 token 效率？
