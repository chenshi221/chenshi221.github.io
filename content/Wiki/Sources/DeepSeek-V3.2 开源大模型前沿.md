---
type: source
status: processed
source_file: '[[Clippings/DeepSeek-V3.2 Pushing the Frontier of Open Large Language Models]]'
title: 'DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models'
site: arXiv
author: DeepSeek-AI
published: 2025-12
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - deepseek
  - sparse-attention
  - rl
  - agent
  - reasoning
aliases: []
---

# DeepSeek-V3.2: 开源大模型前沿

## 核心结论

- DeepSeek-V3.2 通过三大技术突破实现性能跃升，达到与 GPT-5 可比水平：
  1) DeepSeek Sparse Attention (DSA) — 高效稀疏注意力机制，大幅降低长上下文计算复杂度。
  2) 可扩展 RL 框架 — 后训练计算预算超过预训练成本的 10%，Speciale 变体在 IMO 2025 和 IOI 2025 中取得金牌。
  3) 大规模 Agent 任务合成管线 — 生成超 1800 个环境、85000+ 复杂 prompt，驱动 RL 训练。

## 关键事实

- DSA 由闪电索引器（lightning indexer）+ 细粒度 token 选择组成，在 MLA 架构下实现。
- 通过 continued training 从 V3.1-Terminus 起步，经历 dense warm-up（1000 步 / 2.1B tokens）和 sparse training（15000 步 / 943.7B tokens）。
- GRPO 训练的稳定性策略：无偏 KL 估计、Off-Policy Sequence Masking、Keep Routing、Keep Sampling Mask。
- Thinking-in-Tool-Use：设计了一种上下文管理策略，在工具调用场景中保留推理内容，不再冗余重推理。
- V3.2-Speciale 在 IMO 2025 获得 35/42 金牌，ICPC WF 2025 解出 10/12 题，CMO 2025 金牌。
- Code Agent 评估：Terminal Bench 2.0 46.4（使用 Claude Code），SWE Verified 73.1，SWE Multilingual 70.2。
- 搜索 Agent：BrowseComp 67.6（含上下文管理），显著超越开源竞品。
- token 效率仍不及 Gemini-3.0-Pro。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Entities/DeepSeek 系列模型]]、[[Wiki/Concepts/MoE 混合专家模型]]、[[Wiki/Concepts/推理模型与强化学习]]、[[Wiki/Concepts/多模态 Agent]]、[[Wiki/Topics/国产大模型演进]]
- V3.2 是 DeepSeek 在 Agent + 推理方向的集大成者，DSA 是对 MoE 架构注意力瓶颈的重要突破。
- Agent 合成管线是开源社区大规模训练 Agent 能力的重要参考方法。

## 可能的矛盾或待核实点

- DSA 在非 MLA 架构上的有效性仍需验证。
- Agent 合成数据的泛化上限尚未完全探明（论文内合成数据 RL 在 Tau2Bench/MCP-Mark 上提升显著但仍有局限）。

## 后续问题

- token 效率如何与 Gemini-3.0-Pro 对齐？
- 如何更大规模地工业化部署 thinking-in-tool-use？
