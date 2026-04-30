---
type: source
status: processed
source_file: '[[Clippings/Qwen3 Technical Report.md]]'
title: Qwen3 Technical Report
site: Qwen Blog / GitHub
author: Alibaba Qwen Team
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - Qwen
  - 推理模型
  - MoE
  - 蒸馏
  - thinking-budget
aliases:
  - Qwen3
---
# Qwen3 技术报告

## 核心结论

- [[Qwen3]] 是阿里通义千问发布的 Dense + MoE 双架构模型系列，覆盖 0.6B 到 235B 参数范围。
- 核心创新是统一思考/非思考模式：通过 `/think` 和 `/no_think` 标志切换，同一模型即可做深度推理也可快速响应。
- 引入 thinking budget 机制：用户可指定推理 token 预算，控制推理深度与速度的平衡。
- 采用 strong-to-weak distillation：用大模型（strong teacher）蒸馏小模型（weak student），提升小模型推理质量。
- 四阶段后训练管线：长链 CoT 冷启动 -> 推理 RL -> 思考模式融合 -> 通用 RL。
- 训练数据 36T tokens 覆盖 119 种语言，在数学、代码、推理 benchmarks 上达到同规模 SOTA。

## 关键事实

- 模型系列：Dense（0.6B/1.7B/4B/8B/14B/32B）+ MoE（30B-A3B, 235B-A22B）。
- 统一思考切换：`/think` 开启深度推理，`/no_think` 关闭快速响应。
- Thinking budget：用户可限制推理 tokens 数量。
- 后训练四阶段：cold start SFT -> reasoning RL -> thinking fusion -> general RL。
- 训练 36T tokens、119 种语言。
- Strong-to-weak distillation 提升小模型推理能力。

## 方法或论证路径

- 统一思考模式实现"一模型两用"，避免了独立推理模型和普通模型的双轨维护成本。
- Thinking budget 提供了推理质量-速度的用户可控权衡。
- Strong-to-weak distillation 证明了大模型推理能力可以有效传递给小模型。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/国产大模型演进]]、[[Wiki/Concepts/推理模型与强化学习]]、[[Wiki/Concepts/MoE 混合专家模型]]
- 与 [[Wiki/Sources/DeepSeek-R1 强化学习推理]] 对照：Qwen3 的统一思考模式与 R1 的专用推理模型路线不同。
- 与 [[Wiki/Sources/Kimi k1.5 强化学习规模化]] 对照：Qwen3 的 strong-to-weak distillation 与 Kimi 的 long2short 均为推理能力压缩方法。
- 支持：四阶段后训练管线是当前推理模型训练的共同方法论。

## 可能的矛盾或待核实点

- 统一思考模式是否会在两种模式间产生性能折中？是否存在模式切换的开销？
- Strong-to-weak distillation 的保真度上限是多少？是否始终可以逼近 teacher 水平？

## 后续问题

- Qwen3 的 MoE 架构（30B-A3B, 235B-A22B）与 DeepSeek/Kimi 的 MoE 设计有何差异？
- Thinking budget 的细粒度控制是否在复杂推理任务中有效？
