---
type: source
status: processed
source_file: '[[Clippings/DeepSeek-V3 Technical Report]]'
title: DeepSeek-V3 Technical Report
site: arXiv
author: DeepSeek-AI
published: 2024-12
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - deepseek
  - moe
  - mla
  - fp8-training
  - mega-model
aliases: []
---

# DeepSeek-V3 技术报告

## 核心结论

- DeepSeek-V3 是一个 671B 总参数 / 37B 激活参数的 MoE 语言模型，在 14.8T token 上预训练，训练成本仅 278.8 万 H800 GPU 小时（约 557 万美元）。
- 首次在大规模模型上验证 FP8 混合精度训练的可行性和有效性。
- 引入无辅助损失（auxiliary-loss-free）的负载均衡策略，避免负载均衡损失对模型性能的负面影响。
- 提出多 token 预测（MTP）训练目标，提升模型评估性能，并可用于推测解码加速推理。

## 关键事实

- 架构：MLA（多头潜在注意力）+ DeepSeekMoE（细粒度专家 + 共享专家），沿用 V2 设计并加以改进。
- 训练框架：DualPipe 流水线并行算法，实现计算-通信近乎完全重叠。
- 后训练：SFT + RL，并从 DeepSeek-R1 系列蒸馏推理能力。
- 性能：MMLU 88.5、GPQA 59.1、MATH-500 90.2、LiveCodeBench 领先，逼近 GPT-4o 和 Claude-3.5-Sonnet。
- 整个训练过程零不可恢复 loss spike，无需回滚。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Entities/DeepSeek 系列模型]]、[[Wiki/Concepts/MoE 混合专家模型]]、[[Wiki/Topics/国产大模型演进]]
- DeepSeek-V3 是 DeepSeek 系列的里程碑：从 V2 的架构验证走向大规模工程实现，是 V3.2 和 R1 的基础。
- 其 MoE 架构（细粒度专家 + 共享专家 + 无辅助损失负载均衡）是国产大模型的重要技术路线。

## 可能的矛盾或待核实点

- FP8 训练的泛化性尚待其他团队独立验证。

## 后续问题

- MoE 架构在推理场景下的负载均衡如何进一步优化？
- MTP 训练目标对小模型的收益如何？
