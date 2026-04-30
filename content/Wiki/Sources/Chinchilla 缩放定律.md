---
type: source
status: processed
site: arXiv
source_file: '[[Clippings/Training Compute-Optimal Large Language Models.md]]'
title: Training Compute-Optimal Large Language Models
author: Jordan Hoffmann et al. (DeepMind)
published: '2022'
processed: '2026-04-30'
tags:
  - Chinchilla
  - scaling-laws
  - compute-optimal
  - DeepMind
---
# Training Compute-Optimal Large Language Models (Chinchilla 缩放定律)

## 核心结论

- 训练了超过 400 个不同规模的 Transformer 语言模型，发现当前主流 LLM（GPT-3、Gopher、Megatron-Turing NLG 等）明显**训练不足**——模型太大但数据量不够。
- **计算最优缩放定律**：模型参数量和训练 token 数应当等比例增长——模型翻倍，数据也应翻倍。给定 FLOPs 预算，最优模型应该比以前的实践更小、但训练更久。
- 验证实验：训练 Chinchilla（70B 参数，1.4T token），使用与 Gopher（280B）相当的计算预算，但数据多 4 倍。Chinchilla 在所有评估任务上大幅超越 Gopher、GPT-3（175B）和 Megatron-Turing NLG（530B）。
- Chinchilla 在下游微调任务上也一致优于更大模型。

## 关键方法或创新点

- **三族缩放定律分析**：固定训练 FLOPs，通过参数化拟合（三族方法）找到给定计算预算下的最优参数-数据分配。
- 数据重复：最多训练 4 个 epoch（每个 token 最多见 4 次），更多的重复对性能提升有限。
- 实验规模：70M 到 16B 参数，5B 到 500B token。
- 结论公式：给定 C FLOPs，最优参数量 N_opt ∝ C^0.5，最优 token 数 D_opt ∝ C^0.5。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Concepts/Scaling Laws]]、[[Wiki/Topics/大语言模型基础]]
- Chinchilla 定律直接指导了后续模型训练：LLaMA（1.4T token on 65B）、Llama 3（15T token on 405B）都遵循「更多数据、更小模型」的思路。
- 与 Kaplan et al. (2020) 的早期缩放定律形成对比和修正。

## 局限或注意事项

- 发现的是计算瓶颈下的最优，而非推理瓶颈下的最优（推理时小模型更便宜，但可能需要更多次调用）。
- 数据质量的影响未被系统研究（Chinchilla 使用 MassiveText，一个质量较高的数据集）。
- 对于不同的模型架构（如 MoE）和数据混合策略，最优缩放关系可能不同。
