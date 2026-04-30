---
type: source
status: processed
source_file: >-
  [[Clippings/DeepSeek-V2 A Strong, Economical, and Efficient Mixture-of-Experts
  Language Model]]
title: >-
  DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language
  Model
site: arxiv
author: DeepSeek-AI
published: 2024-05
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - DeepSeek-V2
  - MLA
  - MoE
  - KV cache
  - 高效推理
aliases:
  - DeepSeek-V2
---
# DeepSeek-V2 技术报告

## 核心结论

DeepSeek-V2 是 DeepSeek 系列的关键转折点，首次引入两个核心架构创新：**MLA（Multi-head Latent Attention）**和 **DeepSeekMoE**。MLA 通过将 KV cache 压缩为低维潜在向量，大幅降低推理开销（KV cache 减少 93.3%）；DeepSeekMoE 通过稀疏计算实现经济训练。236B 总参/21B 激活，128K 上下文，相比 DeepSeek 67B 性能更强但训练成本降低 42.5%、生成吞吐量提升 5.76 倍。

## 关键事实

- 作者：DeepSeek-AI，2024 年 5 月
- 参数：236B 总 / 21B 激活（MoE），支持 128K 上下文
- 训练数据：8.1T tokens 高质量多源语料
- 训练流程：预训练 → SFT → RL（GRPO 的早期版本）
- **MLA**：将 Key-Value 投影到低维潜空间再展开，大幅减少每 token 的 KV cache 内存占用
- **DeepSeekMoE**：共享专家 + 细粒度路由专家设计，增加专家数量同时保持计算效率
- 对齐：SFT + GRPO 强化学习，Chat 版本在开源模型中达到顶级水平

## 方法或论证路径

- MLA 核心洞察：注意力键/值矩阵是低秩的，可被压缩到远小于每头维度的潜在向量
- 与标准 MHA 相比：MLA 的 KV cache 大小从 2 × n_heads × d_head × n_layers 降至 1 × d_latent × n_layers
- DeepSeekMoE：与传统 MoE（每 token 选 1-2 个专家）不同，使用更细粒度的专家分配
- 经济性对比：与 DeepSeek 67B dense 比较，全面展示推理效率优势

## 与现有 Wiki 的关系

- 关联：[[Wiki/Entities/DeepSeek 系列模型]]——**V2 是 MLA 首次引入的里程碑，必须在此页面详细补充**
- 关联：[[Wiki/Concepts/MoE 混合专家模型]]
- 继承：V2 的技术（MLA + DeepSeekMoE）被 V3 完整继承和优化
- 区别：V2 的基础能力被后续 Coder-V2 代码版本增强

## 可能的矛盾或待核实点

- V2 报告将 GRPO 列为 RL 方法，但 GRPO 的详细公式在 R1 报告中才首次公开

## 后续问题

- MLA 的压缩率是否可以进一步降低（d_latent 的最小有效值）？
- MLA 与 NSA（V3.2 的稀疏注意力）是否冲突或可协同？
