---
type: source
status: processed
source_file: >-
  [[Clippings/Switch Transformers Scaling to Trillion Parameter Models with
  Simple and Efficient Sparsity]]
title: >-
  Switch Transformers: Scaling to Trillion Parameter Models with Simple and
  Efficient Sparsity
site: arxiv
author: 'William Fedus, Barret Zoph, Noam Shazeer (Google)'
published: '2021'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - MoE
  - Switch Transformer
  - 万亿参数
  - Google
  - 架构
aliases:
  - Switch Transformer
---
# Switch Transformer

## 核心结论

Google 提出 Switch Transformer，通过简化的 top-1 门控（每 token 只路由到一个专家）替代多专家软组合，大幅简化 MoE 路由算法并降低通信/计算成本。首次成功训练万亿参数级稀疏模型（Switch-C），预训练速度提升 7 倍，并在 101 种语言的多语言任务上取得增益。解决了 MoE 长期以来的复杂性、通信成本和训练不稳定三大障碍。

## 关键事实

- 作者：William Fedus、Barret Zoph、Noam Shazeer（Google），2021
- 核心简化：每 token 仅路由到 top-1 专家（vs 原始 MoE 的 top-k），门控计算量显著降低
- Switch-C：1.6T 参数，2048 个专家，在 Colossal Clean Crawled Corpus (C4) 上预训练
- 训练技术：选择性精度（bfloat16）、负载均衡损失、专家 dropout
- 基于 T5 架构：T5-Base → Switch-Base 实现 7 倍预训练加速
- 多语言 mT5 版本在全部 101 种语言上均获得提升

## 方法或论证路径

- Top-1 路由：只用 softmax 最大概率的路由决策，门控网络输出退化为一维
- 负载均衡损失：fraction of tokens dispatched × fraction of router probability
- 高容量因子（capacity factor）：控制每个专家处理的 token 上限，防止负载不均导致计算过载
- 蒸馏实验：Switch 模型可以蒸馏回更小的 dense 模型，保持大部分性能
- Mesh-TensorFlow：利用模型分片和数据并行混合策略支持超大规模

## 与现有 Wiki 的关系

- 关联：[[Wiki/Concepts/MoE 混合专家模型]]——本来源是 MoE 从实验到生产的关键过渡
- 补充：Switch Transformer 的 top-1 简化与 DeepSeek/V3 的多专家选择形成设计对比
- 继承：直接继承自 Google Brain 2017 年 MoE 工作，是理论到实用的桥梁

## 可能的矛盾或待核实点

- Top-1 vs Top-k 门控的优劣：Switch 认为多专家选择不必要的复杂度，但 DeepSeek-V3 使用 8 专家/256 总数仍保持高效

## 后续问题

- 万亿参数模型的推理部署成本是否实际可行？
- Switch Transformer 的蒸馏策略与 DeepSeek R1 的蒸馏策略有何异同？
