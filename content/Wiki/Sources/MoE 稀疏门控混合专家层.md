---
type: source
status: processed
source_file: >-
  [[Clippings/Outrageously Large Neural Networks The Sparsely-Gated
  Mixture-of-Experts Layer]]
title: >-
  Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
  Layer
site: arxiv
author: Noam Shazeer et al. (Google Brain)
published: '2017'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - MoE
  - 条件计算
  - 稀疏门控
  - Google Brain
  - 架构
aliases:
  - Sparsely-Gated MoE
---
# MoE 稀疏门控混合专家层

## 核心结论

Google Brain 提出 Sparsely-Gated Mixture-of-Experts（MoE）层，首次在神经网络中实现条件计算（conditional computation）：每 token 仅激活部分子网络（专家），模型容量提升 1000 倍以上，同时保持计算效率。通过引入可微分稀疏门控和负载均衡损失函数，解决了"条件计算长期停留在理论阶段"的实践难题。

## 关键事实

- 作者：Noam Shazeer、Azalia Mirhoseini、Krzysztof Maziarz、Andy Davis、Quoc Le、Geoffrey Hinton、Jeff Dean（Google Brain），2017
- 架构：稀疏门控网络从上千个 FFN 专家中选择 top-k（k=1~4）激活
- 语言建模实验：MoE 层替换 LSTM 中的 FFN 层，模型容量达 137B 参数，但每 token 计算量接近 baseline
- 训练挑战：门控网络软饱和、负载不均、通信开销大
- 贡献：Noisy Top-k Gating（训练时加噪声提高探索）+ 辅助负载均衡损失

## 方法或论证路径

- 门控函数 G(x) = Softmax(KeepTopK(x · Wg + noise, k))
- 噪声作用：训练时打破确定性选择的固有偏差，帮助专家间探索
- 两个辅助损失：负载均衡损失（encourage equal load）+ 重要性损失（encourage equal importance）
- 语言建模（1B words）和机器翻译（WMT'14 En→De）实验验证

## 与现有 Wiki 的关系

- 关联：[[Wiki/Concepts/MoE 混合专家模型]]——本来源是 MoE 概念的开山之作
- 补充：为 DeepSeek MoE、Kimi MoE、Switch Transformer 等后续工作提供了理论和工程基础
- 对比：当前 MoE 页面的负载均衡方案（aux-loss-free bias）是对本论文 auxiliary loss 方案的后续改进

## 可能的矛盾或待核实点

- 论文的 top-k 门控（k>1）与 Switch Transformer 的 top-1 门控形成设计分歧

## 后续问题

- 辅助损失对模型性能的实际损害有多大？DeepSeek 的 aux-loss-free 方案的提升是否可量化？
