---
type: source
status: processed
source_file: >-
  [[Clippings/Native Sparse Attention Hardware-Aligned and Natively Trainable
  Sparse Attention]]
title: >-
  Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse
  Attention
site: arxiv
author: Jingyang Yuan et al. (DeepSeek-AI & PKU)
published: '2025'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - NSA
  - 稀疏注意力
  - DeepSeek
  - 长上下文
  - 硬件对齐
aliases:
  - NSA
---
# NSA 原生稀疏注意力

## 核心结论

DeepSeek 提出 NSA（Native Sparse Attention），一种硬件对齐且可原生训练的稀疏注意力机制。NSA 采用动态层级稀疏策略：粗粒度 token 压缩 + 细粒度 token 选择，既保留全局上下文感知又保持局部精度。关键创新：(1) 算术强度均衡的算法设计，实现与现代硬件的良好对齐；(2) 支持端到端原生训练，无需预定义稀疏模式。

## 关键事实

- 作者：Jingyang Yuan 等（DeepSeek-AI & PKU），2025
- 核心设计：动态层级稀疏 = 粗粒度压缩（block-level）+ 细粒度选择（token-level）
- 长上下文 → 高效：标准注意力的计算量随序列长度平方增长，NSA 通过稀疏化降低
- 硬件对齐：算法设计考虑 GPU 算术强度平衡，避免稀疏计算引入的额外开销超过节省
- 可原生训练：与需要预计算稀疏模式的方法不同，NSA 在训练中端到端学习稀疏选择

## 方法或论证路径

- 粗粒度压缩：将长序列分块，每块压缩为代表性 token，用于全局注意力
- 细粒度选择：基于门控分数在 token 级别选择最重要的 token 进行精确注意力
- 层级组合：粗粒度全局 + 细粒度局部 + 滑动窗口（确保最近邻不被遗漏）
- 在长上下文语言建模和检索任务上验证效果和效率

## 与现有 Wiki 的关系

- 关联：[[Wiki/Concepts/NSA 原生稀疏注意力]]、[[Wiki/Entities/DeepSeek 系列模型]]
- 补充：DSA（DeepSeek Sparse Attention，V3.2 引入）与 NSA 可能有关联或演进关系，需进一步核实
- 与 Transformer 原始注意力、FlashAttention 等形成注意力机制的演进链条

## 可能的矛盾或待核实点

- NSA 与 DSA（DeepSeek-V3.2）的关系：是同一技术还是不同方案？需对照 V3.2 报告确认
- 稀疏注意力在推理 vs 训练中的计算收益是否一致

## 后续问题

- NSA 在极长上下文（1M+ token）下的 scaling 表现
- 与其他稀疏注意力（如 sliding window、Longformer）的定量对比
