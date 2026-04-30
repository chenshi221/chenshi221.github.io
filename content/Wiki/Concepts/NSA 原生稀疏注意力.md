---
type: concept
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - NSA
  - Native Sparse Attention
  - 原生稀疏注意力
tags:
  - NSA
  - 稀疏注意力
  - DeepSeek
  - 长上下文
  - 硬件对齐
sources:
  - '[[Wiki/Sources/NSA 原生稀疏注意力]]'
confidence: medium
---
# NSA 原生稀疏注意力

## 定义

NSA（Native Sparse Attention）是 DeepSeek 提出的硬件对齐且可原生训练的稀疏注意力机制，旨在降低长上下文建模的计算成本。NSA 采用动态层级稀疏策略，将粗粒度 token 压缩与细粒度 token 选择结合，在保持全局上下文感知和局部精度的同时大幅降低计算量。

## 核心设计：动态层级稀疏

```
全序列
  ├── 粗粒度压缩：分块 → 每块压缩为代表 token → 全局注意力
  ├── 细粒度选择：基于门控分数选择最重要 token → 精确注意力
  └── 滑动窗口：确保最近邻不被遗漏
  ──→ 三者结果融合
```

### 三个分支

1. **Token 压缩分支（Compression）**：将长序列分块，每块聚合为一个代表性 token（如通过 MLP 或 attention pooling），用于捕获全局上下文
2. **Token 选择分支（Selection）**：通过门控分数（gate score）在 token 级别动态选择最重要的 K 个 token 进行精确注意力计算
3. **滑动窗口分支（Sliding Window）**：保持最近的 W 个 token 的完整注意力，确保局部信息的精度

### 硬件对齐设计

传统稀疏注意力的问题是：虽然理论计算量少，但稀疏操作在现代 GPU 上的实际效率往往不如密集计算（因为 GPU 为密集矩阵乘法优化）。NSA 的硬件对齐设计通过算术强度均衡（arithmetic intensity balance）解决此问题——确保稀疏模式与 GPU 的 block-level 计算对齐。

## 与标准注意力的关系

| 维度 | 标准 Attention | NSA |
|------|---------------|-----|
| 复杂度 | O(N^2) | O(N)（近似） |
| 全局信息 | 精确 | 近似（压缩分支） |
| 局部精度 | 精确 | 精确（滑动窗口 + 选择） |
| 训练方式 | 原生 | **原生（端到端训练）** |
| 硬件效率 | 在长序列上低 | 硬件对齐，高效 |

"原生可训练"是 NSA 区别于其他稀疏注意力方法的关键——它不需要预定义的稀疏模式（如 Longformer 的固定滑动窗口 + 全局 token），而是在训练中端到端学习稀疏选择策略。

## 与 DeepSeek 技术栈的关系

- **MLA**（DeepSeek-V2）：减少 KV cache 的内存占用
- **NSA**（DeepSeek, 2025）：减少 attention 的计算量
- 两者是互补的：MLA 解决内存瓶颈，NSA 解决计算瓶颈

注意：需区分 NSA 与 **DSA（DeepSeek Sparse Attention）**（在 V3.2 中引入）。两者可能有关联或演进关系，需进一步核实。

## 来源

- [[Wiki/Sources/NSA 原生稀疏注意力]] — NSA 原始论文（DeepSeek, 2025）
- 关联：[[Wiki/Entities/DeepSeek 系列模型]]、[[Wiki/Concepts/MoE 混合专家模型]]
