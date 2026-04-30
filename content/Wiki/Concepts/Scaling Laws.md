---
type: concept
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - 缩放定律
  - Scaling Law
  - Chinchilla定律
  - 计算最优缩放
sources:
  - '[[Wiki/Sources/Chinchilla 缩放定律]]'
  - '[[Wiki/Sources/GPT-3]]'
  - '[[Wiki/Sources/LLaMA]]'
  - '[[Wiki/Sources/Llama 3]]'
tags:
  - scaling
  - LLM
  - compute-optimal
  - Chinchilla
confidence: high
---
# Scaling Laws（缩放定律）

## 定义

Scaling Laws（缩放定律）描述语言模型性能随模型规模、数据量和计算量增长而提升的定量规律。核心问题是：给定计算预算，如何分配模型参数和数据以获得最佳性能？

## 关键研究

### Kaplan et al. Scaling Laws (2020)

- OpenAI 的早期缩放定律研究，基于 GPT 系列模型
- 结论：模型性能与模型大小、数据集大小、训练计算量三者的幂律关系
- 启发 GPT-3 的 175B 参数设计

### Chinchilla 缩放定律 (2022) -- 当前标准

- 来源：[[Wiki/Sources/Chinchilla 缩放定律|Training Compute-Optimal Large Language Models (DeepMind)]]
- 训练了 400+ 模型（70M - 16B 参数，5B - 500B token）
- **核心发现**：
  - 以前的大模型（GPT-3 175B、Gopher 280B）都**严重训练不足**
  - 计算最优时，模型参数量和训练 token 数应**等比例增长**
  - 给定 FLOPs 预算 C，N_opt ∝ C^0.5，D_opt ∝ C^0.5
- 验证：Chinchilla（70B + 1.4T token）使用与 Gopher（280B）同样的预算但全面超越

### 两种缩放定律的对比

| 维度 | Kaplan et al. (2020) | Chinchilla (2022) |
|------|---------------------|-------------------|
| 核心结论 | 参数增长应快于数据 | 参数和数据应等比例增长 |
| 最优关系 | N ∝ C^0.73 | N ∝ C^0.5 |
| 影响 | 促使做大模型（GPT-3 175B 只在 300B token 上训练） | 促使用更多数据训练较小模型（LLaMA 65B + 1.4T token） |
| 数据重复 | 允许较多 epoch | 最多 4 epoch |

## 对工业界的影响

- **LLaMA**：65B 模型 + 1.4T token，直接验证 Chinchilla 结论
- **Llama 3**：405B + 15.6T token，模型和数据等比例增长
- **过度训练策略**：Llama 3 的 8B/70B 小模型故意训练超过计算最优点，以在推理时用低成本达到高水平
- **数据优先**：缩放定律推动行业从「更大模型」转向「更多高质量数据」

## 注意事项

- 缩放定律来自特定数据分布和架构的统计发现，不完全适用于 MoE 等新架构
- 数据质量的影响未被缩放定律完全捕捉：高质量数据可能改变最优比例
- 推理效率考虑不同于训练效率：过度训练小模型可能在推理时更经济

## 相关页面

- [[Wiki/Sources/Chinchilla 缩放定律|Chinchilla 缩放定律来源页]]
- [[Wiki/Sources/GPT-3|GPT-3]]
- [[Wiki/Sources/LLaMA|LLaMA]]
- [[Wiki/Sources/Llama 3|Llama 3]]
- [[Wiki/Topics/大语言模型基础|大语言模型基础]]
