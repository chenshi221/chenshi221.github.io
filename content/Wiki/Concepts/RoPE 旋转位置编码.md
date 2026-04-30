---
type: concept
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - RoPE
  - 旋转位置编码
  - Rotary Position Embedding
sources:
  - '[[Wiki/Sources/RoPE]]'
  - '[[Wiki/Sources/LLaMA]]'
  - '[[Wiki/Sources/Llama 3]]'
tags:
  - position-encoding
  - RoPE
  - transformer
  - LLM
confidence: high
---
# RoPE 旋转位置编码

## 定义

RoPE (Rotary Position Embedding) 是一种位置编码方法，由 Zhuiyi Technology 在 2021 年提出。它通过旋转矩阵编码绝对位置信息，同时在自注意力计算中自然地融入相对位置依赖关系。来源：[[Wiki/Sources/RoPE|RoFormer 论文]]

## 核心原理

### 旋转编码

对每个 query 向量 q 和 key 向量 k，按维度对 (2i, 2i+1) 进行二维旋转：

```
q_m = R_m · q    (位置 m 的 query)
k_n = R_n · k    (位置 n 的 key)
```

其中 R_m 是旋转角度 θ_i · m 的旋转矩阵，θ_i = 10000^(-2i/d)。

### 相对位置的自然涌现

关键性质：注意力分数 q_m^T · k_n 自然包含了相对位置信息：

```
q_m^T · k_n = q^T · R_{n-m} · k
```

即注意力分数只依赖于 query 和 key 的**相对位置 (n-m)**，而非绝对位置。

## 优良性质

1. **相对位置显式编码**：注意力自然携带相对位置信息，而原始正弦编码需要模型通过学习来隐式推导
2. **远程衰减**：随着相对距离增大，注意力权重自然衰减，符合语言的自然特性
3. **任意长度灵活性**：可以在推理时无缝扩展到比训练更长的序列
4. **兼容线性注意力**：可直接集成到线性自注意力中保留位置信息

## 与原始正弦编码的对比

| 特性 | 原始正弦编码 | RoPE |
|------|------------|------|
| 编码方式 | 直接加到 embedding 上 | 在 attention 计算中施加旋转 |
| 相对位置 | 需要模型学习推导 | 自然体现在 attention score 中 |
| 外推能力 | 理论支持，实际有限 | 有限（需 NTK/YaRN 扩展辅助） |
| 远端衰减 | 无 | 内建 |
| 计算开销 | 无 | 极小（旋转矩阵计算） |

## RoPE 的工业采用

### 直接采用 RoPE 的模型

- [[Wiki/Sources/LLaMA|LLaMA]]（Meta, 2023）：系列中首次使用 RoPE
- [[Wiki/Sources/Llama 3|Llama 3]]（Meta, 2024）：延续使用 RoPE，配合 GQA
- Mistral、Qwen、DeepSeek、ChatGLM 等几乎所有主流开源 LLM

### RoPE 的外推扩展

- **NTK-aware Scaled RoPE**：通过调整旋转频率来改善位置外推
- **YaRN**：结合 NTK scaling 和温度调整的 RoPE 扩展方法
- **Position Interpolation**：Linear / NTK 插值方法

## 局限

- 直接外推能力有限——超过训练见过的最大长度时性能下降
- 旋转频率的设置影响不同距离范围的注意力强度，需要针对任务调整
- 按维度对分组旋转的设计要求偶数维度

## 相关页面

- [[Wiki/Sources/RoPE|RoPE 来源页]]
- [[Wiki/Entities/Transformer|Transformer]]
- [[Wiki/Topics/大语言模型基础|大语言模型基础]]
