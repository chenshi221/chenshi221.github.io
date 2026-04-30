---
type: concept
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases:
  - "CTR Prediction"
  - "点击率预估"
  - "Click-Through Rate Prediction"
tags:
  - "recommender-systems"
  - "CTR"
  - "ranking"
sources:
  - "[[Wiki/Sources/OneTrans 2025]]"
  - "[[Wiki/Sources/HyFormer 2025]]"
  - "[[Wiki/Sources/InterFormer 2024]]"
  - "[[Wiki/Topics/生成式推荐系统]]"
confidence: high
---

# CTR 预估 (CTR Prediction)

## 定义

CTR 预估是推荐系统的核心任务之一：给定用户、候选 item 和上下文信息，预测用户点击该 item 的概率。它属于**判别式**任务（预测概率），而非生成式任务。CTR 预估通常位于推荐管线的排序（ranking）阶段，直接决定最终返回给用户的内容。

## 核心问题

### 1. 特征交互 (Feature Interaction)

学习不同特征（用户画像、物品属性、上下文信号）之间的高阶交叉关系。

**经典方法**：
- **FM (Factorization Machine)**：通过隐向量的内积捕获二阶特征交互
- **DeepFM / xDeepFM**：结合 FM 的低阶交互和 DNN 的高阶交互
- **DCN / DCNv2**：Cross Network 显式建模有界阶数的特征交叉
- **DHEN**：层次化集成多种异构交互模块
- **Wukong**：堆叠 FM 层 + 线性压缩，建立交互学习的 scaling law
- **RankMixer**：硬件友好的 token-mixing + per-token FFN

**核心挑战**：随着特征数量增加（工业级数百个特征），高阶交互的组合爆炸和高计算开销。

### 2. 序列建模 (Sequence Modeling)

从用户行为序列中捕获动态兴趣偏好。

**经典方法**：
- **DIN**：使用 target attention 从序列中提取候选相关的兴趣表示
- **SIM / ETA**：扩展序列长度至数千个行为，通过检索式 attention 降低计算
- **LONGER**：工业级长序列建模，因果 Transformer + KV caching + 服务优化
- **HSTU**：生成式推荐场景的高效长序列 backbone

**核心挑战**：超长序列（千级甚至万级行为）的计算效率和实时延迟。

### 3. 特征交互 + 序列建模的融合

传统方法采用分离式 pipeline：

```
用户行为序列 -> 序列编码器 -> 压缩向量
                                    ↓ (拼接)
非序列特征 ----------------> 特征交互模块 -> 预测
```

这种 **encode-then-interaction** 模式的问题是：
- **单向信息流**：非序列特征可以指导序列建模，但序列信息难以反向塑造特征表示
- **延迟融合**：序列信息被压缩后（如平均池化）才与其他特征交互，损失了大量信息

## 统一架构：CTR 预估的新方向

新一代工作试图用统一 Transformer 替代分离式 pipeline：

### InterFormer (Meta, 2024)
- **三层架构**：Interaction Arch ↔ Sequence Arch ↔ Cross Arch
- **双向信息流**：非序列 -> 序列（PFFN）、序列 -> 非序列（交互模块）
- **避免过早聚合**：使用 MHA 保持序列的 one-to-one 映射
- 工业部署：0.15% NE 增益 + 24% QPS 增益

### OneTrans (字节跳动, 2025)
- **Unified Causal Transformer**：S-tokens + NS-tokens 统一输入
- **Mixed Parameterization**：S-tokens 共享参数（同质），NS-tokens 各自 token-specific（异质）
- **Pyramid Compression**：逐层剪枝序列 token 的 query 范围
- **Cross-Request KV Caching**：推理 O(C) -> O(1)
- 工业部署：5.68% GMV/u 提升

### HyFormer (字节跳动, 2025)
- **交替迭代**：Query Decoding（序列解码） ⇄ Query Boosting（特征交互增强）
- **Global Tokens**：作为序列和特征之间的共享语义接口
- **三种序列编码策略**：Full Transformer / LONGER-style / Lightweight
- 工业部署：全量部署，服务数十亿用户

## 演进路线总结

```
传统分离式 pipeline
├── 序列：DIN -> SIM/ETA -> LONGER
├── 交互：FM -> DCNv2 -> DHEN -> Wukong/RankMixer
└── 融合：拼接/池化 -> Target Attention -> ?
             ↓
双向交互（InterFormer）：Interaction Arch ↔ Sequence Arch
             ↓
统一 Transformer（OneTrans）：单一 causal backbone 联合建模
             ↓
交替迭代（HyFormer）：Query Decoding ⇄ Query Boosting
```

## 与生成式推荐的对比

| 维度 | CTR 预估 | 生成式推荐 |
|------|---------|-----------|
| 任务性质 | 判别式（估计概率） | 生成式（生成 item ID） |
| 输出空间 | 标量 p(click) | 离散语义 ID 序列 |
| 架构位置 | 排序阶段 | 端到端（替代全流程） |
| 与 LLM 关系 | 借鉴 Transformer 设计 | 与 LLM 范式深度对齐 |
| 工业成熟度 | 高度成熟 | 快速演进 |
| 代表工作 | OneTrans, HyFormer, InterFormer | OneRec, OpenOneRec |

**交融趋势**：统一 Transformer 虽用于 CTR 预估，但架构思想和具体技术（KV cache、mixed attention、pyramid compression）与生成式推荐高度互通。两者的技术栈正在融合。

## 相关页面

- [[Wiki/Topics/生成式推荐系统]]
- [[Wiki/Concepts/生成式推荐]]
- [[Wiki/Entities/OneRec 系列模型]]
