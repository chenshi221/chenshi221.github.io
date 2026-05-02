---
title: 'OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender'
authors:
  - (ByteDance 团队)
institutions: ByteDance
aliases:
  - OneTrans
date: '2026-04-30'
publish_date: 2025
venue: 暂无正式发表信息 (ByteDance 技术报告)
tags:
  - 论文
  - CTR预估
  - Transformer
  - 序列建模
  - 特征交互
  - 工业部署
  - Scaling-Law
url: ''
code: 未开源
rating: ⭐⭐⭐⭐
status: Read
---
**一句话总结**：ByteDance 提出 OneTrans，一个统一所有特征交互和用户行为序列建模的单一 Transformer 架构，通过混合参数化（序列特征共享参数、非序列特征 token-specific 参数）、金字塔层级结构和跨请求 KV 缓存，在抖音广告场景实现 5.68% GMV/u 提升，并验证了 CTR 预估的 Scaling Law（log-linear 趋势）。

---

## Intro

### Motivation

CTR 预估模型的架构日趋复杂：

1. **特征交互建模**：从 FM、DeepFM、DCN 到 DLRM，单独处理特征间的交互
2. **序列建模**：从 DIN、DIEN 到 SIM、HSTU，单独处理用户历史行为序列
3. **两者分离**：特征交互模块和序列建模模块独立设计、独立调参

这种"两个模块拼接"的架构变得复杂而低效。

### 核心主张

用一个单一 Transformer 统一完成所有任务——用统一 tokenizer 将所有特征（序列特征和非序列特征）转化为 token，用同一个 Transformer 同时建模特征交互和序列依赖。

### 贡献

1. **统一 Transformer 架构**：单一模型替代"特征交互模块 + 序列建模模块"的组合
2. **混合参数化**：序列特征 token（S-tokens）共享参数，非序列特征 token（NS-tokens）使用 token-specific 参数
3. **金字塔层级结构**：上层处理全局交互，下层处理细粒度信息
4. **跨请求 KV 缓存**：广告场景下同一用户在短时间内的多次请求共享 KV cache
5. **CTR Scaling Law**：首次验证 CTR 场景下的缩放定律

---

## Method 核心方法

### 1. 统一 Tokenizer

将所有输入特征转化为统一的 token 序列：

- **序列特征 token (S-tokens)**：用户历史行为中的每个 item（如过去点击的广告），用 item embedding 表示
- **非序列特征 token (NS-tokens)**：用户画像、上下文、广告特征等，每种特征类型一个 token
- 所有 token 拼接为一个长序列输入 Transformer

### 2. 混合参数化

**不应所有 token 共享参数**：
- 序列特征 token（如 50 个历史广告）：相互之间语义相似，应该共享参数（类似 NLP 中同一词表的 token）
- 非序列特征 token（如用户年龄、广告主行业）：语义完全不同，应该用 token-specific 参数

**实现**：
- S-tokens：用共享的 embedding + 共享的 projection 参数
- NS-tokens：每种特征类型有独立的 embedding 和 projection 参数
- 在 Transformer 的 FFN 层中，NS-tokens 也可能使用独立的参数

### 3. 金字塔层级结构

受到视觉 Transformer 中金字塔结构（如 Swin Transformer）的启发：

- **底层**：token 数量多，细粒度信息丰富，但交互范围有限
- **高层**：通过 token 合并减少数量，扩大感受野，进行全局交互
- 实现方式：类似 Swin 的 patch merging，但针对序列特征做 adaption

### 4. 跨请求 KV 缓存

广告场景的特殊优化：
- 用户打开 App 后，短时间内（如 10 分钟）会产生多次广告请求
- 这些请求共享同一用户的历史行为序列
- **KV cache 复用**：用户的 S-tokens 部分的 KV cache 可以在多次请求间复用
- **效果**：在线推理延迟大幅下降

---

## 实验/评估/结果

### 离线

- AUC 超越组合架构（特征交互 + 序列建模分离）的基线
- 消融实验：
  - 混合参数化 vs 全共享参数：显著提升
  - 金字塔 vs 平面 Transformer：提升
  - 跨请求 KV cache：延迟降低明显

### 在线 A/B

- **抖音广告场景**：GMV/u +5.68%
- 验证了统一 Transformer 在工业 CTR 预估中的有效性

### Scaling Law

- CTR 预估任务：AUC 随模型规模的增加呈 log-linear 提升
- 表明更大的模型可以期待更好的效果（在数据量充足的前提下）

---

## 结论

OneTrans 证明了用一个 Transformer 统一特征交互和序列建模是可行的、高效的和可扩展的。混合参数化解决了不同类型 token 不应共享参数的矛盾，金字塔结构保证了高效的全局建模。

---

## 思考

### 优点

1. **架构统一**：用一个 Transformer 替代两个独立模块，是"大道至简"的工业实践
2. **混合参数化精巧**：S-tokens vs NS-tokens 的区分解决了"序列特征语义相似而上下文特征语义不同"的本质矛盾
3. **工业实用性强**：跨请求 KV cache 是切实可行且收益大的工程优化
4. **CTR Scaling Law 有价值**：为推荐模型的规模扩展提供了经验指导

### 缺点与待解决问题

1. **与 OneRec 的关系不清晰**：OneTrans 专注 CTR 预估（判别式），OneRec 专注生成式推荐（生成式）。两者能否用一个模型统一？
2. **序列长度限制**：金字塔合并可能丢失序列中的细粒度信息（如用户特定行为的时间间隔）
3. **离线 + 在线特征的时效性**：跨请求 KV cache 要求用户行为在短时间内不变，对实时特征更新受限
4. **缺乏公开 benchmark 对比**：只在内部数据上评估

### 与已有 Wiki 的连接

- 关联概念：[[Wiki/Concepts/CTR预估]]、[[Wiki/Concepts/序列建模]]、[[Wiki/Concepts/Feature Interaction]]
- 关联实体：[[Wiki/Entities/ByteDance]]
- 关联比较：[[HyFormer]]（也统一了特征交互和序列建模，但在搜索场景）、[[InterFormer]]（双向信息流）
