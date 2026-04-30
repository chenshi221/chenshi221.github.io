---
title: 'InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction'
authors:
  - (Meta AI 团队)
institutions: Meta
aliases:
  - InterFormer
  - Heterogeneous Interaction
date: '2026-04-30'
publish_date: 2025
venue: 暂无正式发表信息 (Meta 技术报告)
tags:
  - 论文
  - CTR预估
  - 特征交互
  - 序列建模
  - Transformer
  - Meta Ads
  - 工业部署
url: ''
code: 未开源
rating: ⭐⭐⭐⭐
status: Read
---
**一句话总结**：Meta 提出 InterFormer，通过引入双向信息流架构（Interaction Arch / Sequence Arch / Cross Arch），让非序列特征和序列特征在建模过程中相互增强——而非各自独立处理再拼接。在 Meta Ads 场景实现 0.15% NE（Normalized Entropy）降低和 24% QPS（吞吐量）提升。

---

## Intro

### Motivation

CTR 预估中两类特征的根本不对称性：
1. **非序列特征**（Non-Sequence）：用户画像、广告特征、上下文——语义独立，每类特征通常 1 个 token
2. **序列特征**（Sequence）：用户历史行为——每类特征 50-200 个 token，语义相似（都是 item）

现有方法的典型做法：
- 两个独立模块：DLRM（处理非序列特征）+ HSTU/DIN（处理序列特征）
- 最后拼接两次的输出
- 信息只在最终层交汇，损失了大量中间交互机会

### 核心主张

通过三个子架构的双向信息流——Interaction Arch（非序列特征的个性化特征交互）、Sequence Arch（序列特征建模）、Cross Arch（两类特征的交叉注意力互增强）——让异构特征在每一层都相互学习。

### 贡献

1. **双向信息流架构**：Interaction Arch + Sequence Arch + Cross Arch 的三模块设计
2. **PFFN（Personalized FFN）**：让 FFN 的参数受用户特征条件化
3. **PMA Tokens**：可学习的聚合 token 作为跨模块信息桥梁
4. **工业部署**：Meta Ads 0.15% NE 降低 + 24% QPS 提升

---

## Method 核心方法

### 1. 整体架构三模块

```
Input → [Interaction Arch | Sequence Arch | Cross Arch] × L → Prediction
```

- **Interaction Arch**：处理非序列特征的混合交互（PFFN + PMA tokens）
- **Sequence Arch**：处理序列特征的时序依赖（Transformer + PMA tokens）
- **Cross Arch**：两类特征的交叉注意力互增强

### 2. Interaction Arch：个性化特征交互

**核心理念**：非序列特征之间的交互应该与当前用户相关。

**PFFN（Personalized FFN）**：
- 标准 FFN：输出 = MLP(input)
- 个性化 FFN：MLP 的权重（或偏置）受用户特征条件化
- 实现：用户特征通过一个小型网络生成 FFN 层的参数调整向量

**意义**：
- 同一广告对不同用户的意义不同
- 传统的 feature interaction（如 DCN）对所有用户一视同仁

### 3. Sequence Arch：序列特征建模

- 用户的历史行为序列（如过去 50 个点击广告）
- 用 Transformer 的双向注意力建模序列内的依赖关系
- 输出通过 PMA tokens 聚合为固定长度的表示

### 4. Cross Arch：双向信息流

这是 InterFormer 最核心的创新：

**非序列 → 序列方向**：
- 用非序列特征的表示（用户画像 + 广告特征）作为 query
- Attend 到序列特征（历史行为）作为 key/value
- 使得"在给定当前候选广告的条件下，用户哪些历史行为最相关"可以建模

**序列 → 非序列方向**：
- 用聚合的用户行为表示（从 Sequence Arch 输出）作为上下文
- 注入到 Interaction Arch 中，调节特征交互的方式
- 使得"用户历史偏好"可以影响特征之间的交互模式

### 5. PMA Tokens

**可学习的聚合 token**（类似 ViT 中的 [CLS] token 但更通用）：

- Sequence Arch 的 PMA tokens：将变长的行为序列压缩为固定长度表示
- Interaction Arch 的 PMA tokens：作为非序列特征交互的"语义汇聚点"
- Cross Arch 通过 PMA tokens 实现信息交换

### 6. 效率提升

- 24% QPS 提升的原因：
  - 通过 PMA tokens 压缩序列表示，在 Cross Arch 中避免了对长序列的全连接
  - 特征交互的个性化（PFFN）比增加更多特征交互层更高效

---

## 实验/评估/结果

### 离线评估

- Meta 内部数据：
  - NE (Normalized Entropy) 降低 0.15%（在 CTR 预估中这是显著提升）
  - 超越 DLRM + HSTU 的组合基线

- 消融实验：
  - Cross Arch 的贡献：双向信息流 > 单向信息流 > 无信息流
  - PFFN 的贡献：个性化 FFN > 标准 FFN
  - PMA Tokens：聚合表示同时保持效果和效率

### 在线 A/B

- Meta Ads：0.15% NE 降低 + 24% QPS 提升
- 部署在 Meta 广告系统的关键位置

---

## 结论

InterFormer 通过架构创新（三类 Arch + 双向信息流）解决了 CTR 预估中异构特征交互的长期问题。PFFN 带来了"特征交互应该因人而异"的关键洞察。在实际部署中同时实现了效果提升和吞吐量优化。

---

## 思考

### 优点

1. **理念先进**：认识到非序列特征和序列特征不应独立处理——双向信息流比"最后拼接"更合理
2. **PFFN 的设计有启发**：个性化特征交互是一个被低估的方向——"广告对不同用户意味着不同的事"
3. **效率与效果的平衡**：24% QPS 提升同时 0.15% NE 降低，表明架构设计是精炼而非堆砌
4. **Meta 的工业验证**：在 Meta Ads 的规模下部署，证明了有效性

### 缺点与待解决问题

1. **三层 Arch 的复杂性**：Interaction + Sequence + Cross 三层架构的参数量和调参难度可能较高
2. **与 HyFormer/OneTrans 的对比**：三者都在解决同一类问题（统一序列建模和特征交互），但设计理念不同，缺乏公平对比
3. **PFFN 的计算开销**：每个样本都要计算个性化的 FFN 权重调整，在网络规模大时开销显著
4. **跨 Arch 的信息瓶颈**：通过 PMA tokens 交换信息虽然高效，但可能存在信息压缩瓶颈

### 与已有 Wiki 的连接

- 关联概念：[[Wiki/Concepts/CTR预估]]、[[Wiki/Concepts/序列建模]]、[[Wiki/Concepts/特征交互]]
- 关联实体：[[Wiki/Entities/Meta]]
- 关联比较：[[HyFormer]]（Query Decoding + Boosting 交替）、[[OneTrans]]（统一但分离参数）
