---
title: 'HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction'
authors:
  - (ByteDance 搜索团队)
institutions: ByteDance
aliases:
  - HyFormer
  - Hybrid Transformer
date: '2026-04-30'
publish_date: 2025
venue: 暂无正式发表信息 (ByteDance 技术报告)
tags:
  - 论文
  - CTR预估
  - 序列建模
  - 特征交互
  - Hybrid Transformer
  - 工业部署
  - 抖音搜索
url: ''
code: 未开源
rating: ⭐⭐⭐⭐
status: Read
---
**一句话总结**：ByteDance 搜索团队提出 HyFormer，一个混合 Transformer 架构统一 CTR 预估中的序列建模和特征交互。通过 Query Decoding（交叉注意力处理长序列）和 Query Boosting（MLP-Mixer 做 token 混合）交替的层叠设计，以及 Global Tokens 作为语义接口，在抖音搜索广告场景实现 0.293% 观看时长和 1.111% 完播数提升。

---

## Intro

### Motivation

CTR 预估中存在两个核心子问题：
1. **用户行为序列建模**：从用户历史行为中提取兴趣表征
2. **特征交互**：建模不同特征（用户、广告、上下文）之间的高阶交互

现有方法通常将两者视为独立模块：
- 序列建模：DIN/DIEN/SIM 等，用 target attention 从行为序列中提取兴趣
- 特征交互：DCN/DLRM 等，用 cross network 或交互层建模特征间关系

这种设计的根本问题是：**序列建模和特征交互之间的关系被忽略了**——"用户对什么类型的广告感兴趣"本身就依赖于"广告特征"和"用户历史行为"的交互。

### 核心主张

用一个统一的 Hybrid Transformer 同时完成两个任务，通过交替的 Query Decoding（序列→特征）和 Query Boosting（特征↔特征）层，让序列建模和特征交互相互增强。

### 贡献

1. **混合 Transformer**：Query Decoding + Query Boosting 交替层叠架构
2. **Global Tokens**：充当序列信息和特征交互之间的语义接口
3. **工业部署**：抖音搜索广告场景 0.293%/1.111% 提升
4. **效率优化**：比分离式架构更简洁，参数利用率更高

---

## Method 核心方法

### 1. Query Decoding

**目的**：从用户行为序列中提取与当前候选广告相关的兴趣信息。

- 使用交叉注意力（cross-attention）
- Query：广告特征 token（当前候选广告的表示）
- Key/Value：用户行为序列 token（历史点击/浏览的广告）
- 输出：融合了历史兴趣的增强特征表示

与传统 target attention（如 DIN）的区别：
- DIN 在 embedding 层做 attention，HyFormer 在多层 Transformer 中重复做
- 注意力可以随着层深越来越精细

### 2. Query Boosting

**目的**：在特征 token 之间进行交互建模。

- 使用 MLP-Mixer 风格的 token mixing
- 将不同特征 token 在特征维度（而非序列维度）上进行混合
- 所有特征 token（用户画像、广告特征、上下文等）相互交互

**为什么不用 self-attention**：
- self-attention 计算复杂度 O(n^2)，但在 token 数量不多的特征交互场景中，MLP-Mixer 更高效
- MLP-Mixer 通过跨 token 的全连接和跨 channel 的全连接交替实现高效交互

### 3. Global Tokens

**核心设计**：在 token 序列中插入一组可学习的 global tokens。

- Global tokens 参与 Query Decoding 和 Query Boosting
- 它们作为"语义接口"，承载序列信息和特征交互之间的信息传递
- 类比 ViT 中的 [CLS] token，但数量更多（多维度语义）

**工作流程**：
1. Query Decoding 层：global tokens 作为额外的 query，从行为序列中提取全局兴趣
2. Query Boosting 层：global tokens 与特征 token 一起做 MLP-Mixer
3. 最终，global tokens 汇总所有信息，输入到预测层

### 4. 交替层叠设计

HyFormer 的每一层是 Query Decoding 和 Query Boosting 的交替：

```
Layer_i = [Query_Decoding + Query_Boosting]
```

这种交替设计保证了：
- 序列信息 → 特征交互：解码出的兴趣信息可以进入特征交互
- 特征交互 → 序列建模：更丰富的特征表示可以在下一层的解码中做更好的 query

---

## 实验/评估/结果

### 离线评估

- AUC 超越分离式架构的多个基线
- 消融实验：
  - Query Decoding + Query Boosting > 仅 Query Decoding：证明特征交互的互补作用
  - Global Tokens > 无 Global Tokens：证明语义接口的价值
  - 交替层叠 > 堆叠相同层型：证明双向信息流的必要性

### 在线 A/B

- 抖音搜索广告：
  - 观看时长 +0.293%
  - 完播数 +1.111%
- 在搜索场景的效果显著（搜索场景下用户意图明确，序列建模和特征交互的结合更重要）

---

## 结论

HyFormer 通过 Query Decoding（交叉注意力处理序列）和 Query Boosting（MLP-Mixer 做特征交互）的交替设计，实现了序列建模和特征交互的深度统一。Global tokens 提供了一种优雅的信息交换接口。该架构在搜索广告场景获得了显著的实验验证。

---

## 思考

### 优点

1. **架构设计有洞察**：认识到序列建模和特征交互应该相互增强，不是独立任务；交替层叠设计让这个洞察落到了实处
2. **Global Tokens 设计精巧**：用一组可学习的 token 作为两个子任务之间的信息接口，简洁而有效
3. **搜索场景的适配**：搜索场景下 query-ad-user 三方的交互比信息流更复杂，HyFormer 的架构天然适合
4. **工业验证**：0.293%/1.111% 的提升在搜索广告场景是显著的

### 缺点与待解决问题

1. **与 OneTrans 的关系**：ByteDance 同时有 OneTrans（信息流）和 HyFormer（搜索），两者是否能统一？
2. **MLP-Mixer 的扩展性**：当特征维度增加时，MLP-Mixer 的 cross-channel 混合可能成为瓶颈
3. **Global Tokens 个数的最优选择**：论文可能缺少对 global token 数量的详细消融
4. **搜索场景专属优化**：是否在信息流场景下也有同样的优势，需要验证

### 与已有 Wiki 的连接

- 关联概念：[[Wiki/Concepts/CTR预估]]、[[Wiki/Concepts/序列建模]]、[[Wiki/Concepts/MLP-Mixer]]
- 关联实体：[[Wiki/Entities/ByteDance]]
- 关联比较：[[OneTrans]]（统一但分离参数空间）、[[InterFormer]]（双向信息流）
