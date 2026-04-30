---
title: 'OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment'
authors:
  - Jiaxin Deng
  - Shiyao Wang
  - Kuo Cai
  - Lejian Ren
  - Qigen Hu
  - Weifeng Ding
  - Qiang Luo
  - Guorui Zhou
institutions: Kuaishou
aliases:
  - OneRec 统一召回排序
  - OneRec IPA
date: '2026-04-30'
publish_date: 2025-02
venue: 'arXiv:2502.18965'
tags:
  - 论文
  - 生成推荐
  - 偏好对齐
  - DPO
  - 召回
  - 排序
url: 'https://arxiv.org/abs/2502.18965'
code: 未开源
rating: ⭐⭐⭐⭐
status: Read
---
**一句话总结**：本文提出 OneRec 的早期版本，将推荐系统中的召回和排序统一为一个生成式 Transformer 模型（encoder-decoder），通过 Session-wise 列表生成、Balanced K-Means 语义 tokenization 和 IPA（Iterative Preference Alignment）DPO 偏好对齐，在快手上实现 1.6% 的观看时长提升。

---

## Intro

### Motivation

传统推荐系统的召回（Retrieval）和排序（Ranking）是两个独立阶段：
- 召回从百万/亿级候选集中快速筛选出数百个候选
- 排序对这数百个候选进行精细化打分和重排

两者的分离带来了级联信息瓶颈：召回阶段的信息损失无法在排序阶段弥补。

作者认为，生成式模型可以统一这两个阶段——用一个模型直接从用户历史序列生成排序好的推荐列表。

### 核心主张

用生成式 Transformer 统一 Recall + Rank，同时引入迭代偏好对齐（IPA）来提升生成列表的质量。

![](https://arxiv.org/html/2502.18965/x1.png)

*Figure 1: 统一架构 vs 传统级联架构——(a) OneRec 统一端到端生成架构直接输出排序列表；(b) 传统级联系统分召回、预排、精排三阶段，各阶段独立优化。*

### 贡献

1. **统一召回排序框架**：首次用生成式模型同时完成召回和排序
2. **Session-wise 列表生成**：每次生成一个完整的 session 内推荐列表
3. **Balanced K-Means tokenization**：在 K-Means 聚类中引入均衡约束，避免码本分布不均
4. **IPA 偏好对齐**：Iterative Preference Alignment——用 DPO 在自生成的困难负样本上迭代对齐

---

## Method 核心方法

### 1. 统一生成式框架

![](https://arxiv.org/html/2502.18965/x2.png)

*Figure 2: OneRec 统一召回排序总框架——两阶段：(i) Session Training 阶段用 session-wise 数据训练 Encoder-Decoder 生成推荐列表；(ii) IPA 阶段用迭代 DPO + self-hard negatives 进行偏好对齐。*

**架构**：Encoder-Decoder Transformer
- Encoder 编码用户行为序列（最近 N 个交互 item）
- Decoder 自回归生成推荐列表

与后续 OneRec 版本不同，此版本的 decoder 是 **session-wise list 生成**：一次生成整个 session 的推荐列表，但不强调 point-wise 逐位置生成。

### 2. Balanced K-Means Tokenization

- 将 item embedding 通过 Balanced K-Means 聚类转化为离散 token
- 在标准 K-Means 基础上添加均衡约束，确保每个聚类大致包含相同数量的 item
- 避免某些码本 token 几乎没有对应 item 的问题
- 每个 item 用 1 个 token 表示（相比后续 RQ-KMeans 的 3 层有所简化）

### 3. IPA（Iterative Preference Alignment）

这是一个迭代的偏好对齐流程：

1. **初始 DPO 训练**：用当前模型（policy）和参考模型（reference），在人工标注的正负样本对上训练
2. **自生成困难负样本**：用当前模型采样一批推荐列表，将 reward 较低的采样作为负样本
3. **迭代**：在新的"正样本-自生成负样本"对上继续 DPO 训练
4. **Self-hard negative mining**：随着模型能力提升，它生成的负样本越来越"困难"，迫使模型学到更精细的偏好

与标准 DPO 的区别：DPO 使用静态负样本，IPA 在训练过程中不断用当前模型生成新负样本。

### 4. 奖励信号

- 观看时长（watch time）：主要优化目标
- 完播率、点赞、分享等互动信号作为辅助

---

## 实验/评估/结果

### 离线

- 在快手真实数据上评估，Recall@K 和 NDCG 超越 DLRM、DSSM 等经典方法
- 消融实验：
  - Balanced K-Means vs 普通 K-Means：token 分布更均匀，覆盖率更高
  - IPA vs 无对齐：DPO 迭代带来的提升是递增的（3-4 轮迭代后 plateau）
  - Self-hard negative vs Random negative：自生成负样本比随机负样本更有效

### 在线 A/B

- 快手 App 首页信息流
- 观看时长 +1.6%（vs 基线级联架构）
- 互动率（点赞、分享、评论）同步提升

---

## 结论

OneRec 统一召回排序方案证明了生成式推荐在工业场景的可行性。Balanced K-Means tokenization 和 IPA 迭代对齐是早期探索中验证的关键技术点，为后续 OneRec 系列工作奠定了基础。

---

## 思考

### 优点

1. **问题定义清晰**：将召回和排序统一为"生成列表"的单一优化问题
2. **Self-hard negative 策略巧妙**：利用模型自身能力提升来构造越来越难的数据，"免费"获得训练信号
3. **工业验证**：在快手真实场景下取得了明确的在线收益

### 缺点与待解决问题

1. **技术深度有限**：相比后续 OneRec V1/V2 的工作，tokenization 和架构设计较为初步
2. **DPO vs PPO 的选择**：论文使用 DPO（offline），但后续 OneRec 系列转向了 GRPO/ECPO（online RL）表明前者可能效果受限
3. **缺乏与 LLM-based 方法的对比**：没有对比基于预训练 LLM 的生成式推荐
4. **推理延迟未讨论**：Encoder-decoder + 列表生成的延迟成本没有详细分析

### 与已有 Wiki 的连接

- 关联概念：[[Wiki/Concepts/DPO]]、[[Wiki/Concepts/K-Means]]
- 关联实体：[[Wiki/Entities/快手]]
- 关联比较：[[OneRec 技术报告]]（后续版本，技术更成熟）
