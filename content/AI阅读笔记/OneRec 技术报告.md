---
title: OneRec Technical Report
authors:
  - Guorui Zhou
  - Jiaxin Deng
  - Jinghao Zhang
  - Kuo Cai
  - Lejian Ren
  - Qiang Luo
  - Qianqian Wang
  - Qigen Hu
  - Rongzhou Zhang
  - Ruiming Tang
institutions: Kuaishou
aliases:
  - OneRec技术报告
  - OneRec V1
date: '2026-04-30'
publish_date: 2025-06
venue: 'arXiv:2506.13695'
tags:
  - 论文
  - 生成推荐
  - MoE
  - RL
  - 工业部署
  - Factorization Machine
  - GRPO
url: 'https://arxiv.org/abs/2506.13695'
code: 未开源
rating: ⭐⭐⭐⭐
status: Read
---
**一句话总结**：Kuaishou 提出 OneRec，一个端到端的生成式推荐系统，用 encoder-decoder Transformer 统一替代传统的召回-粗排-精排-重排级联架构。通过 RQ-KMeans 语义 tokenization、4-scale 特征工程、MoE 扩展和 ECPO（Early Clipped GRPO）偏好对齐，在快手 4 亿 DAU 场景下实现了 0.54%/1.24% 的 App 使用时长增长。

---

## Intro

### Motivation

传统推荐系统采用级联架构（召回-粗排-精排-重排），每级独立建模，存在如下问题：
1. **信息损失**：上游模型的预测传递到下游，产生级联信息瓶颈
2. **目标不一致**：各级模型优化不同目标，缺乏全局最优性
3. **维护成本高**：3-4 个独立模型需要分别训练、调参、维护

最近，以 LLM 为代表的生成式模型展示了统一多任务的能力，生成式推荐（Generative Recommendation）因此成为一个有前景的方向——用一个统一的生成模型端到端地产生推荐列表。

### 核心主张

将推荐问题重新定义为"以用户行为序列为上下文，直接生成推荐 item 列表"的文本生成任务。通过语义 tokenization 将 item 编码为离散 token，用 Transformer 自回归解码。

### 贡献

1. **OneRec 系统**：首个在大规模工业场景（4 亿 DAU）上部署的端到端生成式推荐系统
2. **RQ-KMeans 语义 tokenization**：将 item 特征编码为层级化离散 token，保持语义近邻性
3. **4-scale 特征工程**：保留 Factorization Machine 的交叉特征优势，在 4 个粒度上做特征交互
4. **ECPO 偏好对齐**：将 GRPO 适配到推荐场景，用 Early Clipping 约束策略更新幅度
5. **工业收益**：在快手 App 上实现 0.54% 停留时长提升（单目标）和 1.24% 提升（多目标）

---

## Method 核心方法

### 1. Encoder-Decoder 架构

![](https://arxiv.org/html/2506.13695/x2.png)

*Figure 2: 级联推荐系统 vs OneRec——(a) 传统级联架构包含召回、粗排、精排、重排多个独立阶段，各级信息传递存在瓶颈；(b) OneRec 采用 Encoder-Decoder 架构端到端生成用户偏好视频列表。*

OneRec 采用标准的 encoder-decoder Transformer 架构：

- **Encoder**：处理用户行为序列和上下文特征
  - 输入：用户最近交互的 item 序列（按时间排序）+ 上下文特征（时间、设备等）
  - 每个 item 通过 RQ-KMeans tokenizer 转化为一组离散 token
  - encoder 对所有 token 做双向自注意力
  
- **Decoder**：逐位置自回归生成推荐列表
  - 每一步输出一个推荐 item 的完整 token 序列
  - 解码时引入 cross-attention 关注 encoder 的输出
  - 逐位置生成（point-wise generation）：每次生成一个 item 的所有 token，而非逐 token 看一个 item

- **MoE 扩展**：将 FFN 层替换为 MoE，提升模型容量；
  - Top-2 路由，load balancing loss 保证专家均衡利用

![](https://arxiv.org/html/2506.13695/x4.png)

*Figure 4: OneRec Encoder-Decoder 架构细节——Encoder 处理用户行为序列（RQ-KMeans tokenized items + 4-scale 特征），Decoder 自回归逐位置生成推荐列表 item token。*

### 2. RQ-KMeans 语义 Tokenization

- 将 item 的 embedding 向量通过 Residual Quantized K-Means 转化为多层离散 token
- 每层 1024 个码本，共 3 层 → 每个 item 用 3 个离散 token 表示
- 相比均匀哈希，RQ-KMeans 保持了语义相似性：相似 item 共享更多 token
- 实现了类似"item 的 language"：模型可以泛化到未见过的 item

### 3. 4-Scale 特征工程

借鉴 Factorization Machine 思想，将特征交互分为 4 个尺度：

| Scale | 说明 |
|-------|------|
| S0 | 原始特征（item id、时间、用户画像等） |
| S1 | 用户-item 交互特征（用户对某类 item 的偏好） |
| S2 | item-item 共现特征（item 间的转移概率等） |
| S3 | 全局统计特征（流行度、CTR 等） |

这保证了生成式模型不丢失传统推荐系统积累的特征工程经验。

### 4. ECPO（Early Clipped GRPO）偏好对齐

将强化学习的 GRPO（Group Relative Policy Optimization）适配到推荐场景：

- **奖励设计**：组合多个信号
  - Preference Score：停留时长 + 完播率 + 点赞率等
  - Format Reward：decoder 输出是否满足 item token 格式
  - Industrial Reward：多样性、冷启探索等

- **ECPO 核心改进**：
  - 在 PPO 的 clipped objective 基础上，对策略比率的裁剪范围做了 Early Stopping
  - 当 KL 散度超过阈值时提前停止更新，防止策略崩溃
  - 使用 group sampling（每组采样多个输出）构建相对优势

### 5. 训练和推理

- **预训练**：在数十亿用户交互序列上训练，使用 teacher forcing
- **对齐**：ECPO 阶段使用 on-policy 采样
- **推理**：自回归解码生成 item 序列（list-wise），取 top-K 作为最终推荐列表

---

## 实验/评估/结果

### 离线评估（快手真实用户数据）

| 消融项 | AUC 提升 | 说明 |
|--------|---------|------|
| MoE vs Dense | ~+0.3% | Top-2 路由提升模型容量 |
| RQ-KMeans vs 均匀哈希 | ~+0.2% | 语义 tokenization 保持 item 相似性 |
| 4-scale 特征 | ~+0.4% | 保留传统推荐特征工程经验 |
| ECPO vs 基础模型 | ~+0.3% | 偏好对齐提升推荐质量 |

### 在线 A/B（快手首页信息流，400M DAU）

| 优化目标 | App 使用时长提升 |
|---------|----------------|
| 单目标优化 | **+0.54%** |
| 多目标联合优化 | **+1.24%** |

*同步提升完播率、点赞率、评论率等互动指标。*

---

## 结论

OneRec 证明了端到端生成式架构可以在工业级推荐系统中替代传统的级联架构，不仅简化了系统复杂性，还带来了显著的业务收益。RQ-KMeans 语义 tokenization 和 ECPO 偏好对齐是其中两个关键的创新点。

---

## 思考

### 优点

1. **工业验证扎实**：在 4 亿 DAU 的规模上做了 AB 验证，有真实的业务指标提升
2. **语义 tokenization 设计精巧**：RQ-KMeans 保持了 item embedding 空间的语义结构，比简单哈希 tokenization 有天然优势
3. **特征工程迁移合理**：保留了 FM 的多尺度特征交互，不是简单地把 item id 扔进模型
4. **偏好对齐适配到位**：将 RLHF 思想迁移到推荐，ECPO 的设计考虑了推荐场景的特殊性（如格式约束 reward）

### 缺点与待解决问题

1. **Encoder-Decoder 的延迟问题**：逐位置生成使得推理延迟高于传统级联架构，engineering effort 的上限未知
2. **开源性不足**：技术报告没有释放模型权重和完整训练代码，只提供了论文描述
3. **与 LLM-based 方法的对比缺失**：没有与基于 Llama/Qwen 等 LLM backbone 的生成式推荐方法做系统对比
4. **长尾 item 的表示问题**：RQ-KMeans tokenization 对低频 item 的 token 质量可能较差

### 与已有 Wiki 的连接

- 关联概念：[[Wiki/Concepts/生成式推荐]]、[[Wiki/Concepts/MoE 混合专家模型]]、[[Wiki/Concepts/GRPO]]
- 关联实体：[[Wiki/Entities/快手]]
- 关联比较：[[OneRec-V2]]（后续的 lazy decoder-only 版本）、[[OpenOneRec]]（开源版本）
