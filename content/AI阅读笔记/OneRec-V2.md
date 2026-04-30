---
title: OneRec-V2 Technical Report
authors:
  - Guorui Zhou
  - Honghui Hu
  - Hongtao Cheng
  - Huanjie Wang
  - Jiaxin Deng
  - Jinghao Zhang
  - Kuo Cai
  - Lejian Ren
  - Lu Ren
  - Li Yu
institutions: Kuaishou
aliases:
  - OneRec-V2
  - OneRec V2
date: '2026-04-30'
publish_date: 2025-08
venue: 'arXiv:2508.20900'
tags:
  - 论文
  - 生成推荐
  - MoE
  - Scaling Law
  - Lazy Decoder
  - GRPO
  - 工业部署
url: 'https://arxiv.org/abs/2508.20900'
code: 未开源
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：OneRec-V2 通过 Lazy Decoder-Only 架构替代 V1 的 Encoder-Decoder（训练计算减少 90%，推理计算减少 94%），将模型扩展到 8B 参数并遵循 Chinchilla 缩放定律（拟合 E=3.13, A=3660, alpha=0.489），引入 GBPO（Gradient-Bounded Policy Optimization）和 Duration-Aware Reward Shaping，在快手 App 上实现 0.467%/0.741% 的使用时长提升。

---

## Intro

### Motivation

OneRec V1 虽然验证了生成式推荐的可行性，但存在三个主要问题：

1. **架构效率低**：Encoder-Decoder 架构在推理时需要先编码完整用户序列再逐位置解码，延迟高、计算量大
2. **规模化路径不清晰**：V1 只有一个模型规模，不知道更大模型能带来多少收益
3. **偏好对齐不够稳定**：V1 的 ECPO 有时会导致策略崩溃

### 核心主张

通过架构创新（Lazy Decoder-Only）大幅降低计算开销，让生成式推荐可以扩展到 8B 参数，并通过 scaling law 预测更大规模下的收益。

### 贡献

1. **Lazy Decoder-Only 架构**：训练计算减少 90%，推理计算减少 94%
2. **Scaling Law**：首次在推荐领域拟合 Chinchilla 缩放定律，E=3.13, A=3660, alpha=0.489
3. **GBPO 偏好对齐**：Gradient-Bounded Policy Optimization，比 ECPO 更稳定
4. **Duration-Aware Reward Shaping**：引入时长感知的奖励塑形
5. **工业收益**：0.467%/0.741% 的 App 使用时长提升

---

## Method 核心方法

### 1. Lazy Decoder-Only 架构

这是 OneRec-V2 最核心的创新。V1 的 Encoder-Decoder 存在严重的效率问题：

![](https://arxiv.org/html/2508.20900/x2.png)

*Figure 2: OneRec-V2 整体架构和后训练框架——(a) Lazy Decoder-Only 架构：用户行为序列作为 decoder 前缀 token 参与 causal attention，推理时仅需计算一次 KV cache；(b) 后训练偏好对齐流程。*

**V1 的问题**：
- Encoder 每次推理需要重新编码用户全部行为序列
- Cross-attention 层增加了额外的参数量和计算量

**V2 的 Lazy Decoder-Only**：
- 放弃独立的 Encoder，把所有 token（用户行为 + 推荐列表）拼接为一个长序列
- 用单向 causal attention 统一处理
- "Lazy"的含义：用户行为序列被"懒加载"——它们作为 decoder 的前缀 token 参与自注意力，而非通过交叉注意力注入
- 用户行为部分的 token 在推理时只需要计算一次 KV cache（因为用户行为是固定的）

**效果**：
- 训练计算减少 90%（去掉了 cross-attention 和双向 encoder）
- 推理计算减少 94%（KV cache 复用 + 架构简化）
- 性能没有下降，甚至略好（推测是 causal attention mask 更利于自回归生成）

### 2. Scaling Law

在推荐领域拟合缩放定律：

- **数据**：在可控 token 预算下训练不同规模的模型
- **拟合**：loss = E / C^alpha + A（其中 C 是训练计算量）
- **结果**：E=3.13, A=3660, alpha=0.489

关键发现：
- 推荐数据的缩放定律与语言数据的趋势一致（alpha 约为 0.49-0.5）
- 8B 模型仍处于"可缩放区"，更大的模型可以期待进一步收益
- 对照 Chinchilla 定律：当前数据量与模型规模大致匹配

![](https://arxiv.org/html/2508.20900/x6.png)

*Figure 6: OneRec-V2 Dense 模型 Scaling Law 曲线——训练 loss 随模型参数量 N 的增大呈幂律下降，L^(N) = 3.13 + 3660/N^0.489，验证了推荐领域也遵循 Chinchilla 缩放定律。*

### 3. GBPO（Gradient-Bounded Policy Optimization）

改进 ECPO 的稳定性：

- 在 PPO 框架下，对策略梯度（policy gradient）的幅值施加显式的上界约束
- 相比 ECPO 的隐式约束（Early Stop by KL），GBPO 更直接地控制了每次更新的幅度
- 在训练过程中保持了更稳定的策略改进

### 4. Duration-Aware Reward Shaping

- 原始 reward 信号：使用时长增加
- 问题：不同用户的使用习惯差异大，绝对时长增加不能公平比较
- 改进：奖励信号 = 当前 session 时长 / 用户历史平均 session 时长
- 效果：更公平地跨用户比较，提升了训练稳定性

### 5. 扩展

- 8B 总参数，MoE 架构（多个专家 FFN）
- Top-2 路由，load balancing loss
- 训练数据规模与模型规模匹配

---

## 实验/评估/结果

### 离线评估

- 超越 V1 在所有内部评估指标上
- Scaling law 预测与实际训练结果吻合
- GBPO vs ECPO：更稳定的训练曲线，更低的奖励方差

### 在线 A/B

- 快手 App：使用时长 +0.467%（单目标）/ +0.741%（多目标）
- 在 V1 已经提升的基础上继续增长

---

## 结论

OneRec-V2 通过 Lazy Decoder-Only 架构解决了生成式推荐的核心效率问题，使得将模型扩展到 8B 成为可能。Scaling law 的拟合为未来规模化提供了预测工具，GBPO 和 Duration-Aware Reward Shaping 则提升了训练稳定性。

---

## 思考

### 优点

1. **架构创新意义重大**：Lazy Decoder-Only 将一个架构问题（encoder-decoder 太重）变成了数据组织问题（把所有信息放进一个序列），简单而优雅
2. **Scaling law 的引入**：在推荐领域拟合缩放定律，为资源分配（给数据还是给模型）提供了量化指导
3. **工程与科学的结合**：94% 的计算减少使得生成式推荐从"学术 demo"变成"可规模化部署的系统"
4. **偏好对齐的工程优化**：GBPO 和 Duration-Aware Reward Shaping 是对 RL 方法在推荐场景的务实改进

### 缺点与待解决问题

1. **Lazy 架构的极限**：将所有 token 拼接成一个序列意味着序列长度随着用户行为增长而增长。对于行为特别丰富的高频用户，前缀 KV cache 的成本不可忽略
2. **长尾用户的冷启动**：因果序列前缀对于新用户或低活用户来说信息有限，解码质量可能显著下降
3. **与 OneRec-Think 的集成**：V2 没有融合推理增强（推理增强在 OneRec-Think 中独立探索），两个方向还没有统一
4. **工业权重的不可复现性**：未开源模型，社区无法验证 scaling law 和架构创新

### 与已有 Wiki 的连接

- 关联概念：[[Wiki/Concepts/Scaling Law]]、[[Wiki/Concepts/MoE 混合专家模型]]、[[Wiki/Concepts/GRPO]]
- 关联实体：[[Wiki/Entities/快手]]
- 关联比较：[[OneRec 技术报告]]（V1 Encoder-Decoder）、[[OpenOneRec]]（开源版本）
