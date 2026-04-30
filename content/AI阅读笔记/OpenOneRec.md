---
title: 'OpenOneRec: An Open Foundation Model and Benchmark to Accelerate Generative Recommendation'
authors:
  - Shiyao Wang
  - Jiaxin Deng
  - Junda She
  - Zhiyang Zhang
  - Zixing Zhang
  - Jiaming Huang
  - Jinghao Zhang
  - Honghui Bao
  - Kuo Cai
  - Guorui Zhou
institutions: Kuaishou
aliases:
  - OpenOneRec
  - OpenOneRec-Foundation
date: '2026-04-30'
publish_date: 2025-12
venue: 'arXiv:2512.24762'
tags:
  - 论文
  - 生成推荐
  - 开源模型
  - Scaling Law
  - Benchmark
  - LLM推荐
  - 跨域迁移
url: 'https://arxiv.org/abs/2512.24762'
code: 已开源（模型 + 数据 + 训练框架）
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：OpenOneRec 是首个面向生成式推荐的开源基础模型（基于 Qwen3，1.7B 和 8B），提出了 RecIF-Bench（8 个任务、4 个能力层级的推荐指令遵循评估 benchmark），验证了推荐领域的 Scaling Law（N_opt ∝ C^0.44, D_opt ∝ C^0.56），并通过 on-policy distillation 实现跨域迁移（Amazon 上平均 Recall@10 提升 26.8%）。

---

## Intro

### Motivation

尽管 OneRec 系列在工业场景取得了成功，但：
1. 所有模型闭源，社区无法复现和继续研究
2. 缺乏统一的 benchmark 来评估生成式推荐的"指令遵循"能力
3. 学术数据集（Amazon）上的生成式推荐效果仍然有限

### 核心主张

通过开源基础模型、完整的训练 pipeline 和统一的 benchmark 来加速生成式推荐研究。

### 贡献

1. **OpenOneRec 模型**：开源 1.7B 和 8B 版本，基于 Qwen3 backbone
2. **RecIF-Bench**：首个推荐指令遵循评估 benchmark
3. **Scaling Law**：验证推荐领域的缩放定律
4. **跨域迁移**：on-policy distillation 实现 Amazon 跨域 SOTA
5. **全栈开源**：数据处理 + 预训练 + 后训练 pipeline

---

## Method 核心方法

### 1. 模型架构

- **Backbone**：Qwen3（1.7B 和 8B）
- **Itemic Tokenization**：将推荐 item 转化为特殊 token（通过 learnable embedding），注入 LLM 词表
- **训练分两个阶段**：
  - **Pre-training Stage 1**：仅训练 itemic-related 参数（itoken embedding + itoken head），冻结 LLM 主体
  - **Pre-training Stage 2**：全参数训练

### 2. 数据配方

**推荐领域预训练数据（37.35%）**：
- **Itemic Dense Caption Data**（37.66%）：
  - 对每个 item（视频/商品）生成双语描述
  - 形式：`<item_id> 展示了以下内容：...（中文描述）\n...（英文描述）`
  - 核心价值：让 LLM 的语义空间与 item 的协同过滤信号对齐
- **Sequential User Behavior Data**（0.38%）：用户曝光、长播、点赞等行为序列
- **Interleaved User Persona Grounding Data**（0.21%）：用户画像交错的完整行为数据

**通用领域数据（62.34%）**：数学、代码、推理等数据集（Nemotron_CC_Math、OpenMathReasoning 等）

**关键数据混合策略**：
- 通用数据占比高（62%）以保持 LLM 的通用推理能力
- 推荐数据以 itemic caption 为主（37.66%），行为序列数据少量（<1%）

### 3. RecIF-Bench

**4 个能力层级，8 个任务**：

| 层级 | 能力 | 任务 |
|------|------|------|
| L0: Item Understanding | Item 理解 | Item Captioning |
| L1: Core Recommendation | 核心推荐 | Label Prediction, Video Recommendation, Product Recommendation, Ad Recommendation |
| L2: Advanced Recommendation | 高级推荐 | Label Conditional Recommendation, Interactive Recommendation |
| L3: Reasoning | 推理 | Recommendation Reasoning |

**评估方式**：
- Item Understanding (L0)：LLM-as-Judge（Weighted Information Points 语义匹配）
- 推荐任务 (L1-L3)：Recall@K、NDCG@K
- 推理任务 (L3)：答案质量 + 推理链质量

### 4. Scaling Law

- **实验设置**：在 Amazon 上，控制不同数据量和模型规模
- **结果**：
  - N_opt ∝ C^0.44（最优模型参数规模随计算预算的增长）
  - D_opt ∝ C^0.56（最优数据量随计算预算的增长）
- **含义**：当前与 Chinchilla 规律一致——数据与模型应大致同比例增长

![](https://arxiv.org/html/2512.24762/x5.png)

*Figure 5: 推荐域数据上的 Scaling Law——Top-Left: 不同模型大小在 FLOPs 上的训练 loss 和凸包络线；Top-Right: 固定 token 预算下 loss vs 模型大小的 log-log 图；Bottom: N_opt ∝ C^0.44 和 D_opt ∝ C^0.56，验证推荐领域遵循可预测的幂律缩放。*

### 5. On-Policy Distillation

实现跨域迁移的关键技术：

- **目标**：将在快手上训练的推荐能力迁移到 Amazon 商品推荐
- **方法**：
  - 用 teacher 模型（快手模型）在 Amazon 数据上采样
  - 用 on-policy distillation（reverse KL divergence）将 teacher 行为蒸馏到 student
  - 区别于标准知识蒸馏（forward KL），on-policy 蒸馏让 student 在自己的采样分布上学习
- **实验设置**：
  - OpenOneRec-Pro (8B) 在 Amazon Beauty/Sports/Toys 上
  - 平均 Recall@10：+26.8%
  - 在 4 个能力层级上都取得了提升

![](https://arxiv.org/html/2512.24762/x7.png)

*Figure 7: On-Policy Distillation 流程——Student（Policy Model）对给定 prompt 采样轨迹，Teacher Model 通过 Reverse KL divergence 提供反馈作为 reward，Policy Model 用策略梯度方法迭代优化，实现跨域推荐能力迁移。*

### 6. 多任务 SFT

- 混合 65% 通用推理数据 + 35% 推荐任务数据
- 推荐任务包括：label prediction、item captioning、interactive recommendation、reasoning 等
- 保证模型同时保持推理能力和推荐能力

---

## 实验/评估/结果

### RecIF-Bench

- **L0 Item Understanding**：LLM-Judge F1 score 评估，OpenOneRec 在 caption 语义准确性上超越基线
- **L1-L2 推荐任务**：在 Amazon Beauty/Sports/Toys 上达 SOTA
- **L3 推理推荐**：CoT 只在有限场景下有效（如需要在复杂候选中做比较时）

### On-Policy Distillation

- Amazon 跨域迁移：Recall@10 平均 +26.8%
- 验证了推荐能力可以从快手域迁移到完全不同的商品推荐域

### Scaling Law 验证

- 在 4 个模型规模（0.3B-8B）和多个数据规模上验证
- 缩放定律预测与实验吻合

---

## 结论

OpenOneRec 通过开源模型、benchmark 和训练 pipeline，为生成式推荐研究提供了基础 infrastructure。RecIF-Bench 为评估推荐指令遵循能力提供了统一框架。On-policy distillation 证明了推荐能力的跨域可迁移性。

---

## 思考

### 优点

1. **开源贡献巨大**：在 OneRec 系列全是闭源工业模型的背景下，OpenOneRec 开源完整 pipeline，填补了研究空白
2. **RecIF-Bench 设计合理**：4 个能力层级从基础到推理的递进结构，覆盖了生成式推荐的不同难度层次
3. **Scaling Law 有学术价值**：在推荐领域验证缩放定律，为资源分配提供量化工具
4. **On-Policy Distillation 突破性**：跨域迁移提升 26.8% 是显著的，且方法有通用性

### 缺点与待解决问题

1. **Tokenizer transferability 限制**：论文自我批评——推荐 backbone 的收益受 tokenizer 可迁移性制约，跨域时 item token 语义需要重新对齐
2. **通用能力的保持成本**：62% 通用数据占比说明模型还不能完全"专业化"于推荐，推荐和通用能力的最佳混合比是开放问题
3. **CoT 推理增益有限**：CoT 只在有限场景下有效，test-time scaling 策略需要更深入探索
4. **Pro 版不开放**：OpenOneRec-Pro (8B) 使用了额外的专有数据，开源版本和最优版本之间存在 gap

### 与已有 Wiki 的连接

- 关联概念：[[Wiki/Concepts/Scaling Law]]、[[Wiki/Concepts/知识蒸馏]]、[[Wiki/Concepts/生成式推荐]]
- 关联实体：[[Wiki/Entities/Qwen3]]、[[Wiki/Entities/快手]]
- 关联比较：[[OneRec 技术报告]]（工业版本）、[[OneRec-V2]]（Lazy Decoder-Only 架构）
