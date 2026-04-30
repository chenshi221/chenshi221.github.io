---
title: 'OneRec-Think: In-Text Reasoning for Generative Recommendation'
authors:
  - Zhanyu Liu
  - Shiyao Wang
  - Xinchen Luo
  - Rongzhou Zhang
  - Jiaxin Deng
  - Honghui Bao
  - Jinghao Zhang
  - Wuchao Li
  - Pengfei Zheng
  - Xiangyu Wu
institutions: Kuaishou
aliases:
  - OneRec-Think
  - OneRec Think
date: '2026-04-30'
publish_date: 2025-10
venue: 'arXiv:2510.11639'
tags:
  - 论文
  - 生成推荐
  - 推理增强
  - CoT
  - GRPO
  - LLM
url: 'https://arxiv.org/abs/2510.11639'
code: 未开源
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：OneRec-Think 首次将 LLM 的 CoT 推理能力集成到生成式推荐中，通过 Itemic Token Alignment（多任务预训练对齐）、Reasoning Activation（剪枝上下文激活推理）和 Reasoning Enhancement（GRPO+Rollout-Beam 推理增强）三阶段训练，在快手 App 上实现 0.159% 的 App 使用时长提升，并在 Amazon Beauty/Sports/Toys 上达到 SOTA。

---

## Intro

### Motivation

LLM 的 Chain-of-Thought（CoT）推理在数学、编程等领域展现了显著增益，但推荐系统长期处于"黑盒匹配"范式——模型根据用户行为直接预测偏好，缺乏明确的中间推理步骤。

一个核心挑战是：推荐场景没有类似数学题"标准答案"的明确推理链。用户为什么喜欢某个 item，往往涉及隐性偏好、情境因素和长期兴趣的复合作用。

### 核心主张

通过三阶段训练——对齐 item 理解、激活推理能力、增强推理质量——可以让 LLM-based 的生成式推荐系统学会"先推理、再推荐"。

### 贡献

1. **Itemic Token Alignment**：通过多任务预训练（item captioning + 用户行为理解 + 推荐生成），将 item token 的知识融入 LLM 的语义空间
2. **Reasoning Activation**：用剪枝后的用户行为上下文激发模型的推理能力，使模型开始输出显式推理链
3. **Reasoning Enhancement**：用 GRPO + Rollout-Beam reward 增强推理质量（不仅奖励正确推荐，还奖励推理链的质量）
4. **Think-Ahead 架构**：面向工业部署的推理加速设计

---

## Method 核心方法

### 1. 三阶段训练

![](https://arxiv.org/html/2510.11639/x2.png)

*Figure 2: OneRec-Think 三阶段训练框架——Stage 1: 多任务预训练实现 item 级语义对齐（item captioning + 用户行为理解 + 推荐生成）；Stage 2: 通过剪枝上下文引导激活显式偏好推理链；Stage 3: 用 GRPO + Rollout-Beam Reward 增强推理质量。*

#### Stage 1: Itemic Token Alignment

目标：让 LLM 理解"推荐世界的语言"——item token 和用户行为序列。

- **多任务预训练**：
  - Item captioning：给定 item token → 生成 item 的自然语言描述
  - 用户行为理解：给定行为序列 → 生成用户偏好摘要
  - 推荐生成：给定行为序列 → 生成推荐列表
- 使用 Qwen3 作为 backbone LLM
- 核心思想：在大量推荐语料上进行 domain adaptation，让 LLM 获得对推荐语义的基本理解

#### Stage 2: Reasoning Activation

目标：激活模型的推理能力，使其开始输出显式推理链。

- **剪枝上下文引导**（Pruned Context Bootstrapping）：
  - 取用户完整行为序列中信息密度最高的子集
  - 要求模型先输出推理链（用户偏好分析、类别推断、兴趣转移等），再给出推荐
- 使用少量高质量推理链示例作为引导
- 关键发现：直接用完整行为序列训练推理往往会失败——模型倾向于"跳过推理直接猜"。先让模型在简化任务上学会推理模式，再扩展到完整序列

#### Stage 3: Reasoning Enhancement

目标：用强化学习提升推理质量和推荐准确性。

- **GRPO（Group Relative Policy Optimization）**：
  - 每组（group）采样多个推荐链，比较它们的奖励，构造相对优势
- **Rollout-Beam Reward**：
  - 不仅奖励最终推荐 item 的正确性
  - 还奖励推理链的语义质量（是否连贯、是否充分利用了历史信息、推理逻辑是否合理）
  - 用 roll-out 方式验证推理链是否"导向"正确的推荐

### 2. Think-Ahead 架构

面向工业部署的加速策略：
- 推理链生成和推荐生成解耦
- 推理链可以在离线预计算并缓存
- 在线推理时只执行推荐生成部分，大幅降低延迟

---

## 实验/评估/结果

### 离线评估

- **Amazon Beauty/Sports/Toys**：Recall@10 全面超越基线，达到 SOTA
  - 对比方法：BERT4Rec、SASRec、P5、TIGER 等
- **Kuaishou 内部数据**：超越 OneRec V1 基线
- 关键消融：
  - 三阶段训练 vs 直接 Fine-tune：提升显著
  - 有推理链 vs 无推理链：推理链带来了可解释性和准确性提升
  - Rollout-Beam Reward vs 简单 final reward：推理链质量影响最终推荐效果

### 在线 A/B

- 快手 App：App 使用时长 +0.159%
- 虽然绝对提升不如 OneRec V1（0.54%），但这是在已经高度优化的 OneRec 基础上的 extra gain

---

## 结论

OneRec-Think 首次验证了推理增强在推荐系统中的有效性。三阶段训练策略（对齐→激活→增强）为将 LLM 推理能力迁移到推荐领域提供了可行路径。

---

## 思考

### 优点

1. **推理增强的方法论贡献**：三阶段渐进式训练策略（对齐→激活→增强）是一个可复用的框架，可应用于其他需要引入推理的领域
2. **问题定义新颖**：不是简单套用 CoT 模板，而是认真思考了"推荐推理链"应该长什么样
3. **工业可行性考虑**：Think-Ahead 架构思考了推理的延迟问题
4. **学术与工业双重验证**：在 Amazon 公开数据集上 SOTA + 在快手产线上 AB 验证

### 缺点与待解决问题

1. **推理链的监督信号来路**：推理链的质量标注可能需要大量人工，论文对推理链的训练数据来源描述不够详细
2. **收益规模有限**：0.159% 的使用时长提升在绝对值上有限，推理增强的 scaling 行为未知
3. **推理的可解释性验证不足**：模型输出的推理链是否真的被用于推荐决策，还是只是辅助 token？论文没有做因果分析
4. **LLM 推理的成本**：用 Qwen3 做 backbone 相比 OneRec 的 encoder-decoder 大幅增加了参数量和推理成本

### 与已有 Wiki 的连接

- 关联概念：[[Wiki/Concepts/Chain-of-Thought]]、[[Wiki/Concepts/GRPO]]、[[Wiki/Concepts/推理增强]]
- 关联实体：[[Wiki/Entities/Qwen3]]、[[Wiki/Entities/快手]]
- 关联比较：[[OneRec 技术报告]]（基座模型）、[[OneRec-V2]]（lazy decoder-only 架构）
