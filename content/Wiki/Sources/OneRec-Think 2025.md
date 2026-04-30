---
type: source
status: processed
source_file: "[[Clippings/OneRec-Think In-Text Reasoning for Generative Recommendation]]"
title: "OneRec-Think: In-Text Reasoning for Generative Recommendation"
site: "arxiv.org"
author: "Zhanyu Liu, Shiyao Wang et al. (Kuaishou)"
published: "2025"
processed: "2026-04-30"
updated: "2026-04-30"
tags:
  - "generative-recommendation"
  - "chain-of-thought"
  - "reasoning"
  - "RL-recommendation"
aliases:
  - "OneRec-Think 推理推荐"
---

# OneRec-Think: In-Text Reasoning for Generative Recommendation

## 核心结论

OneRec-Think 将 Chain-of-Thought 推理能力引入生成式推荐，使模型能够生成可解释的文本推理路径。在快手部署后实现 App Stay Time 提升 0.159%。

## 关键事实

- **三阶段训练框架**：(1) Itemic Alignment：通过多任务预训练将 item 语义映射到 LLM 文本空间；(2) Reasoning Activation：从剪枝后的用户上下文中蒸馏推理轨迹，再用于引导原始噪声序列的推理生成；(3) Reasoning Enhancement：使用推荐特化的 reward 函数（Rollout-Beam reward）进行 RL 优化。
- **Itemic Token**：将 item 表示为离散的语义丰富 token，类似于自然语言的 word token，统一推荐和推理的输入空间。
- **Rollout-Beam Reward**：针对推荐场景中标准 pass reward 稀疏的问题（大部分 rollout 命中不了目标 item，都得到零 reward），设计 beam search 内的最优匹配作为 reward，解决 GRPO 等 RL 算法在推荐场景中的 reward 稀疏挑战。
- **Think-Ahead 架构**：为满足工业级延迟要求，将推理分为离线（生成推理路径和初始 item token）和在线（实时 OneRec 基于预生成的前缀 token 完成最终推荐）两阶段。
- **SOTA 性能**：在 Amazon Beauty/Sports/Toys 三个数据集上全面超越 BERT4Rec、SASRec、TIGER、HSTU、ReaRec 等基线。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/生成式推荐系统]]，[[Wiki/Concepts/生成式推荐]]，[[Wiki/Entities/OneRec 系列模型]]
- 关联 AI 知识：[[Wiki/Concepts/Chain-of-Thought]]（如已有），RLHF/GRPO

## 后续问题

- CoT 推理带来的额外推理时间和 token 成本在工业场景的 ROI 如何？
- 推理质量与推荐准确率之间是否存在 trade-off？
