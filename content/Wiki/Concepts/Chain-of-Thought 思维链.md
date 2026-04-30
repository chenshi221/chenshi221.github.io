---
type: concept
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - CoT
  - 思维链
tags:
  - chain-of-thought
  - prompting
  - reasoning
  - emergent-ability
sources:
  - '[[Wiki/Sources/Chain-of-Thought]]'
confidence: high
---
# Chain-of-Thought 思维链

## 定义

Chain-of-Thought (CoT) 是一种提示技术，通过在 few-shot 提示中提供逐步推理过程的示例，**引导大语言模型在回答问题时先生成中间推理步骤，再给出最终答案**。

## 核心机制

1. **推理外化**：将模型隐式的推理过程显式化为自然语言步骤。
2. **步骤分解**：复杂问题被拆解为一系列更简单的中间步骤，每步只需处理较小的推理跨度。
3. **涌现性**：CoT 仅在模型规模超过一定阈值（~100B 参数）后才显著有效，被认为是"涌现能力"之一。

## 典型形式

- **Few-shot CoT**（标准 CoT）：在 prompt 中提供带推理步骤的示例。如："Q: 小红有 3 个苹果，又买了 2 个，她现在有几个？A: 小红一开始有 3 个苹果。她又买了 2 个。3+2=5。所以她有 5 个苹果。"
- **Zero-shot CoT**：仅添加"Let's think step by step"，无需示例也能触发模型逐步推理。
- **Self-Consistency**：生成多条 CoT 路径，取多数投票结果（提升 robust 性）。
- **Auto-CoT**：自动生成 CoT 示例，减少人工编写成本。

## 适用范围

CoT 在以下推理任务上有效：
- 算术推理（math word problems）
- 常识推理（commonsense reasoning）
- 符号推理（symbolic reasoning）
- 多步逻辑推理

但在"直觉型"任务（如情感判断、简单分类）上不一定有帮助，甚至可能引入额外错误。

## 在 CoT → ToT → GoT 演进中的位置

- **CoT**：单条线性推理路径。每个步骤产生一个推理，下一步基于上一步。
- **ToT**：多条路径树状探索。每步可产生多个候选思考，通过评估选择最优路径。
- **GoT**：任意图结构推理。可合并（combine）、提炼（distill）、反馈增强（refine）多条思考。

CoT 是推理增强的起点，也是后续所有方法的基石。

## 局限

- 仅在足够大的模型上有效（小模型可能生成错误推理步骤导致更差结果）。
- 不能探索多种推理路径（单路径策略可能锁死错误方向）。
- 不适用于需要回溯、前瞻或并行探索的任务。

## 关联

- 来源论文：[[Wiki/Sources/Chain-of-Thought]]
- 主题：[[Wiki/Topics/推理增强方法]]
- 多模态延伸：[[Wiki/Entities/GoT]]（GoT 论文 = Generation Chain-of-Thought，多模态推理引导生成）
