---
type: source
status: processed
source_file: >-
  [[Clippings/Graph of Thoughts Solving Elaborate Problems with Large Language
  Models.md]]
title: 'Graph of Thoughts: Solving Elaborate Problems with Large Language Models'
site: arXiv
author: Maciej Besta et al. (ETH Zurich)
published: 2023-08
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - graph-of-thoughts
  - reasoning
  - prompting
aliases:
  - GoT
---
# Graph of Thoughts: Solving Elaborate Problems with Large Language Models

## 核心结论

- **Graph of Thoughts (GoT)** 将 LLM 推理过程建模为**任意有向图**：思考单元是节点，依赖关系是边。这是比 CoT（链）和 ToT（树）更一般的推理结构。
- 图结构使 GoT 能执行更复杂的思考变换：聚合多个思考（combine）、提炼网络核心（distill）、通过反馈循环增强思考（refine）。
- 在排序任务上，GoT 相比 ToT 质量提升 62%，同时成本降低 31%。

## 关键事实

- 作者：Maciej Besta, Nils Blach 等（ETH Zurich, Cledar 等）。
- 发表于 2023（arXiv:2308.09687）。
- GoT 的模块化架构包含三个组件：Prompter（与 LLM 交互生成/变换思考）、Parser（提取结构化信息）、Controller（编排图结构）。
- 核心思考变换操作：Generate（生成新思考）、Aggregate（合并多个思考为新思考）、Refine（基于已有思考做增强）、Score（评分排序）。
- 在 sorting、keyword counting、set operations、document merging 四个任务上验证了优势。

## 方法亮点

- **图抽象**：CoT 和 ToT 都是 GoT 的特例（链是退化的树，树是退化的图），GoT 可表达任意拓扑。
- **成本-质量平衡**：GoT 通过 merge 操作可在提升质量的同时降低 LLM 调用次数（相比 ToT 的独立路径探索）。
- **可扩展架构**：新增思考变换操作和新推理模式时无需修改框架。

## 与现有 Wiki 的关系

- 是 [[Wiki/Topics/推理增强方法]] 中"结构增强"的最高形态：链 → 树 → 图。
- 从 [[Wiki/Sources/Chain-of-Thought]] 和 [[Wiki/Sources/Tree of Thoughts]] 逐步演进而来。

## 后续问题

- 图结构的最优拓扑是否高度任务相关？如何自动搜索最优图结构？
- GoT 在更开放的任务（如多步代码生成、科学推理）上的应用探索。
