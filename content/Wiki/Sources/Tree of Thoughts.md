---
type: source
status: processed
source_file: >-
  [[Clippings/Tree of Thoughts Deliberate Problem Solving with Large Language
  Models.md]]
title: 'Tree of Thoughts: Deliberate Problem Solving with Large Language Models'
site: arXiv
author: 'Shunyu Yao et al. (Princeton, Google DeepMind)'
published: 2023-05
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - tree-of-thoughts
  - reasoning
  - planning
  - search
aliases:
  - ToT
---
# Tree of Thoughts: Deliberate Problem Solving with Large Language Models

## 核心结论

- **Tree of Thoughts (ToT)** 将 CoT 的线性思维链泛化为**树状思维探索**，使 LLM 能同时探索多条推理路径，并通过自我评估进行前瞻和回溯。
- ToT 使 LLM 从"token 级从左到右"的惯性推理跃升为"审慎决策"：考虑多种推理路径，评估每个选择，必要时回溯。
- 在需要规划和搜索的任务上（Game of 24、创意写作、迷你填字），ToT 大幅超越 CoT。Game of 24 上 GPT-4+CoT 仅 4% 成功率，ToT 达 74%。

## 关键事实

- 作者：Shunyu Yao, Dian Yu 等（Princeton, Google DeepMind）。
- 发表于 2023（arXiv:2305.10601，NeurIPS 2023）。
- ToT 四步框架：Thought decomposition（问题分解为思考步骤）→ Thought generator（生成候选思考）→ State evaluator（评估状态/投票）→ Search algorithm（BFS/DFS 选择路径）。
- 三个任务展示了不同的搜索策略：Game of 24（BFS + 前瞻评估）、创意写作（随机采样 + 投票）、填字（DFS + 回溯）。

## 方法亮点

- **模块化框架**：思考分解、生成、评估、搜索四个组件可独立设计和替换。
- **前瞻与回溯**：让 LLM 在推理过程中"看到未来"并"回到过去"，大幅提升复杂规划的准确性。
- 与 CoT 的核心区别：CoT 是单路径生成，ToT 是多路径探索 + 选择。

## 与现有 Wiki 的关系

- 从 [[Wiki/Sources/Chain-of-Thought]] 的线性推理演进到树状探索，见 [[Wiki/Topics/推理增强方法]]。
- 启发了 [[Wiki/Sources/Graph of Thoughts]] 的最一般图结构。

## 后续问题

- ToT 的计算开销远大于 CoT（多次 LLM 调用），如何在效果和成本间平衡？
- ToT 在开放域对话、代码生成等更通用任务上的有效性如何？
