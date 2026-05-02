---
title: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
authors:
  - Shunyu Yao
  - Dian Yu
  - Jeffrey Zhao
  - Izhak Shafran
  - Thomas L. Griffiths
  - Yuan Cao
  - Karthik Narasimhan
institutions: Princeton University, Google DeepMind
aliases:
  - ToT
  - Tree of Thoughts
date: 2026-04-30
publish_date: 2023-05
venue: 'NeurIPS 2023 (Oral)'
tags:
  - 论文
  - 推理
  - 搜索
  - LLM
  - 树搜索
  - CoT扩展
url: 'https://arxiv.org/abs/2305.10601'
code: 'https://github.com/princeton-nlp/tree-of-thought-llm'
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：本文提出 Tree of Thoughts（ToT）框架，将 LLM 的推理过程从 CoT 的单链扩展为树搜索——在每个推理步骤生成多个候选"思维"，利用 LLM 自评估来选择最有希望的路径，支持前瞻（lookahead）和回溯（backtracking），在 Game of 24（4%→74%）、Creative Writing 和 Mini Crosswords 上取得显著提升。

![](https://arxiv.org/html/2305.10601v2/x1.png)

---

## Intro

### Motivation

CoT prompting 虽然显著提升了 LLM 的推理能力，但存在根本性局限：推理是 token-level、从左到右的单链过程，缺乏探索、策略性前瞻和回溯的能力。作者引入认知科学中的"双过程"理论（Kahneman 的 System 1/System 2）：LLM 的 token-level 自回归生成类似于快速、自动的 System 1，而需要更审慎的 System 2——维护和探索多种选择、评估当前状态、必要时回溯——来应对复杂问题。

作者从 AI 先驱 Newell, Shaw, Simon 的经典工作中汲取灵感：问题求解被建模为在组合问题空间中的树搜索。ToT 的核心想法是将这一经典框架与 LLM 的语义能力结合。

### 贡献

1. 提出 ToT 框架：将 LLM 推理视为树搜索，每个节点是一个中间"思维"（thought），支持生成、评估和搜索
2. 定义了 ToT 的四个关键组件：思维分解、思维生成器、状态评估器、搜索算法
3. 在三个新型任务上验证：Game of 24（74% vs CoT 4%）、Creative Writing、Mini Crosswords
4. 证明了 LLM 可以作为自身的启发式评估器（heuristic），无需训练额外的价值模型

---

## Method 核心方法

### 推理范式对比

| 方法 | 结构 | 搜索 | 评估 | 回溯 | 计算成本 |
|------|------|------|------|------|---------|
| CoT | 链 | 无（贪婪） | 无 | 无 | 1x |
| CoT-SC | 链×K | 无（多数投票） | 无 | 无 | K× |
| Self-Refine | 链 | 无（迭代修正） | 自评估 | 无 | ~3x |
| **ToT** | **树** | **BFS/DFS** | **LLM打分/投票** | **有** | **5-100×** |

### ToT 的四个组件

ToT 将任意问题视为树搜索，每个节点 $s = [x, z_1, \ldots, z_i]$ 是一个包含输入和已有思维序列的部分解。

**1. 思维分解（Thought decomposition）**：根据问题性质设计中间思维步骤的粒度。Thought 应"足够小"以保证多样性和可生成性，又"足够大"以保证可评估性。例如：
- Game of 24：每步是"剩余数字 + 一个中间方程"
- Creative Writing：整个写作计划就是一个 thought
- Crosswords：每个单词填充是一个 thought

**2. 思维生成器 $G(p_\theta, s, k)$**：给定当前状态，生成 $k$ 个候选下一步思维。两种策略：
- **独立采样（Sample i.i.d.）**：从 CoT prompt 独立采样 $k$ 次。适用于思维空间丰富、i.i.d. 样本足够多样化的场景（如 Creative Writing）
- **顺序提议（Propose sequentially）**：用一个"提议 prompt"一次性生成 $k$ 个候选。适用于思维空间较受限的场景（如 Game of 24 的数字运算）

**3. 状态评估器 $V(p_\theta, S)$**：评估每个状态的进展，作为搜索的启发式。两种策略：
- **独立估值**：LLM 对每个状态打分（如 1-10）或分类（sure/likely/impossible）。通过少量前瞻模拟 + 常识判断
- **跨状态投票**：让 LLM 比较所有候选状态，投票选出最有希望的一个。类似"逐步自洽性"

**4. 搜索算法**：
- **BFS（广度优先）**：每步保留最佳 $b$ 个状态。适用于深度有限（$T \leq 3$）的问题（Game of 24, Creative Writing）
- **DFS（深度优先）**：优先探索最有希望的状态，到叶节点或评估为不可能后回溯。适用于更深的搜索（Crosswords）

![](https://arxiv.org/html/2305.10601v2/x2.png)

*Figure 2: ToT 方法示意图。CoT（左）是单链推理；ToT（右）在每个步骤生成多个候选思维、评估并选择/回溯，形成树搜索。*

### 与已有方法的关系

CoT、CoT-SC 和 self-refinement 都可以看作 ToT 的特例（有限深度和宽度的树）。

---

## 实验/评估/结果

### Game of 24

- **任务**：给定 4 个数字和 +-\*/ 运算，得到 24
- **结果**：IO 7.3% / CoT 4.0% / CoT-SC 9.0% → **ToT (b=5) 74%**
- 错误分析：~60% 的 CoT 样本在第一步就失败了（前三个 token 出错），说明直接 left-to-right 解码对探索类问题严重不足

### Creative Writing

- **任务**：给定 4 个随机句子，写 4 段连贯文章，每段以对应句子结尾
- **ToT**：先生成 5 个写作计划→投票选最佳→再生成 5 篇文章→投票选最佳（深度 2，宽度 b=1）
- **结果**：GPT-4 评分 IO 6.19 / CoT 6.93 → **ToT 7.56**。人类盲评中 41/100 偏好 ToT，仅 21/100 偏好 CoT

### Mini Crosswords (5x5)

- **ToT**：DFS + 按字谜约束顺序填充 → 回溯 → 剪枝
- **结果**：单词级成功率 IO/CoT <16% → **ToT 60%**，完整解出 4/20 题

### 算力成本分析

ToT 需要的推理 token 约为 CoT 的 5-100 倍。Game of 24 单题约 $0.74，但性能远超 100 次独立 CoT 采样的最佳结果。

---

## 结论

ToT 框架证明了 LLM 的"System 1"级自回归生成可以被基于树搜索的"System 2"有效增强。ToT 将经典 AI 中的规划/搜索思想与当代 LLM 结合，开辟了 LLM 推理增强的新范式。

---

## 思考

### 优点

1. **理论框架的优雅性**：ToT 成功连接了认知科学（双过程理论）、经典 AI（树搜索）和当代 LLM，形成了完整的概念框架。CoT、CoT-SC、self-refinement 都被统一到 ToT 框架下作为特例。

2. **LLM 作为启发式函数**：最创新的洞见是用 LLM 自身的语义理解能力来做状态评估。传统搜索需要人工设计启发式或训练价值模型，而 ToT 直接让 LLM 在自然语言中"推理"状态的优劣，极大降低了搜索的应用门槛。

3. **任务设计的创造性**：Game of 24、Creative Writing、Crosswords 这三个任务精心选择，分别检验 deductive/mathematical、creative、lexical 推理 + search，且都是 GPT-4 + CoT 表现不佳的硬任务。

4. **回溯机制的价值**：在 Crosswords 实验中，删除回溯能力后性能从 60% 降至 20%，清晰证明了回溯（而非简单贪心）对深度搜索问题的重要性。

### 缺点与局限性

1. **推理成本极高**：ToT 需要比 CoT 多 5-100 倍的推理 token 和 API 调用，在实际部署中难以承受。每个中间状态都需要多次 LLM 调用来生成和评估候选。

2. **思维粒度需要人工设计**：每个任务需要手动设计"思维"的粒度和格式，以及搜索策略的配置（深度、宽度、剪枝阈值），缺乏自动化。

3. **无训练，无积累**：ToT 完全在推理时运作，不会让模型变得更好。每次重新运行都是从零开始搜索。后续的 o1/R1 等工作通过 RL 训练将推理能力内化到模型参数中。

4. **与后续 RL 方法的差距**：o1 和 R1 等方法通过 RL 训练让模型自主学会推理和验证，在 AIME 等大规模推理基准上远超 ToT + GPT-4 的效果。ToT 的价值更多在于概念框架而非最终性能。

5. **适用范围有限**：ToT 对需要探索和搜索的封闭式任务（如数学推理、逻辑谜题）效果好，但对开放式对话、创意生成等任务意义不大，因为这类任务不存在明确定义的"正确解"。
