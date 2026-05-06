---
type: concept
status: active
created: '2026-05-06'
updated: '2026-05-06'
tags:
  - agent
  - self-evolving
  - reinforcement-learning
  - lifelong-learning
sources:
  - '[[Wiki/Sources/Self-Evolving Agents 综述]]'
aliases:
  - 自演化智能体
  - Self-Evolving Agent
---

# Self-Evolving Agents 自演化智能体

## 定义

Self-Evolving Agents 是一种让 Agent 以**最小人类监督**自主协调自身改进循环的范式。与依赖人工策划更新的传统系统不同，自演化 Agent 通过闭环机制运作：主动探索问题空间、从推理或交互轨迹中生成自身训练信号、迭代优化策略。

## 两个核心性质

1. **强自主性（Strong Autonomy）**：无需外部监督即可生成学习信号
2. **主动探索（Active Exploration）**：通过内部模型驱动优化或与环境持续交互来改进策略

## Agent 形式化定义

$$A = \langle \Phi, \mathcal{M}, \mathcal{P}, \mathcal{T}, \mathcal{I} \rangle$$

- $\Phi$：核心 LLM（认知大脑）
- $\mathcal{M}$：记忆模块（短期 + 长期）
- $\mathcal{T}$：工具集（API/外部服务）
- $\mathcal{I}$：交互接口（感知 + 行动执行 + 评估反馈）

## 环境形式化定义

$$\mathcal{E} = \langle S, \mathcal{V} \rangle$$

- $S$：状态空间（任务上下文、外部知识库）
- $\mathcal{V}$：验证与反馈机制（环境的核心特征：客观、确定性的反馈信号）

## 三大演化范式

### 模型中心自演化

Agent 通过挖掘内化自身知识提升能力，核心原则是"计算换智能"：
- **推理时演化**：并行采样、顺序自纠正、结构化推理（不更新参数）
- **训练时演化**：合成驱动离线蒸馏、探索驱动在线 RL（更新参数）

### 环境中心自演化

Agent 通过与外部世界持续交互实现演化：
- 静态知识演化（Agentic RAG、深度研究）
- 动态经验演化（离线编译、在线适配、终身演化）
- 模块化架构演化（记忆/工具/交互协议优化）
- 智能体拓扑演化（多智能体结构搜索与动态适配）

### 模型-环境协同演化

Agent 和环境**共同演化**，形成开放式持续增长：
- 多智能体策略协同演化（MARL + 对齐训练）
- 环境训练（自适应课程 + 可扩展环境构建）

## 与传统范式的区别

| 维度 | 传统 Agent | Self-Evolving Agent |
|------|-----------|-------------------|
| 训练信号来源 | 人工标注/人工定义奖励 | 自生成/环境反馈 |
| 可扩展性 | 受限于人类标注成本 | 自主持续改进 |
| 能力上限 | 人类专家水平 | 突破人类上限 |
| 环境角色 | 静态背景 | 可优化的共同演化伙伴 |
| 学习模式 | 离线批量训练 | 在线持续终身学习 |

## 相关页面

- [[Wiki/Concepts/LLM Agent 架构]] — Agent 的基础架构
- [[Wiki/Concepts/推理模型与强化学习]] — RL 训练方法
- [[Wiki/Concepts/测试时计算扩展]] — 推理时演化的核心技术
- [[Wiki/Concepts/RAG 检索增强生成]] — Agentic RAG 基础
- [[Wiki/Sources/Self-Evolving Agents 综述]] — 本概念的系统性综述来源
