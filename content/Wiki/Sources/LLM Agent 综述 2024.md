---
type: source
status: processed
source_file: '[[Clippings/A Survey on Large Language Model based Autonomous Agents]]'
title: A Survey on LLM-based Autonomous Agents
site: Frontiers of Computer Science
author: Lei Wang et al. (Renmin University)
published: '2024'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - agent
  - LLM
  - survey
aliases:
  - LLM Agent 综述
---
# A Survey on LLM-based Autonomous Agents

## 核心结论

- 首次系统性地提出 LLM Agent 统一框架，包含四个核心模块：**Profile（角色设定）、Memory（记忆）、Planning（规划）、Action（行动）**。
- Agent 能力获取分为两大类：需要微调（人工标注/LLM生成/真实数据）和不需要微调（Prompt Engineering / Mechanism Engineering）。
- LLM Agent 应用覆盖社会科学（心理模拟、社会仿真）、自然科学（实验助手、文献管理）和工程（软件开发、机器人）。

## 关键事实

- 来源：Renmin University of China（中国人民大学），发表于 Frontiers of Computer Science 2024。
- 统一框架：Profile 决定角色 → Memory 存储经验 → Planning 分解任务 → Action 执行操作。
- 记忆结构分 Unified Memory（仅短时）和 Hybrid Memory（短时+长时向量库）。
- 规划分为 Planning without Feedback（单路径/多路径/外部规划器）和 Planning with Feedback（环境反馈/人类反馈/模型自检）。
- 常用框架：LangChain、AutoGPT、MetaGPT、ChatDev、Voyager（Minecraft）。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Concepts/LLM Agent 架构]]
- 与 [[Wiki/Sources/Agent AI Survey 2024]] 互补：本综述侧重架构，Agent AI 侧重多模态交互。

## 后续问题

- LLM Agent 的幻觉问题如何在架构层面缓解？
- 记忆模块的长期一致性保证机制仍需探索。
