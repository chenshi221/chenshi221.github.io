---
type: concept
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - Agent 架构
tags:
  - agent
  - LLM
  - architecture
sources:
  - Wiki/Sources/LLM Agent 综述 2024
  - Wiki/Sources/Agent AI Survey 2024
confidence: high
---
# LLM Agent 架构

## 定义

LLM Agent 架构是以大语言模型（LLM）为核心控制器，通过 Profile（角色设定）、Memory（记忆）、Planning（规划）、Action（行动）四大模块协同工作来实现自主任务完成的系统设计范式。

## 四大核心模块

### 1. Profile Module（角色设定）

定义 Agent 的身份、性格、知识和行为边界。三种生成方式：
- **手写**（Handcrafting）：灵活但人力密集。
- **LLM 生成**（LLM-Generation）：用 LLM 自动生成大规模角色，但精确控制有限。
- **数据集对齐**（Dataset Alignment）：从真实数据集提取人口属性，适合社会仿真。

### 2. Memory Module（记忆）

- **结构**：统一记忆（仅短时，依赖 context window）vs 混合记忆（短时 + 长时向量库）。
- **格式**：自然语言、Embedding 向量、数据库、结构化列表。
- **操作**：读取（基于 recency/relevance/importance 加权检索）、写入（去重策略、溢出处理）、反思（从低层记忆生成高层洞见）。

### 3. Planning Module（规划）

- **无反馈规划**：单路径（CoT、Zero-shot-CoT）、多路径（ToT、GoT、RAP/MCTS）、外部规划器（LLM+P、PDDL）。
- **有反馈规划**：环境反馈（ReAct、Voyager）、人类反馈（Inner Monologue）、模型反馈（Self-Refine、Reflexion）。

### 4. Action Module（行动）

- **行动目标**：任务完成、通信交流、环境探索。
- **行动生成**：记忆回忆驱动、计划跟随驱动。
- **行动空间**：外部工具（API、数据库、外部模型）vs LLM 内部知识（规划、对话、常识理解）。

## 能力获取策略

- **需要微调**：人工标注数据、LLM 生成数据、真实数据微调。
- **不需要微调**：Prompt Engineering（CoT、ReAct）、Mechanism Engineering（试错、群体智慧、经验积累、自我进化）。

## 相关来源

- [[Wiki/Sources/LLM Agent 综述 2024]]：本概念的主要出处。
- [[Wiki/Sources/Agent AI Survey 2024]]：多模态 Agent 架构补充。

## Harness 工程：Agent 架构的工程实现层

AHE [[Wiki/Sources/AHE Agentic Harness Engineering]] 将 Agent 架构的四大模块**实例化为可编辑的 harness 组件**：

| Agent 模块 | Harness 组件 | 可演化性 |
|-----------|-------------|---------|
| Profile | System Prompt | 易编辑但单用负增益 |
| Memory | Long-term Memory | 最难但增益最大 (+5.6pp) |
| Planning | （内化在模型推理中） | 不由 harness 直接控制 |
| Action | Tools + Middleware + Skills + Sub-agents | 中等增益 (+2~3pp) |

关键洞察：**Agent 架构的"动作"模块在 harness 中被拆分为 4+ 个解耦组件**，每个可独立编辑、独立回滚。这种文件级解耦是实现 AHE 三大可观测性支柱的前提。

- 详见：[[Wiki/Concepts/Agentic Harness Engineering (AHE)]]
- 详见：[[Wiki/Concepts/Harness 编码 Agent 线束]]

## 与已有 Wiki 的关系

- 关联：[[Wiki/Topics/LLM Agent 与工具使用]]
- 关联：[[Wiki/Concepts/Agentic Harness Engineering (AHE)]]
- 关联：[[Wiki/Concepts/Harness 编码 Agent 线束]]
