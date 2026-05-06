---
type: source
status: complete
created: '2026-05-06'
updated: '2026-05-06'
tags:
  - agent
  - self-evolving
  - reinforcement-learning
  - survey
sources:
  - '[[Clippings/A Systematic Survey of Self-Evolving Agents.md]]'
---

# Self-Evolving Agents 综述

> A Systematic Survey of Self-Evolving Agents: From Model-Centric to Environment-Driven Co-Evolution
> Zhishang Xiang, Chengyi Yang et al. | 厦门大学 & 香港理工大学 | 2026-02 | arXiv: 2502.14683

## 核心论点

传统 Agent 依赖人工监督（SFT 模仿学习 + RL 人工奖励信号），存在**可扩展性瓶颈**和**人类专家上限约束**。Self-Evolving Agents 范式让 Agent 以最小人类监督自主协调自身的改进循环，是突破这一瓶颈的关键方向。

作者强调：**环境不应是静态背景，而应是与 Agent 共同演化的可优化伙伴**——Model-Environment Co-Evolution 是最关键的未来方向。

## 统一分类体系

论文将 Self-Evolving Agents 方法分为三大范式：

### 1. Model-Centric Self-Evolution（模型中心自演化）

Agent 通过挖掘和内化自身知识来自主提升能力，核心原则是"**计算换智能**"。

**推理时演化（Inference-Based）**：
- **并行采样**：Self-Consistency、Best-of-N，通过扩大采样预算让小模型超越大模型
- **顺序自纠正**：Reflexion（verbal RL）、CRITIC（外部工具验证）、Planning Tokens
- **结构化推理**：Tree of Thoughts → Graph of Thoughts → Think-on-Graph（知识图谱对齐）

**训练时演化（Training-Based）**：
- **合成驱动离线蒸馏**：SELF-INSTRUCT → STaR（自洽性迭代）→ ReST-MCTS*（搜索算法+过程奖励）
- **探索驱动在线演化**：R-Zero / Absolute Zero（Challenger-Solver 角色分裂）、WebRL（在线课程 RL）、SPIRAL（多智能体零和博弈）

### 2. Environment-Centric Self-Evolution（环境中心自演化）

Agent 通过与外部世界持续交互实现演化，环境提供静态知识和动态反馈。

**静态知识演化**：
- **Agentic RAG**：从被动检索到自主认知过程（Search-o1、Search-R1、DSPy）
- **深度研究**：自主浏览+证据综合→研究报告（DeepResearcher、WebWeaver）

**动态经验演化**：
- **离线经验编译**：从历史数据构建静态经验库（Agent Workflow Memory、SkillWeaver）
- **在线经验适配**：查询条件化的经验状态（Dynamic Cheatsheet、Memento）
- **终身经验演化**：持久更新的经验库（ReasoningBank、ASI 程序化技能、MemGen 潜在记忆注入、LatentEvolve 昼夜机制）

**模块化架构演化**：
- **交互协议演化**：MemGPT 虚拟内存分页、MemoryBank 遗忘曲线
- **记忆架构演化**：A-MEM Zettelkasten 网络、MemEvolve 元级重构
- **工具增强演化**：VOYAGER 技能库 → LATM/CREATOR 自主工具创建 → Alita 最小预定义架构

**智能体拓扑演化**：
- **离线架构搜索**：AFLOW（MCTS 工作流优化）、ADAS（元智能体写代码）
- **运行时动态适配**：AutoAgents 动态专家团队、G-Designer 图神经网络解码拓扑
- **结构状态演化**：G-Memory 层级图、LatentMAS 隐空间 KV cache 同步

### 3. Model-Environment Co-Evolution（模型-环境协同演化）

Agent 和环境**共同演化**，形成开放式的持续能力增长系统。

**多智能体策略协同演化**：
- OPTIMA（MCTS 指导通信优化）、MAPoRL（验证器反馈长期协作）、CoMAS（同伴评估替代人类反馈）

**环境训练**：
- **自适应课程演化**：GenEnv（模拟器作为动态课程生成器）、RLVE（可验证环境+动态难度匹配）
- **可扩展环境演化**：Dream-Gym（推理世界模型模拟）、Reasoning Gym（100+ 程序化验证任务）、AgentGym（统一交互平台）

## 关键洞察

1. **离线合成 vs 在线探索**：离线蒸馏是高效启动器但受限于基础模型容量；在线探索通过环境反馈突破数据天花板
2. **静态知识 vs 动态经验**：单向从固定环境提取终将受限；Agent-环境关系需从被动提取转向互惠交互
3. **模型中心 vs 协同演化**：模型中心在预定义环境中优化，遇到性能高原；协同演化中环境本身是动态演化系统

## 当前挑战

- **静态非自适应环境**：大多数方法仍在固定规则环境中操作，导致过拟合
- **过度依赖可验证任务**：编译器/单元测试/定理证明器限制了确定性领域之外的进展
- **模拟环境真实性不足**：简化模拟器无法捕获物理世界的不确定性和噪声
- **持续依赖人类初始化**：初始监督的局限或偏差会导致错误累积
- **模型坍缩**：自生成数据递归训练导致输出分布收窄，长尾信息丢失

## 未来方向

- **随 Agent 成长的自适应环境**：Agent 改善自然导致更难的环境
- **更真实、开放的环境**：更丰富的物理动态和开放生成
- **连接多模拟器与真实系统**：统一训练管道
- **超越自动验证的自演化**：创意写作、对话、社会推理等无明确 ground truth 的任务

## 应用领域

- **自动化科学发现**：The AI Scientist、AlphaProof、ChemCrow、GNoME（220 万新材料）、A-Lab（71% 合成成功率）
- **自主软件工程**：SWE-agent（ACI 接口设计）、Claude Code（技能积累）、Manus、Devin、Cursor
- **开放世界模拟**：Voyager（Minecraft 终身学习）、Generative Agents（涌现社交行为）、Project Sid（涌现经济与法律）

## 相关页面

- [[Wiki/Concepts/LLM Agent 架构]] — Agent 五元组定义的基础
- [[Wiki/Concepts/推理模型与强化学习]] — GRPO、RL 训练方法
- [[Wiki/Concepts/测试时计算扩展]] — Inference-Based Evolution 相关
- [[Wiki/Concepts/Chain-of-Thought 思维链]] — 结构化推理基础
- [[Wiki/Concepts/RAG 检索增强生成]] — Agentic RAG 的基础
- [[Wiki/Sources/LLM Agent 综述 2024]] — 前序 Agent 综述
- [[Wiki/Sources/DeepSeek-R1 强化学习推理]] — RL 驱动推理能力
