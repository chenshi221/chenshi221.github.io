---
type: source
status: processed
source_file: >-
  [[Clippings/Agentic Harness Engineering Observability-Driven Automatic
  Evolution of Coding-Agent Harnesses.md]]
title: >-
  Agentic Harness Engineering: Observability-Driven Automatic Evolution of
  Coding-Agent Harnesses
site: arXiv
author: 'Jiahang Lin, Shichun Liu, Chengjun Pan et al.'
published: '2026'
processed: '2026-05-03'
updated: '2026-05-03'
tags:
  - clippings
  - agent
  - coding-agent
  - harness
  - self-evolution
  - observability
aliases:
  - AHE
---
# AHE: Agentic Harness Engineering — 可观测性驱动的编码 Agent 线束自动演化

## 核心结论

- **AHE 是一种闭环系统**，通过与三个可观测性支柱匹配的设计，自动演化编码 Agent 的 harness（线束/脚手架），将每一次编辑变为可证伪的契约。
- **10 次迭代将 Terminal-Bench 2 的 pass@1 从 69.7% 提升至 77.0%**，超越人工设计的 Codex-CLI（71.9%）和自演化基线 ACE、TF-GRPO。
- **冻结的 harness 可跨 benchmark 和模型家族迁移**：在 SWE-bench-verified 上达到最高综合成功率并节省 12% token；跨 3 个替代模型家族获得 +5.1 到 +10.1 pp 增益。
- **增益来源于 tools、middleware 和 long-term memory**，而非 system prompt——表明事实性 harness 结构可迁移，而散文级策略不可。
- **核心洞察**：harness 优化被"可观测性"瓶颈而非 agent 能力瓶颈。一旦演化 agent 获得结构化上下文和清晰的动作空间，就能可靠收敛。

## 关键事实

- **作者**：Jiahang Lin (复旦), Shichun Liu (复旦), Chengjun Pan (北大), Lizhi Lin (上海奇绩智峰) 等
- **机构**：复旦大学、北京大学、上海奇绩智峰有限公司
- **arXiv**: 2604.25850v3, 2026
- **实验模型**：GPT-5.4 (high reasoning)，跨模型测试包括 qwen-3.6-plus, gemini-3.1-flash-lite-preview, deepseek-v4-flash
- **Benchmark**：Terminal-Bench 2（89 任务：4 easy + 55 medium + 30 hard，每任务 1 小时超时），SWE-bench-verified（500 任务）
- **迭代成本**：10 轮迭代约 32 小时，每任务每轮 k=2 rollouts
- **种子 harness**：NexAU₀ — 仅 bash 工具，无 middleware、无 skills、无 sub-agents、无 long-term memory

## 方法或论证路径

### 问题定义

Harness 工程面临三大挑战：
1. **异构动作空间**（heterogeneous action space）：可编辑组件类型多样（prompt、tool、middleware、memory 等）
2. **海量轨迹淹没信号**（voluminous trajectories）：原始轨迹数百万 token，可行动信号被埋没
3. **编辑效果难归因**（hard attribution）：编辑对 pass rate 的影响难以定位

AHE 通过三个对应的可观测性支柱解决：

### 支柱 1：组件可观测性 — NexAU 解耦 harness 基板

- 将 harness 的 7 种可编辑组件暴露为独立文件：system prompt、tool description、tool implementation、middleware、skill、sub-agent configuration、long-term memory
- 每个故障模式映射到单一组件类，每次编辑是一个 git commit
- 种子 harness 故意最小化（仅 bash），迫使每个新增组件通过实测 rollouts 证明价值

### 支柱 2：经验可观测性 — Agent Debugger 分层轨迹证据

- 对每个任务的 k 个 rollout 轨迹进行分层蒸馏
- 轨迹被框架化为可导航的文件环境，debugger agent 用 shell/grep 等工具分析
- 输出：per-task analysis（根因分析 + pass/fail）+ benchmark-level overview（聚合入口）
- 原始轨迹保留供验证，支持渐进式信息披露（progressive disclosure）节省 token

### 支柱 3：决策可观测性 — Evolve Agent 证据驱动、可审计编辑

- 每轮读取证据语料，决定增/改/删哪些组件
- **每条编辑附带自声明预测**（预期修复任务 + 风险回归任务）
- 下一轮验证：将预测与观察到的 task-level delta 交叉比对，形成 per-edit verdict
- 验证失败的编辑在文件粒度回滚

### AHE 外循环算法

```
for t = 1 to N:
  1. Rollout: k 次 rollout/任务
  2. Clean: 去除 base64、去重 tool output
  3. Attribute (t≥2): 验证上一轮 manifest 预测 → 回滚失败编辑
  4. AgentDebugger: 分层蒸馏轨迹
  5. Evolve: workspace 编辑 + 新 manifest
  6. Commit: git 标记迭代
```

## 实验结果

### RQ1: AHE vs 人工 vs 自动化基线

| Method | All (89) | Easy (4) | Medium (55) | Hard (30) |
|--------|----------|----------|-------------|-----------|
| opencode | 47.2% | 75.0% | 52.7% | 33.3% |
| terminus-2 | 62.9% | 75.0% | 74.5% | 40.0% |
| Codex-CLI | 71.9% | 75.0% | 80.0% | 56.7% |
| **NexAU₀** | 69.7% | 87.5% | 78.2% | 51.7% |
| ACE | 68.9% | 91.7% | 78.2% | 48.9% |
| TF-GRPO | 72.3% | 100.0% | 79.4% | 55.6% |
| **AHE** | **77.0%** | **100.0%** | **88.2%** | 53.3% |

### RQ2: 跨 benchmark 和跨模型迁移

- **SWE-bench-verified**：AHE 综合成功率最高（75.6%），token 消耗比种子少 12%，比 ACE 少 32%
- **跨模型**：全部 5 个替代基座模型获得正增益（+2.3 到 +10.1 pp），跨家族增益大于家族内增益
- 远离饱和的模型获益更大：deepseek-v4-flash +10.1 pp, qwen-3.6-plus +6.3 pp

### RQ3: 分析

- **组件消融**：单独替换 AHE 的 tools (+3.3pp), middleware (+2.2pp), long-term memory (+5.6pp) 均超越种子；system prompt 单独替换倒退 -2.3pp
- **组件非加性交互**：三个正组件单独增益之和 (+11.1pp) > 全量增益 (+7.3pp)，因组件间存在冗余验证行为
- **自归因可靠性**：修复预测 precision 33.7%、recall 51.4%（约 5 倍于随机基线）；回归预测 precision 11.8%、recall 11.1%（仅约 2 倍于随机基线）
- **回归盲区**：agent 能可靠预测什么会变好，但几乎无法预测什么会变差

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/LLM Agent 与工具使用]] — AHE 是 Agent 架构在 harness 工程维度的深化
- 关联：[[Wiki/Concepts/LLM Agent 架构]] — AHE 的 7 组件 harness 是 Agent 架构的"外部化"实现
- 关联：[[Wiki/Concepts/推理模型与强化学习]] — AHE 的 self-evolution loop 与 RL-based 推理训练形成互补：前者优化外部 harness，后者优化内部模型
- 关联：[[Wiki/Comparisons/推理模型训练方法比较 DeepSeek-R1 vs Kimi k1.5 vs Qwen3]] — AHE 提供了"模型外优化"的新维度，与模型内 RL 训练形成互补轴
- 支持：Agent 系统性能不仅取决于模型能力，也取决于周围的工程组件——这与 [[Wiki/Concepts/多模态 Agent]] 中 Agent 工具使用的重要性一致
- 新视角：AHE 将 harness 工程从"手工工艺"提升为"可自动化的学习问题"，通过可观测性设计解决了 attribution 问题

## 可能的矛盾或待核实点

- AHE 的超时和步数预算是在 GPT-5.4 high 上拟合的，跨推理层级的增益非单调（medium +2.3, high +7.3, xhigh +2.3），需在多个操作点上重新运行来解耦
- Hard 难度上 AHE 落后于 Codex-CLI，虽然归因于组件交互冗余，但"交互感知演化"仍是开放问题
- 自归因的回归盲区（regression blindness）是核心局限——演化 agent 无法预测哪些编辑会破坏什么

## 后续问题

- "交互感知演化"（interaction-aware evolution）如何实现？能否让演化 agent 意识到组件间的非加性交互？
- 如何在更多的编程语言、仓库级部署和 human-in-the-loop 工作流中验证 AHE？
- 回归盲区（regression blindness）是否可以通过多轮预演（dry-run）或对抗性测试来缓解？
- AHE 的"模型外优化"与模型内 RL 训练能否协同？能否形成一个联合优化循环？
- harness 演化是否会在模型持续升级后产生"知识折旧"？如何衡量 harness 的时效性？
