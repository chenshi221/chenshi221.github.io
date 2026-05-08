---
type: source
created: '2026-05-07'
updated: '2026-05-07'
tags:
  - agent
  - coding-agent
  - harness
  - observability
  - self-evolution
sources:
  - '[[Clippings/Agentic Harness Engineering Observability-Driven Automatic Evolution of Coding-Agent Harnesses]]'
---
# AHE: Agentic Harness Engineering

## 核心论点

- Harness（系统提示词、工具、中间件、长期记忆等模型外部组件）是 coding-agent 性能的关键杠杆
- 手动 harness 工程跟不上基础模型的快速迭代，需要自动化闭环
- AHE 提出三个可观测性支柱驱动 harness 自动演化

## 三个可观测性支柱

1. **组件可观测性 (Component Observability)**
   - 7 类可编辑组件：system prompt、tools、middleware、skills 等
   - 每个组件有文件级表示，编辑空间明确可回滚

2. **经验可观测性 (Experience Observability)**
   - 将百万级原始轨迹 token 蒸馏为分层、可钻取的证据语料
   - 演化 agent 消费结构化根因而非原始日志

3. **决策可观测性 (Decision Observability)**
   - 每次编辑附带自声明预测（change manifest）
   - 下一轮任务结果验证预测，每个编辑成为可证伪契约

## 关键结果

- Terminal-Bench 2: pass@1 从 69.7% 提升至 77.0%
- 超越人工设计 harness Codex-CLI (71.9%) 和自演化基线 ACE、TF-GRPO
- 冻结 harness 无需重新演化即可迁移：
  - SWE-bench-verified: 用更少 token 达到最高成功率
  - 跨 3 个模型家族: +5.1 到 +10.1pp 增益

## 消融实验发现

- **有效组件**: tools、middleware、long-term memory
- **无效组件**: system prompt（单独使用反而回退）
- 结论: 事实性 harness 结构可迁移，散文级策略不可迁移

## 局限性

- harness 组件交互是非加性的，叠加有效编辑有上限
- 自归因对修复可靠但对回退盲，回归预判是未来方向

## 元数据

- 作者: Jiahang Lin, Shichun Liu, Chengjun Pan, Lizhi Lin, Shihan Dou, Xuanjing Huang, Hang Yan, Zhenhua Han, Tao Gui
- 机构: 复旦大学、北京大学、上海奇绩智丰
- 日期: 2026-04-28
- arxiv: 2604.25850
