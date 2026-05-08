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

## Agent Debugger 源码分析

Agent Debugger 不是独立仓库，而是作为 Evolve Agent 的 skill 嵌入在 AHE 代码库中。

**仓库位置**: `agents/evolve_agent/skills/agent-debugger-cli/`（[GitHub](https://github.com/china-qijizhifeng/agentic-harness-engineering/tree/main/agents/evolve_agent/skills/agent-debugger-cli)）

**本质**: 一个 NexAU Agent（`debugger_agent`），不是传统程序化分析器。

- **迭代预算**: 硬限制 20 次工具调用
- **工具集**: `read_file`, `search_file_content`, `glob`, `run_shell_command`, `web_search`, `complete_task`
- **5 阶段工作流**（定义在 system_prompt.md）:
  1. Skim (iter 1-3): read_file 看 trace 头部
  2. Locate (iter 4-10): search_file_content 正则搜索错误关键词
  3. Read in context (iter 11-15): 读命中位置上下文
  4. Cross-trace diff (iter 16-18): 多 trace 对比
  5. Finalize (iter ≤20): complete_task 输出
- **输出 5 种 issue 类型**: `{工具错误, 幻觉, 循环, 不合规, 截断}`
- **两种模式**: `ask`（自由文本）/ `check`（结构化 issues JSON）
- **trace 格式支持**: `openai_messages`（默认）、`langfuse`、`in_memory_tracer`
- **中间件**: LongToolOutputMiddleware（截断长输出）+ ContextCompactionMiddleware（上下文压缩，阈值 0.75）
- **runner.py**: 重试机制（3次指数退避）、预算溢出回退、输出 schema 验证+自动重试

**关键洞察**: Agent Debugger 的分析能力很大程度上来自 trace 的结构化程度（terminal 的 exit code、stack trace、结构化消息序列），而非自身的推理能力。这是一个 20 次工具调用的轻量 Agent，靠 system prompt 中的工作流引导高效提取信号。

## 演化案例研究（4 个关键轨迹）

### 轨迹 1：db-wal-recovery（迭代 2）
- **任务**：从损坏的 WAL 文件重建 SQLite 数据库
- **失败模式**：Agent 从缓存 shell 输出恢复字节、用猜测模式编造缺失行、遗漏对已有行的更新、在自创代理检查（行计数）上提交
- **AHE 修复**：chg-1 在 system prompt 追加 8 条通用规则——契约优先、评估者镜像、最小编辑、泛化不拟合等
- **效果**：0/2 → 2/2，且后续每轮保持 2/2。关键洞察：规则不含 SQLite/WAL 等特定关键词，但通用措辞使其跨任务生效

### 轨迹 2：path-tracing（迭代 5）
- **任务**：实现路径追踪渲染器，输出 PPM 文件
- **失败模式**：Agent 渲染正确图像 → 自检通过 → 执行 `rm -rf` 清理 → 提交 → 验证器找不到输出文件
- **AHE 修复**：发布状态保护机制——prompt 规则（chg-7）+ shell 工具状态守卫（chg-8），阻止删除已验证输出
- **效果**：0/2 → 2/2

### 轨迹 3：mcmc-sampling-stan（迭代 6）
- **任务**：安装 rstan、拟合层次 Beta-Binomial 模型、输出后验均值
- **失败模式**：Agent 用网格积分作为代理验证器、后台启动 MCMC 但未等完成就杀掉、仅检查文件存在性
- **AHE 修复**：两个跨层组件协同——① 工具层：扩展发布状态守卫保护脚本入口点 ② 中间件：ExecutionRiskHintsMiddleware 监控 7 种跨步风险模式
- **效果**：连续 5 轮失败后 0/2 → 2/2

### 轨迹 4：configure-git-webserver（迭代 8）
- **任务**：配置 git 仓库 + web 服务器，验证外部 HTTP 请求
- **失败模式**：部署成功后 Agent 使用 ALLOW_POST_SUCCESS_RESET token 执行破坏性清理
- **AHE 修复**：① 硬化 shell 守卫——删除/重置保护输出变成硬阻塞 ② 中间件 FRAMEWORK 提醒——将风险提示提升到模型推理上下文中
- **效果**：0/2 → 2/2，该轮总分 76.97 为全运行最高

**案例研究揭示的演化模式**：AHE 的演化路径是 prompt → tool → middleware 的层级递进。首先在 prompt 添加通用规则（快速低成本），然后在工具层安装执行时守卫（硬执行保障），最后在中间件添加跨步监控（深层行为模式检测）。

## 自归因分析（附录 D）

- **修复预测**：精度 33.7%、召回 51.4%，约为随机基线的 **5 倍**
- **回归预测**：精度 11.8%、召回 11.1%，仅约随机基线的 **2 倍**
- 9 轮演化中 Agent 发出 43 个唯一回归预测，仅 5 个命中
- 40 个实际发生的回归被完全漏判
- **结论**：Agent 能可靠解释"为什么这个改动有用"，但无法可靠预测"这个改动会破坏什么"

## 局限性

- harness 组件交互是非加性的，叠加有效编辑有上限
- 自归因对修复可靠但对回退盲，回归预判是未来方向
- benchmark 范围有限（Terminal-Bench 2 + SWE-bench-verified），更广泛的编程语言和仓库级部署未测试
- 操作点耦合：超时和步数预算在 GPT-5.4 high 上拟合，跨推理层级增益非单调
- 自修改治理不完整，应视为受控研究原型而非成熟自主改进系统

## 元数据

- 作者: Jiahang Lin, Shichun Liu, Chengjun Pan, Lizhi Lin, Shihan Dou, Xuanjing Huang, Hang Yan, Zhenhua Han, Tao Gui
- 机构: 复旦大学、北京大学、上海奇绩智丰
- 日期: 2026-04-28
- arxiv: 2604.25850
