---
title: "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models"
authors:
  - DeepSeek-AI
institutions: DeepSeek-AI
aliases:
  - DeepSeek-V3.2
  - DSA
date: '2026-04-30'
publish_date: 2025-12
venue: 'arXiv:2512.02556'
tags:
  - 论文
  - LLM
  - 稀疏注意力
  - RL Scaling
  - Agent
  - 推理
url: 'https://arxiv.org/abs/2512.02556'
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：DeepSeek-V3.2 通过 DSA（DeepSeek Sparse Attention）大幅降低长上下文计算复杂度、扩大 RL 后训练计算预算（超过预训练成本的 10%）、以及大规模 agent 任务合成 pipeline，实现了与 GPT-5 可比的推理性能；其 Speciale 变体在 IMO 2025 和 IOI 2025 中达到金牌水平。

![](https://arxiv.org/html/2512.02556v1/x1.png)

*Figure 1: DeepSeek-V3.2 与各模型的 benchmark 对比。*

---

## Intro

### Motivation

推理模型（o1、DeepSeek-R1）发布后，开源与闭源模型之间的差距非但没有缩小，反而在加速扩大。论文识别出三个关键不足：
1. **架构瓶颈**：传统 vanilla attention 在长序列上的 $O(L^2)$ 复杂度严重限制部署和后训练效率
2. **后训练计算投入不足**：开源模型在 post-training 阶段投入的计算资源远少于闭源模型
3. **Agent 泛化能力弱**：开源模型在 tool-use 场景中的泛化和指令遵循能力明显不足

### 贡献

1. **DSA（DeepSeek Sparse Attention）**：结合 lightning indexer 和细粒度 token 选择的高效稀疏注意力，在保持性能的同时大幅降低长上下文计算复杂度
2. **可扩展 RL 框架**：通过 unbiased KL estimate、off-policy sequence masking、keep routing 和 keep sampling mask 等技术稳定扩展 GRPO，后训练计算预算超过预训练成本的 10%
3. **大规模 agent 任务合成 pipeline**：生成 1,800+ 环境和 85,000+ 复杂 prompts，实现 agent 能力的规模化后训练
4. **Speciale 变体**：IMO 2025 和 IOI 2025 金牌级表现

---

## Method 核心方法

### 1. DSA（DeepSeek Sparse Attention）

**架构**：
- **Lightning Indexer**：轻量级打分模块，计算 query token 与之前每个 token 的 index score $I_{t,s}$，使用 FP8 实现极低计算开销
- **细粒度 token 选择**：对每个 query，只保留 top-k index score 对应的 key-value entries 参与注意力计算
- **基于 MLA 的实例化**：使用 MLA 的 MQA 模式，每个 latent vector 被所有 query heads 共享

![](https://arxiv.org/html/2512.02556/x2.png)

*Figure 2: 基于 MLA 实例化的 DSA 架构。绿色部分展示 DSA 如何通过 indexer 选择 top-k key-value entries。*

**继续预训练**：
- **Dense Warm-up 阶段**：冻结所有参数，仅训练 indexer（KL 散度 loss），1000 steps / 16 条 128K 序列
- **Sparse Training 阶段**：全参数训练，选择 top-2048 KV tokens，15000 steps / 480 条 128K 序列（943.7B tokens）

**效果**：
- 核心注意力复杂度从 $O(L^2)$ 降至 $O(Lk)$
- 短/长上下文任务上性能与 DeepSeek-V3.1-Terminus 持平
- 显著降低长上下文推理成本

### 2. 可扩展 RL（GRPO 改进）

**Unbiased KL Estimate**：纠正 K3 estimator 的偏差，使用 importance-sampling 得到无偏 KL 估计

**Off-Policy Sequence Masking**：当 rollout 数据与当前策略的 KL 散度过大时，mask 掉负优势序列，避免高偏差样本误导训练

**Keep Routing**：在训练时保持推理时的 expert routing path，避免 MoE 中训练-推理路由不一致导致的不稳定

**Keep Sampling Mask**：保留采样时的 top-p/top-k truncation mask，确保训练时策略与采样时策略共享相同的 action subspace

### 3. Thinking in Tool-Use

**思考上下文管理**：
- 历史推理内容仅在出现新的 user message 时才丢弃
- 仅 tool 相关消息追加时保留推理内容
- 丢弃推理时仍保留 tool 调用历史

**Cold-Start + 大规模 Agent 任务合成**：
- 通过精心设计的 system prompt 将推理（CoT）和工具调用统一在单条轨迹中
- 4 类 agent 任务：

  | 任务 | 数量 | 环境 | prompt |
  |------|------|------|--------|
  | Code Agent | 24,667 | 真实 | 提取 |
  | Search Agent | 50,275 | 真实 | 合成 |
  | General Agent | 4,417 | 合成 | 合成 |
  | Code Interpreter | 5,908 | 真实 | 提取 |

---

## 实验/评估/结果

### 主结果

- 推理：与 GPT-5-high 可比（AIME 2025: 93.1 vs 94.6, HMMT Feb: 92.5 vs 88.3）
- Agent：SWE-bench Verified 73.1, SWE Multilingual 70.2（开源最强）
- 搜索 Agent：BrowseComp 67.6（+context management），超过 GPT-5
- Tool-Use：$\tau^2$-bench 80.3, MCP-Universe 45.9
- 与 Kimi-K2-Thinking 比，用更少输出 tokens 达到同等分数

### Speciale 变体

| 竞赛 | 成绩 | 奖牌 |
|------|------|------|
| IMO 2025 | 35/42 | 金牌 |
| CMO 2025 | 102/126 | 金牌 |
| IOI 2025 | 492/600 | 金牌 |
| ICPC WF 2025 | 10/12 题 | 金牌 |

但 token 效率显著低于 Gemini-3.0-Pro。

### 合成 Agent 数据的泛化性

- 仅用合成 agent 数据做 RL，在 Tau2Bench、MCP-Mark、MCP-Universe 上均有大幅提升
- 仅用代码和搜索场景的 RL 无法提升这些 benchmark

---

## 结论

DeepSeek-V3.2 通过 DSA 解决了长上下文效率瓶颈，通过扩大 RL 计算预算使推理能力比肩 GPT-5，通过合成 agent 数据大幅提升了工具使用能力。但世界知识广度、token 效率和复杂任务解决能力仍落后于 Gemini-3.0-Pro。

---

## 思考

### 优点

1. **DSA 架构设计实用且完整**：从 indexer warmup 到 sparse training，再到 MLA 实例化，工程实现考虑周全
2. **GRPO 改进的系统性**：4 项改进（unbiased KL、off-policy masking、keep routing、keep sampling mask）层层递进，解决实际训练中的问题
3. **Agent 任务合成 pipeline 是重要方法论贡献**：自动合成 1,800+ 环境 + 验证器，可规模化扩展
4. **Speciale 的竞赛成绩是开源模型的里程碑**
5. **诚实的局限性陈述**：明确提出世界知识、token 效率、复杂任务三方面的不足

### 缺点

1. **DSA 仍然需要 $O(L^2)$ 的 indexer**：虽然 indexer 轻量，但理论上没有完全解决线性复杂度
2. **Speciale 的 token 效率太差**：如 AIME 上 23K tokens vs Gemini 3.0 Pro 的 15K，实用性受限
3. **合成 agent 数据的质量依赖性**：训练数据由 DeepSeek-V3.2 自身生成，存在"自我反馈循环"的风险
4. **与 Gemini-3.0-Pro 的差距仍然明显**：HLE 25.1 vs 37.7，GPQA 82.4 vs 91.9
5. **搜索 Agent 的 context management 方法较粗暴**：discard-all / discard-75% 等策略会丢失信息
