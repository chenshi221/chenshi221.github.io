---
title: "Competitive Programming with Large Reasoning Models"
authors:
  - OpenAI (Ahmed El-Kishky, Alexander Wei, Borys Minaiev, Daniel Selsam, Ignasi Clavera, Jakub Pachocki, Jerry Tworek, Lukasz Kaiser, Mark Chen, Nat McAleese, 等)
institutions: OpenAI
aliases:
  - o1-ioi
  - o3 competitive programming
  - OpenAI竞赛编程
date: 2026-04-30
publish_date: 2025-02
venue: 'arXiv:2502.06807'
tags:
  - 论文
  - 推理
  - 竞赛编程
  - o1
  - o3
  - CodeForces
  - IOI
  - RL
url: 'https://arxiv.org/abs/2502.06807'
code: ''
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：本文展示了 OpenAI o 系列推理模型的竞赛编程能力演进路径——从 o1（CodeForces 1673/89th percentile），到 o1-ioi（手写测试时策略 + RL 微调，IOI 2024 现场 49th percentile/金 medal 门槛），最终到 o3（纯 RL、无手写策略，IOI 金牌、CodeForces 2724/99.8th percentile），证明通用 RL 扩展优于领域特定策略。

---

## Intro

### Motivation

竞争性编程被认为是评估 AI 推理与编码能力的理想测试场：问题复杂、客观可评分、需要高级算法思维。从 Codex 到 AlphaCode/AlphaCode2，AI 编程已取得长足进展。但一个关键问题悬而未决：**领域特定的手工推理策略 vs 通过 RL 训练让模型自主学会推理**，哪种路线更优？

OpenAI 有三个可对比的系统来回答这个问题：o1（通用 RL 推理模型）、o1-ioi（针对 IOI 微调 + 手写测试时策略）、o3（更大规模 RL 训练，无手写策略）。

![](https://arxiv.org/html/2502.06807/x1.png)

*Figure 1: o1-preview 和 o1 与 gpt-4o 在 CodeForces 上的对比。o1 达到 1673 rating（89th percentile），远超 gpt-4o 的 808。*

### 贡献

1. 系统展示了从 o1→o1-ioi→o3 的竞争性编程能力演进曲线
2. 证明了手写测试时策略可以提升性能（o1-ioi CodeForces 2214），但被纯 RL 扩展超越（o3 CodeForces 2724）
3. o3 自主涌现出了复杂测试时策略（如写暴力解验证优化解），无需人工设计
4. 在 IOI 2024 和 SWE-bench Verified 上验证了推理能力从竞赛到真实软件工程的泛化

---

## Method 核心方法

### OpenAI o1

o1 是第一个 large reasoning model，通过 RL 训练让模型在进行 chain-of-thought 推理后再回答。RL 精炼了这个思维链过程：识别和纠正错误、分解复杂任务、在方法失败时探索替代路径。

CodeForces 评估结果：
- gpt-4o: 808（11th percentile）
- o1-preview: 1258（62nd percentile）
- **o1: 1673（89th percentile）**

### o1-ioi：领域特定路线

**Coding RL Fine-tuning**：在 o1 checkpoint 基础上继续 RL，强调挑战性编程问题（C++ 生成和运行时检查）。

**测试时策略**（手写 pipeline）：
- 将 IOI 问题按子任务拆分
- 每个子任务采样 10,000 个解答
- 聚类：基于模型生成的 256 个随机测试输入的输出进行聚类
- 重排序：使用学到的打分函数 + 公测失败惩罚 + 模型生成测试的错误
- 提交策略：从最难子任务开始轮询提交，最多 50 次

**IOI 2024 现场结果**：
- 50 次提交限制：213 分（49th percentile）
- 放宽到 10,000 次提交：362.14 分（超越金牌线）
- 随机选 50 次提交平均仅 156 分 → 选择策略贡献约 60 分

CodeForces：
- 基础：1807（93rd percentile）
- + 公测过滤：2092（96th percentile）
- + 完整策略：**2214（98th percentile）**

### o3：通用 RL 扩展路线

o3 放弃了所有手写测试时策略，完全依赖 RL 训练让模型自主发展推理能力。

**关键发现**：o3 自主涌现出了与 o1-ioi 手写策略类似的测试时行为——写暴力解法验证优化解的输出，交叉检查确保正确性。这证明了 RL 训练可以替代复杂的手工 pipeline。

**IOI 2024 回溯评估**（仅 1K 样本/问题，选 test-time compute 最高的 50 个解答）：
- **395.64 分**——在 50 次提交限制下直接超越金牌线，无需人工策略

**CodeForces**：**2724（99.8th percentile）**，跻身全球前 200 名活跃参赛者

![](https://arxiv.org/html/2502.06807/x5.png)

*Figure 5: o3 在 CodeForces 上达到 2724 rating（99.8th percentile），纯 RL 训练路径超越 o1-ioi 的手写策略路线（2214）。*

---

## 实验/评估/结果

| 系统 | CodeForces Rating | CodeForces Percentile |
|------|-------------------|----------------------|
| gpt-4o | 808 | 11% |
| o1-preview | 1258 | 62% |
| o1 | 1673 | 89% |
| o1-ioi | 1807 | 93% |
| o1-ioi + 策略 | 2214 | 98% |
| **o3** | **2724** | **99.8%** |

| 系统 | IOI 2024 Score | 提交限制 | 策略 |
|------|---------------|---------|------|
| o1-ioi | 213 | 50 | 手写策略 |
| o1-ioi | 362.14 | 10,000 | 手写策略 |
| **o3** | **395.64** | **50** | **无手写策略** |

### 软件工程泛化

- **HackerRank Astra**（65 个工程项目）：o1 63.92% pass@1（vs gpt-4o ~53%）
- **SWE-bench Verified**（500 个真实 GitHub issue）：o1 相比 gpt-4o 提升 8.6%；**o3 相比 o1 提升 22.8%**

---

## 结论

通用 RL 扩展（o3）超越了领域特定策略（o1-ioi），证明了"训练模型学会推理"比"为模型设计推理策略"是更具扩展性的路线。o3 自主涌现出的测试时推理行为（如暴力验证）表明，足够的 RL 训练可以让模型内化复杂的推理策略。

---

## 思考

### 优点

1. **路线对比的清晰性**：论文通过 o1→o1-ioi→o3 三个系统的演进，清晰展示了"手写策略有用但天花板低"和"RL 扩展的上限更高"两条路线的对比。这是方法论层面的重要结论。

2. **涌现行为的实证**：o3 自主发展出写暴力解验证优化解的策略，这个行为恰好对应了 o1-ioi 中最核心的测试逻辑。这说明 RL 训练能覆盖手写策略的收益，且能发现人类设计者可能遗漏的优化。

3. **IOI 现场验证的含金量**：在真实 IOI 竞赛条件下参赛并获得 49th percentile，是对系统能力的严苛检验。IOI 题目不会出现在训练数据中（赛后公开的），有效排除了数据污染的可能性。

4. **从竞赛到工程的泛化**：SWE-bench 和 HackerRank Astra 的结果证明推理能力的提升不限于竞赛题目，能泛化到真实工程任务。

### 缺点与局限性

1. **技术细节严重缺失**：作为 OpenAI 的报告，论文没有披露 RL 算法、奖励设计、训练数据构成、模型架构等关键细节。这使得研究社区无法复现或深入分析。

2. **与 DeepSeek-R1、Kimi k1.5 的对比缺失**：论文发表于 2025 年 2 月，彼时 DeepSeek-R1 和 Kimi k1.5 已经发布。o3 的 CodeForces 2724 与 DeepSeek-R1 的 2029 有显著差距，但论文没有讨论各自方法论的异同。

3. **o3 的具体配置不明**："early checkpoint"意味着最终版 o3 可能更强也可能更弱，结论的稳健性存疑。32,768 token 的 max generation length 是否足够也未知。

4. **成本完全不透明**：o3 的推理成本（推理 token 数、GPU 时）完全没有披露。在实际应用中，2724 rating 如果对应每次提交数千 token 的推理链，性价比可能不如较小的专用模型。

5. **推理能力"涌现"的因果关系不清晰**：o3 的 inference-time 策略是"真正涌现"还是"隐式蒸馏了 o1-ioi 的行为数据"，论文没有给出分析。如果是 RL 训练中包含了 o1-ioi 生成的数据，那么"涌现"的说法需要更谨慎。
