---
title: "DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence"
authors:
  - DeepSeek-AI
institutions: DeepSeek-AI
aliases:
  - DeepSeek-Coder-V2
  - DS-Coder-V2
date: '2026-04-30'
publish_date: 2024-06
venue: 'arXiv:2406.11931'
tags:
  - 论文
  - 代码模型
  - MoE
  - 开源
  - Code Intelligence
url: 'https://arxiv.org/abs/2406.11931'
code: https://github.com/deepseek-ai/DeepSeek-Coder-V2
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：DeepSeek-Coder-V2 是基于 DeepSeek-V2 继续预训练 6T tokens（代码 60% + 数学 10% + 自然语言 30%）得到的开源代码 MoE 模型，支持 338 种编程语言和 128K 上下文，在代码和数学 benchmark 上首次让开源模型达到与 GPT-4 Turbo 同等的性能水平。

![](https://arxiv.org/html/2406.11931v1/x1.png)

*Figure 1: DeepSeek-Coder-V2 在数学和代码 benchmark 上的性能。*

---

## Intro

### Motivation

开源代码模型（StarCoder、CodeLlama、DeepSeek-Coder 等）虽然取得了显著进展，但与 GPT-4 Turbo、Claude 3 Opus、Gemini 1.5 Pro 等闭源模型之间仍存在差距。DeepSeek-Coder-V2 旨在打破这个壁垒。

### 贡献

1. **首个开源千亿参数代码模型**（236B 总参数/21B 激活），在代码和数学上超越 GPT-4 Turbo 等闭源模型
2. **轻量版本 16B/2.4B 激活**，支持 FIM（Fill-In-Middle）代码补全
3. 支持 **338 种编程语言**（vs DeepSeek-Coder 的 86 种），上下文长度从 16K 扩展到 **128K**
4. 采用 **GRPO 强化学习** 对齐，使用编译器反馈 + 奖励模型 + 测试用例

---

## Method 核心方法

### 1. 训练策略

- **继续预训练**：从 DeepSeek-V2 的中间 checkpoint（已训 4.2T tokens）开始，继续训练 6T tokens
- 总训练量：10.2T tokens
- 数据配比：代码 60% + 数学 10% + 自然语言 30%
- 16B 版本使用 FIM（Fill-In-Middle）训练（PSM 模式，FIM 率 0.5），236B 仅用 Next-Token-Prediction
- 优化器：AdamW（β1=0.9, β2=0.95, weight_decay=0.1），余弦衰减至初始值的 10%

### 2. 数据构建

**代码数据（1,170B tokens）**：
- GitHub（2023 年 11 月前公开仓库）：严格过滤 + 近似去重后得 821B 代码 + 185B 代码相关文字
- Common Crawl：用 fastText 多轮召回代码论坛、文档网站等，收集 70B tokens
- 再次用相同 pipeline 从 GitHub 召回高质量代码 94B

**数学数据（221B tokens）**：
- 用 DeepSeekMath 的 pipeline 从 Common Crawl 收集

**消融验证**（1B 模型）：新代码语料在 HumanEval 上提升 6.7%，MBPP 上提升 9.4%。

### 3. 长上下文扩展

用 YaRN 扩展到 128K（scale s=40, α=1, β=32），两阶段继续训练（32K + 128K）。

### 4. 对齐

- SFT：30 万条指令（代码 2 万 + 数学 3 万 + 通用数据），共 300M tokens
- RL：GRPO 算法。**关键发现**：用奖励模型信号优于用原始编译器信号。编译器只能给 0/1 反馈且测试用例覆盖不全，训练一个奖励模型更鲁棒。

![](https://arxiv.org/html/2406.11931/x3.png)

*Figure 3: RL 训练中奖励模型信号 vs 原始编译器信号的对比。奖励模型提供更鲁棒的训练信号，在 LeetCode 和 LeetCode-zh 上均优于编译器 0/1 反馈。*

---

## 实验/评估/结果

### 代码生成（HumanEval + MBPP）

- DeepSeek-Coder-V2 236B：**平均 75.3%**，仅次于 GPT-4o 的 76.4%，在 Java 和 PHP 上最高
- DeepSeek-Coder-V2-Lite 16B：65.6%，超过 33B 的 DeepSeek-Coder（61.9%）

### 竞赛编程（LiveCodeBench + USACO）

- LiveCodeBench 整体 43.4%，与 GPT-4o 持平，仅次于 GPT-4-Turbo（45.7%）
- USACO 12.1%

### 代码补全（FIM）

- DeepSeek-Coder-V2-Lite（2.4B 激活）：FIM 任务平均 86.4%，与 DeepSeek-Coder 33B（86.4%）持平

### 代码修复

- Aider：**73.7%，超越所有模型（包括闭源）**
- SWE-Bench：12.7%
- Defects4J：21.0%

### 数学推理

- MATH：**75.7%**，与 GPT-4o 持平
- AIME 2024：4/30（maj@64 可达 5/30）
- GSM8K：94.9%

### 通用语言能力

- Arena-Hard：**65.0**，远高于 DeepSeek-V2 Chat（41.6），代码/数学/推理训练提升了通用推理能力
- MMLU：79.2，BBH：83.9
- 知识密集型任务（TriviaQA 等）略逊于 DeepSeek-V2 Chat

---

## 结论

DeepSeek-Coder-V2 证明了：在强大的通用基座模型上进行大规模的代码/数学继续预训练，可以让开源模型在代码智能上达到甚至超越闭源模型的水平。但论文也坦承在指令遵循能力上与 GPT-4 Turbo 仍有差距，后续需要加强。

---

## 思考

### 优点

1. **开源里程碑意义重大**：首个开源模型在代码/数学上达到 GPT-4 Turbo 级别的性能
2. **继续预训练策略高效**：基于 DeepSeek-V2 的中间 checkpoint 做 domain-specific 继续训练，比从头训练经济得多
3. **奖励模型优于编译器信号的发现**：对 RLHF 实践有指导意义
4. **语言范围扩展（86→338 种）**：大幅提升了实用性
5. **FIM + 仓库级补全**：16B 版本的 FIM 能力使得轻量级本地代码补全成为可能

### 缺点

1. **SWE-Bench 表现较弱**（12.7%）：论文自己也指出指令遵循能力不足是主要原因
2. **复杂修复任务仍是瓶颈**：Defects4J 上 21% vs GPT-4o 的 26%，仍有差距
3. **知识密集型任务有所下降**：代码/数学训练对通用知识任务有一定负面影响
4. **236B 模型的 FIM 缺失**：最大模型不支持 FIM，限制了在仓库级补全场景的应用
5. **奖励模型训练的偏好数据细节不足**：编译器和测试用例生成的具体方法描述较少
