---
title: GPT-4 Technical Report
authors:
  - OpenAI
institutions: OpenAI
aliases:
  - GPT-4
  - GPT-4论文
  - GPT-4 Technical Report
date: '2026-04-30'
publish_date: 2023-03
venue: 'arXiv:2303.08774'
tags:
  - 论文
  - 大语言模型
  - 多模态
  - GPT-4
  - RLHF
  - Safety
url: 'https://arxiv.org/abs/2303.08774'
code: 未开源（通过 API 和 ChatGPT Plus 提供）
rating: ⭐⭐⭐⭐
status: Read
---
**一句话总结**：本文是 GPT-4 的技术报告，介绍了一个大规模多模态模型——接受图像和文本输入、生成文本输出，在多种专业和学术基准上达到人类水平表现。但报告刻意不公开架构细节、模型规模、训练方法等核心技术信息。

---

## Intro

### Motivation

OpenAI 的目标是推进深度学习在更大规模上的表现，并评估更大模型在能力、安全性、对齐性等维度的表现。GPT-4 是这一目标的产物，重点关注"可预测的缩放（predictable scaling）"——希望能在训练小模型时准确预测大模型的最终性能。

### 核心设计哲学

1. **Predictable Scaling**：用小规模模型的训练行为预测大模型的最终性能。这不仅能提前判断训练的可行性，也能为安全研究提供信息。
2. **安全和对齐**：即使模型强大，也必须安全、可控。GPT-4 投入了大量工作在对抗测试、安全管线和监控上。

### 贡献

1. **GPT-4**：一个多模态模型，在专业考试（律师资格考试前 10%）、学术 benchmark（MMLU 86.4%）上达到人类水平
2. **可预测缩放的成功验证**：训练前通过小模型预测了 GPT-4 的最终 loss 和 HumanEval 性能，预测准确
3. **视觉输入能力**
4. **系统性的安全评估和改进**：红队测试、模型辅助安全管线（RBRMs）
5. **开源 OpenAI Evals 评估框架**

但需要注意的是：这篇 technical report **故意不包含任何架构尺寸、硬件、训练方法、数据集构建、训练方法论等细节**。这是由于"竞争格局和安全 implications 的考虑"。

---

## Method 核心方法（基于报告中透露的部分）

### 1. 可预测缩放（Predictable Scaling）

OpenAI 使用小规模模型的训练曲线预测 GPT-4 的最终 loss 和能力。这包括：
- 预测 final loss：使用较小的模型（如 1/1000 或 1/10000 规模的模型）推断 full-scale 的 loss
- 预测 HumanEval pass rate：从更小模型的性能推断 GPT-4 的代码生成能力
- 预测结果与实际训练结果高度一致

### 2. 架构（极简透露）

- 输入的接受：**图像 + 文本**，输出：**仅文本**
- 基于 Transformer 架构（具体规模、层数、头数一律不公开）
- 接受"任意交错的文本和图像"作为输入

### 3. 训练方法

- 预训练：在大规模语料上训练（数据截止 2021 年 9 月）
- 后训练：RLHF（沿用 InstructGPT 的三步方法论：SFT → RM → PPO）
- 安全训练：额外的安全相关 RLHF 训练 prompt + Rule-Based Reward Models（RBRMs）

### 4. 安全机制

**Rule-Based Reward Models (RBRMs)**：
- 零样本 GPT-4 分类器，用于评估策略模型的输出
- 输入：prompt（可选）、策略模型输出、人工撰写的 scoring rubric
- 根据 rubric 分类：合规拒绝、不合规拒绝（evasive）、包含不允许内容、安全的非拒绝响应
- 为 PPO 训练提供额外奖励信号，精细控制模型行为

**安全训练结果**：
- 对不允许内容的响应率降低 82%（相比 GPT-3.5）
- 敏感请求的合规率提升 29%
- RealToxicityPrompts 上的毒性输出率从 6.48% 降至 0.73%

---

## 实验/评估/结果

### 学术基准（与专用 SOTA 系统对比）

| 基准 | GPT-4 | GPT-3.5 | SOTA（专用系统） |
| --- | --- | --- | --- |
| MMLU | **86.4%** | 70.0% |  |
| ARC (Challenge) | **96.3%** | 85.2% | 86.5% |
| WinoGrande | **87.5%** | 81.6% | 85.1% |
| HumanEval | **67.0%** | 48.1% | 65.8% |
| GSM-8K | **92.0%*** | 57.1% | 87.3% |
| DROP (F1) | 80.9 | 64.1 | **88.4%** |

*GSM-8K 使用了 chain-of-thought prompting，且部分训练数据可能包含 GSM-8K 训练集。

GPT-4 在除 DROP 外的所有 benchmark 上超越专用 SOTA 系统。

### 多语言能力

在 24 种语言的 MMLU 翻译版本上，GPT-4 在大多数语言（包括低资源语言如拉脱维亚语、威尔士语、斯瓦希里语）上超越了 GPT-3.5 的**英语**表现，展示了强大的跨语言泛化。

### 人类偏好评估

在 5,214 个 ChatGPT API prompt 上，GPT-4 的回复在 70.2% 的情况下被偏好于 GPT-3.5。

### 视觉输入能力

GPT-4 能理解并推理图像内容（文档、图表、截图等），few-shot prompting 和 chain-of-thought 等技术同样适用于视觉输入。

### 事实性（Hallucination 改进）

- 内部对抗设计的事实性评估：GPT-4 高于最新 GPT-3.5 约 19 个百分点
- TruthfulQA：GPT-4 base 仅略好于 GPT-3.5，但 RLHF 后大幅改善
- 显式比较中能看到 GPT-4 对常见谬误（"you can't teach an old dog new tricks"）的抵制力更强

---

## 结论

GPT-4 是一个在多种专业和学术 benchmark 上达到人类水平的大规模多模态模型，其能力可以通过可预测缩放提前估算。OpenAI 投入了大量工作在安全性上，包括对抗测试、RBRMs 辅助的 RLHF、安全监控管线等。但由于竞争和安全考虑，该报告未公开核心技术细节。

---

## 思考

### 优点

1. **Predictable Scaling 的价值**：能在训练前准确预测大模型性能，对资源配置、时间规划和安全评估都有重要价值。这个"可预测性"本身是 scaling 研究的核心发现之一。

2. **安全投入的透明度（相对而言）**：在 GPT-4 之前，很少有模型报告对安全做如此系统的评估。RBRMs 的概念（用模型评估模型的安全性）很值得关注。

3. **多模态能力的 demo**：虽然视觉能力只占报告的一小部分，但"接受任意交错的文本和图像"这一描述暗示了信息融合能力的重大进步。

4. **多语言泛化的惊喜**：低资源语言的 MMLU 超过 GPT-3.5 的英语水平，说明大规模预训练产生的跨语言迁移能力超乎预期。

### 缺点与局限

1. **不是一篇真正的技术论文**：OpenAI 自己称之为"technical report"而非 research paper。架构、模型大小、训练数据、硬件、计算量等核心细节全部缺失。这严重限制了学术社区的复现和研究能力。

2. **安全性讨论的不足**：尽管报告讨论了红队测试，但对 jailbreak、系统 prompt 对抗等实用性安全问题触及很少。报告中提到"jailbreak 仍然存在"但没有给出解决方案。

3. **RLHF 后的校准退化**：报告承认预训练模型的校准（calibration）很好，但 RLHF 后校准明显下降。这意味着后训练可能引入新的幻觉模式，但报告没有深入分析。

4. **能力评估的选择性**：报告主要报了 GPT-4 强的 benchmark，对弱的领域（如某些推理任务）着墨较少。这符合技术报告的性质，但不够全面。

5. **不可复现性**：整篇报告的"贡献"不可被学术复现——没有模型权重、没有训练代码、没有数据。它本质上是一份产品发布 docs，不是传统意义上的学术论文。

### 与已有 Wiki 的连接

- 关联概念：[[Predictable Scaling]]、[[RLHF]]、[[RBRM (Rule-Based Reward Model)]]、[[多模态模型]]
- 关联论文：[[AI阅读笔记/InstructGPT]]（GPT-4 的 RLHF 方法论前身）、[[AI阅读笔记/GPT-3]]（GPT-4 的直接前代）、[[AI阅读笔记/GPT-4o System Card]]（GPT-4 的多模态后续）
- 关联实体：[[GPT-4]]、[[ChatGPT]]
- 关联问题：[[Wiki/Questions/OpenAI 不公开 GPT-4 技术细节对研究社区的影响]]
