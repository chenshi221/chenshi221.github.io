---
type: concept
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - GPT
  - GPT系列
  - GPT模型
  - Generative Pre-trained Transformer
sources:
  - '[[Wiki/Sources/GPT-3]]'
  - '[[Wiki/Sources/InstructGPT]]'
  - '[[Wiki/Sources/GPT-4]]'
  - '[[Wiki/Sources/GPT-4o]]'
tags:
  - GPT
  - OpenAI
  - LLM
  - generation
confidence: high
---
# GPT 系列模型

## 定义

GPT（Generative Pre-trained Transformer）是 OpenAI 开发的一系列自回归语言模型，基于 Transformer Decoder 架构。从 GPT-1 到 GPT-4o，该系列模型从纯文本语言模型演变为全模态 AI 系统，是现代 LLM 能力发展的缩影。

## 各代模型概览

### GPT-1 (2018)
- 「Improving Language Understanding by Generative Pre-Training」
- 首次提出 GPT 范式：在大规模无标注文本上做生成式预训练，在下游任务上微调
- 12 层 Transformer Decoder，117M 参数
- 核心贡献：建立「预训练 + 微调」范式的解码器路径

### GPT-2 (2019)
- 「Language Models are Unsupervised Multitask Learners」
- 1.5B 参数，WebText 数据集
- 首次展示 zero-shot 迁移能力（无需微调即可完成部分任务）
- 因「太危险」一度延迟发布（引发了 AI 安全讨论）

### GPT-3 (2020)
- [[Wiki/Sources/GPT-3|Language Models are Few-Shot Learners]]
- 175B 参数，约 300B token 训练数据
- **核心贡献**：系统性地证明了 **in-context learning**——通过上下文中的少量示例即可完成新任务，无需微调
- 8 个不同规模的模型（125M → 175B），验证了性能随规模提升的规律

### InstructGPT (2022)
- [[Wiki/Sources/InstructGPT|Training Language Models to Follow Instructions with Human Feedback]]
- 基于 GPT-3，引入 [[Wiki/Concepts/RLHF|RLHF]] 三阶段对齐流程
- 核心贡献：确立了「对齐」在 LLM 开发中的必要性和标准方法
- 1.3B InstructGPT 的人类偏好评分优于 175B 原始 GPT-3

### GPT-4 (2023)
- [[Wiki/Sources/GPT-4|GPT-4 Technical Report]]
- **多模态输入**（文本+图像），文本输出
- 专业考试 human-level 性能（律师资格考试前 10%）
- predictable scaling 基础架构（小模型预测大模型性能）
- 技术细节高度保密（架构、参数、数据均未公开）

### GPT-4o (2024)
- [[Wiki/Sources/GPT-4o|GPT-4o System Card]]
- **全模态**：文本、音频、图像、视频输入和输出，端到端统一训练
- 语音响应延迟 232ms（接近人类对话速度）
- 视觉和音频理解大幅超越前代

## 关键能力演进

| 代际 | 纯文本 | In-Context Learning | 指令遵循 | 多模态输入 | 全模态 | 端到端多模态 |
|------|--------|-------------------|----------|-----------|--------|------------|
| GPT-1 | Yes | No | No | No | No | No |
| GPT-2 | Yes | Zero-shot | No | No | No | No |
| GPT-3 | Yes | **Few-shot** | No | No | No | No |
| InstructGPT | Yes | Yes | **Yes (RLHF)** | No | No | No |
| GPT-4 | Yes | Yes | Yes | **Yes** | No | No |
| GPT-4o | Yes | Yes | Yes | Yes | **Yes** | **Yes** |

## 架构共性

- 全部基于 **Transformer Decoder**（单向自注意力，自回归生成）
- Next-token prediction 作为预训练目标
- 从纯文本逐渐扩展到多模态（GPT-4 图像输入，GPT-4o 全模态）
- 对齐方法从无 → RLHF → 持续演进

## 与开源模型的对比

- [[Wiki/Sources/LLaMA|LLaMA 系列]]：开源路线的主要竞争者，Llama 3 405B 性能接近 GPT-4
- 开源生态（Mistral、Qwen、DeepSeek 等）：快速追赶，缩小能力差距

## 相关页面

- [[Wiki/Topics/大语言模型基础|大语言模型基础]]
- [[Wiki/Entities/Transformer|Transformer]]
- [[Wiki/Concepts/RLHF|RLHF]]
- [[Wiki/Concepts/Scaling Laws|Scaling Laws]]
