---
aliases:
  - Qwen3 模型
confidence: high
created: '2026-04-30'
sources:
  - '[[Wiki/Sources/Qwen3 技术报告]]'
status: active
tags:
  - Qwen
  - LLM
  - MoE
  - 推理模型
  - 蒸馏
type: entity
updated: '2026-04-30'
---
# Qwen3

## 简介

Qwen3 是阿里通义千问团队发布的第三代大语言模型系列，覆盖 Dense 和 MoE 双架构。核心特色是统一思考/非思考模式、thinking budget 推理预算机制、以及 strong-to-weak distillation 蒸馏策略。

## 模型规格

### Dense 系列

- 0.6B, 1.7B, 4B, 8B, 14B, 32B

### MoE 系列

- Qwen3-30B-A3B：30B 总参数，3B 激活参数
- Qwen3-235B-A22B：235B 总参数，22B 激活参数

### 训练数据

- 36T tokens，覆盖 119 种语言

## 核心创新

### 统一思考/非思考模式

同一模型支持两种模式切换：
- **`/think` 模式**：开启深度推理，模型生成详细的思维链，适用于数学、代码等复杂推理任务。
- **`/no_think` 模式**：关闭深度推理，直接生成简洁回答，适用于简单问答、闲聊等。

这是对 DeepSeek-R1（纯推理专用模型）路线的一种不同选择——一个模型同时服务两种场景，降低维护成本。

### Thinking Budget（推理预算）

用户可以指定推理 tokens 的最大数量，实现推理深度和响应速度之间的细粒度控制。例如限制 thinking budget 为 500 tokens，模型会在推理长度和准确率之间寻找平衡。

### 四阶段后训练管线

1. **Long-CoT 冷启动 SFT**：使用高质量的思维链数据进行监督微调。
2. **推理 RL**：在推理任务上进行强化学习。
3. **Thinking mode fusion**：将思考和非思考两种模式融合到同一模型中。
4. **通用 RL**：在通用任务（写作、对话等）上进行对齐 RL。

### Strong-to-Weak Distillation

用大模型（strong teacher）蒸馏小模型（weak student），将推理能力传递给小参数模型。这不同于 Kimi k1.5 的 long2short（推理链压缩），而是直接在模型间传递能力。

## 与同类模型的对比

| 维度 | Qwen3 | DeepSeek-R1 | Kimi k1.5 |
|------|-------|-------------|-----------|
| 推理模式 | 统一双模式（/think /no_think） | 专用推理模型 | Long2short 蒸馏 |
| 推理预算控制 | Thinking budget | 无类似机制 | Length penalty |
| 模型系列 | Dense + MoE | Dense（蒸馏）+ MoE（V3） | MoE |
| 蒸馏策略 | Strong-to-weak | R1 蒸馏 6 个小模型 | Long2short 四种方法 |
| 训练规模 | 36T tokens | 14.8T (V3) | 15.5T (K2) |

## 设计哲学

1. **一模型多用**：统一思考模式避免了维护独立的推理模型和普通模型。
2. **用户可控**：thinking budget 赋予用户对推理深度的控制权。
3. **全面覆盖**：从 0.6B 到 235B 的广泛参数范围，Dense + MoE 双架构。
4. **实用导向**：四阶段管线保证了模型在推理和通用任务上的双优表现。

## 批判性评估

### 最大优势：实用主义的产品思维

Qwen3 在三家中最不"炫技"，但最"好用"。thinking budget 是一个典型的产品创新——它不追求推理能力的极致，而是让用户能控制"要多少推理深度"。在推理成本是实际部署的核心约束时，这个功能的价值可能比"多 2 个点的准确率"更大。Dense+MoE 双线让 Qwen3 覆盖了从边缘设备到云端的所有部署场景——没有人会因为"模型太大装不下"而放弃 Qwen。

### 原创性的缺失是否致命？

Qwen3 的几乎所有核心概念都已经被提出过：CoT（Google 2022）、RL 推理（DeepSeek 2025）、MoE（Mixtral 2024）、蒸馏（Hinton 2015）。thinking budget 是 Qwen3 少数真正原创的概念之一。问题是：在 AI 领域，"改良者"能活多久？

答案是：商业化能做很久。安卓不是第一个智能手机操作系统，Windows 不是第一个 GUI 操作系统。Qwen3 如果能持续在"工程实用"上领先（部署效率、多语言覆盖、阿里云集成），它可以成为"最广泛使用的大模型"而不需要是"最具原创性的大模型"。

### Thinking budget 的隐含风险

Thinking budget 允许用户限制推理 token 数。但如果 budget 设得太低，模型可能生成不完整的推理链导致错误答案——而用户不知道这个错误是"推理不足"造成的。这引入了一个新的故障模式：用户可能错误地将"推理预算不足导致的错误"归因为"模型能力不行"。Qwen 的文档需要在这一点上做教育。

### Strong-to-weak distillation 与 R1 蒸馏的微妙差异

Qwen3 的 strong-to-weak 蒸馏和 R1 的蒸馏都得出"蒸馏 > 直接 RL"的结论。但它们的"teacher"不同：R1 的 teacher 是经过推理 RL 训练的专用推理模型，Qwen3 的 teacher 可能是一个更大的通用模型（细节未公开）。如果 Qwen3 的 teacher 不是专门的推理模型，那 strong-to-weak 蒸馏传递的"推理能力"来源是什么？这需要更透明的技术细节才能判断。

### Qwen3 在 Agent 上的战略缺失

与 DeepSeek V3.2 和 Kimi K2/K2.5 相比，Qwen3 几乎没有讨论 Agent 能力。在 2025 年这是一个显著的战略缺口。可能的解释：(1) Agent 能力计划在 Qwen4 中推出；(2) 阿里云的 Agent 能力通过 API 层（如 function calling）提供，而非模型原生；(3) 团队认为当前的 Agent 热度是炒作，选择了更务实的路线。无论是哪种，如果 Agent 真的是下一代范式（而不仅仅是当前的炒作），Qwen3 可能正在错过窗口。

## 待核实问题

- Thinking budget 在复杂推理任务中的实际效果（是否存在预算不足导致推理失败的情况）。
- MoE 版本的辅助损失或负载均衡策略与 DeepSeek/Kimi 的差异。
- Qwen3 是否已有对应的 VLM 版本。
