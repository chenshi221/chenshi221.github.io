---
type: source
status: processed
site: arXiv
source_file: '[[Clippings/GPT-4 Technical Report.md]]'
title: GPT-4 Technical Report
author: OpenAI
published: '2023'
processed: '2026-04-30'
tags:
  - GPT-4
  - multimodal
  - scaling
  - safety
  - OpenAI
---
# GPT-4 Technical Report

## 核心结论

- 报告了 **GPT-4**，一个大规模多模态模型（图像+文本输入，文本输出）。虽然比不上人类的真实世界能力，但在许多专业和学术 benchmark 上表现出人类水平性能（如模拟律师资格考试前 10%）。
- 一个核心创新是开发了可预测地跨多种规模运行的基础设施和优化方法，使得在仅使用 GPT-4 1/1000 计算量的小模型上就能准确预测 GPT-4 的某些性能指标。
- 后训练对齐过程显著改善了模型的事实性和行为合规性。

## 关键方法或创新点

- **多模态输入**：支持图像和文本联合作为输入（文本输出），是 GPT 系列首次多模态。
- **predictable scaling**：能够用小规模模型预测大规模模型性能，大幅降低实验成本。
- **后训练对齐**：通过 RLHF 等方式使模型更安全、更符合用户意图（与 [[Wiki/Sources/InstructGPT|InstructGPT]] 一脉相承）。
- 在各种考试 benchmark（AP、SAT、GRE、律师资格考试等）上表现出色。
- 对对抗性和安全性进行了系统评估，包括引入外部红队测试。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Concepts/GPT 系列模型]]、[[Wiki/Concepts/RLHF]]、[[Wiki/Topics/大语言模型基础]]
- GPT-4 的 multimodal 能力是 GPT 系列从纯文本到多模态的转折点。
- 后训练基础设施为 [[Wiki/Sources/GPT-4o|GPT-4o]] 的全模态模型奠定了基础。

## 局限或注意事项

- 技术报告**故意不包含架构细节**（模型大小、训练数据、硬件配置等均未公开），理由是安全考虑和竞争格局。
- 存在与早期 GPT 模型类似的幻觉、偏见和推理错误。
- 上下文长度为 8K 和 32K 版本（GPT-4o 后续大幅扩展了上下文窗口）。
