---
type: source
status: processed
site: arXiv
source_file: '[[Clippings/Language Models are Few-Shot Learners.md]]'
title: Language Models are Few-Shot Learners
author: Tom B. Brown et al. (OpenAI)
published: '2020'
processed: '2026-04-30'
tags:
  - GPT-3
  - few-shot
  - in-context-learning
  - scaling
  - OpenAI
---
# Language Models are Few-Shot Learners (GPT-3)

## 核心结论

- 提出 **GPT-3**，175B 参数的自回归语言模型，证明了扩大语言模型规模可以极大提升 task-agnostic 的 few-shot 性能，部分任务达到甚至超越此前微调方法的 SOTA。
- 无需梯度更新或微调，仅通过上下文中的少量示例（few-shot）即可完成新任务，展示了 **in-context learning** 范式的威力。
- 8 个不同大小的模型（125M 到 175B），系统和可预测的性能随规模提升，验证了 [[Wiki/Concepts/Scaling Laws|Scaling Laws]] 的早期观察。

## 关键方法或创新点

- **Few-shot / One-shot / Zero-shot 评估**：不更新模型参数，仅在 prompt 中提供示例，模型直接推理。
- 训练数据：Common Crawl（过滤后）、WebText2、Books、Wikipedia 等混合语料，约 300B token。
- 模型架构与 GPT-2 基本相同（Transformer Decoder），主要改进：密集和稀疏注意力模式的交替使用。
- 在 TriviaQA、LAMBADA 等任务上达到新的 few-shot SOTA；在部分翻译任务上与监督模型竞争；在 SuperGLUE 上与微调模型接近。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Concepts/GPT 系列模型]]、[[Wiki/Concepts/Scaling Laws]]、[[Wiki/Topics/大语言模型基础]]
- GPT-3 的 in-context learning 能力直接激发了后续指令微调研究（[[Wiki/Sources/InstructGPT]]、ChatGPT）。
- 证明了「涌现能力」的存在：小模型不会的任务，175B 模型通过 few-shot 能做出来。

## 局限或注意事项

- 生成内容存在事实错误、社会偏见、有毒语言等问题。
- few-shot 性能仍然弱于针对特定任务的微调模型（尤其是在自然语言推理等需要深层推理的任务上）。
- 训练成本极高（估计数百万美元），推理成本也很大。
- 文中未提供开源权重，限制了复现性。
