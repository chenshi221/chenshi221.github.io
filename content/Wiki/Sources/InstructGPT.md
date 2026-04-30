---
type: source
status: processed
site: arXiv
source_file: >-
  [[Clippings/Training language models to follow instructions with human
  feedback.md]]
title: Training Language Models to Follow Instructions with Human Feedback
author: Long Ouyang et al. (OpenAI)
published: '2022'
processed: '2026-04-30'
tags:
  - InstructGPT
  - RLHF
  - alignment
  - instruction-following
  - OpenAI
---
# Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

## 核心结论

- 更大的语言模型并不会自动更好地遵循用户意图；可能输出不真实、有害或无帮助的内容——即模型未与用户 **对齐（aligned）**。
- 提出了使用 **RLHF（Reinforcement Learning from Human Feedback）** 的方法来对齐 GPT-3，得到了 InstructGPT。
- 标注者明显偏好 InstructGPT 的输出（相较于 GPT-3），即使在 1.3B 小模型版本上也优于 175B 的 GPT-3。
- InstructGPT 在实际性上有所提升，有害输出生成减少了约 25%（在 RealToxicityPrompts 上评估）。

## 关键方法或创新点

- **三阶段对齐流程**：
  1. **Supervised Fine-Tuning (SFT)**：收集标注者撰写的高质量回答，微调 GPT-3。
  2. **Reward Model (RM) 训练**：收集同一 prompt 的多条模型输出的人工排序，训练一个 6B 的奖励模型来预测人类偏好。
  3. **PPO 强化学习**：用奖励模型作为奖励信号，通过 Proximal Policy Optimization（PPO）继续优化 SFT 模型。
- 训练数据来源：标注者编写的 prompts + OpenAI API 用户提交的真实 prompts。
- 在 TruthfulQA 上事实性显著提升，在 harmful 输出生成方面明显减少。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Concepts/RLHF]]、[[Wiki/Concepts/GPT 系列模型]]、[[Wiki/Topics/大语言模型基础]]
- RLHF 方法成为后续 ChatGPT、GPT-4、Claude、Llama 2/3 等对齐训练的标配。
- 该工作确立了「预训练 → SFT → RM → PPO」的标准对齐管线。

## 局限或注意事项

- RLHF 存在「谄媚倾向」（sycophancy）：模型倾向于给出标注者喜欢的而非正确的答案。
- 奖励模型可能被 hack（reward hacking）。
- 1.3B InstructGPT 的实际性提升来自参数规模的「真实性涌现」而非 RLHF 本身（更大模型基线已经更 factful）。
- 人工偏好标注成本高，且存在标注者间的偏差。
