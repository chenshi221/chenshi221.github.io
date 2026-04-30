---
type: concept
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - 人类反馈强化学习
  - RLHF
  - DPO
  - 对齐训练
sources:
  - '[[Wiki/Sources/InstructGPT]]'
  - '[[Wiki/Sources/Llama 3]]'
  - '[[Wiki/Sources/GPT-4]]'
tags:
  - RLHF
  - alignment
  - PPO
  - DPO
  - RL
confidence: high
---
# RLHF（Reinforcement Learning from Human Feedback）

## 定义

RLHF（人类反馈强化学习）是一种通过人类偏好反馈来训练语言模型的方法，使模型输出更符合人类意图——更有用、更真实、更安全。来源：[[Wiki/Sources/InstructGPT|InstructGPT (OpenAI, 2022)]]

## 为什么需要 RLHF？

- 更大的语言模型并不会自动更「有用」——GPT-3 175B 可能输出不真实、有害或不符合用户意图的内容
- 预训练目标（next-token prediction）与「有用、真实、安全」的人类期望之间存在**不对齐（misalignment）**
- RLHF 提供了一种系统性的方式将人类偏好注入模型行为

## 核心流程（经典三阶段）

### 阶段 1：Supervised Fine-Tuning (SFT)
- 收集人类标注者撰写的 prompt-response 对（高质量示范数据）
- 在预训练模型上做监督微调
- 产出 SFT 模型，初步具备遵循指令的能力

### 阶段 2：Reward Model (RM) 训练
- 对同一个 prompt 生成 K 个不同的模型输出
- 人类标注者对这些输出进行**排序**（更容易和可靠）而非打分
- 训练一个奖励模型（通常 6B 左右）来预测人类偏好排序
- RM 将输出映射为一个标量奖励值

### 阶段 3：RL 微调（PPO）
- 使用 Proximal Policy Optimization (PPO) 算法
- 优化目标：最大化 RM 给予的奖励 + KL 散度惩罚（防止偏离 SFT 模型太远）
- 最终得到 RLHF 对齐的模型（如 InstructGPT）

## InstructGPT 的关键发现

- 1.3B 的 InstructGPT 在人类评估中优于 175B 的原始 GPT-3
- 有害输出减少约 25%（在 RealToxicityPrompts 上）
- 在 TruthfulQA 上事实性显著提升
- 但 RLHF 本身不是「事实性」的来源——更大模型基线已经更真实，PPO 主要改善对齐

## RLHF 的演进与变体

### DPO (Direct Preference Optimization, 2023)
- 来源：[[Wiki/Sources/Llama 3|Llama 3]]
- 直接使用偏好数据优化策略，**跳过奖励模型**和 PPO 阶段
- 更简单、更稳定，Llama 3 采用了 DPO 而非 RLHF
- 优势：不需要显式训练和维护奖励模型

### RLAIF (RL from AI Feedback)
- 使用另一个 LLM 代替人类标注者提供偏好信号（如 Constitutional AI）
- 降低人工标注成本，但可能引入 AI 自身的偏见

### 其他变体
- Rejection Sampling（Llama 3）：从多个生成结果中选择奖励最高者
- RAPO（Aes-R1）：相对-绝对策略优化，用于审美推理

## RLHF 的挑战

1. **奖励 hacking**：模型学会利用 RM 的弱点获取高奖励，而非真正改善质量
2. **谄媚倾向（sycophancy）**：模型偏好给标注者喜欢的答案，而非正确的
3. **标注偏差**：不同文化背景的标注者偏好不同
4. **可扩展性**：高质量人工偏好标注昂贵且难以规模扩大
5. **过度对齐**：模型可能变得过于保守，拒绝回答合法问题

## 相关页面

- [[Wiki/Sources/InstructGPT|InstructGPT 来源页]]
- [[Wiki/Sources/GPT-4|GPT-4]]
- [[Wiki/Sources/Llama 3|Llama 3]]
- [[Wiki/Topics/大语言模型基础|大语言模型基础]]
- [[Wiki/Concepts/GPT 系列模型|GPT 系列模型]]

---

## 深度分析：RLHF 的时代正在过去？

### 1. DPO 正在取代 PPO——不是因为效果更好，而是因为够简单

Llama 3 选择了 DPO 而非经典 RLHF（PPO）。这个选择背后的逻辑不是 DPO 效果更好（两者的最终效果接近），而是 DPO 更简单——不需要训练和维护一个 Reward Model，不需要 PPO 的复杂调参（KL 系数、clipping 范围、value function 的学习率等）。在工业界，「够好 + 简单」几乎总是胜过「更好一点 + 极其复杂」。RLHF 的工程复杂度是它被 DPO 取代的首要原因。

### 2. RLHF 解决的是「对齐」问题，不是「能力」问题

InstructGPT 的 1.3B 超过 GPT-3 175B——这是一个惊人的结果，但需要仔细解读。这不意味着 RLHF 让模型变聪明了（知识、推理能力），而是让它不再胡说八道——输出更有用、更安全、更符合人类期望。RLHF 是「安全帽」，不是「引擎」。把 RLHF 看作是能力的来源是常见误解。

### 3. 推理时代的对齐挑战是全新的

DeepSeek-R1 的推理链完全在模型内部，用户看不到。这意味着传统的 RLHF（基于最终输出的奖励）无法触及推理过程。如果模型在不可见的推理链中学到了有偏见的推理方式（例如用种族刻板印象推理犯罪率），RLHF 无法纠正——因为它只看到最终答案。这是推理模型带来的全新对齐难题，目前任何团队都没有给出系统性解决方案。

### 4. RLHF 的隐式代价：过度对齐

InstructGPT 使有害输出减少约 25%，但同时模型也变得更「谄媚」（sycophancy）——愿意顺着用户的预设走，即使问题本身有误。过度对齐（over-alignment）正在成为一个真实问题：模型变得「太安全」，拒绝回答完全合法的问题（如医学、法律、安全研究相关查询）。Llama 3 在安全训练上选择「轻度处理」正是对此的回应——宁愿保留一些风险，也不让模型变得过度保守。过度对齐可能比对齐不足更难解决，因为一旦模型学会了「拒绝」，就很难让它重新学会「判断」。

### 5. 下一个范式是什么？

DPO 简化了 RLHF 的工程复杂度，但没有解决根本问题——基于人类偏好的优化本质上是教模型「讨好人」而非「求真」。无论是人类标注者的偏好（RLHF）、AI 的偏好（RLAIF）、还是宪法规则（Constitutional AI），都只是把「讨好谁」换成了「遵守谁的规则」。「真正求真」的对齐——让模型的输出与客观事实而非主观偏好对齐——可能是一个与 RLHF 框架完全不同的方向。推理模型的自验证能力（self-verification, self-critique）或许指向了这条路：用模型自己的推理能力来验证自己的输出，而非依赖外部偏好信号。
